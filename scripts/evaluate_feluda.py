"""Frozen pseudo-held-out evaluation on Feluda.

This script does not fit or tune any model/calibration parameter.
It transforms Feluda with the frozen development pipeline, optimizes
breaks under the frozen pacing policy, evaluates catalogue-driven brand
matching, and writes held-out VMAP/debug artifacts.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import wave

import numpy as np
from PIL import Image

from src.audio.indic_asr import IndicConformerASR
from src.audio.speech_safety import speech_context, speech_crosses_boundary
from src.audio.vad import detect_speech
from src.brand_matching.catalog import load_catalogue
from src.brand_matching.ranker import (
    brand_text_descriptor,
    brand_visual_prompt,
    rank_eligible_brands,
)
from src.brand_matching.safety import filter_brands
from src.break_scoring.calibration import EmpiricalCDF
from src.break_scoring.evidence_score import (
    acoustic_pause_margin,
    evidence_aware_break_score,
    text_evidence_reliability,
)
from src.context.clip_text_embeddings import CLIPTextEncoder
from src.context.context_inference import (
    infer_contexts,
    load_taxonomy,
    normalize_vector,
)
from src.context.text_embeddings import MODEL_NAME as TEXT_MODEL_NAME
from src.context.text_embeddings import SemanticTextEncoder
from src.context.visual_embeddings import (
    CLIPVisualEncoder,
    MODEL_NAME as VISUAL_MODEL_NAME,
    context_vector,
)
from src.manifest.debug_manifest import build_debug_manifest
from src.manifest.vmap import write_vmap
from src.optimization.break_optimizer import (
    BreakCandidate,
    PacingPolicy,
    optimize_break_schedule,
    schedule_is_feasible,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT.parent / "hoichoi-assets" / "feluda.mp4"
DEFAULT_OUTPUT = ROOT / "outputs" / "heldout" / "feluda"
DEFAULT_CALIBRATION = ROOT / "configs" / "break_score_calibration.json"
DEFAULT_POLICY = ROOT / "configs" / "pacing.demo.json"
DEFAULT_TAXONOMY = ROOT / "configs" / "context_taxonomy.json"
DEFAULT_BRANDS = ROOT / "configs" / "brands.heldout.json"

GUARD_SECONDS = 0.20
CENTERED_PAUSE_SECONDS = 0.25
ASR_CONTEXT_SECONDS = 10.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {duration}")
    return duration


def ensure_audio(video: Path, wav: Path) -> None:
    if wav.exists():
        return
    wav.parent.mkdir(parents=True, exist_ok=True)
    temp = wav.with_name(wav.stem + ".tmp.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(temp),
        ],
        check=True,
    )
    temp.replace(wav)


def ensure_scene_csv(video: Path, scene_dir: Path) -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scene_dir / f"{video.stem}-Scenes.csv"

    if csv_path.exists():
        return csv_path

    subprocess.run(
        [
            "scenedetect",
            "-i", str(video),
            "detect-adaptive",
            "list-scenes",
            "-o", str(scene_dir),
        ],
        check=True,
    )

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    return csv_path


def load_shots(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        next(handle)
        rows = list(csv.DictReader(handle))

    shots: list[dict] = []

    for row in rows:
        start = float(row["Start Time (seconds)"])
        end = float(row["End Time (seconds)"])
        index = int(row["Scene Number"])

        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError(f"Invalid shot timing in {csv_path}")

        if shots and not math.isclose(
            start,
            shots[-1]["end"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Noncontiguous shot list in {csv_path}")

        shots.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "representative_timestamp": start + (end - start) / 2.0,
            }
        )

    if not shots:
        raise ValueError(f"No shots parsed from {csv_path}")

    expected = list(range(1, len(shots) + 1))
    if [shot["index"] for shot in shots] != expected:
        raise ValueError("Unexpected SceneDetect shot indexes")

    return shots


def build_candidates(shots: list[dict], speech_segments: list[dict]) -> list[dict]:
    candidates: list[dict] = []

    for shot in shots[1:]:
        timestamp = round(float(shot["start"]), 3)

        crosses = speech_crosses_boundary(
            timestamp,
            speech_segments,
            guard_seconds=GUARD_SECONDS,
        )

        context = speech_context(timestamp, speech_segments)

        before = context["seconds_since_speech_end"]
        after = context["seconds_until_speech_start"]

        centered = (
            not crosses
            and not context["inside_detected_speech"]
            and before is not None
            and after is not None
            and before >= CENTERED_PAUSE_SECONDS
            and after >= CENTERED_PAUSE_SECONDS
        )

        candidates.append(
            {
                "timestamp_seconds": timestamp,
                "speech_crosses_boundary": crosses,
                "speech_safe": not crosses,
                "centered_pause": centered,
                "speech_context": context,
            }
        )

    return candidates


def script_quality(text: str) -> tuple[dict[str, int], float]:
    ranges = {
        "bengali": (0x0980, 0x09FF),
        "devanagari": (0x0900, 0x097F),
        "telugu": (0x0C00, 0x0C7F),
        "kannada": (0x0C80, 0x0CFF),
    }
    counts = dict.fromkeys((*ranges, "latin"), 0)
    total = 0

    for character in text:
        if not character.isalpha():
            continue
        total += 1
        for name, (low, high) in ranges.items():
            if low <= ord(character) <= high:
                counts[name] += 1
                break
        if "A" <= character <= "Z" or "a" <= character <= "z":
            counts["latin"] += 1

    ratio = counts["bengali"] / total if total else 0.0
    return counts, ratio


def ensure_clip(source: Path, clip: Path, start: float, end: float) -> None:
    if clip.exists():
        return
    clip.parent.mkdir(parents=True, exist_ok=True)
    temp = clip.with_name(clip.stem + ".tmp.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", str(start), "-i", str(source), "-t", str(end - start),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(temp),
        ],
        check=True,
    )
    temp.replace(clip)


def enrich_asr(
    profile: dict,
    *,
    wav: Path,
    output_root: Path,
    checkpoint_path: Path,
) -> dict:
    with wave.open(str(wav), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16000:
            raise ValueError("Expected 16 kHz mono WAV")
        duration = audio.getnframes() / audio.getframerate()

    # Reuse a prior checkpoint only when the candidate timestamp sequence matches.
    #
    # Important: hydrate the caller-owned profile in place rather than
    # reassigning the local variable. Reassignment would leave main() with
    # the fresh profile lacking cached asr_context fields.
    if checkpoint_path.exists():
        cached = json.loads(
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )

        current_candidates = (
            profile["videos"][0]["candidates"]
        )

        cached_candidates = (
            cached["videos"][0]["candidates"]
        )

        current_times = [
            candidate["timestamp_seconds"]
            for candidate in current_candidates
        ]

        cached_times = [
            candidate["timestamp_seconds"]
            for candidate in cached_candidates
        ]

        if current_times == cached_times:
            for current, previous in zip(
                current_candidates,
                cached_candidates,
                strict=True,
            ):
                if "asr_context" in previous:
                    current["asr_context"] = (
                        previous["asr_context"]
                    )

    candidates = profile["videos"][0]["candidates"]
    safe = [c for c in candidates if c["speech_safe"] is True]

    model = None
    processed_windows = 0
    reused_windows = 0
    inference_seconds = 0.0
    ratios: list[float] = []

    for number, candidate in enumerate(safe, start=1):
        timestamp = float(candidate["timestamp_seconds"])
        context = candidate.get("asr_context") or {
            "window_seconds": ASR_CONTEXT_SECONDS
        }

        for side, start, end in (
            ("left", max(0.0, timestamp - ASR_CONTEXT_SECONDS), timestamp),
            ("right", timestamp, min(duration, timestamp + ASR_CONTEXT_SECONDS)),
        ):
            prior = context.get(side)

            if (
                isinstance(prior, dict)
                and prior.get("start") == start
                and prior.get("end") == end
                and isinstance(prior.get("text"), str)
            ):
                reused_windows += 1
                result = prior
            else:
                if end == start:
                    text = ""
                    elapsed = 0.0
                else:
                    clip = (
                        output_root
                        / "asr-context"
                        / f"{round(timestamp * 1000):09d}_{side}.wav"
                    )
                    ensure_clip(wav, clip, start, end)

                    if model is None:
                        model = IndicConformerASR()

                    started = time.perf_counter()
                    text = model.recognize(clip)
                    elapsed = time.perf_counter() - started
                    inference_seconds += elapsed
                    processed_windows += 1

                counts, ratio = script_quality(text)
                result = {
                    "start": start,
                    "end": end,
                    "text": text,
                    "elapsed_seconds": elapsed,
                    "script_counts": counts,
                    "bengali_script_ratio": ratio,
                }

            context[side] = result

            if result["text"].strip():
                ratios.append(float(result["bengali_script_ratio"]))

        candidate["asr_context"] = context

        if number % 10 == 0:
            write_json(checkpoint_path, profile)
            print(
                f"Feluda ASR: {number}/{len(safe)} speech-safe candidates",
                flush=True,
            )

    write_json(checkpoint_path, profile)

    return {
        "speech_safe_candidates": len(safe),
        "processed_windows": processed_windows,
        "reused_windows": reused_windows,
        "inference_seconds": inference_seconds,
        "nonempty_bengali_ratio_mean": (
            statistics.mean(ratios) if ratios else None
        ),
    }


def enrich_text(profile: dict) -> dict:
    encoder = SemanticTextEncoder()

    candidates = [
        candidate
        for candidate in profile["videos"][0]["candidates"]
        if candidate["speech_safe"] is True
    ]

    texts = list(
        dict.fromkeys(
            candidate["asr_context"][side]["text"]
            for candidate in candidates
            for side in ("left", "right")
            if candidate["asr_context"][side]["text"].strip()
        )
    )

    vectors = (
        dict(
            zip(
                texts,
                encoder.encode(texts, batch_size=32),
                strict=True,
            )
        )
        if texts
        else {}
    )

    available = 0

    for candidate in candidates:
        left = candidate["asr_context"]["left"]["text"]
        right = candidate["asr_context"]["right"]["text"]
        valid = bool(left.strip() and right.strip())

        cosine = (
            float(np.dot(vectors[left], vectors[right]))
            if valid
            else None
        )

        candidate["text_semantics"] = {
            "available": valid,
            "model": TEXT_MODEL_NAME,
            "cosine_similarity": cosine,
            "semantic_change": (1.0 - cosine if valid else None),
            "left_characters": len(left),
            "right_characters": len(right),
        }

        available += int(valid)

    return {
        "speech_safe_candidates": len(candidates),
        "available_text_semantics": available,
        "unavailable_text_semantics": len(candidates) - available,
        "unique_transcripts_encoded": len(texts),
    }


def valid_frame(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def extract_frame(video: Path, path: Path, timestamp: float) -> None:
    if valid_frame(path):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.jpg")

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", str(timestamp), "-i", str(video),
            "-frames:v", "1", "-q:v", "2",
            str(temp),
        ],
        check=True,
    )

    if not valid_frame(temp):
        raise ValueError(f"Invalid extracted frame: {temp}")

    temp.replace(path)


def load_or_encode_shots(
    *,
    video: Path,
    shots: list[dict],
    output_root: Path,
    batch_size: int,
) -> np.ndarray:
    cache = output_root / "visual-cache" / "shot-embeddings.npz"

    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            matrix = data["embeddings"].astype(np.float32)
            indices = data["shot_index"].astype(np.int64)

        if (
            matrix.shape[0] == len(shots)
            and indices.tolist() == [shot["index"] for shot in shots]
            and np.isfinite(matrix).all()
            and np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
        ):
            print("Feluda CLIP: reusing cached shot embeddings", flush=True)
            return matrix

    frame_dir = output_root / "visual-cache" / "frames"
    frame_paths: list[Path] = []

    for number, shot in enumerate(shots, start=1):
        frame = frame_dir / f"shot_{shot['index']:06d}.jpg"
        extract_frame(video, frame, float(shot["representative_timestamp"]))
        frame_paths.append(frame)

        if number % 50 == 0:
            print(
                f"Feluda frames: {number}/{len(shots)} extracted/reused",
                flush=True,
            )

    encoder = CLIPVisualEncoder()
    matrix = encoder.encode(frame_paths, batch_size=batch_size)

    if (
        matrix.shape[0] != len(shots)
        or not np.isfinite(matrix).all()
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Invalid Feluda CLIP embedding matrix")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        model=np.array(VISUAL_MODEL_NAME),
        shot_index=np.asarray([s["index"] for s in shots], dtype=np.int64),
        embeddings=matrix,
        start=np.asarray([s["start"] for s in shots], dtype=np.float64),
        end=np.asarray([s["end"] for s in shots], dtype=np.float64),
        representative_timestamp=np.asarray(
            [s["representative_timestamp"] for s in shots],
            dtype=np.float64,
        ),
    )

    return matrix


def boundary_position(shots: list[dict], timestamp: float) -> int:
    key = round(float(timestamp), 3)
    matches = [
        index
        for index, shot in enumerate(shots)
        if round(float(shot["start"]), 3) == key
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Boundary {timestamp} does not map uniquely to Feluda shot list"
        )

    return matches[0]


def enrich_visual(profile: dict, shots: list[dict], matrix: np.ndarray) -> None:
    candidates = profile["videos"][0]["candidates"]

    for candidate in candidates:
        position = boundary_position(shots, candidate["timestamp_seconds"])

        if not 0 < position < len(shots):
            raise ValueError("Expected internal visual boundary")

        previous = shots[position - 1]
        following = shots[position]

        left_start = max(0, position - 2)
        right_end = min(len(shots), position + 2)

        left = context_vector(matrix[left_start:position])
        right = context_vector(matrix[position:right_end])

        immediate = float(
            np.dot(matrix[position - 1], matrix[position])
        )

        contextual = float(np.dot(left, right))

        candidate["visual_semantics"] = {
            "model": VISUAL_MODEL_NAME,
            "previous_shot": previous,
            "next_shot": following,
            "immediate_cosine_similarity": immediate,
            "immediate_change": 1.0 - immediate,
            "left_context_shots": [
                shot["index"] for shot in shots[left_start:position]
            ],
            "right_context_shots": [
                shot["index"] for shot in shots[position:right_end]
            ],
            "context_cosine_similarity": contextual,
            "context_change": 1.0 - contextual,
        }


def score_frozen(profile: dict, calibration_path: Path) -> dict:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    if "feluda" in " ".join(calibration.get("development_videos", [])).lower():
        raise ValueError("Frozen calibration unexpectedly contains Feluda")

    acoustic_cdf = EmpiricalCDF.from_dict(calibration["acoustic"])
    visual_cdf = EmpiricalCDF.from_dict(calibration["visual"])
    text_cdf = EmpiricalCDF.from_dict(calibration["text"])
    reference_text_length = float(calibration["reference_text_length"])

    candidates = profile["videos"][0]["candidates"]
    eligible_count = 0
    scored_count = 0
    scores: list[float] = []

    for candidate in candidates:
        reasons: list[str] = []

        if candidate.get("speech_crosses_boundary") is True:
            reasons.append("speech_crosses_boundary")

        speech = candidate.get("speech_context") or {}

        if speech.get("inside_detected_speech") is True:
            reasons.append("inside_detected_speech")

        eligible = not reasons

        break_score = {
            "ad_eligible": eligible,
            "rejection_reasons": reasons,
            "scorable": False,
            "acoustic": {
                "raw_pause_margin": None,
                "percentile": None,
            },
            "visual": {
                "raw_context_change": None,
                "percentile": None,
            },
            "text": {
                "available": False,
                "raw_semantic_change": None,
                "effective_length": 0.0,
                "reliability": 0.0,
                "percentile": None,
            },
            "eabs": None,
        }

        candidate["break_score"] = break_score

        if not eligible:
            continue

        eligible_count += 1

        acoustic = acoustic_pause_margin(
            speech.get("seconds_since_speech_end"),
            speech.get("seconds_until_speech_start"),
        )

        visual = (candidate.get("visual_semantics") or {}).get(
            "context_change"
        )

        break_score["acoustic"]["raw_pause_margin"] = acoustic
        break_score["visual"]["raw_context_change"] = visual

        if acoustic is None or visual is None:
            continue

        acoustic_percentile = acoustic_cdf.transform(float(acoustic))
        visual_percentile = visual_cdf.transform(float(visual))

        break_score["acoustic"]["percentile"] = acoustic_percentile
        break_score["visual"]["percentile"] = visual_percentile

        text_percentile = None
        reliability = 0.0
        semantics = candidate.get("text_semantics") or {}

        valid_text = (
            semantics.get("available") is True
            and semantics.get("semantic_change") is not None
            and int(semantics.get("left_characters", 0)) > 0
            and int(semantics.get("right_characters", 0)) > 0
        )

        if valid_text:
            raw_text = float(semantics["semantic_change"])
            effective_length, reliability = text_evidence_reliability(
                int(semantics["left_characters"]),
                int(semantics["right_characters"]),
                reference_text_length,
            )
            text_percentile = text_cdf.transform(raw_text)

            break_score["text"] = {
                "available": True,
                "raw_semantic_change": raw_text,
                "effective_length": effective_length,
                "reliability": reliability,
                "percentile": text_percentile,
            }

        score = evidence_aware_break_score(
            acoustic_percentile,
            visual_percentile,
            text_percentile=text_percentile,
            text_reliability=reliability,
        )

        break_score["scorable"] = True
        break_score["eabs"] = score
        scored_count += 1
        scores.append(score)

    return {
        "calibration_method": calibration["method"],
        "calibration_development_videos": calibration["development_videos"],
        "eligible_candidates": eligible_count,
        "scored_candidates": scored_count,
        "eabs_mean": float(np.mean(scores)) if scores else None,
        "eabs_median": float(np.median(scores)) if scores else None,
        "eabs_max": float(np.max(scores)) if scores else None,
    }


def optimize_frozen(
    profile: dict,
    *,
    duration: float,
    policy_path: Path,
) -> tuple[dict, dict]:
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = PacingPolicy.from_dict(policy_payload)

    source = profile["videos"][0]["candidates"]

    candidates: list[BreakCandidate] = []

    for index, item in enumerate(source):
        score = item.get("break_score") or {}

        if score.get("ad_eligible") is not True:
            continue

        if score.get("scorable") is not True:
            continue

        candidates.append(
            BreakCandidate(
                source_index=index,
                timestamp=float(item["timestamp_seconds"]),
                score=float(score["eabs"]),
            )
        )

    result = optimize_break_schedule(
        candidates,
        video_duration=duration,
        policy=policy,
    )

    selected_records = []

    for item in result["selected"]:
        source_item = source[item.source_index]

        selected_records.append(
            {
                "source_candidate_index": item.source_index,
                "timestamp_seconds": item.timestamp,
                "eabs": item.score,
                "quality_utility": item.score - 50.0,
                "centered_pause": source_item.get("centered_pause"),
                "break_score": source_item.get("break_score"),
                "visual_context_change": (
                    source_item.get("visual_semantics") or {}
                ).get("context_change"),
                "text_semantic_change": (
                    source_item.get("text_semantics") or {}
                ).get("semantic_change"),
            }
        )

    times = [row["timestamp_seconds"] for row in selected_records]

    if not schedule_is_feasible(
        times,
        video_duration=duration,
        policy=policy,
    ):
        raise RuntimeError("Frozen held-out schedule violates pacing policy")

    optimized = {
        "optimizer": "scipy.optimize.milp / HiGHS",
        "objective": "maximize sum(EABS - 50) for positive-quality break opportunities",
        "eabs_neutral_baseline": 50.0,
        "policy": policy_payload,
        "videos": [
            {
                "video": profile["videos"][0]["video"],
                "duration_seconds": duration,
                "input_eabs_candidates": len(candidates),
                "pacing_candidate_count": result["pacing_candidate_count"],
                "quality_candidate_count": result["quality_candidate_count"],
                "rejected_first_break_buffer": result[
                    "rejected_first_break_buffer"
                ],
                "rejected_end_buffer": result["rejected_end_buffer"],
                "rejected_below_neutral_quality": result[
                    "rejected_below_neutral_quality"
                ],
                "selected_break_count": len(selected_records),
                "selected_breaks": selected_records,
                "selected_eabs_sum": result["eabs_sum"],
                "quality_utility": result["quality_utility"],
            }
        ],
        "summary": {
            "video_count": 1,
            "selected_breaks_total": len(selected_records),
            "all_constraints_satisfied": True,
        },
    }

    return optimized, result


def scene_text(candidate: dict) -> str:
    asr = candidate.get("asr_context") or {}
    left = str((asr.get("left") or {}).get("text", "")).strip()
    right = str((asr.get("right") or {}).get("text", "")).strip()
    return " ".join(value for value in (left, right) if value)


def scene_visual_vector(
    candidate: dict,
    shots_by_index: dict[int, np.ndarray],
) -> np.ndarray | None:
    visual = candidate.get("visual_semantics") or {}

    indexes = list(visual.get("left_context_shots") or []) + list(
        visual.get("right_context_shots") or []
    )

    vectors = [
        shots_by_index[int(index)]
        for index in indexes
        if int(index) in shots_by_index
    ]

    if not vectors:
        return None

    return normalize_vector(np.mean(np.stack(vectors), axis=0))


def vector_map(ids: list[str], matrix: np.ndarray) -> dict[str, np.ndarray]:
    if len(ids) != len(matrix):
        raise ValueError("Embedding ID/vector count mismatch")

    return {
        item_id: normalize_vector(vector)
        for item_id, vector in zip(ids, matrix, strict=True)
    }


def match_brands_frozen(
    profile: dict,
    optimized: dict,
    *,
    taxonomy_path: Path,
    brands_path: Path,
    shot_matrix: np.ndarray,
    shots: list[dict],
) -> dict:
    taxonomy = load_taxonomy(taxonomy_path)

    allowed = {entry.context_id for entry in taxonomy.contexts}
    catalogue = load_catalogue(
        brands_path,
        allowed_contexts=allowed,
    )

    if len(catalogue.brands) != 9:
        raise ValueError(
            f"Held-out catalogue must contain exactly 9 brands, got {len(catalogue.brands)}"
        )

    text_encoder = SemanticTextEncoder()
    clip_text_encoder = CLIPTextEncoder()

    taxonomy_ids = [entry.context_id for entry in taxonomy.contexts]

    taxonomy_text = vector_map(
        taxonomy_ids,
        text_encoder.encode(
            [entry.text_description for entry in taxonomy.contexts]
        ),
    )

    taxonomy_visual = vector_map(
        taxonomy_ids,
        clip_text_encoder.encode(
            [entry.visual_prompt for entry in taxonomy.contexts]
        ),
    )

    brand_ids = [brand.brand_id for brand in catalogue.brands]

    brand_text = vector_map(
        brand_ids,
        text_encoder.encode(
            [brand_text_descriptor(brand) for brand in catalogue.brands]
        ),
    )

    brand_visual = vector_map(
        brand_ids,
        clip_text_encoder.encode(
            [brand_visual_prompt(brand) for brand in catalogue.brands]
        ),
    )

    shots_by_index = {
        shot["index"]: normalize_vector(vector)
        for shot, vector in zip(shots, shot_matrix, strict=True)
    }

    candidates = profile["videos"][0]["candidates"]

    selected_brand_counts: Counter[str] = Counter()
    blocked_decisions = 0
    hard_safety_count = 0
    negative_violations = 0
    no_fill_breaks = 0
    ninth_rank_positions: list[int] = []
    ninth_selected = 0

    rows: list[dict] = []

    selected_breaks = optimized["videos"][0]["selected_breaks"]

    for selected in selected_breaks:
        candidate = candidates[int(selected["source_candidate_index"])]
        text = scene_text(candidate)

        text_vector = (
            text_encoder.encode([text])[0]
            if text
            else None
        )

        visual_vector = scene_visual_vector(
            candidate,
            shots_by_index,
        )

        reliability = float(
            (
                candidate.get("break_score")
                or {}
            ).get("text", {}).get("reliability", 0.0)
            or 0.0
        )

        context = infer_contexts(
            taxonomy=taxonomy,
            scene_text=text,
            scene_text_embedding=text_vector,
            scene_visual_embedding=visual_vector,
            taxonomy_text_embeddings=taxonomy_text,
            taxonomy_visual_embeddings=taxonomy_visual,
            text_reliability=reliability,
        )

        hard_safety = set(context["hard_safety_contexts"])
        cues = set(context["context_cues"])

        hard_safety_count += len(hard_safety)

        eligible, blocked = filter_brands(
            catalogue.brands,
            hard_safety,
        )

        blocked_decisions += len(blocked)

        ranked = rank_eligible_brands(
            brands=eligible,
            detected_contexts=cues,
            scene_text_embedding=text_vector,
            scene_visual_embedding=visual_vector,
            brand_text_embeddings=brand_text,
            brand_visual_embeddings=brand_visual,
        )

        selected_brand = ranked[0] if ranked else None

        if selected_brand is None:
            # Fail closed: if every advertiser is blocked by hard
            # safety evidence, deliver no ad at this scheduled break.
            no_fill_breaks += 1
        else:
            selected_brand_counts[selected_brand["brand_id"]] += 1

            blocked_ids = {row["brand_id"] for row in blocked}

            if selected_brand["brand_id"] in blocked_ids:
                negative_violations += 1
                raise RuntimeError(
                    "Selected brand appears in hard-block set"
                )

            if selected_brand["brand_id"] == "nivara_books":
                ninth_selected += 1

        ninth_positions = [
            position
            for position, brand in enumerate(ranked, start=1)
            if brand["brand_id"] == "nivara_books"
        ]
        if ninth_positions:
            ninth_rank_positions.append(ninth_positions[0])

        rows.append(
            {
                "video": profile["videos"][0]["video"],
                "timestamp_seconds": selected["timestamp_seconds"],
                "eabs": selected["eabs"],
                "quality_utility": selected["quality_utility"],
                "scene_text": text,
                "context": context,
                "blocked_brands": blocked,
                "eligible_brand_ids": [brand.brand_id for brand in eligible],
                "ranked_eligible_brands": ranked,
                "selected_brand": selected_brand,
            }
        )

    return {
        "catalogue_name": catalogue.catalogue_name,
        "catalogue_status": catalogue.catalogue_status,
        "taxonomy_version": taxonomy.version,
        "context_method": taxonomy.method,
        "matching_method": (
            "hard negative-context filter, then equal-rank fusion of "
            "multilingual-E5 text compatibility and CLIP visual-text compatibility"
        ),
        "breaks": rows,
        "summary": {
            "break_count": len(rows),
            "delivered_ad_count": len(rows) - no_fill_breaks,
            "no_fill_break_count": no_fill_breaks,
            "brand_count": len(catalogue.brands),
            "blocked_brand_decisions": blocked_decisions,
            "hard_safety_context_count": hard_safety_count,
            "negative_context_violations": negative_violations,
            "selected_brand_counts": dict(sorted(selected_brand_counts.items())),
            "all_selected_brands_safe": negative_violations == 0,
            "catalogue_driven": True,
            "title_specific_brand_logic": False,
            "feluda_processed": True,
            "unseen_ninth_brand_id": "nivara_books",
            "unseen_ninth_brand_ranked_breaks": len(ninth_rank_positions),
            "unseen_ninth_brand_selected_breaks": ninth_selected,
            "unseen_ninth_brand_best_rank": (
                min(ninth_rank_positions) if ninth_rank_positions else None
            ),
        },
    }



def build_delivery_subset(
    optimized: dict,
    brand_matches: dict,
) -> tuple[dict, dict]:
    """
    Return VMAP/debug-manifest inputs containing only safely fillable ads.

    The original optimized schedule and held-out brand-match report remain
    unchanged so no-fill decisions stay visible in evaluation evidence.
    """
    scheduled = optimized["videos"][0]["selected_breaks"]
    matches = brand_matches["breaks"]

    if len(scheduled) != len(matches):
        raise ValueError(
            "Scheduled-break / brand-match count mismatch"
        )

    delivered_breaks = []
    delivered_matches = []

    for break_item, match in zip(
        scheduled,
        matches,
        strict=True,
    ):
        if match.get("selected_brand") is None:
            continue

        delivered_breaks.append(break_item)
        delivered_matches.append(match)

    delivery_video = {
        **optimized["videos"][0],
        "selected_breaks": delivered_breaks,
        "selected_break_count": len(delivered_breaks),
    }

    delivery_optimized = {
        **optimized,
        "videos": [delivery_video],
        "summary": {
            **optimized.get("summary", {}),
            "selected_breaks_total": len(delivered_breaks),
        },
    }

    delivery_matches = {
        **brand_matches,
        "breaks": delivered_matches,
        "summary": {
            **brand_matches["summary"],
            "break_count": len(delivered_matches),
        },
    }

    return delivery_optimized, delivery_matches

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--brands", type=Path, default=DEFAULT_BRANDS)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    args = parser.parse_args()

    started = time.perf_counter()

    video = args.video.resolve()
    output = args.output.resolve()

    if video.stem.lower() != "feluda":
        raise ValueError(
            "Stage 9 is intentionally locked to the pseudo-held-out Feluda asset"
        )

    if not video.exists():
        raise FileNotFoundError(video)

    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)

    if not args.brands.exists():
        raise FileNotFoundError(args.brands)

    calibration_hash_before = sha256(args.calibration)
    policy_hash_before = sha256(args.policy)
    taxonomy_hash_before = sha256(args.taxonomy)

    duration = probe_duration(video)

    audio = output / "audio.wav"
    ensure_audio(video, audio)

    vad_path = output / "vad.json"
    if vad_path.exists():
        speech_segments = json.loads(
            vad_path.read_text(encoding="utf-8")
        )["speech_segments"]
    else:
        speech_segments = detect_speech(audio)
        write_json(
            vad_path,
            {"speech_segments": speech_segments},
        )

    scene_csv = ensure_scene_csv(
        video,
        output / "scenes",
    )

    shots = load_shots(scene_csv)
    candidates = build_candidates(shots, speech_segments)

    profile = {
        "held_out_evaluation": True,
        "guard_seconds": GUARD_SECONDS,
        "centered_pause_seconds": CENTERED_PAUSE_SECONDS,
        "videos": [
            {
                "video": video.name,
                "visual_boundaries": len(candidates),
                "speech_safe": sum(
                    int(candidate["speech_safe"])
                    for candidate in candidates
                ),
                "speech_unsafe": sum(
                    int(not candidate["speech_safe"])
                    for candidate in candidates
                ),
                "centered_pause": sum(
                    int(candidate["centered_pause"])
                    for candidate in candidates
                ),
                "speech_safe_ratio": (
                    sum(int(c["speech_safe"]) for c in candidates)
                    / len(candidates)
                    if candidates
                    else 0.0
                ),
                "candidates": candidates,
            }
        ],
    }

    asr_summary = enrich_asr(
        profile,
        wav=audio,
        output_root=output,
        checkpoint_path=output / "boundary-profile-asr.json",
    )

    text_summary = enrich_text(profile)

    shot_matrix = load_or_encode_shots(
        video=video,
        shots=shots,
        output_root=output,
        batch_size=args.clip_batch_size,
    )

    enrich_visual(profile, shots, shot_matrix)

    score_summary = score_frozen(
        profile,
        args.calibration,
    )

    write_json(
        output / "boundary-profile-scored.json",
        profile,
    )

    optimized, optimizer_result = optimize_frozen(
        profile,
        duration=duration,
        policy_path=args.policy,
    )

    write_json(
        output / "optimized-breaks.json",
        optimized,
    )

    brand_matches = match_brands_frozen(
        profile,
        optimized,
        taxonomy_path=args.taxonomy,
        brands_path=args.brands,
        shot_matrix=shot_matrix,
        shots=shots,
    )

    write_json(
        output / "brand-matches.json",
        brand_matches,
    )

    delivery_optimized, delivery_brand_matches = (
        build_delivery_subset(
            optimized,
            brand_matches,
        )
    )

    debug_manifest = build_debug_manifest(
        optimized=delivery_optimized,
        brand_matches=delivery_brand_matches,
    )

    manifests = output / "manifests"

    write_json(
        manifests / "debug-manifest.json",
        debug_manifest,
    )

    write_vmap(
        path=manifests / "vmap" / "feluda.vmap.xml",
        video_name=video.name,
        breaks=debug_manifest["videos"][0]["breaks"],
        ad_duration_seconds=float(
            optimized["policy"]["nominal_ad_duration_seconds"]
        ),
    )

    calibration_hash_after = sha256(args.calibration)
    policy_hash_after = sha256(args.policy)
    taxonomy_hash_after = sha256(args.taxonomy)

    if calibration_hash_after != calibration_hash_before:
        raise RuntimeError("Frozen calibration changed during held-out evaluation")

    if policy_hash_after != policy_hash_before:
        raise RuntimeError("Frozen pacing policy changed during held-out evaluation")

    if taxonomy_hash_after != taxonomy_hash_before:
        raise RuntimeError("Frozen context taxonomy changed during held-out evaluation")

    selected = optimized["videos"][0]["selected_breaks"]
    selected_times = [float(row["timestamp_seconds"]) for row in selected]

    gaps = [
        right - left
        for left, right in zip(selected_times, selected_times[1:])
    ]

    selected_rows = []

    for row, match in zip(
        selected,
        brand_matches["breaks"],
        strict=True,
    ):
        selected_brand = match.get("selected_brand")

        selected_rows.append(
            {
                "timestamp_seconds": row["timestamp_seconds"],
                "eabs": row["eabs"],
                "quality_utility": row["quality_utility"],
                "delivery_status": (
                    "filled"
                    if selected_brand is not None
                    else "no_fill_brand_safety"
                ),
                "brand_id": (
                    selected_brand["brand_id"]
                    if selected_brand is not None
                    else None
                ),
                "brand_name": (
                    selected_brand["name"]
                    if selected_brand is not None
                    else None
                ),
                "context_cues": match["context"]["context_cues"],
                "hard_safety_contexts": (
                    match["context"]["hard_safety_contexts"]
                ),
                "blocked_brand_count": len(match["blocked_brands"]),
            }
        )

    report = {
        "evaluation": "pseudo-held-out",
        "video": video.name,
        "duration_seconds": duration,
        "development_tuning_after_unlock": False,
        "post_unlock_engineering_fix": (
            "Added fail-closed no-fill behavior when all catalogue "
            "brands are hard safety-blocked. No model, calibration, "
            "threshold, taxonomy, ranking weight, or pacing parameter "
            "was changed."
        ),
        "frozen_assets": {
            "calibration_path": str(args.calibration),
            "calibration_sha256": calibration_hash_before,
            "policy_path": str(args.policy),
            "policy_sha256": policy_hash_before,
            "taxonomy_path": str(args.taxonomy),
            "taxonomy_sha256": taxonomy_hash_before,
        },
        "perception": {
            "visual_shots": len(shots),
            "visual_boundaries": len(candidates),
            "speech_safe_boundaries": profile["videos"][0]["speech_safe"],
            "speech_unsafe_boundaries": profile["videos"][0]["speech_unsafe"],
            "centered_pauses": profile["videos"][0]["centered_pause"],
            "asr": asr_summary,
            "text_semantics": text_summary,
        },
        "scoring": score_summary,
        "optimization": {
            "selected_break_count": len(selected),
            "quality_candidate_count": optimizer_result[
                "quality_candidate_count"
            ],
            "rejected_below_neutral_quality": optimizer_result[
                "rejected_below_neutral_quality"
            ],
            "quality_utility": optimizer_result["quality_utility"],
            "eabs_sum": optimizer_result["eabs_sum"],
            "minimum_selected_gap_seconds": min(gaps) if gaps else None,
            "ad_load_fraction": (
                len(selected)
                * float(optimized["policy"]["nominal_ad_duration_seconds"])
                / duration
            ),
            "all_constraints_satisfied": True,
        },
        "brand_matching": brand_matches["summary"],
        "selected_breaks": selected_rows,
        "artifacts": {
            "scored_profile": str(output / "boundary-profile-scored.json"),
            "optimized_breaks": str(output / "optimized-breaks.json"),
            "brand_matches": str(output / "brand-matches.json"),
            "debug_manifest": str(manifests / "debug-manifest.json"),
            "vmap": str(manifests / "vmap" / "feluda.vmap.xml"),
        },
        "runtime_seconds": time.perf_counter() - started,
    }

    write_json(
        output / "heldout-report.json",
        report,
    )

    print(
        json.dumps(
            {
                "video": report["video"],
                "duration_seconds": report["duration_seconds"],
                "visual_boundaries": report["perception"]["visual_boundaries"],
                "speech_safe_boundaries": report["perception"][
                    "speech_safe_boundaries"
                ],
                "scored_candidates": report["scoring"]["scored_candidates"],
                "selected_break_count": report["optimization"][
                    "selected_break_count"
                ],
                "delivered_ad_count": report["brand_matching"][
                    "delivered_ad_count"
                ],
                "no_fill_break_count": report["brand_matching"][
                    "no_fill_break_count"
                ],
                "selected_breaks": report["selected_breaks"],
                "minimum_selected_gap_seconds": report["optimization"][
                    "minimum_selected_gap_seconds"
                ],
                "ad_load_fraction": report["optimization"]["ad_load_fraction"],
                "negative_context_violations": report["brand_matching"][
                    "negative_context_violations"
                ],
                "all_selected_brands_safe": report["brand_matching"][
                    "all_selected_brands_safe"
                ],
                "unseen_ninth_brand_ranked_breaks": report["brand_matching"][
                    "unseen_ninth_brand_ranked_breaks"
                ],
                "unseen_ninth_brand_selected_breaks": report["brand_matching"][
                    "unseen_ninth_brand_selected_breaks"
                ],
                "unseen_ninth_brand_best_rank": report["brand_matching"][
                    "unseen_ninth_brand_best_rank"
                ],
                "frozen_calibration_unchanged": (
                    calibration_hash_before == calibration_hash_after
                ),
                "frozen_policy_unchanged": (
                    policy_hash_before == policy_hash_after
                ),
                "frozen_taxonomy_unchanged": (
                    taxonomy_hash_before == taxonomy_hash_after
                ),
                "runtime_seconds": report["runtime_seconds"],
                "report": str(output / "heldout-report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
