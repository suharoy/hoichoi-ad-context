"""Immediate and two-shot contextual visual features, without break selection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.stats import pearsonr, spearmanr

from scripts.embed_shots import DEV, STEMS, cache_metadata, load_cache, load_shots, paths_for
from src.context.visual_embeddings import MODEL_NAME, context_vector


INPUT = DEV / "boundary-profile-semantic.json"
OUTPUT = DEV / "boundary-profile-multimodal.json"


def boundary_position(shots: list[dict], timestamp: float) -> int:
    # The profile serialized the same CSV second values to three decimals.
    key = round(timestamp, 3)
    matches = [i for i, shot in enumerate(shots) if round(shot["start"], 3) == key]
    if key == round(shots[-1]["end"], 3):
        matches.append(len(shots))
    if len(matches) != 1:
        raise ValueError(f"Boundary {timestamp} does not map uniquely to CSV shot times")
    position = matches[0]
    if 0 < position < len(shots) and round(shots[position - 1]["end"], 3) != key:
        raise ValueError(f"Nonadjacent shots at {timestamp}")
    return position


def enrich_video(video: dict, shots: list[dict], vectors: np.ndarray) -> dict:
    if Path(video["video"]).stem not in STEMS:
        raise ValueError("Only development videos may receive visual enrichment")
    if len(vectors) != len(shots) or not np.isfinite(vectors).all():
        raise ValueError("Missing or invalid shot embeddings")
    enriched = copy.deepcopy(video)
    for candidate in enriched["candidates"]:
        position = boundary_position(shots, candidate["timestamp_seconds"])
        previous = shots[position - 1] if position else None
        following = shots[position] if position < len(shots) else None
        left_start, right_end = max(0, position - 2), min(len(shots), position + 2)
        left = context_vector(vectors[left_start:position])
        right = context_vector(vectors[position:right_end])
        immediate = float(np.dot(vectors[position - 1], vectors[position])) if previous and following else None
        context = float(np.dot(left, right)) if left is not None and right is not None else None
        candidate["visual_semantics"] = {
            "model": MODEL_NAME, "previous_shot": previous, "next_shot": following,
            "immediate_cosine_similarity": immediate,
            "immediate_change": 1.0 - immediate if immediate is not None else None,
            "left_context_shots": [s["index"] for s in shots[left_start:position]],
            "right_context_shots": [s["index"] for s in shots[position:right_end]],
            "context_cosine_similarity": context,
            "context_change": 1.0 - context if context is not None else None,
        }
    return enriched


def distribution(values: list[float]) -> dict:
    return dict(zip(("mean", "median", "min", "max", "p10", "p25", "p75", "p90"), map(float, (
        np.mean(values), np.median(values), np.min(values), np.max(values),
        *np.percentile(values, [10, 25, 75, 90]),
    )), strict=True))


def diagnostics(profile: dict) -> dict:
    rows = []
    for video in profile["videos"]:
        if Path(video["video"]).stem not in STEMS:
            continue
        for candidate in video["candidates"]:
            if candidate["speech_safe"] is not True:
                continue
            visual = candidate["visual_semantics"]
            text = candidate.get("text_semantics", {})
            previous, following = visual["previous_shot"], visual["next_shot"]
            rows.append({
                "video": video["video"], "timestamp": candidate["timestamp_seconds"],
                "centered_pause": candidate["centered_pause"],
                "inside_detected_speech": candidate["speech_context"]["inside_detected_speech"],
                "text_semantic_change": text.get("semantic_change") if text.get("available") else None,
                "previous_shot_duration": previous["end"] - previous["start"] if previous else None,
                "next_shot_duration": following["end"] - following["start"] if following else None,
                "immediate_change": visual["immediate_change"], "context_change": visual["context_change"],
            })
    if len(rows) != 317:
        raise ValueError(f"Expected 317 speech-safe diagnostic candidates, got {len(rows)}")
    immediate = [r["immediate_change"] for r in rows]
    context = [r["context_change"] for r in rows]
    return {"speech_safe_candidates": len(rows), "immediate_change": distribution(immediate),
            "context_change": distribution(context),
            "pearson": float(pearsonr(immediate, context).statistic),
            "spearman": float(spearmanr(immediate, context).statistic),
            "highest_context_change": sorted(rows, key=lambda r: -r["context_change"])[:10],
            "lowest_context_change": sorted(rows, key=lambda r: r["context_change"])[:10]}


def contact_sheet(name: str, rows: list[dict], profile: dict) -> None:
    canvas = Image.new("RGB", (800, 270 * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    videos = {v["video"]: v for v in profile["videos"]}
    for row_number, row in enumerate(rows):
        y = row_number * 270
        label = f"{row['video']} | {row['timestamp']}s | context_change={row['context_change']:.6f}"
        draw.text((8, y + 5), label, fill="black")
        candidate = next(c for c in videos[row["video"]]["candidates"] if c["timestamp_seconds"] == row["timestamp"])
        stem = Path(row["video"]).stem
        for column, side in enumerate(("previous_shot", "next_shot")):
            shot = candidate["visual_semantics"][side]
            frame = DEV / "visual-cache" / stem / "frames" / f"shot_{shot['index']:06d}.jpg"
            draw.text((column * 400 + 8, y + 22), f"{side} #{shot['index']}", fill="black")
            with Image.open(frame) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (392, 224))
                canvas.paste(thumbnail, (column * 400 + 4, y + 42))
    canvas.save(DEV / "visual-diagnostics" / name)


def main() -> None:
    source_bytes = INPUT.read_bytes()
    original = json.loads(source_bytes)
    enriched = copy.deepcopy(original)
    seen, boundary_count = set(), 0
    for position, video in enumerate(enriched["videos"]):
        stem = Path(video["video"]).stem
        if stem not in STEMS:
            continue
        source, csv_path, cache = paths_for(stem)
        shots = load_shots(csv_path)
        vectors = load_cache(cache / "shot-embeddings.npz", shots, cache_metadata(source, csv_path))
        matrix = np.stack([vectors[s["index"]] for s in shots])
        for candidate in video["candidates"]:
            boundary = boundary_position(shots, candidate["timestamp_seconds"])
            if not 0 < boundary < len(shots):
                raise ValueError("Development profile contains a noninternal boundary")
        enriched["videos"][position] = enrich_video(video, shots, matrix)
        boundary_count += len(video["candidates"])
        seen.add(stem)
    if seen != set(STEMS):
        raise ValueError("Expected all five development videos")
    stripped = copy.deepcopy(enriched)
    for video in stripped["videos"]:
        if Path(video["video"]).stem in STEMS:
            for candidate in video["candidates"]:
                candidate.pop("visual_semantics")
    if stripped != original or INPUT.read_bytes() != source_bytes:
        raise RuntimeError("Prior fields or semantic source changed")
    report = diagnostics(enriched)
    report["validation"] = {"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "source_unchanged": True, "prior_fields_preserved": True,
                            "development_videos": sorted(seen), "internal_boundaries_mapped": boundary_count,
                            "feluda_visual_enrichment": 0}
    # allow_nan=False rejects nonfinite numbers throughout the output.
    output_text = json.dumps(enriched, ensure_ascii=False, indent=2, allow_nan=False)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(output_text, encoding="utf-8")
    temporary.replace(OUTPUT)
    (DEV / "visual-diagnostics" / "transition-summary.json").write_text(report_text, encoding="utf-8")
    contact_sheet("highest-context-change.jpg", report["highest_context_change"], enriched)
    contact_sheet("lowest-context-change.jpg", report["lowest_context_change"], enriched)
    print(report_text, flush=True)
    print(f"Saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
