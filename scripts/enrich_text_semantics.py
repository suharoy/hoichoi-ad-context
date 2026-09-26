"""Extract text semantic change features without selecting or scoring ad breaks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.context.text_embeddings import MODEL_NAME, SemanticTextEncoder


DEV = Path(__file__).resolve().parents[1] / "outputs/dev"
INPUT = DEV / "boundary-profile-asr.json"
OUTPUT = DEV / "boundary-profile-semantic.json"
DIAGNOSTICS = DEV / "boundary-profile-semantic-diagnostics.json"
DEVELOPMENT_STEMS = {
    "bhojon_bilashi", "indubala_bhaater_hotel", "mandaar", "mohanagar", "money_honey",
}
BATCH_SIZE = 32
SANITY_PAIRS = {
    "A": ("আমি আজ কলকাতায় এসেছি", "আজ আমি কলকাতা শহরে এসেছি"),
    "B": ("আমি আজ কলকাতায় এসেছি", "খুনের তদন্ত শুরু হয়েছে"),
    "C": ("এখানকার মানুষজন চলাফেরা খাওয়া দাওয়া", "আমি আপনাদের এখানকার খাবারের কথা জানাতে চাই"),
    "D": ("এখানকার মানুষজন চলাফেরা খাওয়া দাওয়া", "একটি পুরনো খুনের রহস্যের তদন্ত চলছে"),
}


def development_videos(profile: dict) -> list[dict]:
    return [video for video in profile["videos"] if Path(video["video"]).stem in DEVELOPMENT_STEMS]


def sanity_check(encoder: SemanticTextEncoder) -> dict:
    texts = list(dict.fromkeys(text for pair in SANITY_PAIRS.values() for text in pair))
    vectors = dict(zip(texts, encoder.encode(texts, batch_size=BATCH_SIZE), strict=True))
    similarities = {
        name: float(np.dot(vectors[left], vectors[right]))
        for name, (left, right) in SANITY_PAIRS.items()
    }
    print("Sanity-pair cosine similarities:", flush=True)
    print(json.dumps(similarities, indent=2), flush=True)
    if not (similarities["A"] > similarities["B"] and similarities["C"] > similarities["D"]):
        raise RuntimeError("Bengali sanity ordering failed; corpus was not processed")
    return similarities


def enrich(profile: dict, encoder: SemanticTextEncoder) -> tuple[dict, int]:
    enriched = copy.deepcopy(profile)
    candidates = [
        candidate for video in development_videos(enriched)
        for candidate in video["candidates"] if candidate["speech_safe"] is True
    ]
    # Deduplicate exact strings; no normalization or changes to stored transcripts.
    texts = list(dict.fromkeys(
        candidate["asr_context"][side]["text"]
        for candidate in candidates for side in ("left", "right")
        if candidate["asr_context"][side]["text"].strip()
    ))
    print(f"Encoding {len(texts)} unique nonempty corpus transcripts", flush=True)
    vectors = (
        dict(zip(texts, encoder.encode(texts, batch_size=BATCH_SIZE), strict=True))
        if texts else {}
    )
    for candidate in candidates:
        left = candidate["asr_context"]["left"]["text"]
        right = candidate["asr_context"]["right"]["text"]
        available = bool(left.strip() and right.strip())
        cosine = float(np.dot(vectors[left], vectors[right])) if available else None
        candidate["text_semantics"] = {
            "available": available, "model": MODEL_NAME,
            "cosine_similarity": cosine,
            "semantic_change": 1.0 - cosine if available else None,
            "left_characters": len(left), "right_characters": len(right),
        }
    return enriched, len(texts)


def distribution(values: list[float]) -> dict:
    keys = ("mean", "median", "min", "max", "p10", "p25", "p75", "p90")
    if not values:
        return dict.fromkeys(keys)
    return dict(zip(keys, map(float, (
        np.mean(values), np.median(values), np.min(values), np.max(values),
        *np.percentile(values, [10, 25, 75, 90]),
    )), strict=True))


def diagnostics(profile: dict) -> dict:
    available_rows, outliers, per_video = [], [], []
    total = 0
    for video in development_videos(profile):
        safe = [candidate for candidate in video["candidates"] if candidate["speech_safe"] is True]
        total += len(safe)
        changes = []
        for candidate in safe:
            for side in ("left", "right"):
                window = candidate["asr_context"][side]
                if window["text"].strip() and window["bengali_script_ratio"] == 0.0:
                    outliers.append({
                        "video": video["video"], "timestamp": candidate["timestamp_seconds"],
                        "side": side, "raw_text": window["text"],
                        "script_counts": window["script_counts"],
                    })
            semantic = candidate["text_semantics"]
            if semantic["available"]:
                changes.append(semantic["semantic_change"])
                available_rows.append({
                    "video": video["video"], "timestamp": candidate["timestamp_seconds"],
                    "centered_pause": candidate["centered_pause"],
                    "inside_detected_speech": candidate["speech_context"]["inside_detected_speech"],
                    "left_transcript": candidate["asr_context"]["left"]["text"],
                    "right_transcript": candidate["asr_context"]["right"]["text"],
                    "cosine_similarity": semantic["cosine_similarity"],
                    "semantic_change": semantic["semantic_change"],
                })
        stats = distribution(changes)
        per_video.append({
            "video": video["video"], "speech_safe_candidates": len(safe),
            "available": len(changes), "unavailable": len(safe) - len(changes),
            "mean_semantic_change": stats["mean"], "median_semantic_change": stats["median"],
        })
    return {
        "speech_safe_candidates": total,
        "available": len(available_rows), "unavailable": total - len(available_rows),
        "cosine_similarity": distribution([row["cosine_similarity"] for row in available_rows]),
        "semantic_change": distribution([row["semantic_change"] for row in available_rows]),
        "per_video": per_video,
        "highest_semantic_change": sorted(available_rows, key=lambda row: -row["semantic_change"])[:10],
        "lowest_semantic_change": sorted(available_rows, key=lambda row: row["semantic_change"])[:10],
        "non_bengali_asr_outliers": outliers,
    }


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    started = time.perf_counter()
    encoder = SemanticTextEncoder()
    similarities = sanity_check(encoder)
    # No corpus read/encoding occurs until both diagnostic orderings pass.
    original_bytes = INPUT.read_bytes()
    profile = json.loads(original_bytes)
    candidates = [c for v in development_videos(profile) for c in v["candidates"] if c["speech_safe"] is True]
    if len(candidates) != 317:
        raise ValueError(f"Expected 317 enriched development candidates, found {len(candidates)}")
    enriched, unique_count = enrich(profile, encoder)
    if INPUT.read_bytes() != original_bytes:
        raise RuntimeError("ASR input changed during semantic extraction")
    report = diagnostics(enriched)
    report["sanity_pair_similarities"] = similarities
    report["unique_corpus_transcripts_encoded"] = unique_count
    report["source_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    write_json(OUTPUT, enriched)
    report["runtime_seconds"] = time.perf_counter() - started
    write_json(DIAGNOSTICS, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
