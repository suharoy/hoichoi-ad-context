"""Benchmark, then cache one CLIP embedding per existing development shot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from src.context.visual_embeddings import CLIPVisualEncoder, MODEL_NAME


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "outputs/dev"
ASSETS = ROOT.parent / "hoichoi-assets"
STEMS = ("bhojon_bilashi", "indubala_bhaater_hotel", "mandaar", "mohanagar", "money_honey")


def load_shots(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        next(file)  # SceneDetect's timecode-list row.
        rows = list(csv.DictReader(file))
    shots = []
    for row in rows:
        start, end = float(row["Start Time (seconds)"]), float(row["End Time (seconds)"])
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError(f"Invalid shot times in {path}")
        if shots and start != shots[-1]["end"]:
            raise ValueError(f"Noncontiguous shots in {path}")
        shots.append({"index": int(row["Scene Number"]), "start": start, "end": end,
                      "representative_timestamp": start + (end - start) / 2})
    if not shots or [s["index"] for s in shots] != list(range(1, len(shots) + 1)):
        raise ValueError(f"Invalid shot indexes in {path}")
    return shots


def paths_for(stem: str) -> tuple[Path, Path, Path]:
    if stem not in STEMS:
        raise ValueError("Only the five development videos may be processed")
    return (ASSETS / f"{stem}.mp4", DEV / "scenes" / stem / f"{stem}-Scenes.csv",
            DEV / "visual-cache" / stem)


def cache_metadata(video: Path, csv_path: Path) -> str:
    stat = video.stat()
    return json.dumps({"model": MODEL_NAME, "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                       "video_size": stat.st_size, "video_mtime_ns": stat.st_mtime_ns}, sort_keys=True)


def load_cache(path: Path, shots: list[dict], metadata: str) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        if str(data["metadata"]) != metadata:
            raise ValueError(f"Stale shot cache metadata: {path}")
        vectors = data["embeddings"]
        indexes = data["shot_index"]
        if vectors.ndim != 2 or len(vectors) != len(indexes) or len(set(indexes)) != len(indexes):
            raise ValueError(f"Invalid cache dimensions/indexes: {path}")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid cached embeddings: {path}")
        by_index = {s["index"]: s for s in shots}
        for row, index in enumerate(indexes):
            shot = by_index[int(index)]
            for key in ("start", "end", "representative_timestamp"):
                if data[key][row] != shot[key]:
                    raise ValueError(f"Shot metadata mismatch: {path}")
        return {int(index): vector.copy() for index, vector in zip(indexes, vectors, strict=True)}


def save_cache(path: Path, shots: list[dict], vectors: dict, metadata: str) -> None:
    present = [s for s in shots if s["index"] in vectors]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=np.array(metadata), model=np.array(MODEL_NAME),
                        shot_index=np.array([s["index"] for s in present], dtype=np.int64),
                        embeddings=np.stack([vectors[s["index"]] for s in present]),
                        **{key: np.array([s[key] for s in present], dtype=np.float64)
                           for key in ("start", "end", "representative_timestamp")})
    temporary.replace(path)


def valid_frame(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except (OSError, ValueError, SyntaxError):
        return False


def extract_frame(video: Path, path: Path, timestamp: float) -> bool:
    if valid_frame(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.jpg")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", "-q:v", "2",
                    str(temporary)], check=True)
    if not valid_frame(temporary):
        raise ValueError(f"Could not extract a valid frame at {timestamp}: {video}")
    temporary.replace(path)
    return True


def process_shots(stem: str, all_shots: list[dict], selected: list[dict], encoder: CLIPVisualEncoder,
                  batch_size: int, force_encode: bool = False) -> dict:
    video, csv_path, cache = paths_for(stem)
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(video, csv_path)
    # Frame provenance prevents reusing a shot-index image from different source times.
    manifest = cache / "frame-metadata.json"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != metadata:
        raise ValueError(f"Stale representative-frame metadata: {manifest}")
    manifest.write_text(metadata, encoding="utf-8")
    npz = cache / "shot-embeddings.npz"
    vectors = load_cache(npz, all_shots, metadata)
    report = dict(video=stem, shots=len(selected), frames_extracted=0, frames_reused=0,
                  embeddings_produced=0, embeddings_reused=0, extraction_seconds=0., embedding_seconds=0.)
    pending = []
    for shot in selected:
        frame = cache / "frames" / f"shot_{shot['index']:06d}.jpg"
        started = time.perf_counter()
        extracted = extract_frame(video, frame, shot["representative_timestamp"])
        report["extraction_seconds"] += time.perf_counter() - started
        report["frames_extracted" if extracted else "frames_reused"] += 1
        if shot["index"] in vectors and not force_encode:
            report["embeddings_reused"] += 1
        else:
            pending.append((shot, frame))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        started = time.perf_counter()
        encoded = encoder.encode([frame for _, frame in batch], batch_size=batch_size)
        report["embedding_seconds"] += time.perf_counter() - started
        if not np.isfinite(encoded).all() or not np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-5):
            raise ValueError("Embedding sanity check failed")
        for (shot, _), vector in zip(batch, encoded, strict=True):
            vectors[shot["index"]] = vector
        report["embeddings_produced"] += len(batch)
        save_cache(npz, all_shots, vectors, metadata)
        if (offset // batch_size) % 4 == 0:
            print(f"{stem}: {offset + len(batch)}/{len(pending)} new embeddings", flush=True)
    return report


def write_report(name: str, report: dict) -> None:
    path = DEV / "visual-diagnostics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main(batch_size: int = 16, pause_after_benchmark: bool = False) -> None:
    started = time.perf_counter()
    shots = {stem: load_shots(paths_for(stem)[1]) for stem in STEMS}
    total_shots = sum(map(len, shots.values()))
    load_started = time.perf_counter()
    encoder = CLIPVisualEncoder()
    load_seconds = time.perf_counter() - load_started
    first = STEMS[0]
    benchmark = process_shots(first, shots[first], shots[first][:32], encoder, batch_size, force_encode=True)
    video, csv_path, cache = paths_for(first)
    vectors = load_cache(cache / "shot-embeddings.npz", shots[first], cache_metadata(video, csv_path))
    sample = np.stack([vectors[s["index"]] for s in shots[first][:32]])
    norms = np.linalg.norm(sample, axis=1)
    self_cosine = float(np.dot(sample[0], sample[0]))
    if not np.isfinite(sample).all() or not np.isclose(self_cosine, 1, atol=1e-5) or not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("32-frame benchmark sanity check failed")
    images_per_second = len(sample) / benchmark["embedding_seconds"]
    projection = total_shots / images_per_second
    benchmark_report = {**benchmark, "model_load_seconds": load_seconds,
                        "images_per_second": images_per_second, "embedding_dimension": sample.shape[1],
                        "self_cosine": self_cosine, "finite": True,
                        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                        "total_shots": total_shots, "projected_embedding_seconds": projection}
    write_report("benchmark.json", benchmark_report)
    if projection > 20 * 60:
        raise RuntimeError("Projected embedding runtime exceeds 20 minutes; stopped after benchmark")
    if pause_after_benchmark and input("BENCHMARK COMPLETE. Enter continue to process development shots: ").strip() != "continue":
        return
    reports = []
    for stem in STEMS:
        remaining = shots[stem][32:] if stem == first else shots[stem]
        report = process_shots(stem, shots[stem], remaining, encoder, batch_size)
        if stem == first:
            for key in report:
                if key != "video":
                    report[key] += benchmark[key]
        reports.append(report)
        print(json.dumps(report), flush=True)
    embedding_seconds = sum(r["embedding_seconds"] for r in reports)
    overall = {"per_video": reports, "total_shots": total_shots, "model_load_seconds": load_seconds,
               "total_frame_extraction_seconds": sum(r["extraction_seconds"] for r in reports),
               "total_embedding_seconds": embedding_seconds,
               "mean_images_per_second": sum(r["embeddings_produced"] for r in reports) / embedding_seconds,
               "runtime_seconds": time.perf_counter() - started}
    write_report("embedding-summary.json", overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pause-after-benchmark", action="store_true")
    args = parser.parse_args()
    main(args.batch_size, args.pause_after_benchmark)
