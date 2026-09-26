"""Add local raw ASR context to existing speech-safe development boundaries."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import wave

from src.audio.indic_asr import IndicConformerASR


DEV = Path(__file__).resolve().parents[1] / "outputs/dev"
INPUT = DEV / "boundary-profile.json"
OUTPUT = DEV / "boundary-profile-asr.json"
# Configurable V1 hypothesis, not an optimized threshold.
CONTEXT_SECONDS = 10.0
DEVELOPMENT_STEMS = (
    "bhojon_bilashi", "indubala_bhaater_hotel", "mandaar", "mohanagar", "money_honey",
)


def script_quality(text: str) -> tuple[dict[str, int], float]:
    ranges = {
        "bengali": (0x0980, 0x09FF), "devanagari": (0x0900, 0x097F),
        "telugu": (0x0C00, 0x0C7F), "kannada": (0x0C80, 0x0CFF),
    }
    counts = dict.fromkeys((*ranges, "latin"), 0)
    total = 0
    for character in text:
        if not character.isalpha():
            continue
        total += 1
        for script, (lower, upper) in ranges.items():
            if lower <= ord(character) <= upper:
                counts[script] += 1
                break
        if "A" <= character <= "Z" or "a" <= character <= "z":
            counts["latin"] += 1
    return counts, counts["bengali"] / total if total else 0.0


def valid_result(result: object, start: float, end: float) -> bool:
    if not isinstance(result, dict):
        return False
    elapsed = result.get("elapsed_seconds")
    if (
        result.get("start") != start or result.get("end") != end
        or not isinstance(result.get("text"), str)
        or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed) or elapsed < 0
    ):
        return False
    counts, ratio = script_quality(result["text"])
    return result.get("script_counts") == counts and result.get("bengali_script_ratio") == ratio


def read_profile() -> dict:
    profile = json.loads(INPUT.read_text(encoding="utf-8"))
    if OUTPUT.exists():
        cached = json.loads(OUTPUT.read_text(encoding="utf-8"))
        videos = {video["video"]: video for video in cached["videos"]}
        for video in profile["videos"]:
            old = {
                candidate["timestamp_seconds"]: candidate
                for candidate in videos.get(video["video"], {}).get("candidates", [])
            }
            for candidate in video["candidates"]:
                previous = old.get(candidate["timestamp_seconds"], {})
                original = {key: value for key, value in previous.items() if key != "asr_context"}
                if candidate["speech_safe"] is True and original == candidate and "asr_context" in previous:
                    candidate["asr_context"] = copy.deepcopy(previous["asr_context"])
    return profile


def checkpoint(profile: dict) -> None:
    # Replace atomically so a interrupted write does not destroy the prior checkpoint.
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT)


def ensure_clip(source: Path, clip: Path, start: float, end: float) -> None:
    if clip.exists():
        return
    clip.parent.mkdir(parents=True, exist_ok=True)
    temporary = clip.with_name(clip.stem + ".tmp.wav")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", str(start), "-i", str(source), "-t", str(end - start),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary),
        ],
        check=True,
    )
    temporary.replace(clip)


def run(stems: list[str]) -> dict:
    started = time.perf_counter()
    if not stems or any(stem not in DEVELOPMENT_STEMS for stem in stems):
        raise ValueError("Select only approved development stems; held-out videos are forbidden")
    profile = read_profile()
    selected = [video for video in profile["videos"] if Path(video["video"]).stem in stems]
    if {Path(video["video"]).stem for video in selected} != set(stems):
        raise ValueError("Selected video is missing from the boundary profile")

    model = None
    processed = reused = 0
    empty_transcript_windows = nonempty_transcript_windows = 0
    inference_seconds = 0.0
    rtfs, ratios, per_video = [], [], []
    for video in selected:
        stem = Path(video["video"]).stem
        source = DEV / "corpus" / stem / "audio.wav"
        with wave.open(str(source), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != 16000:
                raise ValueError(f"Expected 16 kHz mono corpus WAV: {source}")
            duration = audio.getnframes() / audio.getframerate()
        safe = [candidate for candidate in video["candidates"] if candidate["speech_safe"] is True]
        names = [round(candidate["timestamp_seconds"] * 1000) for candidate in safe]
        if len(names) != len(set(names)):
            raise ValueError(f"Candidate timestamps collide in millisecond clip names: {stem}")
        video_seconds = 0.0
        enriched = 0
        for candidate in safe:
            timestamp = candidate["timestamp_seconds"]
            if not math.isfinite(timestamp) or not 0 <= timestamp <= duration:
                raise ValueError(f"Candidate outside audio: {stem} at {timestamp}")
            context = candidate.get("asr_context")
            if not isinstance(context, dict) or context.get("window_seconds") != CONTEXT_SECONDS:
                context = {"window_seconds": CONTEXT_SECONDS}
            for side, start, end in (
                ("left", max(0.0, timestamp - CONTEXT_SECONDS), timestamp),
                ("right", timestamp, min(duration, timestamp + CONTEXT_SECONDS)),
            ):
                clip = DEV / "asr-context" / stem / f"{round(timestamp * 1000):09d}_{side}.wav"
                result = context.get(side)
                if clip.exists() and valid_result(result, start, end):
                    reused += 1
                else:
                    # Endpoint boundaries have no audio on one side.
                    if end == start:
                        text, elapsed = "", 0.0
                    else:
                        ensure_clip(source, clip, start, end)
                        if model is None:
                            model = IndicConformerASR()
                        inference_started = time.perf_counter()
                        text = model.recognize(clip)
                        elapsed = time.perf_counter() - inference_started
                        processed += 1
                        inference_seconds += elapsed
                        video_seconds += elapsed
                        rtfs.append(elapsed / (end - start))
                    counts, ratio = script_quality(text)
                    result = {
                        "start": start, "end": end, "text": text,
                        "elapsed_seconds": elapsed, "script_counts": counts,
                        "bengali_script_ratio": ratio,
                    }
                context[side] = result
                if result["text"].strip():
                    nonempty_transcript_windows += 1
                    ratios.append(result["bengali_script_ratio"])
                else:
                    empty_transcript_windows += 1
            candidate["asr_context"] = context
            enriched += 1
            if enriched % 10 == 0:
                print(f"{stem}: {enriched}/{len(safe)} candidates enriched", flush=True)
        checkpoint(profile)
        per_video.append({
            "video": video["video"], "speech_safe_candidates": len(safe),
            "enriched_candidates": enriched, "asr_inference_seconds": video_seconds,
        })
        print(json.dumps(per_video[-1], ensure_ascii=False), flush=True)

    summary = {
        "scope": "selected videos; inference times and RTF cover newly processed windows",
        "total_visual_boundaries": sum(len(video["candidates"]) for video in selected),
        "total_speech_safe_candidates": sum(row["speech_safe_candidates"] for row in per_video),
        "total_candidates_enriched": sum(row["enriched_candidates"] for row in per_video),
        "windows_processed": processed, "windows_reused": reused,
        "total_asr_inference_seconds": inference_seconds,
        "mean_rtf": statistics.mean(rtfs) if rtfs else None,
        "transcript_windows": {
            "empty": empty_transcript_windows,
            "nonempty": nonempty_transcript_windows,
        },
        "nonempty_bengali_script_ratio": {
            "mean": statistics.mean(ratios) if ratios else None,
            "median": statistics.median(ratios) if ratios else None,
            "minimum": min(ratios) if ratios else None,
        },
        "per_video": per_video,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs="+", required=True, choices=DEVELOPMENT_STEMS)
    run(parser.parse_args().video)
