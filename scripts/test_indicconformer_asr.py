"""Standalone Bengali-script diagnostic; no production placement logic."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import wave

import onnx_asr


MODEL = "OpenVoiceOS/ai4bharat-indicconformer-bn-onnx"
CORPUS = Path(__file__).resolve().parents[1] / "outputs/dev/corpus/bhojon_bilashi"
SOURCE = CORPUS / "audio.wav"
CLIPS = CORPUS / "asr-benchmark"
OUTPUT = CORPUS / "indicconformer-benchmark.json"
# These windows and the script gate are benchmark diagnostics only.
WINDOWS = ((40, 60), (75, 95), (100, 120))


def script_quality(transcript: str) -> tuple[dict[str, int], float]:
    counts = dict.fromkeys(("bengali", "devanagari", "telugu", "kannada", "latin"), 0)
    ranges = {
        "bengali": (0x0980, 0x09FF),
        "devanagari": (0x0900, 0x097F),
        "telugu": (0x0C00, 0x0C7F),
        "kannada": (0x0C80, 0x0CFF),
    }
    total_letters = 0
    for character in transcript:
        if not character.isalpha():
            continue
        total_letters += 1
        for script, (lower, upper) in ranges.items():
            if lower <= ord(character) <= upper:
                counts[script] += 1
                break
        if "A" <= character <= "Z" or "a" <= character <= "z":
            counts["latin"] += 1
    ratio = counts["bengali"] / total_letters if total_letters else 0.0
    return counts, ratio


def main() -> None:
    benchmark_started = time.perf_counter()
    CLIPS.mkdir(parents=True, exist_ok=True)
    clips = []
    for start, end in WINDOWS:
        clip = CLIPS / f"indicconformer-{start}-{end}.wav"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-y", "-ss", str(start), "-i", str(SOURCE),
                "-t", str(end - start), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(clip),
            ],
            check=True,
        )
        with wave.open(str(clip), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getframerate() != 16000
                or audio.getsampwidth() != 2
                or audio.getnframes() != 20 * 16000
            ):
                raise ValueError(f"Unexpected benchmark audio format or duration: {clip}")
        clips.append(clip)

    print(f"Loading {MODEL}: int8, CPUExecutionProvider", flush=True)
    model = onnx_asr.load_model(
        MODEL,
        quantization="int8",
        providers=["CPUExecutionProvider"],
    )
    print("MODEL LOAD OK", flush=True)

    samples = []
    for (start, end), clip in zip(WINDOWS, clips):
        started = time.perf_counter()
        transcript = model.recognize(str(clip))
        elapsed_seconds = time.perf_counter() - started
        counts, ratio = script_quality(transcript)
        sample = {
            "start": start,
            "end": end,
            "duration": end - start,
            "transcript": transcript,
            "elapsed_seconds": elapsed_seconds,
            "rtf": elapsed_seconds / 20.0,
            "script_counts": counts,
            "bengali_script_ratio": ratio,
            "script_pass": ratio >= 0.90,
        }
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False, indent=2), flush=True)

    OUTPUT.write_text(
        json.dumps(
            {"model": MODEL, "quantization": "int8", "samples": samples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT}", flush=True)
    print(f"All three script_pass: {all(sample['script_pass'] for sample in samples)}", flush=True)
    print(f"Total benchmark runtime seconds: {time.perf_counter() - benchmark_started:.6f}", flush=True)


if __name__ == "__main__":
    main()
