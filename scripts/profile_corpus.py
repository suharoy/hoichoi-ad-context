from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.audio.vad import detect_speech


ASSETS = Path("../hoichoi-assets")
OUTPUT_ROOT = Path("outputs/dev/corpus")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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

    return float(result.stdout.strip())


def extract_audio(video: Path, wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(video),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(wav),
        ],
        check=True,
    )


results = []

for video in sorted(ASSETS.glob("*.mp4")):
    print(f"\nProcessing {video.name}")

    title_dir = OUTPUT_ROOT / video.stem
    title_dir.mkdir(parents=True, exist_ok=True)

    wav = title_dir / "audio.wav"

    duration = probe_duration(video)

    if not wav.exists():
        print("  extracting audio...")
        extract_audio(video, wav)

    print("  running VAD...")
    segments = detect_speech(wav)

    speech_seconds = sum(
        segment["end"] - segment["start"]
        for segment in segments
    )

    speech_ratio = speech_seconds / duration

    result = {
        "video": video.name,
        "duration_seconds": round(duration, 3),
        "speech_segments": len(segments),
        "speech_seconds": round(speech_seconds, 3),
        "speech_ratio": round(speech_ratio, 4),
    }

    results.append(result)

    print(
        f"  speech={speech_seconds:.1f}s "
        f"({speech_ratio:.1%}), "
        f"segments={len(segments)}"
    )


OUTPUT = OUTPUT_ROOT / "corpus-profile.json"

OUTPUT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print(f"\nSaved corpus profile to {OUTPUT}")