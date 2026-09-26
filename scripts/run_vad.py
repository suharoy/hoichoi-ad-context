from __future__ import annotations

import json
from pathlib import Path

from src.audio.vad import detect_speech


AUDIO_PATH = Path(
    "outputs/dev/audio/bhojon_bilashi.wav"
)

OUTPUT_PATH = Path(
    "outputs/dev/audio/bhojon_bilashi-vad.json"
)


segments = detect_speech(AUDIO_PATH)

speech_seconds = sum(
    segment["end"] - segment["start"]
    for segment in segments
)

result = {
    "audio": AUDIO_PATH.name,
    "speech_segments": segments,
    "speech_segment_count": len(segments),
    "speech_duration_seconds": round(speech_seconds, 3),
}

OUTPUT_PATH.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print(f"Speech segments: {len(segments)}")
print(f"Detected speech: {speech_seconds:.1f} seconds")
print(f"Saved: {OUTPUT_PATH}")