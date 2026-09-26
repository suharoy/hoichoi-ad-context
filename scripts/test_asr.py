from __future__ import annotations

import json
from pathlib import Path

from src.audio.asr import ASRModel


AUDIO = Path(
    "outputs/dev/corpus/"
    "bhojon_bilashi/audio-test-120s.wav"
)

OUTPUT = Path(
    "outputs/dev/corpus/"
    "bhojon_bilashi/asr-test-120s.json"
)


def main() -> None:
    model = ASRModel(
        model_size="small",
        device="cpu",
        compute_type="int8",
    )

    segments = model.transcribe(AUDIO)

    OUTPUT.write_text(
        json.dumps(
            {
                "language": "bn",
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Segments: {len(segments)}")
    print(f"Saved: {OUTPUT}")
    print()

    print("First transcript segments:")
    print("-" * 80)

    for segment in segments[:10]:
        print(
            f"[{segment['start']:7.2f} - "
            f"{segment['end']:7.2f}] "
            f"{segment['text']}"
        )


if __name__ == "__main__":
    main()
