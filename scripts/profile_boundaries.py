from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from src.audio.speech_safety import (
    speech_context,
    speech_crosses_boundary,
)
from src.audio.vad import detect_speech


ASSETS = Path("../hoichoi-assets")

CORPUS_ROOT = Path("outputs/dev/corpus")
SCENES_ROOT = Path("outputs/dev/scenes")
OUTPUT = Path("outputs/dev/boundary-profile.json")

DEV_STEMS = [
    "bhojon_bilashi",
    "indubala_bhaater_hotel",
    "mandaar",
    "mohanagar",
    "money_honey",
]

HELD_OUT = "feluda"

GUARD_SECONDS = 0.20
CENTERED_PAUSE_SECONDS = 0.25


def ensure_scene_csv(video: Path) -> Path:
    out_dir = SCENES_ROOT / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{video.stem}-Scenes.csv"

    if csv_path.exists():
        return csv_path

    print("  detecting visual shots...")

    subprocess.run(
        [
            "scenedetect",
            "-i",
            str(video),
            "detect-adaptive",
            "list-scenes",
            "-o",
            str(out_dir),
        ],
        check=True,
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected SceneDetect output not found: {csv_path}"
        )

    return csv_path


def load_boundaries(csv_path: Path) -> list[float]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        # PySceneDetect puts the Timecode List on the first row.
        next(f)

        reader = csv.DictReader(f)
        rows = list(reader)

    # Scene 1 begins at t=0, so only subsequent scene starts
    # correspond to internal visual boundaries.
    return [
        float(row["Start Time (seconds)"])
        for row in rows[1:]
    ]


def ensure_vad_segments(
    stem: str,
) -> list[dict[str, float]]:
    title_dir = CORPUS_ROOT / stem
    wav_path = title_dir / "audio.wav"
    vad_path = title_dir / "vad.json"

    if not wav_path.exists():
        raise FileNotFoundError(
            f"Missing corpus WAV: {wav_path}"
        )

    if vad_path.exists():
        return json.loads(
            vad_path.read_text(encoding="utf-8")
        )["speech_segments"]

    print("  running VAD...")

    segments = detect_speech(wav_path)

    vad_path.write_text(
        json.dumps(
            {
                "speech_segments": segments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return segments


def main() -> None:
    all_results = []

    print(
        f"Held out from tuning: {HELD_OUT}.mp4\n"
    )

    for stem in DEV_STEMS:
        video = ASSETS / f"{stem}.mp4"

        print(f"Processing {video.name}")

        csv_path = ensure_scene_csv(video)
        boundaries = load_boundaries(csv_path)
        speech_segments = ensure_vad_segments(stem)

        candidates = []

        for timestamp in boundaries:
            crosses = speech_crosses_boundary(
                timestamp,
                speech_segments,
                guard_seconds=GUARD_SECONDS,
            )

            context = speech_context(
                timestamp,
                speech_segments,
            )

            inside_speech = context[
                "inside_detected_speech"
            ]

            before = context[
                "seconds_since_speech_end"
            ]

            after = context[
                "seconds_until_speech_start"
            ]

            centered_pause = (
                not crosses
                and not inside_speech
                and before is not None
                and after is not None
                and before >= CENTERED_PAUSE_SECONDS
                and after >= CENTERED_PAUSE_SECONDS
            )

            candidates.append(
                {
                    "timestamp_seconds": round(
                        timestamp,
                        3,
                    ),
                    "speech_crosses_boundary": crosses,
                    "speech_safe": not crosses,
                    "centered_pause": centered_pause,
                    "speech_context": context,
                }
            )

        total = len(candidates)

        unsafe = sum(
            candidate["speech_crosses_boundary"]
            for candidate in candidates
        )

        safe = total - unsafe

        centered = sum(
            candidate["centered_pause"]
            for candidate in candidates
        )

        summary = {
            "video": video.name,
            "visual_boundaries": total,
            "speech_safe": safe,
            "speech_unsafe": unsafe,
            "centered_pause": centered,
            "speech_safe_ratio": round(
                safe / total if total else 0.0,
                4,
            ),
            "candidates": candidates,
        }

        all_results.append(summary)

        print(
            f"  visual boundaries : {total}"
        )
        print(
            f"  speech-safe       : {safe}"
        )
        print(
            f"  speech-unsafe     : {unsafe}"
        )
        print(
            f"  centered pauses   : {centered}"
        )
        print()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "held_out": f"{HELD_OUT}.mp4",
                "guard_seconds": GUARD_SECONDS,
                "centered_pause_seconds":
                    CENTERED_PAUSE_SECONDS,
                "videos": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()