"""Build copyright-safe static GitHub Pages demo."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = (
    ROOT
    / "outputs"
    / "dev"
    / "manifests"
    / "debug-manifest.json"
)

SITE = ROOT / "docs"

DATA = SITE / "data"

MEDIA = SITE / "media"

MODEL_MANIFEST = (
    DATA
    / "model-manifest.json"
)

FIXTURE_MANIFEST = (
    DATA
    / "fixture-manifest.json"
)

BASELINE_SOURCE = (
    ROOT
    / "outputs"
    / "dev"
    / "baseline-evaluation.json"
)

BASELINE_PUBLIC = (
    DATA
    / "baseline-summary.json"
)

FIXTURE_VIDEO = (
    MEDIA
    / "synthetic-fixture.mp4"
)


def sanitize_manifest(
    source: dict,
) -> dict:
    """
    Keep model decisions and explanations, but remove material
    unnecessary for the public evidence explorer.
    """
    manifest = copy.deepcopy(
        source
    )

    for video in manifest.get(
        "videos",
        [],
    ):
        video.pop(
            "media_url",
            None,
        )

        video.pop(
            "media_available",
            None,
        )

        for item in video.get(
            "breaks",
            [],
        ):
            # Raw transcript is unnecessary for the public demo.
            item.pop(
                "scene_text",
                None,
            )

            # Full taxonomy ranking is bulky; top-level cues and
            # brand ranking are enough for auditability here.
            context = (
                item.get(
                    "context"
                )
                or {}
            )

            context.pop(
                "ranked_contexts",
                None,
            )

            selected = (
                item.get(
                    "selected_brand"
                )
            )

            if selected:
                # Public static demo renders the synthetic ad itself.
                selected[
                    "creative_uri"
                ] = None

    manifest.pop(
        "runtime",
        None,
    )

    manifest[
        "public_demo"
    ] = {
        "media_hosted": False,
        "reason": (
            "Supplied hackathon source videos are not "
            "redistributed by this repository."
        ),
        "local_file_playback_supported": True,
    }

    return manifest


def fixture_manifest() -> dict:
    """
    Synthetic fixture proves player mechanics only.

    These timestamps are NOT model evaluation results and are explicitly
    separated from the development/held-out decision manifests.
    """
    return {
        "fixture_type": (
            "synthetic_player_mechanics_only"
        ),
        "model_evaluation": False,
        "video": (
            "synthetic-fixture.mp4"
        ),
        "duration_seconds": 45.0,
        "breaks": [
            {
                "break_id": (
                    "fixture-break-01"
                ),
                "timestamp_seconds": 15.0,
                "eabs": None,
                "quality_utility": None,
                "delivery_status": (
                    "filled"
                ),
                "context": {
                    "context_cues": [
                        "technology",
                        "conversation",
                    ],
                    "hard_safety_contexts": [],
                },
                "brand_safety": {
                    "blocked_brands": [],
                },
                "selected_brand": {
                    "brand_id": (
                        "orivox_tech"
                    ),
                    "name": (
                        "Orivox Tech"
                    ),
                },
                "brand_ranking": [],
                "semantic_scene": {
                    "relation": (
                        "synthetic_fixture"
                    ),
                },
            }
        ],
    }


def generate_fixture() -> None:
    ffmpeg = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH"
        )

    MEDIA.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        (
            "testsrc2="
            "size=1280x720:"
            "rate=25:"
            "duration=45"
        ),
        "-f",
        "lavfi",
        "-i",
        (
            "sine="
            "frequency=220:"
            "sample_rate=48000:"
            "duration=45"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "32",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(
            FIXTURE_VIDEO
        ),
    ]

    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            "Generate the development debug manifest first: "
            f"{SOURCE_MANIFEST}"
        )

    source = json.loads(
        SOURCE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        source.get(
            "summary",
            {},
        ).get(
            "negative_context_violations",
            1,
        )
        != 0
    ):
        raise RuntimeError(
            "Refusing to publish unsafe manifest"
        )

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanitized = (
        sanitize_manifest(
            source
        )
    )

    MODEL_MANIFEST.write_text(
        json.dumps(
            sanitized,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    FIXTURE_MANIFEST.write_text(
        json.dumps(
            fixture_manifest(),
            indent=2,
        ),
        encoding="utf-8",
    )

    if not BASELINE_SOURCE.is_file():
        raise FileNotFoundError(
            "Baseline evaluation missing: "
            f"{BASELINE_SOURCE}"
        )

    baseline = json.loads(
        BASELINE_SOURCE.read_text(
            encoding="utf-8"
        )
    )

    BASELINE_PUBLIC.write_text(
        json.dumps(
            {
                "evaluation": baseline.get(
                    "evaluation"
                ),
                "scope": baseline.get(
                    "scope"
                ),
                "baselines": baseline.get(
                    "baselines"
                ),
                "aggregate": baseline.get(
                    "aggregate"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    generate_fixture()

    print(
        "Public demo evidence manifest:",
        MODEL_MANIFEST,
    )

    print(
        "Synthetic playback fixture:",
        FIXTURE_VIDEO,
    )


if __name__ == "__main__":
    main()
