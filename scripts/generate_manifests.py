"""Generate VMAP documents and unified debug manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.manifest.debug_manifest import (
    build_debug_manifest,
)
from src.manifest.vmap import (
    write_vmap,
)


ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_OPTIMIZED = (
    ROOT
    / "outputs"
    / "dev"
    / "optimized-breaks.json"
)

DEFAULT_MATCHES = (
    ROOT
    / "outputs"
    / "dev"
    / "brand-matches.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "manifests"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate VMAP and unified "
            "debug manifests."
        )
    )

    parser.add_argument(
        "--optimized",
        type=Path,
        default=DEFAULT_OPTIMIZED,
    )

    parser.add_argument(
        "--matches",
        type=Path,
        default=DEFAULT_MATCHES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    optimized = json.loads(
        args.optimized.read_text(
            encoding="utf-8"
        )
    )

    brand_matches = json.loads(
        args.matches.read_text(
            encoding="utf-8"
        )
    )

    manifest = (
        build_debug_manifest(
            optimized=optimized,
            brand_matches=(
                brand_matches
            ),
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path = (
        args.output_dir
        / "debug-manifest.json"
    )

    debug_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy = optimized[
        "policy"
    ]

    ad_duration = float(
        policy[
            "nominal_ad_duration_seconds"
        ]
    )

    vmap_dir = (
        args.output_dir
        / "vmap"
    )

    vmap_files: list[str] = []

    for video in manifest[
        "videos"
    ]:
        path = (
            vmap_dir
            / (
                Path(
                    video[
                        "video"
                    ]
                ).stem
                + ".vmap.xml"
            )
        )

        write_vmap(
            path=path,
            video_name=(
                video["video"]
            ),
            breaks=(
                video["breaks"]
            ),
            ad_duration_seconds=(
                ad_duration
            ),
        )

        vmap_files.append(
            str(
                path.relative_to(
                    args.output_dir
                )
            ).replace(
                "\\",
                "/",
            )
        )

    summary = {
        **manifest[
            "summary"
        ],
        "vmap_file_count": (
            len(
                vmap_files
            )
        ),
        "vmap_files": (
            vmap_files
        ),
        "debug_manifest": (
            str(
                debug_path
            )
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
