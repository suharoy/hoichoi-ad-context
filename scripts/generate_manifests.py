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
from src.segmentation.scene_lookup import (
    annotate_manifest_with_scenes,
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

DEFAULT_SEMANTIC_SCENES = (
    ROOT
    / "outputs"
    / "dev"
    / "semantic-scenes.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "manifests"
)


def load_json(
    path: Path,
    *,
    label: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate VMAP, semantic-scene-aware "
            "debug manifest, and delivery diagnostics."
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
        "--semantic-scenes",
        type=Path,
        default=DEFAULT_SEMANTIC_SCENES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    optimized = load_json(
        args.optimized,
        label="Optimized-break file",
    )

    brand_matches = load_json(
        args.matches,
        label="Brand-match file",
    )

    semantic_scenes = load_json(
        args.semantic_scenes,
        label="Semantic-scene file",
    )

    if (
        semantic_scenes.get(
            "summary",
            {},
        ).get(
            "held_out_used_for_fit"
        )
        is not False
    ):
        raise ValueError(
            "Semantic-scene calibration does not "
            "explicitly confirm held-out exclusion"
        )

    manifest = build_debug_manifest(
        optimized=optimized,
        brand_matches=brand_matches,
    )

    manifest = annotate_manifest_with_scenes(
        manifest=manifest,
        semantic_scenes=semantic_scenes,
    )

    # Hard postcondition: semantic-scene annotation must
    # actually be present before anything is written.
    scene_meta = manifest.get(
        "semantic_scene_segmentation"
    )

    if not scene_meta:
        raise RuntimeError(
            "Semantic-scene annotation was not attached"
        )

    for video in manifest["videos"]:
        if (
            video.get(
                "semantic_scene_count"
            )
            is None
        ):
            raise RuntimeError(
                "Missing semantic_scene_count for "
                f"{video['video']}"
            )

        for item in video["breaks"]:
            if not item.get(
                "semantic_scene"
            ):
                raise RuntimeError(
                    "Missing semantic-scene annotation for "
                    f"{video['video']} @ "
                    f"{item['timestamp_seconds']}"
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

    policy = optimized["policy"]

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

    for video in manifest["videos"]:
        output_path = (
            vmap_dir
            / (
                Path(
                    video["video"]
                ).stem
                + ".vmap.xml"
            )
        )

        write_vmap(
            path=output_path,
            video_name=video["video"],
            breaks=video["breaks"],
            ad_duration_seconds=(
                ad_duration
            ),
        )

        vmap_files.append(
            str(
                output_path.relative_to(
                    args.output_dir
                )
            ).replace(
                "\\",
                "/",
            )
        )

    summary = {
        **manifest["summary"],
        "semantic_scene_method": (
            scene_meta.get(
                "method"
            )
        ),
        "semantic_scene_video_count": (
            len(
                manifest["videos"]
            )
        ),
        "semantic_scene_counts": {
            video["video"]: (
                video[
                    "semantic_scene_count"
                ]
            )
            for video
            in manifest["videos"]
        },
        "vmap_file_count": (
            len(
                vmap_files
            )
        ),
        "vmap_files": (
            vmap_files
        ),
        "debug_manifest": str(
            debug_path
        ),
    }

    print(
        "Semantic scenes attached successfully.",
        flush=True,
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
