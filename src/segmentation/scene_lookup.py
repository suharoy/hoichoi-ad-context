"""Map ad-break timestamps onto semantic scene intervals."""

from __future__ import annotations

import math


BOUNDARY_TOLERANCE_SECONDS = 0.001


def validate_scenes(
    scenes: list[dict],
) -> None:
    if not scenes:
        raise ValueError(
            "Semantic scene list cannot be empty"
        )

    previous_end = None

    for index, scene in enumerate(
        scenes
    ):
        start = float(
            scene["start_seconds"]
        )

        end = float(
            scene["end_seconds"]
        )

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
        ):
            raise ValueError(
                "Invalid semantic scene interval"
            )

        if (
            previous_end is not None
            and abs(
                start - previous_end
            )
            > BOUNDARY_TOLERANCE_SECONDS
        ):
            raise ValueError(
                "Semantic scenes are not contiguous"
            )

        previous_end = end


def locate_scene(
    *,
    timestamp_seconds: float,
    scenes: list[dict],
) -> dict:
    """
    Describe the relationship between one timestamp and semantic scenes.

    No arbitrary 'near scene boundary' threshold is used. We record the
    exact boundary distance and whether the timestamp is exactly on a
    semantic transition.
    """
    validate_scenes(
        scenes
    )

    timestamp = float(
        timestamp_seconds
    )

    if not math.isfinite(
        timestamp
    ):
        raise ValueError(
            "Timestamp must be finite"
        )

    start = float(
        scenes[0][
            "start_seconds"
        ]
    )

    end = float(
        scenes[-1][
            "end_seconds"
        ]
    )

    if (
        timestamp
        < start
        - BOUNDARY_TOLERANCE_SECONDS
        or timestamp
        > end
        + BOUNDARY_TOLERANCE_SECONDS
    ):
        raise ValueError(
            "Timestamp falls outside semantic scenes"
        )

    internal_boundaries = []

    for index in range(
        len(scenes) - 1
    ):
        left = scenes[index]
        right = scenes[index + 1]

        boundary = float(
            left[
                "end_seconds"
            ]
        )

        internal_boundaries.append(
            (
                boundary,
                left,
                right,
            )
        )

        if (
            abs(
                timestamp - boundary
            )
            <= BOUNDARY_TOLERANCE_SECONDS
        ):
            return {
                "relation": (
                    "exact_scene_boundary"
                ),
                "scene_id": (
                    right["scene_id"]
                ),
                "previous_scene_id": (
                    left["scene_id"]
                ),
                "next_scene_id": (
                    right["scene_id"]
                ),
                "scene_start_seconds": float(
                    right[
                        "start_seconds"
                    ]
                ),
                "scene_end_seconds": float(
                    right[
                        "end_seconds"
                    ]
                ),
                "scene_duration_seconds": float(
                    right[
                        "duration_seconds"
                    ]
                ),
                "nearest_scene_boundary_seconds": (
                    boundary
                ),
                "distance_to_nearest_scene_boundary_seconds": (
                    0.0
                ),
                "seconds_from_scene_start": (
                    0.0
                ),
                "seconds_until_scene_end": float(
                    right[
                        "end_seconds"
                    ]
                )
                - boundary,
            }

    containing = None

    for index, scene in enumerate(
        scenes
    ):
        scene_start = float(
            scene[
                "start_seconds"
            ]
        )

        scene_end = float(
            scene[
                "end_seconds"
            ]
        )

        is_last = (
            index
            == len(scenes) - 1
        )

        if (
            scene_start
            <= timestamp
            < scene_end
        ) or (
            is_last
            and timestamp
            <= scene_end
            + BOUNDARY_TOLERANCE_SECONDS
        ):
            containing = scene
            break

    if containing is None:
        raise RuntimeError(
            "Could not locate containing semantic scene"
        )

    candidates = [
        boundary
        for (
            boundary,
            _,
            _,
        )
        in internal_boundaries
    ]

    if candidates:
        nearest = min(
            candidates,
            key=lambda value: abs(
                timestamp - value
            ),
        )

        distance = abs(
            timestamp - nearest
        )
    else:
        nearest = None
        distance = None

    scene_start = float(
        containing[
            "start_seconds"
        ]
    )

    scene_end = float(
        containing[
            "end_seconds"
        ]
    )

    return {
        "relation": "within_scene",
        "scene_id": (
            containing[
                "scene_id"
            ]
        ),
        "previous_scene_id": None,
        "next_scene_id": None,
        "scene_start_seconds": (
            scene_start
        ),
        "scene_end_seconds": (
            scene_end
        ),
        "scene_duration_seconds": float(
            containing[
                "duration_seconds"
            ]
        ),
        "nearest_scene_boundary_seconds": (
            nearest
        ),
        "distance_to_nearest_scene_boundary_seconds": (
            float(distance)
            if distance is not None
            else None
        ),
        "seconds_from_scene_start": (
            timestamp
            - scene_start
        ),
        "seconds_until_scene_end": (
            scene_end
            - timestamp
        ),
    }


def annotate_manifest_with_scenes(
    *,
    manifest: dict,
    semantic_scenes: dict,
) -> dict:
    """
    Attach semantic-scene structure to an existing delivery manifest.

    This is explanatory metadata only. It does not alter break selection,
    ranking, safety filtering, or VMAP delivery.
    """
    scene_lookup = {
        item["video"]: item
        for item in semantic_scenes[
            "videos"
        ]
    }

    for video in manifest[
        "videos"
    ]:
        video_name = (
            video["video"]
        )

        if (
            video_name
            not in scene_lookup
        ):
            raise ValueError(
                "Semantic scenes missing for "
                f"{video_name}"
            )

        scene_video = (
            scene_lookup[
                video_name
            ]
        )

        scenes = (
            scene_video[
                "scenes"
            ]
        )

        validate_scenes(
            scenes
        )

        video[
            "semantic_scene_count"
        ] = len(scenes)

        video[
            "semantic_scene_method"
        ] = semantic_scenes.get(
            "method"
        )

        for item in video[
            "breaks"
        ]:
            item[
                "semantic_scene"
            ] = locate_scene(
                timestamp_seconds=(
                    item[
                        "timestamp_seconds"
                    ]
                ),
                scenes=scenes,
            )

    manifest[
        "semantic_scene_segmentation"
    ] = {
        "schema_version": (
            semantic_scenes.get(
                "schema_version"
            )
        ),
        "method": (
            semantic_scenes.get(
                "method"
            )
        ),
        "calibration_mode": (
            semantic_scenes.get(
                "calibration_mode"
            )
        ),
        "held_out_used_for_fit": (
            semantic_scenes.get(
                "summary",
                {},
            ).get(
                "held_out_used_for_fit"
            )
        ),
        "role": (
            "explanatory narrative structure; "
            "does not modify frozen EABS or "
            "break optimization"
        ),
    }

    return manifest
