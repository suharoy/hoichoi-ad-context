"""Fit and apply development-only semantic scene segmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from src.break_scoring.calibration import (
    harmonic_mean,
)
from src.segmentation.semantic_scenes import (
    SceneCalibration,
    build_scene_intervals,
    collapse_adjacent_strong_boundaries,
    otsu_threshold,
    transition_evidence,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "boundary-profile-multimodal.json"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "semantic-scenes.json"
)

CONFIG = (
    ROOT
    / "configs"
    / "scene_segmentation.json"
)

DEVELOPMENT_STEMS = {
    "bhojon_bilashi",
    "indubala_bhaater_hotel",
    "mandaar",
    "mohanagar",
    "money_honey",
}


def stem_of(
    video: dict,
) -> str:
    return Path(
        video["video"]
    ).stem


def distribution(
    values: list[float],
) -> dict:
    if not values:
        return {
            key: None
            for key in (
                "mean",
                "median",
                "p10",
                "p90",
                "min",
                "max",
            )
        }

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(array)
        ),
        "median": float(
            np.median(array)
        ),
        "p10": float(
            np.percentile(
                array,
                10,
            )
        ),
        "p90": float(
            np.percentile(
                array,
                90,
            )
        ),
        "min": float(
            np.min(array)
        ),
        "max": float(
            np.max(array)
        ),
    }


def collect_calibration(
    profile: dict,
) -> tuple[
    list[float],
    list[float],
    list[float],
]:
    visual_values = []
    text_values = []
    effective_lengths = []

    for video in profile["videos"]:
        if (
            stem_of(video)
            not in DEVELOPMENT_STEMS
        ):
            continue

        for candidate in video[
            "candidates"
        ]:
            visual = (
                candidate.get(
                    "visual_semantics"
                )
                or {}
            )

            change = visual.get(
                "context_change"
            )

            if change is None:
                raise ValueError(
                    "Missing development visual change"
                )

            visual_values.append(
                float(change)
            )

            semantic = (
                candidate.get(
                    "text_semantics"
                )
                or {}
            )

            if not semantic.get(
                "available",
                False,
            ):
                continue

            text_change = (
                semantic.get(
                    "semantic_change"
                )
            )

            if text_change is None:
                continue

            left = int(
                semantic.get(
                    "left_characters",
                    0,
                )
                or 0
            )

            right = int(
                semantic.get(
                    "right_characters",
                    0,
                )
                or 0
            )

            if left <= 0 or right <= 0:
                continue

            text_values.append(
                float(
                    text_change
                )
            )

            effective_lengths.append(
                harmonic_mean(
                    float(left),
                    float(right),
                )
            )

    return (
        visual_values,
        text_values,
        effective_lengths,
    )


def fit_or_load(
    *,
    profile: dict,
    source_hash: str,
) -> tuple[
    SceneCalibration,
    dict,
    str,
]:
    (
        visual_values,
        text_values,
        effective_lengths,
    ) = collect_calibration(
        profile
    )

    if len(visual_values) < 2:
        raise ValueError(
            "Insufficient visual calibration data"
        )

    if len(text_values) < 2:
        raise ValueError(
            "Insufficient text calibration data"
        )

    if not effective_lengths:
        raise ValueError(
            "No reliable text-length calibration data"
        )

    fitted = {
        "version": 1,
        "method": (
            "development_otsu_multimodal_change"
        ),
        "development_videos": sorted(
            DEVELOPMENT_STEMS
        ),
        "source_sha256": (
            source_hash
        ),
        "visual_reference_count": (
            len(
                visual_values
            )
        ),
        "text_reference_count": (
            len(
                text_values
            )
        ),
        "visual_otsu_threshold": (
            otsu_threshold(
                visual_values
            )
        ),
        "text_otsu_threshold": (
            otsu_threshold(
                text_values
            )
        ),
        "reference_text_length": float(
            statistics.median(
                effective_lengths
            )
        ),
        "fusion": (
            "(visual_change/visual_otsu + "
            "text_reliability*"
            "(text_change/text_otsu)) / "
            "(1+text_reliability)"
        ),
        "boundary_rule": (
            "transition_strength >= 1.0"
        ),
        "adjacent_suppression": (
            "collapse each consecutive run "
            "of strong shot boundaries to "
            "the maximum-strength boundary"
        ),
        "held_out_used_for_fit": False,
    }

    if CONFIG.exists():
        frozen = json.loads(
            CONFIG.read_text(
                encoding="utf-8"
            )
        )

        # Once written, this calibration is frozen.
        fields = (
            "method",
            "development_videos",
            "source_sha256",
            "visual_reference_count",
            "text_reference_count",
            "visual_otsu_threshold",
            "text_otsu_threshold",
            "reference_text_length",
            "fusion",
            "boundary_rule",
            "adjacent_suppression",
            "held_out_used_for_fit",
        )

        for field in fields:
            if frozen.get(field) != fitted.get(
                field
            ):
                raise RuntimeError(
                    "Frozen scene calibration "
                    f"does not match development fit: {field}"
                )

        payload = frozen
        mode = "reused_frozen"

    else:
        CONFIG.write_text(
            json.dumps(
                fitted,
                indent=2,
            ),
            encoding="utf-8",
        )

        payload = fitted
        mode = "fitted_and_frozen"

    calibration = SceneCalibration(
        visual_threshold=float(
            payload[
                "visual_otsu_threshold"
            ]
        ),
        text_threshold=float(
            payload[
                "text_otsu_threshold"
            ]
        ),
        reference_text_length=float(
            payload[
                "reference_text_length"
            ]
        ),
    )

    return (
        calibration,
        payload,
        mode,
    )


def main() -> None:
    source_bytes = (
        INPUT.read_bytes()
    )

    source_hash = hashlib.sha256(
        source_bytes
    ).hexdigest()

    profile = json.loads(
        source_bytes
    )

    development = [
        video
        for video in profile[
            "videos"
        ]
        if stem_of(video)
        in DEVELOPMENT_STEMS
    ]

    stems = {
        stem_of(video)
        for video in development
    }

    if stems != DEVELOPMENT_STEMS:
        raise ValueError(
            "Expected exactly all five "
            "development titles"
        )

    if any(
        "feluda"
        in stem_of(video).lower()
        for video in development
    ):
        raise RuntimeError(
            "Held-out title entered "
            "scene calibration"
        )

    (
        calibration,
        calibration_payload,
        calibration_mode,
    ) = fit_or_load(
        profile=profile,
        source_hash=source_hash,
    )

    videos_out = []

    all_scene_durations = []
    all_transition_strengths = []

    total_visual_boundaries = 0
    total_raw_strong = 0
    total_selected = 0

    highest = []

    for video in development:
        rows = []

        candidates = (
            video[
                "candidates"
            ]
        )

        for index, candidate in enumerate(
            candidates
        ):
            evidence = (
                transition_evidence(
                    candidate,
                    calibration,
                )
            )

            row = {
                "candidate_index": (
                    index
                ),
                "timestamp_seconds": float(
                    candidate[
                        "timestamp_seconds"
                    ]
                ),
                "speech_safe": bool(
                    candidate.get(
                        "speech_safe",
                        False,
                    )
                ),
                "centered_pause": bool(
                    candidate.get(
                        "centered_pause",
                        False,
                    )
                ),
                **evidence,
            }

            rows.append(row)

        selected = (
            collapse_adjacent_strong_boundaries(
                rows
            )
        )

        scenes = build_scene_intervals(
            video_name=video[
                "video"
            ],
            candidates=candidates,
            selected_boundaries=selected,
        )

        durations = [
            float(
                scene[
                    "duration_seconds"
                ]
            )
            for scene in scenes
        ]

        video_summary = {
            "video": video[
                "video"
            ],
            "visual_boundary_count": (
                len(rows)
            ),
            "raw_strong_transition_count": sum(
                row["strong"]
                for row in rows
            ),
            "selected_scene_boundary_count": (
                len(selected)
            ),
            "scene_count": (
                len(scenes)
            ),
            "scene_duration_seconds": (
                distribution(
                    durations
                )
            ),
        }

        videos_out.append(
            {
                **video_summary,
                "boundaries": rows,
                "selected_boundaries": (
                    selected
                ),
                "scenes": scenes,
            }
        )

        total_visual_boundaries += (
            len(rows)
        )

        total_raw_strong += sum(
            row["strong"]
            for row in rows
        )

        total_selected += len(
            selected
        )

        all_scene_durations.extend(
            durations
        )

        all_transition_strengths.extend(
            row[
                "transition_strength"
            ]
            for row in rows
        )

        highest.extend(
            {
                "video": (
                    video["video"]
                ),
                **row,
            }
            for row in selected
        )

    highest = sorted(
        highest,
        key=lambda row: (
            -row[
                "transition_strength"
            ]
        ),
    )[:20]

    result = {
        "schema_version": "1.0",
        "method": (
            calibration_payload[
                "method"
            ]
        ),
        "calibration_mode": (
            calibration_mode
        ),
        "calibration": (
            calibration_payload
        ),
        "videos": videos_out,
        "summary": {
            "video_count": (
                len(
                    videos_out
                )
            ),
            "visual_boundary_count": (
                total_visual_boundaries
            ),
            "raw_strong_transition_count": (
                total_raw_strong
            ),
            "selected_scene_boundary_count": (
                total_selected
            ),
            "scene_count": (
                total_selected
                + len(videos_out)
            ),
            "scene_duration_seconds": (
                distribution(
                    all_scene_durations
                )
            ),
            "transition_strength": (
                distribution(
                    all_transition_strengths
                )
            ),
            "held_out_used_for_fit": (
                False
            ),
        },
        "highest_strength_boundaries": (
            highest
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "calibration_mode": (
                    calibration_mode
                ),
                "visual_otsu_threshold": (
                    calibration.visual_threshold
                ),
                "text_otsu_threshold": (
                    calibration.text_threshold
                ),
                "reference_text_length": (
                    calibration.reference_text_length
                ),
                **result["summary"],
                "top_boundaries": [
                    {
                        "video": row[
                            "video"
                        ],
                        "timestamp_seconds": (
                            row[
                                "timestamp_seconds"
                            ]
                        ),
                        "transition_strength": (
                            row[
                                "transition_strength"
                            ]
                        ),
                        "visual_change": (
                            row[
                                "visual_change"
                            ]
                        ),
                        "text_change": (
                            row[
                                "text_change"
                            ]
                        ),
                        "text_reliability": (
                            row[
                                "text_reliability"
                            ]
                        ),
                    }
                    for row in highest[:10]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"Saved: {OUTPUT}"
    )

    print(
        f"Frozen calibration: {CONFIG}"
    )


if __name__ == "__main__":
    main()
