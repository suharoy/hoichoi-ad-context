"""Sequential semantic-scene grouping from multimodal boundary evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.break_scoring.evidence_score import (
    text_evidence_reliability,
)


@dataclass(frozen=True)
class SceneCalibration:
    visual_threshold: float
    text_threshold: float
    reference_text_length: float


def otsu_threshold(
    values: list[float],
) -> float:
    """
    Exact one-dimensional Otsu split.

    Chooses the threshold that maximizes between-class variance.
    No histogram bin-count hyperparameter is required.
    """
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim != 1
        or len(array) < 2
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            "Otsu requires at least two finite values"
        )

    ordered = np.sort(array)

    unique = np.unique(ordered)

    if len(unique) < 2:
        raise ValueError(
            "Otsu requires at least two distinct values"
        )

    cumulative = np.cumsum(ordered)
    total_sum = float(cumulative[-1])
    n = len(ordered)

    best_threshold = None
    best_variance = -1.0

    # A legal split can occur only between unequal adjacent values.
    for index in range(n - 1):
        left_value = ordered[index]
        right_value = ordered[index + 1]

        if left_value == right_value:
            continue

        left_count = index + 1
        right_count = n - left_count

        left_sum = float(cumulative[index])
        right_sum = total_sum - left_sum

        left_mean = left_sum / left_count
        right_mean = right_sum / right_count

        left_weight = left_count / n
        right_weight = right_count / n

        between_variance = (
            left_weight
            * right_weight
            * (
                left_mean
                - right_mean
            )
            ** 2
        )

        threshold = float(
            (
                left_value
                + right_value
            )
            / 2.0
        )

        if (
            between_variance
            > best_variance
        ):
            best_variance = (
                between_variance
            )
            best_threshold = threshold

    if best_threshold is None:
        raise RuntimeError(
            "Otsu failed to find a split"
        )

    return best_threshold


def transition_evidence(
    candidate: dict,
    calibration: SceneCalibration,
) -> dict:
    """
    Score one visual shot boundary as semantic-scene evidence.

    Strength=1 means the fused evidence lies exactly at the
    development-derived modality separation point.

    This is an interpretable relative transition strength,
    not a probability.
    """
    visual = (
        candidate.get(
            "visual_semantics"
        )
        or {}
    )

    visual_change = visual.get(
        "context_change"
    )

    if visual_change is None:
        raise ValueError(
            "Scene grouping requires visual context change"
        )

    visual_change = float(
        visual_change
    )

    if (
        not math.isfinite(
            visual_change
        )
        or visual_change < 0
    ):
        raise ValueError(
            "Invalid visual context change"
        )

    if calibration.visual_threshold <= 0:
        raise ValueError(
            "Visual threshold must be positive"
        )

    visual_ratio = (
        visual_change
        / calibration.visual_threshold
    )

    semantic = (
        candidate.get(
            "text_semantics"
        )
        or {}
    )

    text_change = None
    text_ratio = None
    text_reliability = 0.0
    effective_text_length = 0.0

    if semantic.get(
        "available",
        False,
    ):
        raw_text_change = (
            semantic.get(
                "semantic_change"
            )
        )

        if raw_text_change is not None:
            text_change = float(
                raw_text_change
            )

            if (
                not math.isfinite(
                    text_change
                )
                or text_change < 0
            ):
                raise ValueError(
                    "Invalid text semantic change"
                )

            if calibration.text_threshold <= 0:
                raise ValueError(
                    "Text threshold must be positive"
                )

            (
                effective_text_length,
                text_reliability,
            ) = text_evidence_reliability(
                semantic.get(
                    "left_characters"
                ),
                semantic.get(
                    "right_characters"
                ),
                calibration.reference_text_length,
            )

            if text_reliability > 0:
                text_ratio = (
                    text_change
                    / calibration.text_threshold
                )

    numerator = visual_ratio
    denominator = 1.0

    if (
        text_ratio is not None
        and text_reliability > 0
    ):
        numerator += (
            text_reliability
            * text_ratio
        )
        denominator += (
            text_reliability
        )

    strength = (
        numerator / denominator
    )

    return {
        "transition_strength": (
            float(strength)
        ),
        "visual_change": (
            visual_change
        ),
        "visual_threshold_ratio": (
            float(visual_ratio)
        ),
        "text_change": (
            text_change
        ),
        "text_threshold_ratio": (
            float(text_ratio)
            if text_ratio is not None
            else None
        ),
        "text_reliability": (
            float(text_reliability)
        ),
        "effective_text_length": (
            float(effective_text_length)
        ),
        "strong": bool(
            strength >= 1.0
        ),
    }


def collapse_adjacent_strong_boundaries(
    rows: list[dict],
) -> list[dict]:
    """
    Keep one transition from each consecutive run of strong boundaries.

    This suppresses multiple neighbouring shot cuts representing the same
    semantic transition without introducing a duration hyperparameter.
    """
    strong = [
        row
        for row in rows
        if row["strong"]
    ]

    if not strong:
        return []

    selected: list[dict] = []
    run: list[dict] = [
        strong[0]
    ]

    for row in strong[1:]:
        if (
            int(row["candidate_index"])
            == int(
                run[-1][
                    "candidate_index"
                ]
            )
            + 1
        ):
            run.append(row)
            continue

        selected.append(
            max(
                run,
                key=lambda item: (
                    item[
                        "transition_strength"
                    ],
                    -item[
                        "candidate_index"
                    ],
                ),
            )
        )

        run = [row]

    selected.append(
        max(
            run,
            key=lambda item: (
                item[
                    "transition_strength"
                ],
                -item[
                    "candidate_index"
                ],
            ),
        )
    )

    return sorted(
        selected,
        key=lambda item: (
            item["timestamp_seconds"]
        ),
    )


def build_scene_intervals(
    *,
    video_name: str,
    candidates: list[dict],
    selected_boundaries: list[dict],
) -> list[dict]:
    """
    Convert selected internal shot boundaries into contiguous scene intervals.
    """
    if not candidates:
        raise ValueError(
            "Cannot build scenes without shot boundaries"
        )

    visual_records = [
        candidate["visual_semantics"]
        for candidate in candidates
    ]

    shot_records = []

    for visual in visual_records:
        for side in (
            "previous_shot",
            "next_shot",
        ):
            shot = visual.get(side)

            if shot is not None:
                shot_records.append(
                    shot
                )

    if not shot_records:
        raise ValueError(
            "No shot metadata available"
        )

    duration = max(
        float(
            shot["end"]
        )
        for shot in shot_records
    )

    shot_count = (
        max(
            int(
                shot["index"]
            )
            for shot in shot_records
        )
        + 1
    )

    cuts = []

    for row in selected_boundaries:
        candidate = candidates[
            int(
                row[
                    "candidate_index"
                ]
            )
        ]

        next_shot = (
            candidate[
                "visual_semantics"
            ][
                "next_shot"
            ]
        )

        if next_shot is None:
            raise ValueError(
                "Selected internal boundary lacks next shot"
            )

        cuts.append(
            {
                "timestamp_seconds": float(
                    row[
                        "timestamp_seconds"
                    ]
                ),
                "next_shot_index": int(
                    next_shot[
                        "index"
                    ]
                ),
                "transition_strength": float(
                    row[
                        "transition_strength"
                    ]
                ),
                "visual_change": (
                    row["visual_change"]
                ),
                "text_change": (
                    row["text_change"]
                ),
                "text_reliability": (
                    row[
                        "text_reliability"
                    ]
                ),
            }
        )

    scenes = []

    start_seconds = 0.0
    start_shot = 0

    for number, cut in enumerate(
        cuts,
        start=1,
    ):
        end_seconds = (
            cut[
                "timestamp_seconds"
            ]
        )

        next_shot = (
            cut[
                "next_shot_index"
            ]
        )

        if (
            end_seconds
            <= start_seconds
            or next_shot
            <= start_shot
        ):
            raise ValueError(
                "Non-monotonic scene cut"
            )

        scenes.append(
            {
                "scene_id": (
                    f"{video_name.rsplit('.', 1)[0]}"
                    f"-scene-{number:03d}"
                ),
                "start_seconds": (
                    start_seconds
                ),
                "end_seconds": (
                    end_seconds
                ),
                "duration_seconds": (
                    end_seconds
                    - start_seconds
                ),
                "start_shot_index": (
                    start_shot
                ),
                "end_shot_index": (
                    next_shot - 1
                ),
                "shot_count": (
                    next_shot
                    - start_shot
                ),
                "representative_shot_index": (
                    (
                        start_shot
                        + next_shot
                        - 1
                    )
                    // 2
                ),
                "ending_transition": (
                    cut
                ),
            }
        )

        start_seconds = end_seconds
        start_shot = next_shot

    number = len(scenes) + 1

    scenes.append(
        {
            "scene_id": (
                f"{video_name.rsplit('.', 1)[0]}"
                f"-scene-{number:03d}"
            ),
            "start_seconds": (
                start_seconds
            ),
            "end_seconds": (
                duration
            ),
            "duration_seconds": (
                duration
                - start_seconds
            ),
            "start_shot_index": (
                start_shot
            ),
            "end_shot_index": (
                shot_count - 1
            ),
            "shot_count": (
                shot_count
                - start_shot
            ),
            "representative_shot_index": (
                (
                    start_shot
                    + shot_count
                    - 1
                )
                // 2
            ),
            "ending_transition": None,
        }
    )

    return scenes
