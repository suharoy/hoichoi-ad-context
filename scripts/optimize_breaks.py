"""Optimize globally feasible ad-break schedules from frozen EABS scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.optimization.break_optimizer import (
    EABS_NEUTRAL_BASELINE,
    BreakCandidate,
    PacingPolicy,
    greedy_break_schedule,
    optimize_break_schedule,
    schedule_is_feasible,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "boundary-profile-scored.json"
)

DEFAULT_POLICY = (
    ROOT
    / "configs"
    / "pacing.demo.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "optimized-breaks.json"
)


def derive_video_duration(video: dict) -> float:
    """
    Recover video duration from shot metadata already stored by Stage 3.

    The final shot end timestamp should approximate the source duration
    without reopening the media file.
    """
    ends: list[float] = []

    for candidate in video["candidates"]:
        visual = candidate.get("visual_semantics") or {}

        previous_shot = (
            visual.get("previous_shot") or {}
        )
        next_shot = (
            visual.get("next_shot") or {}
        )

        for shot in (
            previous_shot,
            next_shot,
        ):
            end = shot.get("end")

            if end is not None:
                ends.append(float(end))

    if not ends:
        raise ValueError(
            f"Cannot derive duration for {video['video']}"
        )

    duration = max(ends)

    if (
        not np.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(
            f"Invalid duration for "
            f"{video['video']}: {duration}"
        )

    return duration


def source_candidates(
    video: dict,
) -> list[BreakCandidate]:
    """
    Extract frozen Stage-4 candidates eligible for optimization.

    Stage 5 does not recalculate or alter EABS.
    """
    candidates: list[BreakCandidate] = []

    for index, candidate in enumerate(
        video["candidates"]
    ):
        break_score = (
            candidate.get("break_score") or {}
        )

        if (
            break_score.get("ad_eligible")
            is not True
        ):
            continue

        if (
            break_score.get("scorable")
            is not True
        ):
            continue

        score = break_score.get("eabs")

        if score is None:
            continue

        score = float(score)

        if not np.isfinite(score):
            raise ValueError(
                f"Non-finite EABS in "
                f"{video['video']}"
            )

        if not 0.0 <= score <= 100.0:
            raise ValueError(
                f"EABS outside [0,100] "
                f"in {video['video']}"
            )

        candidates.append(
            BreakCandidate(
                source_index=index,
                timestamp=float(
                    candidate[
                        "timestamp_seconds"
                    ]
                ),
                score=score,
            )
        )

    return candidates


def minimum_selected_gap(
    timestamps: list[float],
) -> float | None:
    ordered = sorted(timestamps)

    if len(ordered) < 2:
        return None

    return min(
        right - left
        for left, right in zip(
            ordered,
            ordered[1:],
        )
    )


def selected_record(
    candidate: BreakCandidate,
    source_candidate: dict,
) -> dict:
    """
    Preserve enough source evidence for audit/debug output
    without recomputing any frozen Stage-4 feature.
    """
    break_score = source_candidate[
        "break_score"
    ]

    return {
        "source_candidate_index": (
            candidate.source_index
        ),
        "timestamp_seconds": (
            candidate.timestamp
        ),
        "eabs": candidate.score,
        "quality_utility": (
            candidate.score
            - EABS_NEUTRAL_BASELINE
        ),
        "centered_pause": (
            source_candidate.get(
                "centered_pause"
            )
        ),
        "break_score": break_score,
        "visual_context_change": (
            source_candidate
            .get("visual_semantics", {})
            .get("context_change")
        ),
        "text_semantic_change": (
            source_candidate
            .get("text_semantics", {})
            .get("semantic_change")
        ),
    }


def selected_timestamps(
    result: dict,
) -> list[float]:
    return [
        candidate.timestamp
        for candidate in result["selected"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Globally optimize ad-break schedules "
            "from frozen EABS scores."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    payload = json.loads(
        args.input.read_text(
            encoding="utf-8"
        )
    )

    policy_payload = json.loads(
        args.policy.read_text(
            encoding="utf-8"
        )
    )

    policy = PacingPolicy.from_dict(
        policy_payload
    )

    results: list[dict] = []

    total_optimal_utility = 0.0
    total_greedy_utility = 0.0

    total_optimal_eabs = 0.0
    total_greedy_eabs = 0.0

    total_selected = 0
    total_quality_candidates = 0
    total_below_neutral_rejected = 0

    for video in payload["videos"]:
        video_name = video["video"]

        duration = derive_video_duration(
            video
        )

        candidates = source_candidates(
            video
        )

        optimal = optimize_break_schedule(
            candidates,
            video_duration=duration,
            policy=policy,
        )

        greedy = greedy_break_schedule(
            candidates,
            video_duration=duration,
            policy=policy,
        )

        optimal_times = (
            selected_timestamps(optimal)
        )

        greedy_times = (
            selected_timestamps(greedy)
        )

        if not schedule_is_feasible(
            optimal_times,
            video_duration=duration,
            policy=policy,
        ):
            raise RuntimeError(
                f"Invalid MILP schedule "
                f"for {video_name}"
            )

        if not schedule_is_feasible(
            greedy_times,
            video_duration=duration,
            policy=policy,
        ):
            raise RuntimeError(
                f"Invalid greedy schedule "
                f"for {video_name}"
            )

        selected = optimal["selected"]

        records = [
            selected_record(
                candidate,
                video["candidates"][
                    candidate.source_index
                ],
            )
            for candidate in selected
        ]

        selected_count = len(selected)

        ad_load_fraction = (
            selected_count
            * policy.nominal_ad_duration_seconds
            / duration
        )

        optimal_utility = float(
            optimal["quality_utility"]
        )

        greedy_utility = float(
            greedy["quality_utility"]
        )

        optimal_eabs = float(
            optimal["eabs_sum"]
        )

        greedy_eabs = float(
            greedy["eabs_sum"]
        )

        optimizer_gain = (
            optimal_utility
            - greedy_utility
        )

        if optimizer_gain < -1e-8:
            raise RuntimeError(
                f"MILP objective is worse "
                f"than greedy for {video_name}"
            )

        total_optimal_utility += (
            optimal_utility
        )

        total_greedy_utility += (
            greedy_utility
        )

        total_optimal_eabs += (
            optimal_eabs
        )

        total_greedy_eabs += (
            greedy_eabs
        )

        total_selected += (
            selected_count
        )

        total_quality_candidates += (
            optimal[
                "quality_candidate_count"
            ]
        )

        total_below_neutral_rejected += (
            optimal[
                "rejected_below_neutral_quality"
            ]
        )

        results.append(
            {
                "video": video_name,
                "duration_seconds": duration,

                "input_eabs_candidates": len(
                    candidates
                ),

                "pacing_candidate_count": (
                    optimal[
                        "pacing_candidate_count"
                    ]
                ),

                "quality_candidate_count": (
                    optimal[
                        "quality_candidate_count"
                    ]
                ),

                "rejected_first_break_buffer": (
                    optimal[
                        "rejected_first_break_buffer"
                    ]
                ),

                "rejected_end_buffer": (
                    optimal[
                        "rejected_end_buffer"
                    ]
                ),

                "rejected_below_neutral_quality": (
                    optimal[
                        "rejected_below_neutral_quality"
                    ]
                ),

                "selected_break_count": (
                    selected_count
                ),

                "selected_breaks": (
                    records
                ),

                "selected_eabs_sum": (
                    optimal_eabs
                ),

                "quality_utility": (
                    optimal_utility
                ),

                "greedy_selected_eabs_sum": (
                    greedy_eabs
                ),

                "greedy_quality_utility": (
                    greedy_utility
                ),

                "optimizer_gain_over_greedy_utility": (
                    optimizer_gain
                ),

                "minimum_selected_gap_seconds": (
                    minimum_selected_gap(
                        optimal_times
                    )
                ),

                "ad_load_fraction": (
                    ad_load_fraction
                ),

                "solver_success": (
                    optimal[
                        "solver_success"
                    ]
                ),

                "solver_message": (
                    optimal[
                        "solver_message"
                    ]
                ),
            }
        )

    summary_gain = (
        total_optimal_utility
        - total_greedy_utility
    )

    if summary_gain < -1e-8:
        raise RuntimeError(
            "Corpus MILP utility is worse "
            "than greedy baseline"
        )

    report = {
        "optimizer": (
            "scipy.optimize.milp / HiGHS"
        ),

        "objective": (
            "maximize sum(EABS - 50) "
            "for positive-quality "
            "break opportunities"
        ),

        "eabs_neutral_baseline": (
            EABS_NEUTRAL_BASELINE
        ),

        "neutral_baseline_rationale": (
            "EABS=50 corresponds to "
            "median-strength multimodal evidence "
            "under percentile calibration; "
            "candidates at or below 50 have "
            "non-positive quality utility."
        ),

        "policy": policy_payload,

        "videos": results,

        "summary": {
            "video_count": len(
                results
            ),

            "selected_breaks_total": (
                total_selected
            ),

            "quality_candidates_total": (
                total_quality_candidates
            ),

            "rejected_below_neutral_quality_total": (
                total_below_neutral_rejected
            ),

            "optimal_selected_eabs_sum_total": (
                total_optimal_eabs
            ),

            "greedy_selected_eabs_sum_total": (
                total_greedy_eabs
            ),

            "optimal_quality_utility_total": (
                total_optimal_utility
            ),

            "greedy_quality_utility_total": (
                total_greedy_utility
            ),

            "optimizer_gain_over_greedy_utility_total": (
                summary_gain
            ),

            "all_constraints_satisfied": True,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nSaved: {args.output}"
    )


if __name__ == "__main__":
    main()