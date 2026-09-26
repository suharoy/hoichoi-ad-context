"""Exact global optimization of ad-break schedules under pacing constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


EABS_NEUTRAL_BASELINE = 50.0


def eabs_utility(score: float) -> float:
    """Return break quality utility relative to neutral multimodal evidence."""
    score = float(score)

    if not np.isfinite(score):
        raise ValueError("EABS must be finite")

    if not 0.0 <= score <= 100.0:
        raise ValueError("EABS must be in [0, 100]")

    return score - EABS_NEUTRAL_BASELINE


@dataclass(frozen=True)
class PacingPolicy:
    policy_name: str
    minimum_first_break_seconds: float
    minimum_end_buffer_seconds: float
    minimum_gap_seconds: float
    max_breaks_per_hour: int
    maximum_ad_load_fraction: float
    nominal_ad_duration_seconds: float

    @classmethod
    def from_dict(cls, payload: dict) -> "PacingPolicy":
        policy = cls(
            policy_name=str(payload["policy_name"]),
            minimum_first_break_seconds=float(
                payload["minimum_first_break_seconds"]
            ),
            minimum_end_buffer_seconds=float(
                payload["minimum_end_buffer_seconds"]
            ),
            minimum_gap_seconds=float(
                payload["minimum_gap_seconds"]
            ),
            max_breaks_per_hour=int(
                payload["max_breaks_per_hour"]
            ),
            maximum_ad_load_fraction=float(
                payload["maximum_ad_load_fraction"]
            ),
            nominal_ad_duration_seconds=float(
                payload["nominal_ad_duration_seconds"]
            ),
        )

        policy.validate()
        return policy

    def validate(self) -> None:
        if self.minimum_first_break_seconds < 0:
            raise ValueError(
                "minimum_first_break_seconds cannot be negative"
            )

        if self.minimum_end_buffer_seconds < 0:
            raise ValueError(
                "minimum_end_buffer_seconds cannot be negative"
            )

        if self.minimum_gap_seconds < 0:
            raise ValueError(
                "minimum_gap_seconds cannot be negative"
            )

        if self.max_breaks_per_hour < 1:
            raise ValueError(
                "max_breaks_per_hour must be >= 1"
            )

        if not 0 < self.maximum_ad_load_fraction <= 1:
            raise ValueError(
                "maximum_ad_load_fraction must be in (0, 1]"
            )

        if self.nominal_ad_duration_seconds <= 0:
            raise ValueError(
                "nominal_ad_duration_seconds must be positive"
            )


@dataclass(frozen=True)
class BreakCandidate:
    source_index: int
    timestamp: float
    score: float


def schedule_is_feasible(
    timestamps: list[float],
    *,
    video_duration: float,
    policy: PacingPolicy,
) -> bool:
    """Validate all pacing constraints for a proposed timestamp set."""
    ordered = sorted(float(value) for value in timestamps)

    if not ordered:
        return True

    if ordered[0] < policy.minimum_first_break_seconds:
        return False

    latest_allowed = (
        video_duration - policy.minimum_end_buffer_seconds
    )

    if ordered[-1] > latest_allowed:
        return False

    for left, right in zip(ordered, ordered[1:]):
        if right - left < policy.minimum_gap_seconds:
            return False

    # Rolling one-hour limit.
    for start in ordered:
        count = sum(
            1
            for timestamp in ordered
            if start <= timestamp < start + 3600.0
        )

        if count > policy.max_breaks_per_hour:
            return False

    ad_load = (
        len(ordered)
        * policy.nominal_ad_duration_seconds
        / video_duration
    )

    if (
        ad_load
        > policy.maximum_ad_load_fraction + 1e-12
    ):
        return False

    return True


def policy_window_candidates(
    candidates: list[BreakCandidate],
    *,
    video_duration: float,
    policy: PacingPolicy,
) -> tuple[list[BreakCandidate], dict[str, int]]:
    """
    Remove candidates impossible because of beginning/end buffers.

    Other pacing constraints are solved globally by MILP.
    """
    valid: list[BreakCandidate] = []

    rejected_first = 0
    rejected_end = 0

    latest_allowed = (
        video_duration - policy.minimum_end_buffer_seconds
    )

    for candidate in candidates:
        if (
            candidate.timestamp
            < policy.minimum_first_break_seconds
        ):
            rejected_first += 1
            continue

        if candidate.timestamp > latest_allowed:
            rejected_end += 1
            continue

        valid.append(candidate)

    return valid, {
        "rejected_first_break_buffer": rejected_first,
        "rejected_end_buffer": rejected_end,
    }


def positive_quality_candidates(
    candidates: list[BreakCandidate],
) -> tuple[list[BreakCandidate], int]:
    """
    Keep only candidates with EABS strictly above the neutral baseline.

    EABS=50 is the mathematically neutral point of the percentile-calibrated
    evidence score. A candidate at or below 50 has non-positive quality
    utility and therefore should not consume ad inventory.
    """
    kept: list[BreakCandidate] = []

    for candidate in candidates:
        if eabs_utility(candidate.score) > 0.0:
            kept.append(candidate)

    return kept, len(candidates) - len(kept)


def optimize_break_schedule(
    candidates: list[BreakCandidate],
    *,
    video_duration: float,
    policy: PacingPolicy,
) -> dict:
    """
    Solve the exact maximum-quality feasible schedule using binary MILP.

    The optimized utility is:

        sum(EABS - 50)

    over selected candidates with EABS > 50.

    This allows the optimizer to leave ad inventory unused instead of
    selecting below-neutral break opportunities merely because every raw
    EABS value is positive.
    """
    candidates, filter_counts = policy_window_candidates(
        candidates,
        video_duration=video_duration,
        policy=policy,
    )

    pacing_candidate_count = len(candidates)

    candidates, rejected_below_neutral_quality = (
        positive_quality_candidates(candidates)
    )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.timestamp,
            candidate.source_index,
        ),
    )

    quality_candidate_count = len(candidates)
    n = quality_candidate_count

    if n == 0:
        return {
            "selected": [],
            "eabs_sum": 0.0,
            "quality_utility": 0.0,
            "solver_success": True,
            "solver_message": (
                "No positive-quality feasible candidates"
            ),
            "pacing_candidate_count": (
                pacing_candidate_count
            ),
            "quality_candidate_count": 0,
            "rejected_below_neutral_quality": (
                rejected_below_neutral_quality
            ),
            "minimum_gap_constraints": 0,
            **filter_counts,
        }

    times = np.asarray(
        [
            candidate.timestamp
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    scores = np.asarray(
        [
            candidate.score
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    if not np.isfinite(times).all():
        raise ValueError(
            "Candidate timestamps must be finite"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Candidate scores must be finite"
        )

    # All candidates have already passed eabs_utility(), so these values
    # are strictly positive and mathematically tied to the EABS neutral point.
    utilities = scores - EABS_NEUTRAL_BASELINE

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    minimum_gap_constraint_count = 0

    # ---------------------------------------------------------
    # Minimum-gap conflicts.
    # ---------------------------------------------------------

    for i in range(n):
        for j in range(i + 1, n):
            if (
                times[j] - times[i]
                >= policy.minimum_gap_seconds
            ):
                break

            row = np.zeros(
                n,
                dtype=np.float64,
            )
            row[i] = 1.0
            row[j] = 1.0

            rows.append(row)
            lower_bounds.append(-np.inf)
            upper_bounds.append(1.0)

            minimum_gap_constraint_count += 1

    # ---------------------------------------------------------
    # Maximum breaks in every rolling 60-minute window.
    #
    # Anchoring windows at each candidate is sufficient for
    # discrete candidate timestamps: any violating window can be
    # shifted to its earliest contained selected candidate.
    # ---------------------------------------------------------

    for start_index in range(n):
        start = times[start_index]

        indices = np.flatnonzero(
            (times >= start)
            & (times < start + 3600.0)
        )

        if (
            len(indices)
            <= policy.max_breaks_per_hour
        ):
            continue

        row = np.zeros(
            n,
            dtype=np.float64,
        )
        row[indices] = 1.0

        rows.append(row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(
            float(policy.max_breaks_per_hour)
        )

    # ---------------------------------------------------------
    # Total ad-load constraint.
    # ---------------------------------------------------------

    row = np.full(
        n,
        policy.nominal_ad_duration_seconds,
        dtype=np.float64,
    )

    rows.append(row)
    lower_bounds.append(-np.inf)
    upper_bounds.append(
        policy.maximum_ad_load_fraction
        * video_duration
    )

    matrix = np.vstack(rows)

    constraints = LinearConstraint(
        matrix,
        np.asarray(
            lower_bounds,
            dtype=np.float64,
        ),
        np.asarray(
            upper_bounds,
            dtype=np.float64,
        ),
    )

    # scipy.optimize.milp minimizes, so negate positive quality utility.
    objective = -utilities

    result = milp(
        c=objective,
        integrality=np.ones(
            n,
            dtype=np.int32,
        ),
        bounds=Bounds(
            np.zeros(
                n,
                dtype=np.float64,
            ),
            np.ones(
                n,
                dtype=np.float64,
            ),
        ),
        constraints=constraints,
        options={
            "disp": False,
        },
    )

    if not result.success or result.x is None:
        raise RuntimeError(
            f"MILP failed: {result.message}"
        )

    selected_indices = np.flatnonzero(
        result.x > 0.5
    )

    selected = [
        candidates[index]
        for index in selected_indices
    ]

    selected_times = [
        candidate.timestamp
        for candidate in selected
    ]

    if not schedule_is_feasible(
        selected_times,
        video_duration=video_duration,
        policy=policy,
    ):
        raise RuntimeError(
            "MILP returned a schedule that violates pacing constraints"
        )

    selected_eabs_sum = float(
        sum(
            candidate.score
            for candidate in selected
        )
    )

    selected_quality_utility = float(
        sum(
            eabs_utility(candidate.score)
            for candidate in selected
        )
    )

    return {
        "selected": selected,
        "eabs_sum": selected_eabs_sum,
        "quality_utility": (
            selected_quality_utility
        ),
        "solver_success": True,
        "solver_message": str(result.message),
        "pacing_candidate_count": (
            pacing_candidate_count
        ),
        "quality_candidate_count": (
            quality_candidate_count
        ),
        "rejected_below_neutral_quality": (
            rejected_below_neutral_quality
        ),
        "minimum_gap_constraints": (
            minimum_gap_constraint_count
        ),
        **filter_counts,
    }


def greedy_break_schedule(
    candidates: list[BreakCandidate],
    *,
    video_duration: float,
    policy: PacingPolicy,
) -> dict:
    """
    Score-descending greedy baseline under the same constraints.

    Only candidates with positive EABS quality utility are considered.
    This is an evaluation baseline, not the production selector.
    """
    candidates, filter_counts = policy_window_candidates(
        candidates,
        video_duration=video_duration,
        policy=policy,
    )

    pacing_candidate_count = len(candidates)

    candidates, rejected_below_neutral_quality = (
        positive_quality_candidates(candidates)
    )

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.timestamp,
            candidate.source_index,
        ),
    )

    selected: list[BreakCandidate] = []

    for candidate in ranked:
        proposal = selected + [candidate]

        if schedule_is_feasible(
            [
                item.timestamp
                for item in proposal
            ],
            video_duration=video_duration,
            policy=policy,
        ):
            selected.append(candidate)

    selected.sort(
        key=lambda candidate: (
            candidate.timestamp,
            candidate.source_index,
        )
    )

    selected_eabs_sum = float(
        sum(
            candidate.score
            for candidate in selected
        )
    )

    selected_quality_utility = float(
        sum(
            eabs_utility(candidate.score)
            for candidate in selected
        )
    )

    return {
        "selected": selected,
        "eabs_sum": selected_eabs_sum,
        "quality_utility": (
            selected_quality_utility
        ),
        "pacing_candidate_count": (
            pacing_candidate_count
        ),
        "quality_candidate_count": len(
            candidates
        ),
        "rejected_below_neutral_quality": (
            rejected_below_neutral_quality
        ),
        **filter_counts,
    }
