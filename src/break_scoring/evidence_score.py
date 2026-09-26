"""Evidence-aware multimodal break suitability scoring."""

from __future__ import annotations

import math

from src.break_scoring.calibration import harmonic_mean


def text_evidence_reliability(
    left_characters: int | None,
    right_characters: int | None,
    reference_text_length: float,
) -> tuple[float, float]:
    """
    Return:
        effective_text_length,
        reliability in [0, 1]

    Reliability is zero when either side lacks usable text.
    """
    if reference_text_length <= 0:
        raise ValueError("reference_text_length must be positive")

    if not left_characters or not right_characters:
        return 0.0, 0.0

    effective_length = harmonic_mean(
        float(left_characters),
        float(right_characters),
    )

    reliability = min(
        1.0,
        effective_length / reference_text_length,
    )

    return effective_length, reliability


def evidence_aware_break_score(
    acoustic_percentile: float,
    visual_percentile: float,
    *,
    text_percentile: float | None = None,
    text_reliability: float = 0.0,
) -> float:
    """
    Evidence-Aware Break Suitability Score (EABS).

    This is a relative suitability index, NOT a probability.
    """
    acoustic_percentile = float(acoustic_percentile)
    visual_percentile = float(visual_percentile)
    text_reliability = float(text_reliability)

    for name, value in (
        ("acoustic_percentile", acoustic_percentile),
        ("visual_percentile", visual_percentile),
        ("text_reliability", text_reliability),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    numerator = acoustic_percentile + visual_percentile
    denominator = 2.0

    if text_reliability > 0.0:
        if text_percentile is None:
            raise ValueError(
                "text_percentile is required when text reliability > 0"
            )

        text_percentile = float(text_percentile)

        if not math.isfinite(text_percentile):
            raise ValueError("text_percentile must be finite")

        if not 0.0 <= text_percentile <= 1.0:
            raise ValueError("text_percentile must be in [0, 1]")

        numerator += text_reliability * text_percentile
        denominator += text_reliability

    score = 100.0 * numerator / denominator

    # Floating-point defensive check.
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"Computed EABS outside [0,100]: {score}")

    return score


def acoustic_pause_margin(
    seconds_since_speech_end: float | None,
    seconds_until_speech_start: float | None,
) -> float | None:
    """
    Distance to the nearest detected speech on either available side.

    Missing sides are ignored rather than treated as infinity.
    If neither side is available, acoustic evidence is unavailable.
    """
    available = []

    for value in (
        seconds_since_speech_end,
        seconds_until_speech_start,
    ):
        if value is not None:
            value = float(value)

            if not math.isfinite(value):
                raise ValueError("Speech-distance values must be finite")

            if value < 0:
                raise ValueError("Speech-distance values cannot be negative")

            available.append(value)

    if not available:
        return None

    return min(available)