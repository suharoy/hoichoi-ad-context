from __future__ import annotations


def speech_crosses_boundary(
    timestamp: float,
    speech_segments: list[dict[str, float]],
    *,
    guard_seconds: float = 0.20,
) -> bool:
    """
    Return True when one continuous detected-speech region spans
    both sides of a candidate boundary.

    This is evidence that cutting here could interrupt an utterance.
    """
    left = timestamp - guard_seconds
    right = timestamp + guard_seconds

    return any(
        segment["start"] <= left
        and segment["end"] >= right
        for segment in speech_segments
    )


def speech_context(
    timestamp: float,
    speech_segments: list[dict[str, float]],
) -> dict[str, float | bool | None]:

    containing = next(
        (
            segment
            for segment in speech_segments
            if segment["start"] <= timestamp <= segment["end"]
        ),
        None,
    )

    if containing is not None:
        return {
            "inside_detected_speech": True,
            "seconds_since_speech_end": None,
            "seconds_until_speech_start": None,
            "seconds_since_speech_start": round(
                timestamp - containing["start"], 3
            ),
            "seconds_until_speech_end": round(
                containing["end"] - timestamp, 3
            ),
        }

    previous_end = max(
        (
            segment["end"]
            for segment in speech_segments
            if segment["end"] < timestamp
        ),
        default=None,
    )

    next_start = min(
        (
            segment["start"]
            for segment in speech_segments
            if segment["start"] > timestamp
        ),
        default=None,
    )

    return {
        "inside_detected_speech": False,
        "seconds_since_speech_end": (
            round(timestamp - previous_end, 3)
            if previous_end is not None
            else None
        ),
        "seconds_until_speech_start": (
            round(next_start - timestamp, 3)
            if next_start is not None
            else None
        ),
        "seconds_since_speech_start": None,
        "seconds_until_speech_end": None,
    }