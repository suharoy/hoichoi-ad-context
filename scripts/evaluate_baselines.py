"""Compare production placement against naive development baselines."""

from __future__ import annotations

import json
from pathlib import Path

from src.audio.speech_safety import (
    speech_context,
    speech_crosses_boundary,
)
from src.optimization.break_optimizer import (
    PacingPolicy,
    schedule_is_feasible,
)


ROOT = Path(__file__).resolve().parents[1]

PROFILE = (
    ROOT
    / "outputs"
    / "dev"
    / "boundary-profile-multimodal.json"
)

OPTIMIZED = (
    ROOT
    / "outputs"
    / "dev"
    / "optimized-breaks.json"
)

POLICY = (
    ROOT
    / "configs"
    / "pacing.demo.json"
)

CORPUS = (
    ROOT
    / "outputs"
    / "dev"
    / "corpus"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "baseline-evaluation.json"
)

GUARD_SECONDS = 0.20


def vad_segments(
    video_name: str,
) -> list[dict]:
    stem = Path(video_name).stem

    path = (
        CORPUS
        / stem
        / "vad.json"
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return payload[
        "speech_segments"
    ]


def evaluate_timestamp(
    timestamp: float,
    segments: list[dict],
) -> dict:
    crosses = speech_crosses_boundary(
        timestamp,
        segments,
        guard_seconds=GUARD_SECONDS,
    )

    context = speech_context(
        timestamp,
        segments,
    )

    inside = bool(
        context[
            "inside_detected_speech"
        ]
    )

    return {
        "timestamp_seconds": float(
            timestamp
        ),
        "speech_crosses_boundary": (
            bool(crosses)
        ),
        "inside_detected_speech": (
            inside
        ),
        "speech_safe": (
            not crosses
            and not inside
        ),
    }


def periodic_baseline(
    *,
    duration: float,
    policy: PacingPolicy,
) -> list[float]:
    """
    Fixed five-minute placement, subject only to beginning/end buffers.

    This intentionally ignores video/audio content.
    """
    timestamps = []

    timestamp = max(
        300.0,
        policy.minimum_first_break_seconds,
    )

    latest = (
        duration
        - policy.minimum_end_buffer_seconds
    )

    while timestamp <= latest:
        proposal = (
            timestamps
            + [timestamp]
        )

        if schedule_is_feasible(
            proposal,
            video_duration=duration,
            policy=policy,
        ):
            timestamps.append(
                timestamp
            )

        timestamp += 300.0

    return timestamps


def visual_only_baseline(
    *,
    video: dict,
    duration: float,
    target_count: int,
    policy: PacingPolicy,
) -> list[float]:
    """
    Greedily choose the strongest CLIP contextual shot transitions.

    Speech evidence and text evidence are deliberately ignored.
    Inventory count is matched to the production schedule where feasible.
    """
    ranked = []

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
            continue

        ranked.append(
            (
                float(change),
                float(
                    candidate[
                        "timestamp_seconds"
                    ]
                ),
            )
        )

    ranked.sort(
        key=lambda row: (
            -row[0],
            row[1],
        )
    )

    selected = []

    for _, timestamp in ranked:
        proposal = (
            selected
            + [timestamp]
        )

        if not schedule_is_feasible(
            proposal,
            video_duration=duration,
            policy=policy,
        ):
            continue

        selected.append(
            timestamp
        )

        if (
            len(selected)
            == target_count
        ):
            break

    return sorted(selected)


def summarize(
    rows: list[dict],
) -> dict:
    count = len(rows)

    unsafe = sum(
        not row[
            "speech_safe"
        ]
        for row in rows
    )

    crossing = sum(
        row[
            "speech_crosses_boundary"
        ]
        for row in rows
    )

    inside = sum(
        row[
            "inside_detected_speech"
        ]
        for row in rows
    )

    return {
        "break_count": count,
        "speech_safe_count": (
            count - unsafe
        ),
        "speech_unsafe_count": (
            unsafe
        ),
        "speech_safe_rate": (
            (count - unsafe) / count
            if count
            else None
        ),
        "speech_interruption_rate": (
            unsafe / count
            if count
            else None
        ),
        "speech_crossing_count": (
            crossing
        ),
        "inside_detected_speech_count": (
            inside
        ),
    }


def main() -> None:
    profile = json.loads(
        PROFILE.read_text(
            encoding="utf-8"
        )
    )

    optimized = json.loads(
        OPTIMIZED.read_text(
            encoding="utf-8"
        )
    )

    policy_payload = json.loads(
        POLICY.read_text(
            encoding="utf-8"
        )
    )

    policy = PacingPolicy.from_dict(
        policy_payload
    )

    profile_lookup = {
        item["video"]: item
        for item in profile[
            "videos"
        ]
    }

    videos = []

    aggregate = {
        "periodic": [],
        "visual_only": [],
        "production": [],
    }

    for optimized_video in optimized[
        "videos"
    ]:
        name = (
            optimized_video[
                "video"
            ]
        )

        source = (
            profile_lookup[
                name
            ]
        )

        duration = float(
            optimized_video[
                "duration_seconds"
            ]
        )

        target_count = len(
            optimized_video[
                "selected_breaks"
            ]
        )

        segments = vad_segments(
            name
        )

        periodic_times = (
            periodic_baseline(
                duration=duration,
                policy=policy,
            )
        )

        visual_times = (
            visual_only_baseline(
                video=source,
                duration=duration,
                target_count=(
                    target_count
                ),
                policy=policy,
            )
        )

        production_times = [
            float(
                row[
                    "timestamp_seconds"
                ]
            )
            for row
            in optimized_video[
                "selected_breaks"
            ]
        ]

        methods = {}

        for method, timestamps in (
            (
                "periodic",
                periodic_times,
            ),
            (
                "visual_only",
                visual_times,
            ),
            (
                "production",
                production_times,
            ),
        ):
            rows = [
                evaluate_timestamp(
                    timestamp,
                    segments,
                )
                for timestamp
                in timestamps
            ]

            aggregate[
                method
            ].extend(
                rows
            )

            methods[
                method
            ] = {
                **summarize(
                    rows
                ),
                "timestamps": (
                    rows
                ),
            }

        videos.append(
            {
                "video": name,
                "duration_seconds": (
                    duration
                ),
                "production_inventory_count": (
                    target_count
                ),
                "methods": methods,
            }
        )

    result = {
        "evaluation": (
            "development_baseline_comparison"
        ),
        "scope": (
            "speech-interruption proxy; "
            "not human perceptual quality"
        ),
        "baselines": {
            "periodic": (
                "fixed five-minute placement "
                "with pacing constraints; "
                "no content understanding"
            ),
            "visual_only": (
                "strongest CLIP contextual "
                "shot transitions, matched to "
                "production inventory count "
                "where feasible; no speech/text"
            ),
            "production": (
                "hard speech safety + "
                "multimodal EABS + MILP pacing"
            ),
        },
        "policy": (
            policy_payload
        ),
        "videos": videos,
        "aggregate": {
            method: summarize(
                rows
            )
            for method, rows
            in aggregate.items()
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result[
                "aggregate"
            ],
            indent=2,
        )
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
