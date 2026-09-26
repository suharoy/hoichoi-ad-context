from src.optimization.break_optimizer import (
    BreakCandidate,
    PacingPolicy,
    greedy_break_schedule,
    optimize_break_schedule,
    schedule_is_feasible,
)


def policy(
    *,
    gap=300,
    max_per_hour=10,
    ad_load=1.0,
    ad_duration=30,
    first=0,
    end=0,
):
    return PacingPolicy(
        policy_name="test",
        minimum_first_break_seconds=first,
        minimum_end_buffer_seconds=end,
        minimum_gap_seconds=gap,
        max_breaks_per_hour=max_per_hour,
        maximum_ad_load_fraction=ad_load,
        nominal_ad_duration_seconds=ad_duration,
    )


def candidate(index, timestamp, score):
    return BreakCandidate(
        source_index=index,
        timestamp=timestamp,
        score=score,
    )


def test_optimizer_can_beat_greedy():
    candidates = [
        candidate(0, 300, 58),
        candidate(1, 500, 60),
        candidate(2, 700, 58),
    ]

    p = policy(gap=300)

    optimal = optimize_break_schedule(
        candidates,
        video_duration=1800,
        policy=p,
    )

    greedy = greedy_break_schedule(
        candidates,
        video_duration=1800,
        policy=p,
    )

    # Utilities relative to neutral EABS = 50:
    #
    # 300s -> 8
    # 500s -> 10
    # 700s -> 8
    #
    # Greedy selects 500s first and blocks both 300s and 700s.
    # MILP instead selects 300s + 700s:
    # 8 + 8 = 16 > 10.
    assert optimal["quality_utility"] == 16
    assert greedy["quality_utility"] == 10

    assert optimal["eabs_sum"] == 116
    assert greedy["eabs_sum"] == 60

    assert [
        item.timestamp
        for item in optimal["selected"]
    ] == [300, 700]

    assert [
        item.timestamp
        for item in greedy["selected"]
    ] == [500]


def test_below_neutral_quality_is_not_selected():
    candidates = [
        candidate(0, 300, 49),
        candidate(1, 700, 25),
    ]

    p = policy(gap=300)

    result = optimize_break_schedule(
        candidates,
        video_duration=1800,
        policy=p,
    )

    assert result["selected"] == []
    assert result["eabs_sum"] == 0.0
    assert result["quality_utility"] == 0.0
    assert result["quality_candidate_count"] == 0
    assert result["rejected_below_neutral_quality"] == 2


def test_neutral_quality_is_not_selected():
    candidates = [
        candidate(0, 300, 50),
    ]

    p = policy(gap=300)

    result = optimize_break_schedule(
        candidates,
        video_duration=1800,
        policy=p,
    )

    # EABS=50 is exactly neutral evidence and has zero utility.
    assert result["selected"] == []
    assert result["quality_utility"] == 0.0
    assert result["rejected_below_neutral_quality"] == 1


def test_minimum_gap_is_enforced():
    p = policy(gap=300)

    assert schedule_is_feasible(
        [300, 600],
        video_duration=1800,
        policy=p,
    )

    assert not schedule_is_feasible(
        [300, 599],
        video_duration=1800,
        policy=p,
    )


def test_first_and_end_buffers_are_enforced():
    p = policy(
        first=120,
        end=120,
    )

    assert schedule_is_feasible(
        [120, 1680],
        video_duration=1800,
        policy=p,
    )

    assert not schedule_is_feasible(
        [119],
        video_duration=1800,
        policy=p,
    )

    assert not schedule_is_feasible(
        [1681],
        video_duration=1800,
        policy=p,
    )


def test_rolling_hour_limit_is_enforced():
    p = policy(
        gap=0,
        max_per_hour=2,
    )

    assert not schedule_is_feasible(
        [100, 200, 300],
        video_duration=4000,
        policy=p,
    )

    assert schedule_is_feasible(
        [100, 200, 3800],
        video_duration=4000,
        policy=p,
    )


def test_ad_load_limit_is_enforced():
    p = policy(
        gap=0,
        ad_load=0.05,
        ad_duration=30,
    )

    # 1800 * 0.05 = 90 seconds => maximum 3 x 30-second ads.
    assert schedule_is_feasible(
        [100, 200, 300],
        video_duration=1800,
        policy=p,
    )

    assert not schedule_is_feasible(
        [100, 200, 300, 400],
        video_duration=1800,
        policy=p,
    )