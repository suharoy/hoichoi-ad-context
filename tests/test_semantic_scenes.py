from src.segmentation.semantic_scenes import (
    SceneCalibration,
    build_scene_intervals,
    collapse_adjacent_strong_boundaries,
    otsu_threshold,
    transition_evidence,
)


def test_otsu_separates_two_clusters():
    threshold = otsu_threshold(
        [
            0.10,
            0.11,
            0.12,
            0.80,
            0.81,
            0.82,
        ]
    )

    assert 0.12 < threshold < 0.80


def test_visual_only_threshold_has_unit_strength():
    candidate = {
        "visual_semantics": {
            "context_change": 0.20,
        },
    }

    calibration = SceneCalibration(
        visual_threshold=0.20,
        text_threshold=0.10,
        reference_text_length=50.0,
    )

    evidence = transition_evidence(
        candidate,
        calibration,
    )

    assert evidence[
        "transition_strength"
    ] == 1.0

    assert evidence["strong"] is True


def test_short_text_has_reduced_influence():
    calibration = SceneCalibration(
        visual_threshold=0.20,
        text_threshold=0.10,
        reference_text_length=100.0,
    )

    candidate = {
        "visual_semantics": {
            "context_change": 0.10,
        },
        "text_semantics": {
            "available": True,
            "semantic_change": 0.30,
            "left_characters": 5,
            "right_characters": 5,
        },
    }

    evidence = transition_evidence(
        candidate,
        calibration,
    )

    assert (
        0.0
        < evidence[
            "text_reliability"
        ]
        < 1.0
    )

    # Text can help, but a tiny fragment cannot dominate.
    assert (
        evidence[
            "transition_strength"
        ]
        < 1.0
    )


def test_adjacent_strong_boundaries_collapse_to_max():
    rows = [
        {
            "candidate_index": 0,
            "timestamp_seconds": 10.0,
            "transition_strength": 1.2,
            "strong": True,
        },
        {
            "candidate_index": 1,
            "timestamp_seconds": 12.0,
            "transition_strength": 1.8,
            "strong": True,
        },
        {
            "candidate_index": 2,
            "timestamp_seconds": 20.0,
            "transition_strength": 0.4,
            "strong": False,
        },
        {
            "candidate_index": 3,
            "timestamp_seconds": 30.0,
            "transition_strength": 1.1,
            "strong": True,
        },
    ]

    selected = (
        collapse_adjacent_strong_boundaries(
            rows
        )
    )

    assert [
        item["candidate_index"]
        for item in selected
    ] == [1, 3]


def test_scene_intervals_are_contiguous():
    candidates = [
        {
            "timestamp_seconds": 10.0,
            "visual_semantics": {
                "previous_shot": {
                    "index": 0,
                    "start": 0.0,
                    "end": 10.0,
                },
                "next_shot": {
                    "index": 1,
                    "start": 10.0,
                    "end": 20.0,
                },
            },
        },
        {
            "timestamp_seconds": 20.0,
            "visual_semantics": {
                "previous_shot": {
                    "index": 1,
                    "start": 10.0,
                    "end": 20.0,
                },
                "next_shot": {
                    "index": 2,
                    "start": 20.0,
                    "end": 30.0,
                },
            },
        },
    ]

    selected = [
        {
            "candidate_index": 0,
            "timestamp_seconds": 10.0,
            "transition_strength": 1.5,
            "visual_change": 0.5,
            "text_change": None,
            "text_reliability": 0.0,
        }
    ]

    scenes = build_scene_intervals(
        video_name="demo.mp4",
        candidates=candidates,
        selected_boundaries=selected,
    )

    assert len(scenes) == 2

    assert (
        scenes[0]["end_seconds"]
        == scenes[1]["start_seconds"]
        == 10.0
    )

    assert scenes[0]["shot_count"] == 1
    assert scenes[1]["shot_count"] == 2
