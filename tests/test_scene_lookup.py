from src.segmentation.scene_lookup import (
    annotate_manifest_with_scenes,
    locate_scene,
)


def sample_scenes():
    return [
        {
            "scene_id": "demo-scene-001",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "duration_seconds": 10.0,
        },
        {
            "scene_id": "demo-scene-002",
            "start_seconds": 10.0,
            "end_seconds": 30.0,
            "duration_seconds": 20.0,
        },
        {
            "scene_id": "demo-scene-003",
            "start_seconds": 30.0,
            "end_seconds": 50.0,
            "duration_seconds": 20.0,
        },
    ]


def test_exact_scene_boundary():
    result = locate_scene(
        timestamp_seconds=10.0,
        scenes=sample_scenes(),
    )

    assert (
        result["relation"]
        == "exact_scene_boundary"
    )

    assert (
        result["previous_scene_id"]
        == "demo-scene-001"
    )

    assert (
        result["next_scene_id"]
        == "demo-scene-002"
    )

    assert (
        result[
            "distance_to_nearest_scene_boundary_seconds"
        ]
        == 0.0
    )


def test_intra_scene_location():
    result = locate_scene(
        timestamp_seconds=23.0,
        scenes=sample_scenes(),
    )

    assert (
        result["relation"]
        == "within_scene"
    )

    assert (
        result["scene_id"]
        == "demo-scene-002"
    )

    assert (
        result[
            "seconds_from_scene_start"
        ]
        == 13.0
    )

    assert (
        result[
            "seconds_until_scene_end"
        ]
        == 7.0
    )

    assert (
        result[
            "distance_to_nearest_scene_boundary_seconds"
        ]
        == 7.0
    )


def test_manifest_annotation_does_not_change_break():
    manifest = {
        "videos": [
            {
                "video": "demo.mp4",
                "breaks": [
                    {
                        "break_id": (
                            "demo-break-01"
                        ),
                        "timestamp_seconds": (
                            23.0
                        ),
                        "eabs": 80.0,
                    }
                ],
            }
        ]
    }

    semantic = {
        "schema_version": "1.0",
        "method": "test",
        "calibration_mode": (
            "fitted_and_frozen"
        ),
        "summary": {
            "held_out_used_for_fit": False,
        },
        "videos": [
            {
                "video": "demo.mp4",
                "scenes": sample_scenes(),
            }
        ],
    }

    original_eabs = (
        manifest[
            "videos"
        ][0][
            "breaks"
        ][0][
            "eabs"
        ]
    )

    annotated = (
        annotate_manifest_with_scenes(
            manifest=manifest,
            semantic_scenes=semantic,
        )
    )

    item = (
        annotated[
            "videos"
        ][0][
            "breaks"
        ][0]
    )

    assert (
        item["eabs"]
        == original_eabs
    )

    assert (
        item[
            "semantic_scene"
        ][
            "relation"
        ]
        == "within_scene"
    )
