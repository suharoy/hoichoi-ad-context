import xml.etree.ElementTree as ET

from src.manifest.debug_manifest import (
    build_debug_manifest,
)
from src.manifest.vmap import (
    VMAP_NS,
    build_vmap,
    format_time,
)


def sample_optimized():
    return {
        "policy": {
            "nominal_ad_duration_seconds": 30,
        },
        "videos": [
            {
                "video": "demo.mp4",
                "duration_seconds": 1000.0,
                "selected_breaks": [
                    {
                        "source_candidate_index": 1,
                        "timestamp_seconds": 120.0,
                        "eabs": 80.0,
                        "quality_utility": 30.0,
                        "centered_pause": True,
                        "visual_context_change": 0.2,
                        "text_semantic_change": 0.1,
                        "break_score": {
                            "ad_eligible": True,
                            "eabs": 80.0,
                        },
                    }
                ],
            }
        ],
    }


def sample_matches():
    return {
        "catalogue_name": "demo",
        "catalogue_status": "synthetic",
        "context_method": "test",
        "matching_method": "test",
        "breaks": [
            {
                "video": "demo.mp4",
                "timestamp_seconds": 120.0,
                "scene_text": "?????",
                "context": {
                    "context_cues": [
                        "food"
                    ],
                    "hard_safety_contexts": [],
                    "lexical_safety_hits": [],
                    "consensus_safety_hits": [],
                    "ranked_contexts": [],
                },
                "blocked_brands": [],
                "eligible_brand_ids": [
                    "brand_a"
                ],
                "ranked_eligible_brands": [
                    {
                        "brand_id": "brand_a",
                        "match_score": 90.0,
                    }
                ],
                "selected_brand": {
                    "brand_id": "brand_a",
                    "name": "Brand A",
                    "creative_uri": (
                        "/creatives/"
                        "brand_a.mp4"
                    ),
                    "match_score": 90.0,
                },
            }
        ],
        "summary": {
            "negative_context_violations": 0,
        },
    }


def test_time_format():
    assert (
        format_time(129.72)
        == "00:02:09.720"
    )


def test_debug_manifest_joins_break_and_brand():
    manifest = (
        build_debug_manifest(
            optimized=(
                sample_optimized()
            ),
            brand_matches=(
                sample_matches()
            ),
        )
    )

    assert (
        manifest[
            "summary"
        ][
            "break_count"
        ]
        == 1
    )

    item = (
        manifest[
            "videos"
        ][0][
            "breaks"
        ][0]
    )

    assert (
        item[
            "selected_brand"
        ][
            "brand_id"
        ]
        == "brand_a"
    )

    assert (
        item[
            "context"
        ][
            "context_cues"
        ]
        == ["food"]
    )


def test_vmap_contains_absolute_ad_break():
    manifest = (
        build_debug_manifest(
            optimized=(
                sample_optimized()
            ),
            brand_matches=(
                sample_matches()
            ),
        )
    )

    tree = build_vmap(
        video_name="demo.mp4",
        breaks=(
            manifest[
                "videos"
            ][0][
                "breaks"
            ]
        ),
        ad_duration_seconds=30,
    )

    root = tree.getroot()

    assert (
        root.tag
        == (
            f"{{{VMAP_NS}}}"
            "VMAP"
        )
    )

    breaks = root.findall(
        f"{{{VMAP_NS}}}AdBreak"
    )

    assert len(breaks) == 1

    assert (
        breaks[0].attrib[
            "timeOffset"
        ]
        == "00:02:00.000"
    )

    vast_data = (
        breaks[0]
        .find(
            f"{{{VMAP_NS}}}"
            "AdSource"
        )
        .find(
            f"{{{VMAP_NS}}}"
            "VASTAdData"
        )
    )

    vast = vast_data.find(
        "VAST"
    )

    assert vast is not None

    media = vast.find(
        "./Ad/InLine/Creatives/"
        "Creative/Linear/"
        "MediaFiles/MediaFile"
    )

    assert media is not None

    assert (
        media.text
        == "/creatives/brand_a.mp4"
    )


def test_manifest_rejects_unsafe_summary():
    matches = sample_matches()

    matches[
        "summary"
    ][
        "negative_context_violations"
    ] = 1

    try:
        build_debug_manifest(
            optimized=(
                sample_optimized()
            ),
            brand_matches=matches,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unsafe manifest "
            "should have failed"
        )
