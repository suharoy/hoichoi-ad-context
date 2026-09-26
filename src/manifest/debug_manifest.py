"""Unified audit/debug manifest for optimized break and brand decisions."""

from __future__ import annotations

from pathlib import Path


def match_key(
    video: str,
    timestamp: float,
) -> tuple[str, float]:
    return (
        video,
        round(
            float(timestamp),
            6,
        ),
    )


def build_debug_manifest(
    *,
    optimized: dict,
    brand_matches: dict,
) -> dict:
    """
    Join Stage-5 break decisions with Stage-6 context and brand decisions.
    """
    if (
        brand_matches.get(
            "summary",
            {},
        ).get(
            "negative_context_violations",
            0,
        )
        != 0
    ):
        raise ValueError(
            "Cannot generate manifest: "
            "negative-context violations exist"
        )

    match_lookup: dict[
        tuple[str, float],
        dict,
    ] = {}

    for item in brand_matches[
        "breaks"
    ]:
        key = match_key(
            item["video"],
            item[
                "timestamp_seconds"
            ],
        )

        if key in match_lookup:
            raise ValueError(
                f"Duplicate brand match: {key}"
            )

        match_lookup[
            key
        ] = item

    output_videos: list[dict] = []

    total_breaks = 0
    matched_breaks = 0

    for video in optimized[
        "videos"
    ]:
        video_name = video[
            "video"
        ]

        output_breaks: list[
            dict
        ] = []

        ordered = sorted(
            video[
                "selected_breaks"
            ],
            key=lambda item: (
                float(
                    item[
                        "timestamp_seconds"
                    ]
                )
            ),
        )

        for number, break_item in enumerate(
            ordered,
            start=1,
        ):
            timestamp = float(
                break_item[
                    "timestamp_seconds"
                ]
            )

            key = match_key(
                video_name,
                timestamp,
            )

            if key not in match_lookup:
                raise ValueError(
                    "Missing Stage-6 brand "
                    f"match for {key}"
                )

            match = (
                match_lookup[
                    key
                ]
            )

            selected_brand = (
                match.get(
                    "selected_brand"
                )
            )

            if selected_brand is None:
                raise ValueError(
                    f"No brand selected for {key}"
                )

            blocked_ids = {
                item[
                    "brand_id"
                ]
                for item in match.get(
                    "blocked_brands",
                    [],
                )
            }

            if (
                selected_brand[
                    "brand_id"
                ]
                in blocked_ids
            ):
                raise ValueError(
                    "Selected brand also appears "
                    f"in hard-block list: {key}"
                )

            context = (
                match.get(
                    "context"
                )
                or {}
            )

            break_id = (
                f"{Path(video_name).stem}"
                f"-break-{number:02d}"
            )

            output_breaks.append(
                {
                    "break_id": (
                        break_id
                    ),
                    "timestamp_seconds": (
                        timestamp
                    ),
                    "eabs": (
                        break_item[
                            "eabs"
                        ]
                    ),
                    "quality_utility": (
                        break_item[
                            "quality_utility"
                        ]
                    ),
                    "boundary_evidence": {
                        "centered_pause": (
                            break_item.get(
                                "centered_pause"
                            )
                        ),
                        "visual_context_change": (
                            break_item.get(
                                "visual_context_change"
                            )
                        ),
                        "text_semantic_change": (
                            break_item.get(
                                "text_semantic_change"
                            )
                        ),
                        "break_score": (
                            break_item.get(
                                "break_score"
                            )
                        ),
                    },
                    "context": {
                        "context_cues": (
                            context.get(
                                "context_cues",
                                [],
                            )
                        ),
                        "hard_safety_contexts": (
                            context.get(
                                "hard_safety_contexts",
                                [],
                            )
                        ),
                        "lexical_safety_hits": (
                            context.get(
                                "lexical_safety_hits",
                                [],
                            )
                        ),
                        "consensus_safety_hits": (
                            context.get(
                                "consensus_safety_hits",
                                [],
                            )
                        ),
                        "ranked_contexts": (
                            context.get(
                                "ranked_contexts",
                                [],
                            )
                        ),
                    },
                    "brand_safety": {
                        "blocked_brands": (
                            match.get(
                                "blocked_brands",
                                [],
                            )
                        ),
                        "eligible_brand_ids": (
                            match.get(
                                "eligible_brand_ids",
                                [],
                            )
                        ),
                    },
                    "selected_brand": (
                        selected_brand
                    ),
                    "brand_ranking": (
                        match.get(
                            "ranked_eligible_brands",
                            [],
                        )
                    ),
                    "scene_text": (
                        match.get(
                            "scene_text",
                            "",
                        )
                    ),
                }
            )

            total_breaks += 1
            matched_breaks += 1

        output_videos.append(
            {
                "video": (
                    video_name
                ),
                "duration_seconds": (
                    video[
                        "duration_seconds"
                    ]
                ),
                "vmap_file": (
                    "vmap/"
                    f"{Path(video_name).stem}"
                    ".vmap.xml"
                ),
                "break_count": (
                    len(
                        output_breaks
                    )
                ),
                "breaks": (
                    output_breaks
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "pipeline": (
            "multimodal evidence -> "
            "EABS -> MILP pacing -> "
            "context retrieval -> "
            "hard brand safety -> "
            "brand ranking -> VMAP"
        ),
        "catalogue_name": (
            brand_matches.get(
                "catalogue_name"
            )
        ),
        "catalogue_status": (
            brand_matches.get(
                "catalogue_status"
            )
        ),
        "context_method": (
            brand_matches.get(
                "context_method"
            )
        ),
        "matching_method": (
            brand_matches.get(
                "matching_method"
            )
        ),
        "videos": (
            output_videos
        ),
        "summary": {
            "video_count": (
                len(
                    output_videos
                )
            ),
            "break_count": (
                total_breaks
            ),
            "matched_break_count": (
                matched_breaks
            ),
            "unmatched_break_count": (
                total_breaks
                - matched_breaks
            ),
            "negative_context_violations": (
                0
            ),
            "all_breaks_have_brand": (
                matched_breaks
                == total_breaks
            ),
            "all_selected_brands_safe": (
                True
            ),
        },
    }
