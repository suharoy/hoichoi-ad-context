"""Unified audit/debug manifest for optimized break and delivery decisions."""

from __future__ import annotations

from pathlib import Path


FILLED = "filled"
NO_FILL_BRAND_SAFETY = "no_fill_brand_safety"


def match_key(
    video: str,
    timestamp: float,
) -> tuple[str, float]:
    return (
        video,
        round(float(timestamp), 6),
    )


def _resolve_delivery(
    *,
    key: tuple[str, float],
    match: dict,
    brand_count: int,
) -> tuple[str, str | None]:
    """
    Resolve one scheduled break to either a filled ad or a fail-closed no-fill.

    A no-fill is valid only if every catalogue brand was explicitly rejected
    by the hard negative-context filter. Missing ranking output must never be
    silently converted into a no-fill.
    """
    selected_brand = match.get("selected_brand")

    blocked = match.get("blocked_brands", [])
    blocked_ids = {
        str(item["brand_id"])
        for item in blocked
    }

    eligible_ids = {
        str(item)
        for item in match.get(
            "eligible_brand_ids",
            [],
        )
    }

    ranked = match.get(
        "ranked_eligible_brands",
        [],
    )

    if selected_brand is not None:
        brand_id = str(
            selected_brand["brand_id"]
        )

        if brand_id in blocked_ids:
            raise ValueError(
                "Selected brand also appears "
                f"in hard-block list: {key}"
            )

        if eligible_ids and brand_id not in eligible_ids:
            raise ValueError(
                "Selected brand is absent from "
                f"eligible brand IDs: {key}"
            )

        return FILLED, None

    # No selected brand is allowed only as a hard-safety no-fill.
    if eligible_ids:
        raise ValueError(
            "No brand selected despite safety-eligible "
            f"brands existing: {key}"
        )

    if ranked:
        raise ValueError(
            "No brand selected but ranked brands exist: "
            f"{key}"
        )

    if brand_count < 1:
        raise ValueError(
            "Cannot verify no-fill without brand_count"
        )

    if len(blocked_ids) != brand_count:
        raise ValueError(
            "No-fill requires every catalogue brand "
            f"to be explicitly hard-blocked: {key}; "
            f"blocked={len(blocked_ids)}, "
            f"catalogue={brand_count}"
        )

    return (
        NO_FILL_BRAND_SAFETY,
        "all_catalogue_brands_hard_blocked",
    )


def build_debug_manifest(
    *,
    optimized: dict,
    brand_matches: dict,
) -> dict:
    """
    Join optimized break opportunities with context, safety and delivery.

    Every scheduled opportunity must resolve to exactly one of:
      - filled
      - no_fill_brand_safety

    No-fill opportunities remain in the audit manifest but are omitted from
    VMAP delivery by the VMAP builder.
    """
    summary = (
        brand_matches.get("summary")
        or {}
    )

    if (
        summary.get(
            "negative_context_violations",
            0,
        )
        != 0
    ):
        raise ValueError(
            "Cannot generate manifest: "
            "negative-context violations exist"
        )

    if (
        summary.get(
            "all_selected_brands_safe",
            True,
        )
        is False
    ):
        raise ValueError(
            "Cannot generate manifest: "
            "upstream safety check failed"
        )

    brand_count = int(
        summary.get(
            "brand_count",
            0,
        )
        or 0
    )

    match_lookup: dict[
        tuple[str, float],
        dict,
    ] = {}

    for item in brand_matches["breaks"]:
        key = match_key(
            item["video"],
            item["timestamp_seconds"],
        )

        if key in match_lookup:
            raise ValueError(
                f"Duplicate brand match: {key}"
            )

        match_lookup[key] = item

    output_videos: list[dict] = []

    total_breaks = 0
    delivered_ads = 0
    no_fill_breaks = 0

    for video in optimized["videos"]:
        video_name = video["video"]

        output_breaks: list[dict] = []

        video_delivered = 0
        video_no_fill = 0

        ordered = sorted(
            video["selected_breaks"],
            key=lambda item: float(
                item["timestamp_seconds"]
            ),
        )

        for number, break_item in enumerate(
            ordered,
            start=1,
        ):
            timestamp = float(
                break_item["timestamp_seconds"]
            )

            key = match_key(
                video_name,
                timestamp,
            )

            if key not in match_lookup:
                raise ValueError(
                    "Missing brand match for "
                    f"{key}"
                )

            match = match_lookup[key]

            (
                delivery_status,
                delivery_reason,
            ) = _resolve_delivery(
                key=key,
                match=match,
                brand_count=brand_count,
            )

            selected_brand = (
                match.get("selected_brand")
            )

            if delivery_status == FILLED:
                delivered_ads += 1
                video_delivered += 1
            else:
                no_fill_breaks += 1
                video_no_fill += 1

            context = (
                match.get("context")
                or {}
            )

            break_id = (
                f"{Path(video_name).stem}"
                f"-break-{number:02d}"
            )

            output_breaks.append(
                {
                    "break_id": break_id,
                    "timestamp_seconds": timestamp,
                    "eabs": break_item["eabs"],
                    "quality_utility": (
                        break_item[
                            "quality_utility"
                        ]
                    ),
                    "delivery_status": (
                        delivery_status
                    ),
                    "delivery_reason": (
                        delivery_reason
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

        output_videos.append(
            {
                "video": video_name,
                "duration_seconds": (
                    video["duration_seconds"]
                ),
                "vmap_file": (
                    "vmap/"
                    f"{Path(video_name).stem}"
                    ".vmap.xml"
                ),
                "break_count": (
                    len(output_breaks)
                ),
                "delivered_ad_count": (
                    video_delivered
                ),
                "no_fill_break_count": (
                    video_no_fill
                ),
                "breaks": output_breaks,
            }
        )

    resolved = (
        delivered_ads
        + no_fill_breaks
    )

    if resolved != total_breaks:
        raise RuntimeError(
            "Not every scheduled break received "
            "a delivery decision"
        )

    return {
        "schema_version": "1.1",
        "pipeline": (
            "multimodal evidence -> "
            "EABS -> MILP pacing -> "
            "context retrieval -> "
            "hard brand safety -> "
            "brand ranking -> "
            "fail-closed delivery -> VMAP"
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
        "videos": output_videos,
        "summary": {
            "video_count": len(
                output_videos
            ),
            "break_count": total_breaks,
            "scheduled_break_count": (
                total_breaks
            ),
            "delivered_ad_count": (
                delivered_ads
            ),
            "no_fill_break_count": (
                no_fill_breaks
            ),
            # Backward-compatible names:
            "matched_break_count": (
                delivered_ads
            ),
            "unmatched_break_count": (
                no_fill_breaks
            ),
            "resolved_break_count": (
                resolved
            ),
            "all_breaks_resolved": (
                resolved == total_breaks
            ),
            "negative_context_violations": 0,
            "all_breaks_have_brand": (
                delivered_ads
                == total_breaks
            ),
            "all_selected_brands_safe": True,
        },
    }
