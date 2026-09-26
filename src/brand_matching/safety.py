"""Hard negative-context brand safety filtering."""

from __future__ import annotations

from src.brand_matching.catalog import Brand


def filter_brands(
    brands: tuple[Brand, ...] | list[Brand],
    detected_contexts: set[str],
) -> tuple[list[Brand], list[dict]]:
    """
    Remove brands whose negative_contexts intersect detected contexts.

    This happens before any relevance ranking.
    """
    eligible: list[Brand] = []
    blocked: list[dict] = []

    for brand in brands:
        hits = sorted(
            set(
                brand.negative_contexts
            )
            & detected_contexts
        )

        if hits:
            blocked.append(
                {
                    "brand_id": (
                        brand.brand_id
                    ),
                    "hard_block": True,
                    "negative_context_hits": (
                        hits
                    ),
                    "reason": (
                        "negative_context:"
                        + ",".join(hits)
                    ),
                }
            )
        else:
            eligible.append(
                brand
            )

    return eligible, blocked
