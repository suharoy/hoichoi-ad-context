"""Rank safety-eligible brands by multimodal semantic compatibility."""

from __future__ import annotations

import numpy as np

from src.brand_matching.catalog import Brand
from src.context.context_inference import (
    normalize_vector,
    rank_percentiles,
)


def brand_text_descriptor(
    brand: Brand,
) -> str:
    return (
        f"{brand.description} "
        f"Positive contexts: "
        f"{', '.join(brand.positive_contexts)}. "
        f"Activities: "
        f"{', '.join(brand.activities)}."
    ).strip()


def brand_visual_prompt(
    brand: Brand,
) -> str:
    concepts = (
        list(
            brand.positive_contexts
        )
        + list(
            brand.activities
        )
    )

    return (
        "an advertising-compatible "
        "scene involving "
        + ", ".join(concepts)
    )


def rank_eligible_brands(
    *,
    brands: list[Brand],
    detected_contexts: set[str],
    scene_text_embedding: np.ndarray | None,
    scene_visual_embedding: np.ndarray | None,
    brand_text_embeddings: dict[str, np.ndarray],
    brand_visual_embeddings: dict[str, np.ndarray],
) -> list[dict]:
    """
    Rank only brands that have already passed hard safety filtering.

    Text and visual similarities are rank-normalized separately and then
    equally fused across the available modalities.
    """
    if not brands:
        return []

    text_raw: dict[str, float] = {}
    visual_raw: dict[str, float] = {}

    if scene_text_embedding is not None:
        scene_text_embedding = (
            normalize_vector(
                scene_text_embedding
            )
        )

        for brand in brands:
            text_raw[
                brand.brand_id
            ] = float(
                np.dot(
                    scene_text_embedding,
                    normalize_vector(
                        brand_text_embeddings[
                            brand.brand_id
                        ]
                    ),
                )
            )

    if scene_visual_embedding is not None:
        scene_visual_embedding = (
            normalize_vector(
                scene_visual_embedding
            )
        )

        for brand in brands:
            visual_raw[
                brand.brand_id
            ] = float(
                np.dot(
                    scene_visual_embedding,
                    normalize_vector(
                        brand_visual_embeddings[
                            brand.brand_id
                        ]
                    ),
                )
            )

    text_rank = rank_percentiles(
        text_raw
    )

    visual_rank = rank_percentiles(
        visual_raw
    )

    ranked: list[dict] = []

    for brand in brands:
        ranks: list[float] = []

        if brand.brand_id in text_rank:
            ranks.append(
                text_rank[
                    brand.brand_id
                ]
            )

        if brand.brand_id in visual_rank:
            ranks.append(
                visual_rank[
                    brand.brand_id
                ]
            )

        if not ranks:
            continue

        positive_overlap = sorted(
            set(
                brand.positive_contexts
            )
            & detected_contexts
        )

        activity_overlap = sorted(
            set(
                brand.activities
            )
            & detected_contexts
        )

        match_score = (
            100.0
            * float(
                np.mean(ranks)
            )
        )

        ranked.append(
            {
                "brand_id": (
                    brand.brand_id
                ),
                "name": (
                    brand.name
                ),
                "creative_uri": (
                    brand.creative_uri
                ),
                "text_cosine": (
                    text_raw.get(
                        brand.brand_id
                    )
                ),
                "text_rank": (
                    text_rank.get(
                        brand.brand_id
                    )
                ),
                "visual_cosine": (
                    visual_raw.get(
                        brand.brand_id
                    )
                ),
                "visual_rank": (
                    visual_rank.get(
                        brand.brand_id
                    )
                ),
                "positive_context_overlap": (
                    positive_overlap
                ),
                "activity_overlap": (
                    activity_overlap
                ),
                "match_score": (
                    match_score
                ),
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["match_score"],
            row["brand_id"],
        )
    )

    return ranked
