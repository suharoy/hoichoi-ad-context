"""Multimodal context retrieval over a fixed, inspectable taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContextEntry:
    context_id: str
    text_description: str
    visual_prompt: str
    keywords: tuple[str, ...]
    safety_sensitive: bool


@dataclass(frozen=True)
class ContextTaxonomy:
    version: int
    context_top_k: int
    safety_consensus_percentile: float
    method: str
    contexts: tuple[ContextEntry, ...]


def load_taxonomy(path: Path) -> ContextTaxonomy:
    payload = json.loads(path.read_text(encoding="utf-8"))

    contexts = tuple(
        ContextEntry(
            context_id=str(item["context_id"]),
            text_description=str(item["text_description"]),
            visual_prompt=str(item["visual_prompt"]),
            keywords=tuple(
                str(value)
                for value in item.get("keywords", [])
            ),
            safety_sensitive=bool(
                item.get("safety_sensitive", False)
            ),
        )
        for item in payload["contexts"]
    )

    ids = [item.context_id for item in contexts]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Context taxonomy contains duplicate context_id values"
        )

    top_k = int(payload["context_top_k"])

    if top_k < 1:
        raise ValueError(
            "context_top_k must be >= 1"
        )

    safety_consensus_percentile = float(
        payload.get("safety_consensus_percentile", 0.90)
    )

    if not 0.0 <= safety_consensus_percentile <= 1.0:
        raise ValueError(
            "safety_consensus_percentile must be in [0,1]"
        )

    return ContextTaxonomy(
        version=int(payload["version"]),
        context_top_k=top_k,
        safety_consensus_percentile=safety_consensus_percentile,
        method=str(payload["method"]),
        contexts=contexts,
    )


def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional vector"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Vector contains non-finite values"
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0:
        raise ValueError(
            "Cannot normalize zero vector"
        )

    return vector / norm


def rank_percentiles(
    scores: dict[str, float],
) -> dict[str, float]:
    """
    Convert arbitrary higher-is-better scores to [0,1] rank percentiles.

    Ties receive their average rank. Best=1 and worst=0 when n>1.
    """
    if not scores:
        return {}

    keys = list(scores)

    values = np.asarray(
        [
            float(scores[key])
            for key in keys
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Rank inputs must all be finite"
        )

    if len(keys) == 1:
        return {
            keys[0]: 1.0
        }

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    sorted_values = values[order]

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            start + end - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    percentiles = (
        ranks / (len(values) - 1)
    )

    return {
        key: float(
            percentiles[index]
        )
        for index, key
        in enumerate(keys)
    }


def lexical_context_hits(
    text: str,
    taxonomy: ContextTaxonomy,
) -> set[str]:
    lowered = text.casefold()

    hits: set[str] = set()

    if not lowered.strip():
        return hits

    for context in taxonomy.contexts:
        if any(
            keyword.casefold()
            in lowered
            for keyword
            in context.keywords
        ):
            hits.add(
                context.context_id
            )

    return hits


def infer_contexts(
    *,
    taxonomy: ContextTaxonomy,
    scene_text: str,
    scene_text_embedding: np.ndarray | None,
    scene_visual_embedding: np.ndarray | None,
    taxonomy_text_embeddings: dict[str, np.ndarray],
    taxonomy_visual_embeddings: dict[str, np.ndarray],
    text_reliability: float = 0.0,
) -> dict:
    """
    Retrieve scene contexts by rank-fusing available E5 and CLIP evidence.

    Safety-sensitive lexical hits are always retained even when outside top-k.
    """
    text_raw: dict[str, float] = {}
    visual_raw: dict[str, float] = {}

    if scene_text_embedding is not None:
        scene_text_embedding = normalize_vector(
            scene_text_embedding
        )

        for context in taxonomy.contexts:
            vector = normalize_vector(
                taxonomy_text_embeddings[
                    context.context_id
                ]
            )

            text_raw[
                context.context_id
            ] = float(
                np.dot(
                    scene_text_embedding,
                    vector,
                )
            )

    if scene_visual_embedding is not None:
        scene_visual_embedding = normalize_vector(
            scene_visual_embedding
        )

        for context in taxonomy.contexts:
            vector = normalize_vector(
                taxonomy_visual_embeddings[
                    context.context_id
                ]
            )

            visual_raw[
                context.context_id
            ] = float(
                np.dot(
                    scene_visual_embedding,
                    vector,
                )
            )

    text_rank = rank_percentiles(
        text_raw
    )

    visual_rank = rank_percentiles(
        visual_raw
    )

    lexical_hits = lexical_context_hits(
        scene_text,
        taxonomy,
    )

    rows: list[dict] = []

    for context in taxonomy.contexts:
        available_ranks: list[float] = []

        if (
            context.context_id
            in text_rank
        ):
            available_ranks.append(
                text_rank[
                    context.context_id
                ]
            )

        if (
            context.context_id
            in visual_rank
        ):
            available_ranks.append(
                visual_rank[
                    context.context_id
                ]
            )

        fused_rank = (
            float(
                np.mean(
                    available_ranks
                )
            )
            if available_ranks
            else 0.0
        )

        lexical_hit = (
            context.context_id
            in lexical_hits
        )

        rows.append(
            {
                "context_id": (
                    context.context_id
                ),
                "safety_sensitive": (
                    context.safety_sensitive
                ),
                "text_cosine": (
                    text_raw.get(
                        context.context_id
                    )
                ),
                "text_rank": (
                    text_rank.get(
                        context.context_id
                    )
                ),
                "visual_cosine": (
                    visual_raw.get(
                        context.context_id
                    )
                ),
                "visual_rank": (
                    visual_rank.get(
                        context.context_id
                    )
                ),
                "lexical_hit": (
                    lexical_hit
                ),
                "fused_rank": (
                    fused_rank
                ),
            }
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            -row["fused_rank"],
            row["context_id"],
        ),
    )

    # Ordinary context retrieval is top-k over non-safety concepts only.
    # Safety concepts are deliberately NOT promoted merely because every
    # relative ranking has a "top" item.
    normal_ranked = [
        row
        for row in ordered
        if not row["safety_sensitive"]
    ]

    context_cues = {
        row["context_id"]
        for row in normal_ranked[: taxonomy.context_top_k]
    }

    lexical_safety_hits = {
        row["context_id"]
        for row in rows
        if row["lexical_hit"] and row["safety_sensitive"]
    }

    # Semantic safety detection requires cross-modal agreement when both
    # modalities exist. This avoids turning a single noisy relative rank
    # into a hard commercial block.
    consensus_safety_hits: set[str] = set()

    for row in rows:
        if not row["safety_sensitive"]:
            continue

        text_value = row["text_rank"]
        visual_value = row["visual_rank"]

        if (
            text_reliability >= 1.0
            and text_value is not None
            and visual_value is not None
            and text_value >= taxonomy.safety_consensus_percentile
            and visual_value >= taxonomy.safety_consensus_percentile
        ):
            consensus_safety_hits.add(row["context_id"])

    hard_safety_contexts = (
        lexical_safety_hits
        | consensus_safety_hits
    )

    detected = (
        context_cues
        | hard_safety_contexts
    )

    return {
        "method": taxonomy.method,
        "top_k": taxonomy.context_top_k,
        "safety_consensus_percentile": (
            taxonomy.safety_consensus_percentile
        ),
        "context_cues": sorted(context_cues),
        "hard_safety_contexts": sorted(hard_safety_contexts),
        "detected_contexts": sorted(detected),
        "ranked_contexts": ordered,
        "lexical_safety_hits": sorted(lexical_safety_hits),
        "consensus_safety_hits": sorted(consensus_safety_hits),
    }
