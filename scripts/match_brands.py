"""Infer break context, hard-filter unsafe brands, then rank safe brands."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from src.brand_matching.catalog import (
    load_catalogue,
)
from src.brand_matching.ranker import (
    brand_text_descriptor,
    brand_visual_prompt,
    rank_eligible_brands,
)
from src.brand_matching.safety import (
    filter_brands,
)
from src.context.clip_text_embeddings import (
    CLIPTextEncoder,
)
from src.context.context_inference import (
    infer_contexts,
    load_taxonomy,
    normalize_vector,
)
from src.context.text_embeddings import (
    SemanticTextEncoder,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OPTIMIZED = (
    ROOT
    / "outputs"
    / "dev"
    / "optimized-breaks.json"
)

DEFAULT_SCORED = (
    ROOT
    / "outputs"
    / "dev"
    / "boundary-profile-scored.json"
)

DEFAULT_TAXONOMY = (
    ROOT
    / "configs"
    / "context_taxonomy.json"
)

DEFAULT_BRANDS = (
    ROOT
    / "configs"
    / "brands.demo.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "dev"
    / "brand-matches.json"
)


def scene_text(
    candidate: dict,
) -> str:
    asr = (
        candidate.get(
            "asr_context"
        )
        or {}
    )

    left = str(
        (
            asr.get("left")
            or {}
        ).get("text", "")
    ).strip()

    right = str(
        (
            asr.get("right")
            or {}
        ).get("text", "")
    ).strip()

    return " ".join(
        value
        for value
        in (left, right)
        if value
    )


def load_visual_cache(
    stem: str,
) -> dict[int, np.ndarray]:
    path = (
        ROOT
        / "outputs"
        / "dev"
        / "visual-cache"
        / stem
        / "shot-embeddings.npz"
    )

    data = np.load(path)

    indices = (
        data["shot_index"]
        .astype(int)
    )

    embeddings = (
        data["embeddings"]
        .astype(np.float32)
    )

    return {
        int(index): (
            normalize_vector(vector)
        )
        for index, vector
        in zip(
            indices,
            embeddings,
        )
    }


def scene_visual_vector(
    candidate: dict,
    cache: dict[int, np.ndarray],
) -> np.ndarray | None:
    """
    Build a local visual-context vector from the Stage-3 context shots.

    Stage-3 stores left_context_shots / right_context_shots as shot indices.
    This also tolerates dict-shaped records for forward compatibility.
    """
    visual = (
        candidate.get("visual_semantics")
        or {}
    )

    shot_records = (
        list(
            visual.get(
                "left_context_shots"
            )
            or []
        )
        + list(
            visual.get(
                "right_context_shots"
            )
            or []
        )
    )

    vectors: list[np.ndarray] = []

    for shot in shot_records:
        if isinstance(shot, dict):
            if "index" not in shot:
                raise ValueError(
                    f"Shot record missing index: {shot}"
                )
            index = int(shot["index"])
        else:
            index = int(shot)

        if index not in cache:
            raise KeyError(
                f"Shot index {index} "
                f"missing from visual cache"
            )

        vectors.append(
            cache[index]
        )

    if not vectors:
        return None

    return normalize_vector(
        np.mean(
            np.stack(vectors),
            axis=0,
        )
    )

def vector_map(
    ids: list[str],
    matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(ids) != len(matrix):
        raise ValueError(
            "Embedding ID/vector count mismatch"
        )

    return {
        item_id: normalize_vector(
            vector
        )
        for item_id, vector
        in zip(ids, matrix)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Context-aware, hard-safe "
            "brand matching for optimized breaks."
        )
    )

    parser.add_argument(
        "--optimized",
        type=Path,
        default=DEFAULT_OPTIMIZED,
    )

    parser.add_argument(
        "--scored",
        type=Path,
        default=DEFAULT_SCORED,
    )

    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )

    parser.add_argument(
        "--brands",
        type=Path,
        default=DEFAULT_BRANDS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    optimized = json.loads(
        args.optimized.read_text(
            encoding="utf-8"
        )
    )

    scored = json.loads(
        args.scored.read_text(
            encoding="utf-8"
        )
    )

    taxonomy = load_taxonomy(
        args.taxonomy
    )

    allowed_contexts = {
        context.context_id
        for context
        in taxonomy.contexts
    }

    catalogue = load_catalogue(
        args.brands,
        allowed_contexts=(
            allowed_contexts
        ),
    )

    text_encoder = (
        SemanticTextEncoder()
    )

    clip_text_encoder = (
        CLIPTextEncoder()
    )

    taxonomy_ids = [
        context.context_id
        for context
        in taxonomy.contexts
    ]

    taxonomy_text_matrix = (
        text_encoder.encode(
            [
                context.text_description
                for context
                in taxonomy.contexts
            ]
        )
    )

    taxonomy_visual_matrix = (
        clip_text_encoder.encode(
            [
                context.visual_prompt
                for context
                in taxonomy.contexts
            ]
        )
    )

    taxonomy_text_embeddings = (
        vector_map(
            taxonomy_ids,
            taxonomy_text_matrix,
        )
    )

    taxonomy_visual_embeddings = (
        vector_map(
            taxonomy_ids,
            taxonomy_visual_matrix,
        )
    )

    brand_ids = [
        brand.brand_id
        for brand
        in catalogue.brands
    ]

    brand_text_matrix = (
        text_encoder.encode(
            [
                brand_text_descriptor(
                    brand
                )
                for brand
                in catalogue.brands
            ]
        )
    )

    brand_visual_matrix = (
        clip_text_encoder.encode(
            [
                brand_visual_prompt(
                    brand
                )
                for brand
                in catalogue.brands
            ]
        )
    )

    brand_text_embeddings = (
        vector_map(
            brand_ids,
            brand_text_matrix,
        )
    )

    brand_visual_embeddings = (
        vector_map(
            brand_ids,
            brand_visual_matrix,
        )
    )

    scored_by_video = {
        item["video"]: item
        for item
        in scored["videos"]
    }

    visual_caches: dict[
        str,
        dict[int, np.ndarray],
    ] = {}

    results: list[dict] = []

    selected_brand_counts: (
        Counter[str]
    ) = Counter()

    detected_context_counts: (
        Counter[str]
    ) = Counter()

    blocked_brand_decisions = 0
    negative_context_violations = 0

    for optimized_video in (
        optimized["videos"]
    ):
        video_name = (
            optimized_video["video"]
        )

        if (
            "feluda"
            in video_name.lower()
        ):
            raise ValueError(
                "Feluda must remain "
                "pseudo-held-out during "
                "Stage 6 development"
            )

        source_video = (
            scored_by_video[
                video_name
            ]
        )

        stem = Path(
            video_name
        ).stem

        if (
            stem
            not in visual_caches
        ):
            visual_caches[
                stem
            ] = load_visual_cache(
                stem
            )

        cache = (
            visual_caches[
                stem
            ]
        )

        for selected_break in (
            optimized_video[
                "selected_breaks"
            ]
        ):
            source_index = int(
                selected_break[
                    "source_candidate_index"
                ]
            )

            candidate = (
                source_video[
                    "candidates"
                ][source_index]
            )

            timestamp = float(
                candidate[
                    "timestamp_seconds"
                ]
            )

            if abs(
                timestamp
                - float(
                    selected_break[
                        "timestamp_seconds"
                    ]
                )
            ) > 1e-6:
                raise ValueError(
                    "Source candidate mismatch "
                    f"for {video_name}"
                )

            text = scene_text(
                candidate
            )

            text_vector = (
                text_encoder.encode(
                    [text]
                )[0]
                if text
                else None
            )

            visual_vector = (
                scene_visual_vector(
                    candidate,
                    cache,
                )
            )

            break_text_evidence = (
                (
                    candidate.get("break_score")
                    or {}
                ).get("text")
                or {}
            )

            text_reliability = float(
                break_text_evidence.get(
                    "reliability",
                    0.0,
                )
                or 0.0
            )

            context_result = (
                infer_contexts(
                    taxonomy=taxonomy,
                    scene_text=text,
                    scene_text_embedding=(
                        text_vector
                    ),
                    scene_visual_embedding=(
                        visual_vector
                    ),
                    taxonomy_text_embeddings=(
                        taxonomy_text_embeddings
                    ),
                    taxonomy_visual_embeddings=(
                        taxonomy_visual_embeddings
                    ),
                    text_reliability=(
                        text_reliability
                    ),
                )
            )

            context_cues = set(
                context_result[
                    "context_cues"
                ]
            )

            hard_safety_contexts = set(
                context_result[
                    "hard_safety_contexts"
                ]
            )

            for context_id in (
                context_cues
            ):
                detected_context_counts[
                    context_id
                ] += 1

            eligible, blocked = (
                filter_brands(
                    catalogue.brands,
                    hard_safety_contexts,
                )
            )

            blocked_brand_decisions += (
                len(blocked)
            )

            ranked = (
                rank_eligible_brands(
                    brands=eligible,
                    detected_contexts=(
                        context_cues
                    ),
                    scene_text_embedding=(
                        text_vector
                    ),
                    scene_visual_embedding=(
                        visual_vector
                    ),
                    brand_text_embeddings=(
                        brand_text_embeddings
                    ),
                    brand_visual_embeddings=(
                        brand_visual_embeddings
                    ),
                )
            )

            selected_brand = (
                ranked[0]
                if ranked
                else None
            )

            if (
                selected_brand
                is not None
            ):
                selected_brand_counts[
                    selected_brand[
                        "brand_id"
                    ]
                ] += 1

                brand = next(
                    item
                    for item
                    in catalogue.brands
                    if (
                        item.brand_id
                        == selected_brand[
                            "brand_id"
                        ]
                    )
                )

                violation = sorted(
                    set(
                        brand.negative_contexts
                    )
                    & hard_safety_contexts
                )

                if violation:
                    negative_context_violations += 1

                    raise RuntimeError(
                        "Hard negative-context "
                        "filter failed: "
                        f"{brand.brand_id} "
                        f"/ {violation}"
                    )

            results.append(
                {
                    "video": (
                        video_name
                    ),
                    "timestamp_seconds": (
                        timestamp
                    ),
                    "eabs": (
                        selected_break[
                            "eabs"
                        ]
                    ),
                    "quality_utility": (
                        selected_break[
                            "quality_utility"
                        ]
                    ),
                    "scene_text": (
                        text
                    ),
                    "context": (
                        context_result
                    ),
                    "blocked_brands": (
                        blocked
                    ),
                    "eligible_brand_ids": [
                        brand.brand_id
                        for brand
                        in eligible
                    ],
                    "ranked_eligible_brands": (
                        ranked
                    ),
                    "selected_brand": (
                        selected_brand
                    ),
                }
            )

    report = {
        "catalogue_name": (
            catalogue.catalogue_name
        ),
        "catalogue_status": (
            catalogue.catalogue_status
        ),
        "taxonomy_version": (
            taxonomy.version
        ),
        "context_method": (
            taxonomy.method
        ),
        "matching_method": (
            "hard negative-context filter, "
            "then equal-rank fusion of "
            "multilingual-E5 text compatibility "
            "and CLIP visual-text compatibility"
        ),
        "breaks": results,
        "summary": {
            "break_count": (
                len(results)
            ),
            "brand_count": (
                len(
                    catalogue.brands
                )
            ),
            "blocked_brand_decisions": (
                blocked_brand_decisions
            ),
            "negative_context_violations": (
                negative_context_violations
            ),
            "selected_brand_counts": dict(
                sorted(
                    selected_brand_counts.items()
                )
            ),
            "context_cue_counts": dict(
                detected_context_counts.most_common()
            ),
            "all_selected_brands_safe": (
                negative_context_violations
                == 0
            ),
            "catalogue_driven": True,
            "title_specific_brand_logic": False,
            "feluda_processed": False,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
