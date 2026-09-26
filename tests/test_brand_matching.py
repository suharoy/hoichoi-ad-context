import numpy as np

from src.brand_matching.catalog import Brand
from src.brand_matching.ranker import (
    rank_eligible_brands,
)
from src.brand_matching.safety import (
    filter_brands,
)
from src.context.context_inference import (
    ContextEntry,
    ContextTaxonomy,
    infer_contexts,
    rank_percentiles,
)


def brand(
    brand_id,
    *,
    positive=(),
    negative=(),
    activities=(),
):
    return Brand(
        brand_id=brand_id,
        name=brand_id,
        description=(
            f"Synthetic {brand_id}"
        ),
        positive_contexts=tuple(
            positive
        ),
        negative_contexts=tuple(
            negative
        ),
        activities=tuple(
            activities
        ),
        creative_uri=(
            f"/{brand_id}.mp4"
        ),
    )


def test_rank_percentiles_are_scale_free():
    result = rank_percentiles(
        {
            "a": 1000.0,
            "b": 5.0,
            "c": -20.0,
        }
    )

    assert result["a"] == 1.0
    assert result["b"] == 0.5
    assert result["c"] == 0.0


def test_negative_context_is_hard_blocked_before_ranking():
    safe = brand(
        "safe",
        positive=("food",),
    )

    unsafe = brand(
        "unsafe",
        positive=("food",),
        negative=("violence",),
    )

    eligible, blocked = (
        filter_brands(
            [safe, unsafe],
            {"food", "violence"},
        )
    )

    assert [
        item.brand_id
        for item in eligible
    ] == ["safe"]

    assert (
        blocked[0]["brand_id"]
        == "unsafe"
    )

    assert (
        blocked[0]["hard_block"]
        is True
    )

    assert (
        blocked[0][
            "negative_context_hits"
        ]
        == ["violence"]
    )


def test_lexical_safety_hit_is_always_detected():
    taxonomy = ContextTaxonomy(
        version=1,
        context_top_k=1,
        safety_consensus_percentile=0.90,
        method="test",
        contexts=(
            ContextEntry(
                context_id="food",
                text_description="food",
                visual_prompt="food",
                keywords=("খাবার",),
                safety_sensitive=False,
            ),
            ContextEntry(
                context_id="violence",
                text_description="violence",
                visual_prompt="violence",
                keywords=("খুন",),
                safety_sensitive=True,
            ),
        ),
    )

    result = infer_contexts(
        taxonomy=taxonomy,
        scene_text=(
            "এখানে খুন হয়েছে"
        ),
        scene_text_embedding=np.array(
            [1.0, 0.0]
        ),
        scene_visual_embedding=np.array(
            [1.0, 0.0]
        ),
        taxonomy_text_embeddings={
            "food": np.array(
                [1.0, 0.0]
            ),
            "violence": np.array(
                [0.0, 1.0]
            ),
        },
        taxonomy_visual_embeddings={
            "food": np.array(
                [1.0, 0.0]
            ),
            "violence": np.array(
                [0.0, 1.0]
            ),
        },
    )

    assert (
        "violence"
        in result[
            "detected_contexts"
        ]
    )

    assert (
        "violence"
        in result[
            "lexical_safety_hits"
        ]
    )


def test_ranker_uses_only_eligible_brand_list():
    first = brand(
        "first",
        positive=("food",),
    )

    second = brand(
        "second",
        positive=("travel",),
    )

    ranked = (
        rank_eligible_brands(
            brands=[first],
            detected_contexts={
                "food"
            },
            scene_text_embedding=np.array(
                [1.0, 0.0]
            ),
            scene_visual_embedding=np.array(
                [1.0, 0.0]
            ),
            brand_text_embeddings={
                "first": np.array(
                    [1.0, 0.0]
                ),
                "second": np.array(
                    [0.0, 1.0]
                ),
            },
            brand_visual_embeddings={
                "first": np.array(
                    [1.0, 0.0]
                ),
                "second": np.array(
                    [0.0, 1.0]
                ),
            },
        )
    )

    assert len(ranked) == 1

    assert (
        ranked[0]["brand_id"]
        == "first"
    )


def test_unseen_ninth_brand_requires_no_code_path():
    existing = brand(
        "existing",
        positive=("food",),
    )

    unseen_ninth = brand(
        "unseen_09",
        positive=("travel",),
    )

    ranked = (
        rank_eligible_brands(
            brands=[
                existing,
                unseen_ninth,
            ],
            detected_contexts={
                "travel"
            },
            scene_text_embedding=np.array(
                [0.0, 1.0]
            ),
            scene_visual_embedding=np.array(
                [0.0, 1.0]
            ),
            brand_text_embeddings={
                "existing": np.array(
                    [1.0, 0.0]
                ),
                "unseen_09": np.array(
                    [0.0, 1.0]
                ),
            },
            brand_visual_embeddings={
                "existing": np.array(
                    [1.0, 0.0]
                ),
                "unseen_09": np.array(
                    [0.0, 1.0]
                ),
            },
        )
    )

    assert (
        ranked[0]["brand_id"]
        == "unseen_09"
    )

    assert (
        ranked[0]["match_score"]
        == 100.0
    )


def test_multimodal_safety_consensus_is_detected():
    taxonomy = ContextTaxonomy(
        version=1,
        context_top_k=1,
        safety_consensus_percentile=0.90,
        method="test",
        contexts=(
            ContextEntry(
                context_id="food",
                text_description="food",
                visual_prompt="food",
                keywords=(),
                safety_sensitive=False,
            ),
            ContextEntry(
                context_id="violence",
                text_description="violence",
                visual_prompt="violence",
                keywords=(),
                safety_sensitive=True,
            ),
        ),
    )

    result = infer_contexts(
        taxonomy=taxonomy,
        scene_text="",
        scene_text_embedding=np.array([0.0, 1.0]),
        scene_visual_embedding=np.array([0.0, 1.0]),
        taxonomy_text_embeddings={
            "food": np.array([1.0, 0.0]),
            "violence": np.array([0.0, 1.0]),
        },
        taxonomy_visual_embeddings={
            "food": np.array([1.0, 0.0]),
            "violence": np.array([0.0, 1.0]),
        },
        text_reliability=1.0,
    )

    assert "violence" in result["detected_contexts"]
    assert "violence" in result["consensus_safety_hits"]


def test_visual_context_shot_indices_can_be_integers():
    from scripts.match_brands import scene_visual_vector

    candidate = {
        "visual_semantics": {
            "left_context_shots": [1, 2],
            "right_context_shots": [3],
        }
    }

    cache = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
    }

    result = scene_visual_vector(
        candidate,
        cache,
    )

    assert result.shape == (2,)
    assert np.isfinite(result).all()
    assert np.isclose(
        np.linalg.norm(result),
        1.0,
    )