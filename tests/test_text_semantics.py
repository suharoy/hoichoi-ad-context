import copy
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts import enrich_text_semantics as semantics
from src.context.text_embeddings import MODEL_NAME, SemanticTextEncoder


def test_encoder_prefixes_both_sides_and_requests_normalized_cpu_batches():
    with patch("src.context.text_embeddings.SentenceTransformer") as loader:
        encoder = SemanticTextEncoder()
        encoder.encode(["বাংলা", "English"])
        encoder.encode(["আরও"])
        loader.assert_called_once_with(MODEL_NAME, device="cpu")
        assert loader.return_value.encode.call_args_list[0].args[0] == [
            "query: বাংলা", "query: English",
        ]
        assert loader.return_value.encode.call_args_list[0].kwargs == {
            "batch_size": 32, "normalize_embeddings": True,
            "convert_to_numpy": True, "show_progress_bar": True,
        }


def candidate(left, right, safe=True):
    return {
        "timestamp_seconds": 1.0, "speech_safe": safe,
        "asr_context": {"left": {"text": left}, "right": {"text": right}},
    }


def test_dedup_missing_evidence_and_preservation():
    profile = {"videos": [
        {"video": "mandaar.mp4", "candidates": [
            candidate("এক", "দুই"), candidate("এক", "এক"),
            candidate(" ", "দুই"), candidate("এক", ""),
            candidate("unsafe", "unsafe", safe=False),
        ]},
        {"video": "feluda.mp4", "candidates": [candidate("heldout", "heldout")]},
    ]}
    before = copy.deepcopy(profile)
    encoder = Mock()
    encoder.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    enriched, unique_count = semantics.enrich(profile, encoder)
    encoder.encode.assert_called_once_with(["এক", "দুই"], batch_size=32)
    assert unique_count == 2
    assert profile == before
    cs = enriched["videos"][0]["candidates"]
    assert cs[0]["text_semantics"]["cosine_similarity"] == 0
    assert cs[0]["text_semantics"]["semantic_change"] == 1
    assert cs[1]["text_semantics"]["semantic_change"] == 0
    for c in cs[2:4]:
        assert c["text_semantics"]["available"] is False
        assert c["text_semantics"]["cosine_similarity"] is None
        assert c["text_semantics"]["semantic_change"] is None
    assert cs[2]["text_semantics"]["left_characters"] == 1
    assert cs[4] == before["videos"][0]["candidates"][4]
    assert enriched["videos"][1] == before["videos"][1]
    for c in cs:
        c.pop("text_semantics", None)
    assert enriched == before


def test_sanity_failure_stops_before_reading_corpus(tmp_path, monkeypatch):
    encoder = Mock()
    # Identical vectors for all diagnostic texts fail the strict ordering.
    encoder.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 1))
    monkeypatch.setattr(semantics, "SemanticTextEncoder", lambda: encoder)
    monkeypatch.setattr(semantics, "INPUT", tmp_path / "missing.json")
    output = tmp_path / "semantic.json"
    monkeypatch.setattr(semantics, "OUTPUT", output)
    with pytest.raises(RuntimeError, match="sanity"):
        semantics.main()
    assert not output.exists()
    assert encoder.encode.call_count == 1


def test_summary_percentiles_and_empty_statistics():
    stats = semantics.distribution([0.0, 1.0])
    assert stats == pytest.approx({
        "mean": 0.5, "median": 0.5, "min": 0, "max": 1,
        "p10": 0.1, "p25": 0.25, "p75": 0.75, "p90": 0.9,
    })
    assert all(value is None for value in semantics.distribution([]).values())
