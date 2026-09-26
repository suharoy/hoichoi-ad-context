import copy
import json
import wave
from unittest.mock import Mock, patch

import pytest

from scripts import enrich_boundaries_asr as enrichment
from src.audio.indic_asr import IndicConformerASR


def test_backend_loads_once_and_only_strips_edges():
    with patch("src.audio.indic_asr.onnx_asr.load_model") as load:
        load.return_value.recognize.return_value = "  বাংলা  English! \n"
        model = IndicConformerASR()
        assert model.recognize("left.wav") == "বাংলা  English!"
        assert model.recognize("right.wav") == "বাংলা  English!"
        load.assert_called_once_with(
            "OpenVoiceOS/ai4bharat-indicconformer-bn-onnx",
            quantization="int8", providers=["CPUExecutionProvider"],
        )


def test_script_counts_include_other_alphabets_in_denominator():
    counts, ratio = enrichment.script_quality("কकకಕAzΩ ১২3 !")
    assert counts == dict(bengali=1, devanagari=1, telugu=1, kannada=1, latin=2)
    assert ratio == pytest.approx(1 / 7)
    assert enrichment.script_quality(" ১২3!")[1] == 0


def test_enrichment_preserves_fields_and_resumes_per_window(tmp_path, monkeypatch):
    dev = tmp_path / "dev"
    corpus = dev / "corpus" / "bhojon_bilashi"
    corpus.mkdir(parents=True)
    with wave.open(str(corpus / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 16000 * 3)
    profile = {
        "held_out": "feluda.mp4",
        "videos": [
            {"video": "bhojon_bilashi.mp4", "candidates": [
                {"timestamp_seconds": 1.0, "speech_safe": True, "keep": {"x": 1}},
                {"timestamp_seconds": 2.0, "speech_safe": False},
            ]},
            {"video": "mandaar.mp4", "candidates": [
                {"timestamp_seconds": 1.0, "speech_safe": True},
            ]},
        ],
    }
    source = dev / "boundary-profile.json"
    source.write_text(json.dumps(profile), encoding="utf-8")
    output = dev / "boundary-profile-asr.json"
    monkeypatch.setattr(enrichment, "DEV", dev)
    monkeypatch.setattr(enrichment, "INPUT", source)
    monkeypatch.setattr(enrichment, "OUTPUT", output)
    model = Mock()
    # Empty output is valid, and Latin output is not rejected.
    model.recognize.side_effect = ["", "English", "ক"]
    factory = Mock(return_value=model)
    monkeypatch.setattr(enrichment, "IndicConformerASR", factory)

    result = enrichment.run(["bhojon_bilashi"])
    assert result["windows_processed"] == 2
    assert result["windows_reused"] == 0
    factory.assert_called_once()
    enriched = json.loads(output.read_text(encoding="utf-8"))
    context = enriched["videos"][0]["candidates"][0]["asr_context"]
    assert (context["left"]["start"], context["left"]["end"]) == (0, 1)
    assert (context["right"]["start"], context["right"]["end"]) == (1, 3)
    assert context["right"]["text"] == "English"
    assert "script_pass" not in context["right"]
    stripped = copy.deepcopy(enriched)
    del stripped["videos"][0]["candidates"][0]["asr_context"]
    assert stripped == profile
    assert json.loads(source.read_text()) == profile

    clips = list((dev / "asr-context" / "bhojon_bilashi").glob("*.wav"))
    mtimes = {clip: clip.stat().st_mtime_ns for clip in clips}
    resumed = enrichment.run(["bhojon_bilashi"])
    assert resumed["windows_processed"] == 0
    assert resumed["windows_reused"] == 2
    assert model.recognize.call_count == 2
    assert all(clip.stat().st_mtime_ns == stamp for clip, stamp in mtimes.items())
    assert json.loads(output.read_text(encoding="utf-8")) == enriched

    # A missing clip invalidates only its own cached inference.
    (dev / "asr-context" / "bhojon_bilashi" / "000001000_left.wav").unlink()
    resumed = enrichment.run(["bhojon_bilashi"])
    assert resumed["windows_processed"] == 1
    assert resumed["windows_reused"] == 1


def test_held_out_cannot_be_selected():
    with pytest.raises(ValueError, match="development"):
        enrichment.run(["feluda"])
