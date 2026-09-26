import json
import pytest
from src.api.app import load_manifest,safe_child

def good(): return {'videos':[],'summary':{'break_count':0,'negative_context_violations':0,'all_breaks_have_brand':True}}

def test_safe_child_accepts_filename(tmp_path): assert safe_child(tmp_path,'video.mp4')==(tmp_path/'video.mp4').resolve()

def test_safe_child_rejects_traversal(tmp_path):
    with pytest.raises(ValueError): safe_child(tmp_path,'../secret.mp4')

def test_rejects_unsafe_manifest(tmp_path):
    p=good(); p['summary']['negative_context_violations']=1; f=tmp_path/'m.json'; f.write_text(json.dumps(p))
    with pytest.raises(ValueError): load_manifest(f)

def test_accepts_safe_manifest(tmp_path):
    f=tmp_path/'m.json'; f.write_text(json.dumps(good())); assert load_manifest(f)['summary']['all_breaks_have_brand'] is True



def test_accepts_resolved_no_fill_manifest(tmp_path):
    payload = {
        "videos": [],
        "summary": {
            "break_count": 1,
            "delivered_ad_count": 0,
            "no_fill_break_count": 1,
            "all_breaks_resolved": True,
            "all_breaks_have_brand": False,
            "negative_context_violations": 0,
        },
    }

    path = tmp_path / "resolved.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = load_manifest(path)

    assert (
        loaded["summary"][
            "no_fill_break_count"
        ]
        == 1
    )


def test_rejects_unresolved_manifest(tmp_path):
    payload = {
        "videos": [],
        "summary": {
            "break_count": 1,
            "all_breaks_resolved": False,
            "negative_context_violations": 0,
        },
    }

    path = tmp_path / "unresolved.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_manifest(path)
