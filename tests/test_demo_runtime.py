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
