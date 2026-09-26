"""FastAPI runtime for the playable hackathon demo."""
from __future__ import annotations
import json, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[2]
FRONTEND=ROOT/'frontend'
CREATIVES=FRONTEND/'creatives'
DEFAULT_MANIFEST=ROOT/'outputs'/'dev'/'manifests'/'debug-manifest.json'
DEFAULT_VIDEO_ROOT=ROOT.parent/'hoichoi-assets'

def manifest_path()->Path:
    v=os.getenv('HOICHOI_MANIFEST_PATH')
    return Path(v).expanduser().resolve() if v else DEFAULT_MANIFEST.resolve()

def video_root()->Path:
    v=os.getenv('HOICHOI_VIDEO_ROOT')
    return Path(v).expanduser().resolve() if v else DEFAULT_VIDEO_ROOT.resolve()

def safe_child(root:Path, filename:str)->Path:
    if not filename or Path(filename).name!=filename:
        raise ValueError('Only plain filenames are allowed')
    root=root.resolve(); target=(root/filename).resolve()
    if target.parent!=root: raise ValueError('Requested path escapes configured root')
    return target

def load_manifest(path:Path)->dict:
    if not path.exists(): raise FileNotFoundError(path)
    payload=json.loads(path.read_text(encoding='utf-8'))
    s=payload.get('summary') or {}
    if s.get('negative_context_violations',0)!=0:
        raise ValueError('Manifest contains negative-context violations')
    if not s.get('all_breaks_have_brand',False):
        raise ValueError('Manifest contains unmatched breaks')
    return payload

def runtime_manifest()->dict:
    payload=load_manifest(manifest_path()); vr=video_root(); videos=[]
    for v in payload.get('videos',[]):
        name=str(v['video'])
        try: available=safe_child(vr,name).is_file()
        except ValueError: available=False
        videos.append({**v,'media_url':f'/media/{name}','media_available':available})
    return {**payload,'videos':videos,'runtime':{'video_root':str(vr),'manifest_path':str(manifest_path())}}

app=FastAPI(title='Hoichoi Ad Context Demo',version='0.1.0',docs_url='/api/docs',redoc_url=None)
app.mount('/static',StaticFiles(directory=str(FRONTEND)),name='static')

@app.get('/api/health')
def health():
    try: m=runtime_manifest()
    except Exception as e: return {'status':'degraded','detail':str(e)}
    return {'status':'ok','video_count':len(m['videos']),'available_video_count':sum(1 for v in m['videos'] if v['media_available']),'break_count':m['summary']['break_count'],'negative_context_violations':m['summary']['negative_context_violations']}

@app.get('/api/manifest')
def manifest():
    try: return JSONResponse(runtime_manifest())
    except FileNotFoundError as e: raise HTTPException(503,f'Debug manifest not found: {e}') from e
    except ValueError as e: raise HTTPException(500,str(e)) from e

@app.get('/media/{filename}')
def media(filename:str):
    try: p=safe_child(video_root(),filename)
    except ValueError as e: raise HTTPException(400,str(e)) from e
    if not p.is_file(): raise HTTPException(404,f'Video not found: {filename}')
    return FileResponse(p,media_type='video/mp4',filename=filename,content_disposition_type='inline')

@app.get('/creatives/{filename}')
def creative(filename:str):
    try: p=safe_child(CREATIVES,filename)
    except ValueError as e: raise HTTPException(400,str(e)) from e
    if not p.is_file(): raise HTTPException(404,f'Creative not found: {filename}')
    return FileResponse(p,media_type='video/mp4',filename=filename,content_disposition_type='inline')

@app.get('/')
def index(): return FileResponse(FRONTEND/'index.html')
