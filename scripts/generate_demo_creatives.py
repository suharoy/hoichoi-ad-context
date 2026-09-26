"""Generate synthetic 30-second MP4 creatives used by the live demo."""
from __future__ import annotations
import hashlib,json,shutil,subprocess,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]; CATALOGUE=ROOT/'configs'/'brands.demo.json'; OUT=ROOT/'frontend'/'creatives'
W,H,DURATION=1280,720,30

def font(size):
    for p in [Path('C:/Windows/Fonts/segoeuib.ttf'),Path('C:/Windows/Fonts/arialbd.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]:
        if p.exists(): return ImageFont.truetype(str(p),size=size)
    return ImageFont.load_default()

def colour(bid):
    d=hashlib.sha256(bid.encode()).digest(); return (45+d[0]%90,35+d[1]%90,55+d[2]%100)

def card(brand,path):
    im=Image.new('RGB',(W,H),colour(str(brand['brand_id']))); d=ImageDraw.Draw(im)
    d.text((70,60),'SYNTHETIC DEMO CREATIVE',font=font(24),fill='white')
    d.text((70,125),str(brand.get('name',brand['brand_id'])),font=font(76),fill='white')
    desc='\n'.join(textwrap.wrap(str(brand['description']),54)); d.multiline_text((74,265),desc,font=font(30),fill='white',spacing=10)
    ctx=', '.join(brand.get('positive_contexts',[])); d.text((74,535),f'Context fit: {ctx}',font=font(22),fill='white')
    d.text((74,630),'AdContext · hackathon-generated placeholder creative',font=font(20),fill='white'); im.save(path)

def main():
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('ffmpeg not found on PATH')
    data=json.loads(CATALOGUE.read_text(encoding='utf-8')); OUT.mkdir(parents=True,exist_ok=True)
    for b in data['brands']:
        bid=str(b['brand_id']); png=OUT/f'{bid}.png'; mp4=OUT/f'{bid}.mp4'; print('Generating',bid,flush=True); card(b,png)
        subprocess.run([ffmpeg,'-y','-loop','1','-i',str(png),'-f','lavfi','-i','anullsrc=channel_layout=stereo:sample_rate=48000','-t',str(DURATION),'-c:v','libx264','-preset','veryfast','-tune','stillimage','-crf','30','-pix_fmt','yuv420p','-c:a','aac','-b:a','64k','-shortest','-movflags','+faststart',str(mp4)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        png.unlink(missing_ok=True)
    print(f'Generated {len(data["brands"])} creatives in {OUT}')
if __name__=='__main__': main()
