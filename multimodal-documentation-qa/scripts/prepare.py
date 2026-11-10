"""Freeze a searchable documentation pool and light local copies of its illustrations."""
import concurrent.futures
import gzip
import hashlib
import io
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin,urlsplit

from lxml import html
from PIL import Image,ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer

from collect import read_questions, DATA,DOMAINS,ROOT

STORE=DATA/"assets"
STORE.mkdir(exist_ok=True)
CACHE=DATA/"pages"
CACHE.mkdir(exist_ok=True)
MAX_WORDS=115


def get(url,limit=18000000):
    return subprocess.check_output(["curl","-fgsSL","--compressed","--retry","1","--connect-timeout","8","--max-time","25","--max-filesize",str(limit),url],stderr=subprocess.DEVNULL)


def page_pool(name,rows):
    idx=json.loads((DATA/f"{name}-pages.json").read_text())["pages"]
    gold={t for r in rows for t in r["targets"]}
    if name=="matplotlib":
        topical=[p for p in idx if p.startswith(("gallery/","api/","users/explain/","tutorials/","plot_types/"))]
    elif name=="blender": topical=[p for p in idx if p.startswith(("modeling/geometry_nodes/","editors/geometry_node","interface/controls/nodes/"))]
    else: topical=[p for p in idx if p.startswith(("docs/user_manual/","docs/training_manual/","docs/pyqgis_developer_cookbook/"))]
    topical=sorted(p for p in topical if not p.endswith(("/index","sg_execution_times")))
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,4))
    x=v.fit_transform([idx[p]+" "+p.rsplit("/",1)[-1].replace("_"," ") for p in topical])
    sim=(v.transform([r["title"] for r in rows])@x.T).toarray()
    hard={topical[i] for scores in sim for i in scores.argsort()[-1:]}
    random_pages=set(random.Random(42).sample(topical,min(100,len(topical))))
    return sorted(gold|hard|random_pages),idx


def parse(raw,url,page):
    tree=html.fromstring(raw)
    nodes=tree.xpath('//article[@role="main"]|//article[contains(@class,"bd-article")]|//div[@role="main"]|//main|//div[contains(concat(" ",normalize-space(@class)," ")," body ")]')
    if not nodes: raise ValueError("main content not found")
    root=next((el for el in nodes if el.tag=="article"),nodes[0])
    contents_links = [" ".join(a.text_content().split()) for a in root.xpath('.//*[contains(@class,"toctree-wrapper")]//a') if a.text_content().strip()]
    for bad in root.xpath('.//nav|.//script|.//style|.//aside|.//*[contains(@class,"headerlink") or contains(@class,"sphx-glr-footer") or contains(@class,"sidebar") or contains(@class,"toctree-wrapper") or contains(@class,"admonition-versionchanged")]'):
        if bad.getparent() is not None: bad.drop_tree()
    title=" ".join(root.xpath("string(.//h1[1])").split()) or page.rsplit("/",1)[-1]
    sections=[]; heading=title; tokens=[]; images=[]
    def flush():
        if not tokens and not images: return
        for start in range(0,max(1,len(tokens)),MAX_WORDS):
            sections.append({"section":heading,"text":" ".join(tokens[start:start+MAX_WORDS]),"image_urls":list(dict.fromkeys(images)) if start==0 else []})
        tokens.clear();images.clear()
    def walk(el):
        nonlocal heading
        if not isinstance(el.tag,str): return
        if el.tag in ("h1","h2","h3","h4","h5","h6"):
            flush(); heading=" ".join(el.text_content().split()) or title; return
        if el.tag=="img":
            img=urljoin(url,el.get("src",""))
            if "/_images/" in img and not img.lower().endswith(".svg"):
                images.append(img)
            if el.get("alt","").strip(): tokens.extend(el.get("alt").split())
            return
        tokens.extend((el.text or "").split())
        for ch in el:
            walk(ch)
            tokens.extend((ch.tail or "").split())
    walk(root);flush()
    if not sections and contents_links:
        sections.append({"section":title,"text":" · ".join(contents_links),"image_urls":[]})
    if not sections: raise ValueError("empty article")
    return {"slug":page,"url":url,"title":title,"chunks":sections}


def load_page(task):
    name,slug=task
    url=DOMAINS[name][2]+slug+".html"
    dst=CACHE/name/(hashlib.sha256(slug.encode()).hexdigest()[:20]+".json.gz")
    if dst.exists(): return json.loads(gzip.decompress(dst.read_bytes()))
    try:
        p=parse(get(url),url,slug)
    except (ValueError,subprocess.CalledProcessError) as err:
        return {"slug":slug,"url":url,"error":str(err)}
    dst.parent.mkdir(exist_ok=True)
    dst.write_bytes(gzip.compress(json.dumps(p,ensure_ascii=False).encode(),mtime=0))
    return p


def image_path(url): return STORE/(hashlib.sha256(url.encode()).hexdigest()[:24]+".jpg")


def ensure_image(url):
    target=image_path(url)
    if target.exists(): return True
    u=urlsplit(url)
    if u.scheme not in ("https","http") or u.hostname not in {"i.sstatic.net","i.stack.imgur.com","docs.blender.org","docs.qgis.org","matplotlib.org"}: return False
    try:
        with Image.open(io.BytesIO(get(url,16000000))) as img:
            if min(img.size)<60 or max(img.size)<140: return False
            img=ImageOps.exif_transpose(img)
            img.thumbnail((512,512), Image.Resampling.LANCZOS)
            if img.mode in ("RGBA","LA","P") or "transparency" in img.info:
                rgba=img.convert("RGBA"); white=Image.new("RGB",img.size,"white"); white.paste(rgba,mask=rgba.getchannel("A")); img=white
            img.convert("RGB").save(target,quality=88)
        return True
    except (subprocess.CalledProcessError,OSError,ValueError,Image.DecompressionBombError): return False


def prepare(name):
    rows=read_questions(name)
    if not rows: raise ValueError(name+" empty sample")
    pool,idx=page_pool(name,rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        pages=list(executor.map(load_page,[(name,p) for p in pool]))
    errors=[x for x in pages if "error" in x]
    pages=[x for x in pages if "error" not in x]
    candidates=set(u for r in rows for u in r["images"][:2])
    for p in pages:
        ordered=list(dict.fromkeys(u for c in p["chunks"] for u in c["image_urls"]))
        candidates.update(ordered[:3])
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        status=dict(zip(sorted(candidates),executor.map(ensure_image,sorted(candidates))))
    for p in pages:
        valid=list(dict.fromkeys(u for c in p["chunks"] for u in c["image_urls"] if status.get(u)))[:2]
        for c in p["chunks"]: c["assets"]=[u for u in c["image_urls"] if u in valid][:1]
    usable=sorted(url for url,ok in status.items() if ok)
    result={"pages":pages,"failed":errors,"planned_pool":pool,"available_assets":{u:image_path(u).name for u in usable},
            "unavailable_or_small_images":sorted(u for u,ok in status.items() if not ok)}
    (DATA/f"{name}-corpus.json.gz").write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False).encode(),mtime=0))
    print(name,"pages",len(pages),"/",len(pool),"chunks",sum(len(p["chunks"]) for p in pages),"readable images",len(usable),"failed",len(errors),flush=True)


if __name__=="__main__":
    for name in sys.argv[1:] or DOMAINS: prepare(name)
