"""Build the accepted-answer dataset; the notebook reads only the local snapshots."""
import argparse
import datetime as dt
import gzip
import hashlib
import html
import json
import os
import random
import re
import subprocess
import tempfile
import time
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = Path(tempfile.gettempdir()) / "docqa-api-accepted-v2"
DOMAINS = {
    "matplotlib": ("stackoverflow","matplotlib","https://matplotlib.org/3.10.7/"),
    "blender": ("blender","geometry-nodes","https://docs.blender.org/manual/en/4.5/"),
    "qgis": ("gis","qgis;raster","https://docs.qgis.org/3.44/en/"),
}
FILTER = "!20aKG._8Oscv*6djs8Pgm"


def fetch(url):
    return subprocess.check_output(["curl","-fgsSL","--compressed","--connect-timeout","12","--max-time","90","--retry","2",url])


def api(path,**params):
    url="https://api.stackexchange.com/2.3/"+path+"?"+urlencode(params)
    CACHE.mkdir(exist_ok=True)
    file=CACHE/(hashlib.sha256(url.encode()).hexdigest()+".gz")
    if file.exists():
        return json.loads(gzip.decompress(file.read_bytes()))
    raw=fetch(url)
    d=json.loads(raw)
    if "items" not in d: raise ValueError(d)
    tmp=file.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(raw))
    tmp.replace(file)
    if d.get("backoff"): time.sleep(d["backoff"]+1)
    elif d.get("quota_remaining",1)==0: print("API quota exhausted",flush=True)
    else: time.sleep(.12)
    return d


class Markup(HTMLParser):
    def __init__(self,body):
        super().__init__()
        self.links=[]; self.images=[]; self.body=[]; self.code=[]; self.pre=0
        self.feed(body)
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=="a" and d.get("href"): self.links.append(d["href"])
        if tag=="img" and d.get("src"): self.images.append(d["src"])
        if tag=="pre": self.pre+=1
    def handle_endtag(self,tag):
        if tag=="pre": self.pre=max(0,self.pre-1)
    def handle_data(self,data):
        (self.code if self.pre else self.body).append(data)


def read_questions(domain=None):
    with (DATA / "questions.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows if domain is None else [row for row in rows if row["domain"] == domain]


def docmap(name):
    return json.loads((DATA / f"{name}-pages.json").read_text())


def canonical(link,name,idx):
    try: u=urlsplit(link)
    except ValueError: return None
    host=(u.hostname or "").removeprefix("www.")
    path=unquote(u.path).strip("/")
    if name=="matplotlib" and host=="matplotlib.org":
        path=re.sub(r"^(stable|devdocs|main|\d+\.\d+(?:\.\d+)?(?:rc\d+)?)/","",path)
    elif name=="blender" and host=="docs.blender.org":
        m=re.match(r"^manual/[^/]+/(?:latest|dev|[0-9][\w.-]*)/(.*)",path)
        if not m: return None
        path=m[1]
    elif name=="qgis" and host=="docs.qgis.org":
        m=re.match(r"^(?:testing|latest|[0-9][\w.-]*)/[^/]+/(docs/.*)",path)
        if not m: return None
        path=m[1]
    else: return None
    path=path.removesuffix(".html").rstrip("/")
    if path not in idx["pages"]: path=idx["aliases"].get(path,path)
    if path not in idx["pages"] and path+"/index" in idx["pages"]: path+="/index"
    return path if path in idx["pages"] else None


def epoch(year,month): return int(dt.datetime(year,month,1,tzinfo=dt.timezone.utc).timestamp())


def scan(name):
    site,tags,base=DOMAINS[name]
    idx=docmap(name); scanned={}; eligible={}; counts=Counter()
    for y in range(2021,2026):
        for m in (1,4,7,10):
            windows=[(epoch(y,m), (epoch(y+1,1) if m==10 else epoch(y,m+3))-1)]
            while windows:
                lo,hi=windows.pop()
                for page in range(1,26):
                    response=api("search/advanced",site=site,tagged=tags,accepted="true",fromdate=lo,todate=hi,sort="creation",order="asc",pagesize=100,page=page,filter=FILTER)
                    for q in response["items"]:
                        qid=q["question_id"]
                        if qid in scanned: continue
                        scanned[qid]=True
                        aid=q.get("accepted_answer_id")
                        a=next((a for a in q.get("answers",[]) if a.get("answer_id")==aid and a.get("is_accepted")),None)
                        if not a and aid:
                            extra=api(f"answers/{aid}",site=site,filter="withbody")["items"]
                            a=next((v for v in extra if v.get("is_accepted") and v["question_id"]==qid),None)
                        if not a: counts["accepted_unavailable"]+=1;continue
                        am=Markup(a["body"]); qm=Markup(q["body"])
                        links={u:t for u in set(am.links) if (t:=canonical(u,name,idx))}
                        if not links: continue
                        counts["matching_page_in_answer"]+=1
                        already={canonical(u,name,idx) for u in qm.links}
                        use=sorted(set(links.values())-already)
                        if not use: continue
                        owner=lambda o:{"name":html.unescape(o.get("display_name","unknown")),"url":o.get("link"),"id":o.get("user_id")}
                        eligible[qid]={"domain":name,"site":site,"id":qid,"accepted_answer_id":aid,"url":q["link"],
                          "answer_url":f"https://{urlsplit(q['link']).netloc}/a/{aid}","accepted":True,
                          "title":html.unescape(q["title"]),"body_html":q["body"],"answer_html":a["body"],
                          "body_text":" ".join(qm.body).strip(),"code":"\n".join(qm.code).strip(),"images":list(dict.fromkeys(qm.images)),
                          "author":owner(q.get("owner",{})),"answer_author":owner(a.get("owner",{})),
                          "license":q.get("content_license"),"answer_license":a.get("content_license"),
                          "creation_date":q["creation_date"],"score":q["score"],"tags":q["tags"],
                          "answer_doc_links":dict(sorted(links.items())),"question_doc_pages":sorted(t for t in already if t),
                          "targets":use,"target_urls":[base+t+".html" for t in use]}
                    if not response["has_more"]: break
                    if page==25:
                        mid=(lo+hi)//2
                        windows.extend([(lo,mid),(mid+1,hi)])
            print(name,y,m,"checked",len(scanned),"eligible",len(eligible),"quota",response.get("quota_remaining"),flush=True)
    order=sorted(eligible.values(),key=lambda x:(-x["score"],x["id"]))
    size=min(200,len(order))
    top=order[:size//2]; others=order[size//2:]
    draw=random.Random(42).sample(sorted(others,key=lambda x:x["id"]),size-len(top))
    for group,records in [("popular",top),("random_remaining",draw)]:
        for r in records: r["group"]=group
    manifest={"site":site,"tags":tags,"documentation":base,"seen_accepted_questions":len(scanned),
              "eligible":len(eligible),"selected":size,"seed":42,"popular":len(top),"random_remaining":len(draw),
              "first_utc": "2021-01-01","last_utc":"2025-12-31","snapshot":dt.datetime.now(dt.timezone.utc).isoformat(),
              "counts":dict(counts)}
    (DATA/(name+"-selection.json")).write_text(json.dumps(manifest,indent=2))
    print(manifest,flush=True)
    return top + draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("domains", nargs="*", metavar="DOMAIN", help="matplotlib, blender, qgis")
    selected = parser.parse_args().domains or list(DOMAINS)
    if set(selected) - DOMAINS.keys():
        parser.error("Unknown domain")
    path = DATA / "questions.jsonl"
    existing = read_questions() if path.exists() else []
    by_domain = {name: [r for r in existing if r["domain"] == name] for name in DOMAINS}
    for name in selected:
        by_domain[name] = scan(name)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for name in DOMAINS:
            for row in by_domain[name]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
