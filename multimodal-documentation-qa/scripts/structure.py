"""Read the official HTML structure for the same fixed document pool."""
import concurrent.futures
import gzip
import json
import re
import sys
from urllib.parse import urljoin,urldefrag

from lxml import html
from collect import read_questions, DATA,DOMAINS,canonical,docmap
from prepare import get,page_pool

MENTION=re.compile(r"\b(?:as (?:shown|illustrated|seen) (?:above|below)|(?:as shown in |see )?(?:fig(?:ure)?\.?\s*\d+)|(?:the )?(?:figure|image|screenshot) (?:above|below))\b",re.I)


def inspect(task):
    name,slug,idx=task
    url=DOMAINS[name][2]+slug+".html"
    try: tree=html.fromstring(get(url))
    except Exception as err: return {"slug":slug,"error":str(err)}
    nodes=tree.xpath('//article[@role="main"]|//article[contains(@class,"bd-article")]|//div[@role="main"]|//main')
    if not nodes: return {"slug":slug,"error":"missing main"}
    root=next((x for x in nodes if x.tag=="article"),nodes[0])
    for el in root.xpath('.//nav|.//aside|.//*[contains(@class,"headerlink") or contains(@class,"toctree-wrapper")]'):
        if el.getparent() is not None: el.drop_tree()
    figures=root.xpath('.//figure[.//img]|.//div[contains(concat(" ",normalize-space(@class)," ")," figure ")][.//img]')
    ids=set()
    for f in figures:
        if f.get("id"): ids.add(f.get("id"))
        ids.update(f.xpath('.//*[@id]/@id'))
    nref=0; links=set()
    for a in root.xpath('.//a[@href]'):
        dst,frag=urldefrag(urljoin(url,a.get("href")))
        if dst==url and frag in ids and not a.xpath(".//img"): nref+=1
        target=canonical(dst,name,idx)
        if target and target!=slug: links.add(target)
    paras=[" ".join(p.text_content().split()) for p in root.xpath(".//p[not(ancestor::figure)]")]
    return {"slug":slug,"figure_blocks":len(figures),"captioned":sum(bool(f.xpath('.//figcaption[normalize-space()]|.//*[contains(concat(" ",normalize-space(@class)," ")," caption ")][normalize-space()]')) for f in figures),
            "paragraphs":len(paras),"paragraph_markers":sum(bool(MENTION.search(p)) for p in paras),
            "exact_figure_links":nref,"outlinks":sorted(links)}


def main():
    for name in DOMAINS:
        rows=read_questions(name)
        pool,_=page_pool(name,rows); idx=docmap(name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            data=list(ex.map(inspect,[(name,s,idx) for s in pool]))
        (DATA/f"{name}-structure.json.gz").write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0))
        print(name,len(data),sum('error'in p for p in data),flush=True)


if __name__=="__main__":main()
