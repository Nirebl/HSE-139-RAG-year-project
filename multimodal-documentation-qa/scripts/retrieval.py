"""Local ONNX encoders, lexical retrieval and paired multimodal page ranking."""
import gzip
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from collect import read_questions

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
CACHE=ROOT/".cache"/"embeddings"
METHODS=("BM25","BGE","CLIP A","CLIP B","CLIP C","CLIP D")
CLIP_MEAN=np.array([.48145466,.4578275,.40821073],dtype=np.float32)
CLIP_STD=np.array([.26862954,.26130258,.27577711],dtype=np.float32)


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)


def session(group,filename):
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
    return ort.InferenceSession(str(ROOT/"models"/group/filename),sess_options=opt,providers=["CPUExecutionProvider"])


def memo(tag,values,compute):
    CACHE.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()[:16]
    path=CACHE/f"v1-{tag}-{digest}.npy"
    if not path.exists():
        arr=compute()
        temp=path.with_suffix(".tmp.npy")
        np.save(temp,arr);temp.replace(path)
    return np.load(path)


def encode_text(group,texts):
    def calculate():
        is_bge=group=="bge"
        tok=Tokenizer.from_file(str(ROOT/"models"/group/"tokenizer.json"))
        tok.enable_truncation(max_length=160 if is_bge else 77)
        encoded=tok.encode_batch(texts)
        model=session(group,"model_quantized.onnx" if is_bge else "text_model_quantized.onnx")
        ids=sorted(range(len(encoded)),key=lambda i:len(encoded[i].ids))
        result=np.empty((len(encoded),384 if is_bge else 512),dtype=np.float32)
        for begin in range(0,len(ids),16):
            ind=ids[begin:begin+16]
            n=max(len(encoded[i].ids) for i in ind)
            input_ids=np.full((len(ind),n),0 if is_bge else 49407,dtype=np.int64)
            mask=np.zeros_like(input_ids)
            for row,i in enumerate(ind):
                seq=encoded[i].ids;input_ids[row,:len(seq)]=seq;mask[row,:len(seq)]=1
            feed={"input_ids":input_ids}
            if is_bge: feed.update(attention_mask=mask,token_type_ids=np.zeros_like(input_ids))
            out=model.run(None,feed)[0]
            result[ind]=normalize(out[:,0,:] if is_bge else out)
        return result
    return memo(group+"-text",texts,calculate)


def encode_images(paths):
    def calculate():
        out=[]
        model=session("clip","vision_model_quantized.onnx")
        for begin in range(0,len(paths),8):
            batch=[]
            for name in paths[begin:begin+8]:
                with Image.open(DATA/"assets"/name) as im:
                    im=im.convert("RGB")
                    scale=224/min(im.size)
                    im=im.resize((max(224,round(im.width*scale)),max(224,round(im.height*scale))),Image.Resampling.BICUBIC)
                    x=(im.width-224)//2;y=(im.height-224)//2
                    arr=np.asarray(im.crop((x,y,x+224,y+224)),dtype=np.float32)/255
                    batch.append(((arr-CLIP_MEAN)/CLIP_STD).transpose(2,0,1))
            out.extend(normalize(model.run(None,{"pixel_values":np.stack(batch)})[0]))
        return np.asarray(out,dtype=np.float32).reshape((-1,512))
    return memo("clip-vision",paths,calculate)


def bm25(docs,queries):
    cv=CountVectorizer(token_pattern=r"(?u)\b\w{2,}\b",stop_words="english")
    matrix=cv.fit_transform(docs).astype(np.float32).tocsr()
    length=np.asarray(matrix.sum(1)).ravel()
    df=np.asarray((matrix>0).sum(0)).ravel()
    rowids=np.repeat(np.arange(len(docs)),np.diff(matrix.indptr))
    idf=np.log1p((len(docs)-df+.5)/(df+.5))
    matrix.data=matrix.data*2.2/(matrix.data+1.2*(.25+.75*length[rowids]/length.mean()))*idf[matrix.indices]
    q=cv.transform(queries).astype(np.float32);q.data[:]=1
    return (q@matrix.T).toarray()


def corpus_inputs(name):
    corpus=json.loads(gzip.decompress((DATA/f"{name}-corpus.json.gz").read_bytes()))
    rows=read_questions(name)
    pages=sorted(corpus["pages"],key=lambda p:p["slug"])
    docs=[]; offsets=[]; doc_assets=[]
    for p in pages:
        offsets.append(len(docs))
        for c in p["chunks"]:
            docs.append(" ".join(dict.fromkeys([p["title"],c["section"]]))+". "+c["text"])
            doc_assets.append(c["assets"])
    lookup=corpus["available_assets"]
    q_assets=[[u for u in r["images"][:2] if u in lookup] for r in rows]
    queries=[re.sub(r"https?://\S+"," ",r["title"]+"\n"+r["body_text"]+"\n"+r["code"]) for r in rows]
    return rows, pages, docs, offsets, doc_assets, lookup, q_assets, queries


def evaluate(name):
    rows,pages,docs,offsets,doc_assets,lookup,q_assets,queries=corpus_inputs(name)
    page_ids=[p["slug"] for p in pages]
    assets=sorted(set(lookup[u] for seq in q_assets+doc_assets for u in seq))
    print(name,len(docs),"chunks,",len(assets),"distinct images",flush=True)
    ranks={}; first_chunks={}
    def add(method,scores):
        page_scores=np.maximum.reduceat(scores,offsets,axis=1)
        order=np.argsort(-page_scores,axis=1,kind="stable")
        first_chunks[method]=[docs[int(i)] for i in np.argmax(scores,axis=1)]
        ranks[method]=[[page_ids[i] for i in line[:10]] for line in order]
    add("BM25",bm25(docs,queries))
    b=encode_text("bge",docs)
    bq=encode_text("bge",["Represent this sentence for searching relevant passages: "+q for q in queries])
    add("BGE",bq@b.T)
    print(name,"text baselines complete",flush=True)
    v=encode_images(assets); vectors=dict(zip(assets,v))
    qt=encode_text("clip",queries);dt=encode_text("clip",docs)
    def fused(base,items):
        out=base.copy()
        for i,seq in enumerate(items):
            if seq:
                visual=normalize(np.mean([vectors[lookup[u]] for u in seq],axis=0))
                out[i]=normalize(base[i]+visual)
        return out
    qm=fused(qt,q_assets);dm=fused(dt,doc_assets)
    for method,left,right in [("CLIP A",qt,dt),("CLIP B",qm,dt),("CLIP C",qt,dm),("CLIP D",qm,dm)]:
        add(method,left@right.T)
    have=set(page_ids)
    visible={p["slug"] for p in pages if any(c["assets"] for c in p["chunks"])}
    details=[]
    for i,r in enumerate(rows):
        targets=set(r["targets"])
        # Keep complete qrels as the denominator. Report target availability explicitly.
        record={"domain":name,"id":r["id"],"group":r["group"],"targets":sorted(targets),
                "any_gold_loaded":bool(targets&have),"all_gold_loaded":targets<=have,
                "query_image":bool(q_assets[i]),"target_image":bool(targets&visible),"methods":{}}
        for method,lists in ranks.items():
            top=lists[i];matches=[rank for rank,slug in enumerate(top,1) if slug in targets]
            record["methods"][method]={"top10":top,"top1_chunk":first_chunks[method][i],"hit5":int(any(rank<=5 for rank in matches)),
                        "rr10":1/min(matches) if matches else 0,"recall5":len(targets&set(top[:5]))/len(targets)}
        details.append(record)
    dest=DATA/f"{name}-rankings.json"
    dest.write_text(json.dumps(details,ensure_ascii=False))
    print(name,"done",flush=True)
    return details


if __name__=="__main__":
    import sys
    for domain in sys.argv[1:]: evaluate(domain)
