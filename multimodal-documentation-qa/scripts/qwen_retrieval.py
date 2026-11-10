"""Local exact dense retrieval with Qwen3-VL-Embedding-2B. No remote inference."""
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from retrieval import CACHE, DATA, ROOT, corpus_inputs

MODEL = ROOT / "models" / "Qwen3-VL-Embedding-2B"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
WEIGHT_HASH = "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query."
DOC_INSTRUCTION = "Represent the user's input."
TEXT_LIMIT = 512
DIM = 2048
MODES = {"A": ("qt", "dt"), "B": ("qm", "dt"), "C": ("qt", "dm"), "D": ("qm", "dm")}
PROTOCOL = {"model": "Qwen/Qwen3-VL-Embedding-2B", "revision": REVISION, "text_limit": TEXT_LIMIT,
            "query_instruction": QUERY_INSTRUCTION, "document_instruction": DOC_INSTRUCTION,
            "pool": "last non-padding token, L2", "dtype": "float16", "transformers": "4.57.6",
            "processor": "native Qwen3VL; image(s) before text; official chat template; right padding",
            "implementation": 1}
NAMESPACE = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:18]


@lru_cache(maxsize=2000)
def asset_hash(filename):
    return hashlib.sha256((DATA / "assets" / filename).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Input:
    key: str
    role: str
    text: str
    images: tuple

    @classmethod
    def build(cls, role, text, images=()):
        payload = [NAMESPACE, role, text, [asset_hash(f) for f in images]]
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
        return cls(key, role, text, tuple(images))


def inputs_for(name):
    rows, pages, docs, offsets, doc_assets, lookup, q_assets, queries = corpus_inputs(name)
    dt = [Input.build("doc", value) for value in docs]
    qt = [Input.build("query", value) for value in queries]
    def with_images(items, attachments):
        return [Input.build(v.role, v.text, [lookup[u] for u in urls]) if urls else v
                for v, urls in zip(items, attachments, strict=True)]
    inputs = {"dt": dt, "dm": with_images(dt, doc_assets), "qt": qt, "qm": with_images(qt, q_assets)}
    return rows, pages, offsets, inputs


class Encoder:
    def __init__(self, device):
        import torch
        import transformers
        from transformers import Qwen3VLModel, Qwen3VLProcessor
        if transformers.__version__ != PROTOCOL["transformers"]:
            raise RuntimeError("Install the versions in requirements.lock for this cache.")
        with (MODEL / "model.safetensors").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHT_HASH:
                raise ValueError("Unexpected Qwen checkpoint checksum")
        self.device = device
        self.torch = torch
        self.processor = Qwen3VLProcessor.from_pretrained(MODEL, local_files_only=True, padding_side="right")
        self.model, info = Qwen3VLModel.from_pretrained(
            MODEL, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True,
            output_loading_info=True, key_mapping={r"^model\.": ""})
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Checkpoint did not load exactly: {info}")
        self.model.eval().to(device)
        self.trimmed = {}

    def text(self, entry):
        k = entry.text
        if k not in self.trimmed:
            tok = self.processor.tokenizer
            ids = tok(k, add_special_tokens=False, truncation=False)["input_ids"]
            self.trimmed[k] = (len(ids), k if len(ids) <= TEXT_LIMIT else tok.decode(
                ids[:TEXT_LIMIT], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        return self.trimmed[k]

    def tokens(self, entry):
        estimate = min(self.text(entry)[0], TEXT_LIMIT) + 32
        for filename in entry.images:
            with Image.open(DATA / "assets" / filename) as im:
                estimate += max(4, round(im.width / 32) * round(im.height / 32))
        return estimate

    def batch(self, entries):
        torch = self.torch
        prompts, images = [], []
        for entry in entries:
            content = []
            for filename in entry.images:
                with Image.open(DATA / "assets" / filename) as im:
                    images.append(im.convert("RGB"))
                content.append({"type": "image"})
            content.append({"type":"text","text":self.text(entry)[1]})
            instruction = QUERY_INSTRUCTION if entry.role == "query" else DOC_INSTRUCTION
            messages = [{"role":"system","content":[{"type":"text","text":instruction}]},
                        {"role":"user","content":content}]
            prompts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        batch = self.processor(
            text=prompts, images=images or None, padding=True, return_tensors="pt",
        )
        if images and len(batch["image_grid_thw"]) != len(images):
            raise ValueError("Mismatch between image placeholders and images")
        batch = {k: v.to(device=self.device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
                 for k,v in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch, use_cache=False).last_hidden_state
            idx = batch["attention_mask"].long().sum(1) - 1
            final = hidden[torch.arange(len(entries),device=self.device), idx]
            emb = torch.nn.functional.normalize(final.float(), dim=-1).cpu().numpy()
        if emb.shape != (len(entries), DIM) or not np.allclose(np.linalg.norm(emb,axis=1),1,atol=1e-4):
            raise ValueError("Invalid embedding output")
        return emb


class VectorStore:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE / f"qwen-{NAMESPACE}.sqlite", timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def missing(self, entries):
        seen = {x[0] for x in self.db.execute("SELECT key FROM vectors")}
        return [entry for key,entry in entries.items() if key not in seen]

    def save(self, entries, vectors):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO vectors VALUES (?,?)",
                                [(entry.key, v.astype("<f4").tobytes()) for entry,v in zip(entries,vectors,strict=True)])

    def matrix(self, entries):
        result = np.empty((len(entries),DIM),dtype=np.float32)
        for i,entry in enumerate(entries):
            row = self.db.execute("SELECT vector FROM vectors WHERE key=?", (entry.key,)).fetchone()
            if row is None: raise KeyError(entry.key)
            result[i] = np.frombuffer(row[0],dtype="<f4")
        return result


def fill(store, model, pending, token_budget=1800):
    sizes={v.key:model.tokens(v) for v in pending}
    ordered=sorted(pending,key=lambda v:(bool(v.images),len(v.images),sizes[v.key],v.key))
    index=0
    start=time.monotonic()
    while index < len(ordered):
        end=index+1
        while end < min(len(ordered),index+32):
            if len(ordered[end].images)!=len(ordered[index].images) or sizes[ordered[end].key]*(end-index+1)>token_budget:
                break
            end+=1
        entries=ordered[index:end]
        emb=model.batch(entries)
        store.save(entries,emb)
        if index//128!=(end//128) or end==len(ordered) or index==0:
            print(f"embeddings {end}/{len(ordered)}  {time.monotonic()-start:.0f}s",flush=True)
        index=end


def rank(name, store, data):
    rows,pages,offsets,items=data
    slugs=[p["slug"] for p in pages]
    matrices={k:store.matrix(values) for k,values in items.items()}
    results=[{"domain":name,"id":q["id"],"targets":q["targets"],"methods":{}} for q in rows]
    for mode,(left,right) in MODES.items():
        chunk_scores=matrices[left] @ matrices[right].T
        page_scores=np.maximum.reduceat(chunk_scores,offsets,axis=1)
        top=np.argsort(-page_scores,axis=1,kind="stable")[:,:10]
        for i,ids in enumerate(top):
            targets=set(rows[i]["targets"])
            top10=[slugs[j] for j in ids]
            matches=[j+1 for j,s in enumerate(top10) if s in targets]
            results[i]["methods"][mode]={"top10":top10,"hit5":int(any(j<=5 for j in matches)),
                "rr10":1/min(matches) if matches else 0., "recall5":len(set(top10[:5]) & targets)/len(targets)}
    dest = DATA / f"{name}-qwen-rankings.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(results,ensure_ascii=False))
    tmp.replace(dest)
    print(name, {k:sum(r["methods"][k]["hit5"] for r in results) for k in MODES}, flush=True)
    return results


def run(domains=("matplotlib","blender","qgis"), device="mps"):
    store=VectorStore()
    model=None
    result={}
    try:
        for name in domains:
            data=inputs_for(name)
            entries={i.key:i for values in data[-1].values() for i in values}
            pending=store.missing(entries)
            print(name,len(entries),"unique inputs,",len(pending),"uncached",flush=True)
            if pending:
                if model is None: model=Encoder(device)
                fill(store,model,pending)
            result[name]=rank(name,store,data)
    finally:
        store.db.close()
    return result


if __name__=="__main__":
    run()
