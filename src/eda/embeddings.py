"""Pinned local model, tokenizer-aware windows and text-bound embedding cache."""
import hashlib
import json
import os

import numpy as np

from .loaders import CONFIG, ROOT, dump


def download_model():
    from huggingface_hub import snapshot_download
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    snapshot_download(CONFIG['model'], revision=manifest['model_revision'], local_dir=ROOT / 'models/minilm',
                      allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'],
                      ignore_patterns=['onnx/*', 'openvino/*'], max_workers=4)


def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(ROOT / 'models/minilm'), device=CONFIG['device'], local_files_only=True)
    return model


def chunks(documents, tokenizer):
    result = []
    width, overlap = CONFIG['chunk_tokens'], CONFIG['overlap_tokens']
    for d in documents:
        enc = tokenizer(d['content'], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = enc['offset_mapping']
        for start in range(0, len(offsets), width-overlap):
            stop = min(start+width, len(offsets))
            left, right = offsets[start][0], offsets[stop-1][1]
            result.append(dict(document_id=d['document_id'], chunk_id=f'{d["document_id"]}:{start}:{stop}',
                               source_type=d['source_type'], token_start=start, token_end=stop,
                               char_start=left, char_end=right, content=d['content'][left:right]))
            if stop == len(offsets):
                break
    return result


def encode(model, texts, label):
    revision = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))['model_revision']
    spec = dict(texts=texts, model=CONFIG['model'], revision=revision, preprocessing='conservative-v1',
                normalized=True, max_seq_length=model.max_seq_length, pooling='model-config mean pooling')
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    path = ROOT / f'data/cache/{label}-{key}.npy'
    if path.exists():
        return np.load(path, allow_pickle=False)
    # Fail rather than silently truncate any unit.
    lengths = [len(model.tokenizer(t, add_special_tokens=True, truncation=False)['input_ids']) for t in texts]
    if max(lengths, default=0) > model.max_seq_length:
        raise ValueError(f'{label}: token window exceeds model limit: {max(lengths)}')
    vectors = model.encode(texts, batch_size=CONFIG['batch_size'], normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    dump(path.with_suffix('.json'), dict(spec_hash=key, revision=revision, model=CONFIG['model'],
         count=len(texts), dimensions=vectors.shape[1], device=CONFIG['device'], batch_size=CONFIG['batch_size'],
         max_seq_length=model.max_seq_length, pooling=str(model[1]), normalization=True))
    return vectors


if __name__ == '__main__':
    download_model()
