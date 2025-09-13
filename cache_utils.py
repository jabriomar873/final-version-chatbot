import os, json, hashlib, time
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from config import CACHING_CONFIG, TEXT_CONFIG


def _hash_docs(docs: List[Document], extra: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    for d in docs:
        meta = d.metadata or {}
        m.update(d.page_content.encode('utf-8', 'ignore'))
        for k in sorted(meta.keys()):
            m.update(str(k).encode()); m.update(str(meta[k]).encode())
    for k in sorted(extra.keys()):
        m.update(str(k).encode()); m.update(str(extra[k]).encode())
    return m.hexdigest()[:32]


def cache_key(documents: List[Document]) -> str:
    params = {}
    if CACHING_CONFIG.get('respect_chunk_params'):
        params['chunk_size'] = TEXT_CONFIG.get('chunk_size')
        params['chunk_overlap'] = TEXT_CONFIG.get('chunk_overlap')
    return _hash_docs(documents, params)


def load_cache(key: str):
    base = CACHING_CONFIG['path']
    meta_path = os.path.join(base, key + '.json')
    vec_path = os.path.join(base, key + CACHING_CONFIG.get('ext', '.npz'))
    if not (os.path.exists(meta_path) and os.path.exists(vec_path)):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        data = np.load(vec_path)
        vectors = data['vectors']
        return {'meta': meta, 'vectors': vectors}
    except Exception:
        return None


def save_cache(key: str, documents: List[Document], vectors: List[List[float]]):
    base = CACHING_CONFIG['path']
    os.makedirs(base, exist_ok=True)
    meta_path = os.path.join(base, key + '.json')
    vec_path = os.path.join(base, key + CACHING_CONFIG.get('ext', '.npz'))
    payload = [{'content': d.page_content, 'metadata': d.metadata} for d in documents]
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'documents': payload, 'ts': time.time()}, f)
        np.savez_compressed(vec_path, vectors=np.array(vectors, dtype='float32'))
    except Exception:
        pass
