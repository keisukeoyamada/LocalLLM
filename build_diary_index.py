import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # ローカル配置名に置き換え
texts, meta = [], []

import re


def split_chunks(text, max_chars=280, min_chars=60):
    parts = []
    for seg in re.split(r"[。\n!?！？]+", text):
        seg = seg.strip()
        if seg:
            parts.append(seg + "。")
    chunks, buf = [], ""
    for seg in parts:
        if len(buf) + len(seg) <= max_chars:
            buf += seg
        else:
            if len(buf) >= min_chars:
                chunks.append(buf.strip())
            buf = seg
    if len(buf) >= min_chars:
        chunks.append(buf.strip())
    return chunks


for p in sorted(Path("data/diaries").glob("*")):
    if p.suffix.lower() not in {".txt", ".md"}:
        continue
    date = p.stem
    content = p.read_text(encoding="utf-8")
    print(p.name)
    # 簡易チャンク: 文を適当に分割（本番は句点区切り＋長さ制御が望ましい）
    for i, chunk in enumerate(split_chunks(content)):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 見出しっぽい短文を除外（記号や空白を削って10文字未満ならスキップ）
        if len(chunk.strip("#*- 　")) < 3:
            continue
        texts.append(chunk)
        meta.append({"date": date, "source": p.name, "chunk_id": i})

embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(np.array(embs, dtype="float32"))
faiss.write_index(index, "data/diary.index")
Path("data/diary_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
