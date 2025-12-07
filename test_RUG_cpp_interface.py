# test_RUG_cpp_interface.py
import json
from pathlib import Path

import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ===== Llama (GGUF) 初期化 =====
LLM_MODEL_PATH = "./models/mmnga-elyza-8b/Llama-3-ELYZA-JP-8B-Q5_K_M.gguf"
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,  # GPUを使わないなら 0
)

# ===== 埋め込み・FAISS =====
LOCAL_EMBED_DIR = Path("models/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_MODEL_ID = str(LOCAL_EMBED_DIR) if LOCAL_EMBED_DIR.exists() else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(EMBED_MODEL_ID)

INDEX_PATH = Path("data/diary.index")
META_PATH = Path("data/diary_meta.json")
index = faiss.read_index(str(INDEX_PATH))
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
print("meta:", len(meta))


def load_diary_chunks(diaries_dir: Path = Path("data/diaries")):
    lookup = {}
    for p in sorted(diaries_dir.glob("*.txt")):
        content = p.read_text(encoding="utf-8")
        for i, chunk in enumerate(content.split("。")):
            chunk = chunk.strip()
            if not chunk:
                continue
            lookup[(p.name, i)] = chunk
    return lookup


CHUNKS = load_diary_chunks()


def retrieve(query: str, k: int = 10, top_n: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, k)
    contexts = []
    for idx in ids[0]:
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        chunk_text = CHUNKS.get((m["source"], m["chunk_id"]), "")
        if not chunk_text.strip():
            continue
        contexts.append(f"[{m['date']}/{m['source']} #{m['chunk_id']}] {chunk_text}")
    return contexts


SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。" "以下のコンテキストから事実に基づき簡潔に答えてください。" "不明な場合は「わかりません」と答えてください。"


def build_prompt(user_input: str, contexts):
    context_block = "\n".join(contexts) if contexts else "（検索結果なし）"
    return f"{SYSTEM_PROMPT}\n\n" f"コンテキスト:\n{context_block}\n\n" f"ユーザー: {user_input}\nアシスタント:"


if __name__ == "__main__":
    while True:
        user_q = input(">>> ")
        ctx = retrieve(user_q, k=20, top_n=5)
        full_prompt = build_prompt(user_q, ctx)
        resp = llm(
            full_prompt,
            max_tokens=200,
            temperature=0.7,  # 必要なら 0.0〜0.3 で保守的に
        )
        print(full_prompt)
        print(resp["choices"][0]["text"])
