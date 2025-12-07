# LocalLLM RAG/微調整メモ

## 構成
- `test_interface.py` : HF Transformers でベースモデルを対話実行するシンプルREPL。
- `test_RUG_interface.py` : HFモデル + SentenceTransformer + FAISS で日記RAG対話。
- `test_RUG_cpp_interface.py` : `llama_cpp` + GGUFモデル + SentenceTransformer + FAISS で日記RAG対話。
- `build_diary_index.py` : `data/diaries` 配下のテキスト/MDをチャンク化→埋め込み→`data/diary.index` と `data/diary_meta.json` を生成。
- `finetune.py` : `./mistral-1b` をLoRAで指示微調整するサンプル。
- モデル: `Llama-3.2-1B`, `mistral-1b` (HF形式)、`models/mmnga-elyza-8b/...gguf` (GGUF量子化)。
- データ: `data/train.jsonl`（instruction/input/output形式）、日記ソース `data/diaries/*`。

## RAG処理フロー
1. `build_diary_index.py` で日記をチャンク化し、SentenceTransformerで埋め込み → FAISSに格納。
2. `test_RUG_interface.py` / `test_RUG_cpp_interface.py` で質問を埋め込み検索 → 上位チャンクをプロンプトに差し込み生成。
3. コンテキスト件数は検索kを広めに取り、上位N件だけプロンプトに入れるとノイズが減る。

### チャンク化のポイント
- 句点・改行・句読点で分割し、`max_chars`/`min_chars` を設定して長すぎ/短すぎを調整。
- 見出しや記号だけの短文は `len(chunk.strip("#*- 　")) < 10` などで除外。
- mdがノイズになる場合は拡張子で絞るか、メタに `kind` を入れて検索時にフィルタ。

## llama_cpp 版 (例)
- モデル初期化: `Llama(model_path="./models/mmnga-elyza-8b/Llama-3-ELYZA-JP-8B-Q5_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)`
- プロンプトはシンプルに: システム文 + コンテキスト + 質問。
- 呼び出し: `resp = llm(full_prompt, max_tokens=200, temperature=0.7); resp["choices"][0]["text"]`

## HF版 (例)
- `AutoTokenizer/AutoModelForCausalLM.from_pretrained(MODEL_PATH)` を同じフォルダからロード。
- `model.generation_config.pad_token_id = tokenizer.eos_token_id` を設定。
- 生成は `do_sample=False` + `max_new_tokens` を短めにすると日本語安定。

## 埋め込みモデル
- 推奨: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（ローカルに置くなら `models/paraphrase-multilingual-MiniLM-L12-v2/` を指定）。
- オフライン運用では事前にモデルフォルダを配置し、環境変数 `HF_HUB_OFFLINE=1` を設定すると無駄なダウンロード待ちを避けられる。

## gitignore の扱い
- `data/` は追跡対象外にする (`data/` を `.gitignore` に記載、追跡済みなら `git rm -r --cached data`）。
- `models/` を残しつつ中身を無視する場合: `models/.gitignore` に
