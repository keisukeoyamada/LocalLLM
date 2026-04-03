# import torch
import time
from llama_cpp import Llama

# llm = Llama(model_path="./models/mmnga-elyza-8b/Llama-3-ELYZA-JP-8B-Q5_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)
# model = AutoModelForCausalLM.from_pretrained(model_path)
llm = Llama(model_path="./models/gemma-3-27b-gguf/google_gemma-3-27b-it-Q5_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)

SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。常に日本語で答えてください。"

while True:
    user_input = input(">>> ")
    full_prompt = f"{SYSTEM_PROMPT}\n\nユーザー: {user_input}\nアシスタント:"
    start = time.time()
    response = llm(
        full_prompt,
        max_tokens=2000,
        temperature=0.7,
    )
    elapsed = time.time() - start
    tokens = response["usage"]["completion_tokens"]
    print(response["choices"][0]["text"])
    print(f"\n[{tokens} tokens, {tokens / elapsed:.1f} tok/s]")
