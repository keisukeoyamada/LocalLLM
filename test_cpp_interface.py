# import torch
from llama_cpp import Llama

llm = Llama(model_path="./models/mmnga-elyza-8b/Llama-3-ELYZA-JP-8B-Q5_K_M.gguf", n_ctx=4096, n_gpu_layers=-1)
# model = AutoModelForCausalLM.from_pretrained(model_path)

SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。常に日本語で答えてください。回答する前は必ず「アシスタント:」と記載してください。"

while True:
    user_input = input(">>> ")
    full_prompt = f"{SYSTEM_PROMPT}\n\nユーザー: {user_input}\nアシスタント:"
    resp = llm(full_prompt, max_tokens=500, temperature=0.7)
    print(resp["choices"][0]["text"])
