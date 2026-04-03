# test_mlx_interface.py
# Simple MLX REPL without RAG. Apple Silicon (M4) only.
# For other platforms, use test_cpp_interface.py (llama_cpp).

from mlx_lm import generate, load
from mlx_lm.generate import make_sampler

MLX_MODEL_PATH = "./models/gemma-3-27b-mlx"
print(f"Loading MLX model: {MLX_MODEL_PATH}")
model, tokenizer = load(MLX_MODEL_PATH)

SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。常に日本語で答えてください。"

while True:
    user_input = input(">>> ")
    full_prompt = f"{SYSTEM_PROMPT}\n\nユーザー: {user_input}\nアシスタント:"
    response = generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=2000,
        sampler=make_sampler(temp=0.7),
        verbose=True,
    )
