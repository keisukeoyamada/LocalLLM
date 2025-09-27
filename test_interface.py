# test_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。常に日本語で答えてください。"

while True:
    prompt = input(">>> ")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, repetition_penalty=1.2)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
