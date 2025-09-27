import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# モデル指定（4bit量子化モデルを使うなら別途対応）
model_name = "./mistral-1b"

# データセット読み込み
dataset = load_dataset("json", data_files="./data/train.jsonl", split="train")

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# LoRAの設定
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)  # LLaMA系のLoRA対象

# LoRAラップ
model = get_peft_model(model, lora_config)


# データ整形（シンプルな例）
def tokenize(entry):
    prompt = f"{entry['instruction']}\n{entry['input']}\n"
    target = entry["output"]
    full = prompt + target
    tokenized = tokenizer(full, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# トレーニング設定
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,  # Metalで動かす場合はfp16非対応のためFalseにする
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer)

# 学習開始
trainer.train()
