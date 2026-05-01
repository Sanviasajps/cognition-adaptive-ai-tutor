from __future__ import annotations

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "training_data" / "llm_tutor"
OUTPUT_DIR = ROOT / "models" / "llm_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_data(example):
    return {
        "text": f"""### Instruction
{example['instruction']}

### Input
{example['input']}

### Response
{example['output']}"""
    }


def tokenize(example, tokenizer):
    return tokenizer(example["text"], truncation=True, max_length=256)


def main():
    print("Loading dataset (FAST MODE)...")

    # 🔥 Reduced dataset for fast training
    train_data = load_jsonl(DATA_DIR / "tutor_train.jsonl")[:2000]
    val_data = load_jsonl(DATA_DIR / "tutor_val.jsonl")[:200]

    dataset = {
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    }

    dataset["train"] = dataset["train"].map(format_data)
    dataset["validation"] = dataset["validation"].map(format_data)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Tokenizing...")
    tokenized = {
        "train": dataset["train"].map(lambda x: tokenize(x, tokenizer)),
        "validation": dataset["validation"].map(lambda x: tokenize(x, tokenizer)),
    }

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,  # 🔥 fast
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print("Starting FAST training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete! 🚀")


if __name__ == "__main__":
    main()