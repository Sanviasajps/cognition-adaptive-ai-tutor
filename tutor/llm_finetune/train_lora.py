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

OUTPUT_DIR = (
    ROOT
    / "models"
    / "llm_finetuned"
    / "qwen_coder_05b_lora"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


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

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )


def main():

    print("📦 Loading dataset...")

    # safer smaller subset first
    train_data = load_jsonl(
        DATA_DIR / "tutor_train.jsonl"
    )[:2000]

    val_data = load_jsonl(
        DATA_DIR / "tutor_val.jsonl"
    )[:200]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    dataset = {
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    }

    print("🧹 Formatting dataset...")

    dataset["train"] = dataset["train"].map(
        format_data
    )

    dataset["validation"] = dataset["validation"].map(
        format_data
    )

    print("🔤 Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🧠 Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    print("⚙️ Applying LoRA...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    print("🔄 Tokenizing dataset...")

    tokenized = {
        "train": dataset["train"].map(
            lambda x: tokenize(x, tokenizer)
        ),
        "validation": dataset["validation"].map(
            lambda x: tokenize(x, tokenizer)
        ),
    }

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("⚙️ Preparing training arguments...")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        num_train_epochs=1,

        learning_rate=1e-4,

        logging_steps=10,

        evaluation_strategy="steps",
        eval_steps=50,

        save_steps=50,
        save_total_limit=1,

        fp16=torch.cuda.is_available(),

        report_to="none",

        logging_dir=str(OUTPUT_DIR / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print("🚀 Starting Qwen-Coder LoRA training...")

    trainer.train()

    print("📊 Running evaluation...")

    eval_results = trainer.evaluate()

    print("Evaluation Results:")
    print(eval_results)

    print("💾 Saving model...")

    model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Qwen-Coder training complete!")


if __name__ == "__main__":
    main()