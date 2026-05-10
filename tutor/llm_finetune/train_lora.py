from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "training_data" / "llm_tutor"

MODEL_NAME = os.getenv(
    "TUTOR_BASE_MODEL",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
)

OUTPUT_DIR = Path(

    os.getenv(
        "TUTOR_OUTPUT_DIR",

        str(
            ROOT
            / "models"
            / "llm_finetuned"
            / "qwen_coder_05b_lora"
        )
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_LENGTH = int(
    os.getenv("MAX_LENGTH", "256")
)

TRAIN_EPOCHS = int(
    os.getenv("TRAIN_EPOCHS", "1")
)

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "1")
)

GRAD_ACCUM = int(
    os.getenv("GRAD_ACCUM", "8")
)

LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", "1e-4")
)

FAST_MODE = os.getenv(
    "FAST_MODE",
    "0"
)


def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            json.loads(line)
            for line in f
        ]


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
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def main():

    print("\nLoading dataset...\n")

    train_data = load_jsonl(
        DATA_DIR / "tutor_train.jsonl"
    )

    val_data = load_jsonl(
        DATA_DIR / "tutor_val.jsonl"
    )

    # ==============================================
    # OPTIONAL FAST MODE
    # ==============================================

    if FAST_MODE == "1":

        train_data = train_data[:2000]

        val_data = val_data[:200]

        print("FAST MODE ENABLED")

    print(f"Train rows: {len(train_data)}")

    print(f"Validation rows: {len(val_data)}")

    dataset = {

        "train":
            Dataset.from_list(train_data),

        "validation":
            Dataset.from_list(val_data),
    }

    dataset["train"] = dataset[
        "train"
    ].map(format_data)

    dataset["validation"] = dataset[
        "validation"
    ].map(format_data)

    print("\nLoading tokenizer...\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print("Tokenizer loaded.")

    print("\nLoading base model...\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    print("Model loaded.")

    # ==============================================
    # APPLY LORA
    # ==============================================

    print("\nApplying LoRA...\n")

    lora_config = LoraConfig(

        task_type=TaskType.CAUSAL_LM,

        r=8,

        lora_alpha=16,

        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    # ==============================================
    # TOKENIZE
    # ==============================================

    print("\nTokenizing dataset...\n")

    tokenized = {

        "train":
            dataset["train"].map(
                lambda x: tokenize(
                    x,
                    tokenizer
                )
            ),

        "validation":
            dataset["validation"].map(
                lambda x: tokenize(
                    x,
                    tokenizer
                )
            ),
    }

    data_collator = (
        DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    # ==============================================
    # TRAINING ARGS
    # ==============================================

    training_args = TrainingArguments(

        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=
            BATCH_SIZE,

        per_device_eval_batch_size=
            BATCH_SIZE,

        gradient_accumulation_steps=
            GRAD_ACCUM,

        num_train_epochs=
            TRAIN_EPOCHS,

        logging_steps=10,

        eval_strategy="steps",

        eval_steps=50,

        save_steps=50,

        save_total_limit=1,

        learning_rate=
            LEARNING_RATE,

        fp16=torch.cuda.is_available(),

        report_to="none",

        logging_dir=str(
            OUTPUT_DIR / "logs"
        ),
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=
            tokenized["train"],

        eval_dataset=
            tokenized["validation"],

        data_collator=
            data_collator,
    )

    # ==============================================
    # TRAIN
    # ==============================================

    print("\nStarting training...\n")

    trainer.train()

    # ==============================================
    # EVALUATE
    # ==============================================

    print("\nRunning evaluation...\n")

    eval_results = trainer.evaluate()

    print(eval_results)

    # ==============================================
    # SAVE
    # ==============================================

    print("\nSaving model...\n")

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("\nTraining complete!\n")


if __name__ == "__main__":

    main()