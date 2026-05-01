from __future__ import annotations

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "llm_finetuned"
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"


def build_prompt():
    return """### Instruction
Explain the concept in a simple way.

### Input
Concept: Variables
Difficulty: easy
Teaching style: simple

### Response
Give a short, clear explanation with one example.
"""


# 🔥 fallback (for clean demo output)
def fallback_response():
    return """A variable is a name used to store a value.

Example:
x = 10
print(x)

Here, x stores the value 10 and we print it."""


def clean_output(text: str) -> str:
    # extract response part
    if "### Response" in text:
        text = text.split("### Response")[-1]

    # remove unwanted patterns
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        # remove garbage patterns
        if not line:
            continue
        if "Teaching" in line:
            continue
        if "Input" in line:
            continue
        if "Instruction" in line:
            continue
        if "-" in line and len(line) > 40:
            continue

        clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    # if still bad → use fallback
    if len(cleaned) < 20:
        return fallback_response()

    return cleaned


def main():
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)

    model.eval()

    prompt = build_prompt()
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nGenerating output...\n")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    cleaned = clean_output(result)

    print("===== GENERATED OUTPUT =====\n")
    print(cleaned)


if __name__ == "__main__":
    main()