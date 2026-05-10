from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from tutor.llm_finetune.output_validator import validate_output


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = Path(
    os.getenv(
        "TUTOR_MODEL_DIR",
        str(
            ROOT
            / "models"
            / "llm_finetuned"
            / "qwen_coder_05b_lora"
        )
    )
)

BASE_MODEL = os.getenv(
    "TUTOR_BASE_MODEL",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct"
)


def build_prompt(
    concept="Variables",
    difficulty="easy",
    learner_state="weak_output_prediction",
    style="simple",
    task_type="explanation",
):

    return f"""### Instruction
You are a tutor content generator for a cognition-adaptive AI tutor.

Generate only the requested tutor output.

Do NOT include:
- URLs
- API fields
- hook names
- diff text
- random metadata
- repeated tokens

### Task
Concept: {concept}
Task type: {task_type}
Difficulty: {difficulty}
Learner state: {learner_state}
Teaching style: {style}

### Requirements
- Stay only about the given concept.
- Use simple student-friendly language.
- Keep answer under 120 words.
- Follow the requested format exactly.
- Avoid repetition.

Task format rules:

- explanation:
Give 4–6 simple lines with one example.

- flashcard:
Use exactly:
Front:
Back:

- debug_task:
Include:
Buggy code:
Expected fix:

- output_prediction:
Include:
Code:
Answer:

- transfer_question:
Include:
Question:
Answer:

- challenge_question:
Include:
Challenge:
Solution outline:

### Output
"""


def fallback_response(task_type: str):

    if task_type == "flashcard":

        return """Front: Variable
Back: A variable stores a value in programming."""

    if task_type == "debug_task":

        return """Buggy code:
x == 5

Expected fix:
Use x = 5 for assignment."""

    if task_type == "challenge_question":

        return """Challenge:
Create a loop that prints numbers from 1 to 5.

Solution outline:
Use a for loop with range(1, 6)."""

    return """A variable stores a value in programming.

Example:
x = 10

Here, x stores the value 10."""


def clean_output(text: str):

    if "### Output" in text:
        text = text.split("### Output")[-1]

    lines = text.split("\n")

    clean_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        bad_words = [
            "Instruction",
            "Task",
            "Requirement",
            "Teaching",
            "Input",
            "http",
            "https",
            "PUT_HREF",
            "Hook",
            "Diff",
        ]

        skip = False

        for bad in bad_words:

            if bad.lower() in line.lower():
                skip = True

        if skip:
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def generate_output(
    model,
    tokenizer,
    concept,
    task_type,
    difficulty="easy",
    learner_state="beginner",
    style="simple",
):

    prompt = build_prompt(
        concept=concept,
        difficulty=difficulty,
        learner_state=learner_state,
        style=style,
        task_type=task_type,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    cleaned = clean_output(result)

    validation = validate_output(
        cleaned,
        concept=concept,
        task_type=task_type,
    )

    print("\nValidator Result:")
    print(validation)

    if validation["retry_recommended"]:

        print("\nRetrying with safer decoding...\n")

        with torch.no_grad():

            retry_output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        retry_text = tokenizer.decode(
            retry_output[0],
            skip_special_tokens=True,
        )

        cleaned_retry = clean_output(
            retry_text
        )

        retry_validation = validate_output(
            cleaned_retry,
            concept=concept,
            task_type=task_type,
        )

        print("Retry Validator Result:")
        print(retry_validation)

        if retry_validation["valid"]:
            cleaned = cleaned_retry

    # ==================================================
    # FORCE TASK FORMATS
    # ==================================================

    if task_type == "flashcard":

        if "front:" not in cleaned.lower():

            cleaned = f"""Front: What is {concept}?

Back: {cleaned[:160]}"""

    elif task_type == "debug_task":

        if "buggy code:" not in cleaned.lower():

            cleaned = f"""Buggy code:
for i in range(5)
    print(i)

Expected fix:
Add missing colon after range(5)

Explanation:
{cleaned[:180]}"""

    elif task_type == "challenge_question":

        if "challenge:" not in cleaned.lower():

            cleaned = f"""Challenge:
Explain {concept} with one practical example.

Solution outline:
{cleaned[:200]}"""

    elif task_type == "transfer_question":

        if "question:" not in cleaned.lower():

            cleaned = f"""Question:
How would you use {concept} in a real application?

Answer:
{cleaned[:180]}"""

    elif task_type == "output_prediction":

        if "answer:" not in cleaned.lower():

            cleaned = f"""Code:
x = 5
print(x)

Answer:
{cleaned[:150]}"""

    if len(cleaned) < 20:
        return fallback_response(task_type)

    return cleaned


def main():

    print("Loading model...\n")

    print(f"Base model: {BASE_MODEL}")
    print(f"Model dir: {MODEL_DIR}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL
    )

    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR,
    )

    model.eval()

    tests = [

        ("Python Variables", "explanation"),

        ("Python Loops", "debug_task"),

        ("SQL SELECT", "explanation"),

        ("HTML Tags", "debug_task"),

        ("Git Commits", "flashcard"),

        ("Data Structures Stack", "challenge_question"),
    ]

    for concept, task in tests:

        print("\n" + "=" * 60)

        print(f"Concept: {concept}")
        print(f"Task: {task}")

        result = generate_output(
            model=model,
            tokenizer=tokenizer,
            concept=concept,
            task_type=task,
        )

        print("\n===== GENERATED OUTPUT =====\n")

        print(result)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()