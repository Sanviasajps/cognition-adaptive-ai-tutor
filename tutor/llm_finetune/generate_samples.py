from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TASK_TYPES = [
    "explanation",
    "flashcard",
    "mcq",
    "debug_task",
    "output_prediction",
    "transfer_question",
    "challenge_question",
    "hint",
    "feedback",
    "revision_note",
]

CONCEPTS = [
    "Python Variables",
    "Python Loops",
    "SQL SELECT",
    "HTML Tags",
    "Git Commits",
]


def build_sample(task_type, concept):

    if task_type == "explanation":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": f"{concept} is an important programming concept used in software development."
        }

    elif task_type == "flashcard":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "front": f"What is {concept}?",
                "back": f"{concept} is a programming concept."
            }
        }

    elif task_type == "mcq":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "question": f"What is {concept} mainly used for?",
                "options": [
                    "Programming",
                    "Cooking",
                    "Drawing",
                    "Gaming"
                ],
                "answer": "Programming",
                "explanation": f"{concept} is related to programming."
            }
        }

    elif task_type == "debug_task":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "buggy_code": "for i in range(5)\n    print(i)",
                "expected_fix": "for i in range(5):\n    print(i)",
                "hint": "Missing colon after range(5)"
            }
        }

    elif task_type == "output_prediction":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "code": "x = 5\nprint(x)",
                "answer": "5"
            }
        }

    elif task_type == "transfer_question":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "question": f"How is {concept} used in real projects?",
                "answer": f"{concept} is commonly used in software systems."
            }
        }

    elif task_type == "challenge_question":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": {
                "challenge": f"Explain {concept} with a real-world example.",
                "solution_outline": "Use a simple practical scenario."
            }
        }

    elif task_type == "hint":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": f"Think carefully about how {concept} works."
        }

    elif task_type == "feedback":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": "Good attempt. Review the syntax carefully."
        }

    elif task_type == "revision_note":
        return {
            "task_type": task_type,
            "concept": concept,
            "output": f"Remember the key rules of {concept}."
        }

    return {
        "task_type": task_type,
        "concept": concept,
        "output": "Sample output"
    }


def generate_samples(output_file):

    samples = []

    for task in TASK_TYPES:

        for concept in CONCEPTS:

            sample = build_sample(task, concept)

            samples.append(sample)

    with open(output_file, "w", encoding="utf-8") as f:

        for sample in samples:

            f.write(json.dumps(sample) + "\n")

    return len(samples)


def main():

    smol_path = OUTPUT_DIR / "smollm2_135m_samples.jsonl"

    qwen_path = OUTPUT_DIR / "qwen_coder_05b_samples.jsonl"

    smol_count = generate_samples(smol_path)

    qwen_count = generate_samples(qwen_path)

    print("\nSample generation completed.\n")

    print(f"SmolLM2 samples: {smol_count}")
    print(f"Qwen samples: {qwen_count}")

    print("\nSaved files:")

    print(smol_path)
    print(qwen_path)


if __name__ == "__main__":
    main()