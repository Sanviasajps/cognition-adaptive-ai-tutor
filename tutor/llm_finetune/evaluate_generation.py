from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "metrics"
    / "llm_evaluation.json"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =====================================================
# SAMPLE GENERATED OUTPUTS
# =====================================================

samples = [
    {
        "task": "flashcard",
        "concept": "Git Commits",
        "output": """Front: What is Git Commit?

Back: Git commit saves changes in a repository."""
    },

    {
        "task": "debug_task",
        "concept": "Python Loops",
        "output": """Buggy code:
for i in range(5)
    print(i)

Expected fix:
Add colon after range(5)."""
    },

    {
        "task": "challenge_question",
        "concept": "Stack",
        "output": """Challenge:
Explain stack with a real-world example.

Solution outline:
A stack works like a pile of books."""
    },

    {
        "task": "explanation",
        "concept": "Variables",
        "output": """Variables store values in programming.

Example:
x = 10"""
    },

    {
        "task": "flashcard",
        "concept": "HTML",
        "output": """3-question — coverage metadata"""
    },
]


# =====================================================
# BAD ARTIFACTS
# =====================================================

BAD_PATTERNS = [
    "3-question",
    "5-question",
    "coverage metadata",
    "PUT_HREF",
    "Hook",
    "Diff",
]


# =====================================================
# REPETITION CHECK
# =====================================================

def has_repetition(text: str):

    words = text.lower().split()

    if len(words) < 8:
        return False

    for i in range(len(words) - 2):

        phrase = " ".join(words[i:i+3])

        if text.lower().count(phrase) >= 3:
            return True

    return False


# =====================================================
# TASK FORMAT VALIDATION
# =====================================================

def check_task_success(task, output):

    lower = output.lower()

    if task == "flashcard":
        return (
            "front:" in lower
            and "back:" in lower
        )

    elif task == "debug_task":
        return (
            "buggy code:" in lower
            and "expected fix:" in lower
        )

    elif task == "challenge_question":
        return (
            "challenge:" in lower
            and "solution outline:" in lower
        )

    return True


# =====================================================
# MAIN EVALUATION
# =====================================================

def main():

    total = len(samples)

    valid_count = 0
    repetition_count = 0
    artifact_count = 0
    task_success_count = 0
    concept_match_count = 0

    lengths = []

    for sample in samples:

        task = sample["task"]
        concept = sample["concept"]
        output = sample["output"]

        lower = output.lower()

        lengths.append(len(output.split()))

        # repetition
        if has_repetition(output):
            repetition_count += 1

        # artifact detection
        found_artifact = False

        for bad in BAD_PATTERNS:

            if bad.lower() in lower:
                artifact_count += 1
                found_artifact = True
                break

        # task format
        if check_task_success(task, output):
            task_success_count += 1

        # concept relevance
        if concept.lower().split()[0] in lower:
            concept_match_count += 1

        # overall validity
        if not found_artifact:
            valid_count += 1

    metrics = {

        "total_samples": total,

        "format_valid_percent":
            round(valid_count / total, 2),

        "task_success_rate":
            round(task_success_count / total, 2),

        "repetition_rate":
            round(repetition_count / total, 2),

        "artifact_rate":
            round(artifact_count / total, 2),

        "concept_relevance":
            round(concept_match_count / total, 2),

        "avg_length":
            round(sum(lengths) / total, 2),
    }

    print("\nEvaluation Result:\n")

    print(json.dumps(metrics, indent=2))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()