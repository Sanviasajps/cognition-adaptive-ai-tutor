from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAMPLE_PATH = os.getenv(
    "TUTOR_SAMPLE_PATH",
    str(ROOT / "outputs" / "samples" / "smollm2_135m_samples.jsonl")
)

METRICS_PATH = os.getenv(
    "TUTOR_METRICS_PATH",
    str(ROOT / "outputs" / "metrics" / "smollm2_135m_metrics.json")
)

SAMPLE_PATH = Path(SAMPLE_PATH)
METRICS_PATH = Path(METRICS_PATH)

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


BAD_PATTERNS = [
    "3-question",
    "5-question",
    "coverage metadata",
    "PUT_HREF",
    "Hook",
    "Diff",
    "Task_instance",
    "base_url",
    "http://",
    "https://",
]


def has_repetition(text: str) -> bool:

    words = text.lower().split()

    if len(words) < 8:
        return False

    for i in range(len(words) - 2):

        phrase = " ".join(words[i:i + 3])

        if text.lower().count(phrase) >= 3:
            return True

    return False


def validate_flashcard(output):

    return (
        isinstance(output, dict)
        and "front" in output
        and "back" in output
    )


def validate_mcq(output):

    return (
        isinstance(output, dict)
        and "question" in output
        and "options" in output
        and isinstance(output["options"], list)
        and len(output["options"]) == 4
        and "answer" in output
        and "explanation" in output
    )


def validate_debug(output):

    return (
        isinstance(output, dict)
        and "buggy_code" in output
        and "expected_fix" in output
        and "hint" in output
    )


def validate_mindmap(output):

    return (
        isinstance(output, dict)
        and "center" in output
        and "branches" in output
        and isinstance(output["branches"], list)
    )


def main():

    if not SAMPLE_PATH.exists():

        print(f"Sample file missing: {SAMPLE_PATH}")
        return

    samples = []

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                samples.append(json.loads(line))

            except Exception:
                continue

    total = len(samples)

    if total == 0:

        print("No samples found.")
        return

    repetition_count = 0
    bad_pattern_count = 0
    invalid_output_count = 0
    task_success_count = 0
    concept_match_count = 0

    flashcard_valid = 0
    mcq_valid = 0
    debug_valid = 0
    mindmap_valid = 0

    lengths = []

    for sample in samples:

        task = str(sample.get("task_type", "")).strip()

        concept = str(sample.get("concept", "")).strip()

        output = sample.get("output", "")

        if isinstance(output, dict):
            output_text = json.dumps(output)
        else:
            output_text = str(output)

        lower = output_text.lower()

        lengths.append(len(output_text.split()))

        # repetition detection
        if has_repetition(output_text):
            repetition_count += 1

        # artifact detection
        for bad in BAD_PATTERNS:

            if bad.lower() in lower:
                bad_pattern_count += 1
                break

        # concept relevance
        if concept:

            concept_tokens = concept.lower().split()

            if concept_tokens:

                if concept_tokens[0] in lower:
                    concept_match_count += 1

        # structured validation
        valid_task = True

        if task == "flashcard":

            if validate_flashcard(output):
                flashcard_valid += 1
            else:
                valid_task = False

        elif task == "mcq":

            if validate_mcq(output):
                mcq_valid += 1
            else:
                valid_task = False

        elif task == "debug_task":

            if validate_debug(output):
                debug_valid += 1
            else:
                valid_task = False

        elif task == "mindmap":

            if validate_mindmap(output):
                mindmap_valid += 1
            else:
                valid_task = False

        if valid_task:
            task_success_count += 1
        else:
            invalid_output_count += 1

    metrics = {

        "total_samples": total,

        "format_validity":
            round((total - invalid_output_count) / total, 2),

        "task_success_rate":
            round(task_success_count / total, 2),

        "repetition_rate":
            round(repetition_count / total, 2),

        "avg_length":
            round(sum(lengths) / total, 2),

        "invalid_output_count":
            invalid_output_count,

        "bad_pattern_count":
            bad_pattern_count,

        "json_validity":
            round((total - invalid_output_count) / total, 2),

        "mcq_validity":
            mcq_valid,

        "debug_validity":
            debug_valid,

        "flashcard_validity":
            flashcard_valid,

        "mindmap_validity":
            mindmap_valid,

        "concept_relevance_proxy":
            round(concept_match_count / total, 2),

        "manual_quality_placeholder":
            "human review required"
    }

    print("\nEvaluation Result:\n")

    print(json.dumps(metrics, indent=2))

    with open(METRICS_PATH, "w", encoding="utf-8") as f:

        json.dump(metrics, f, indent=2)

    print(f"\nSaved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()