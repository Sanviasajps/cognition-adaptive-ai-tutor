from __future__ import annotations

import re
from typing import Dict, List


BAD_PATTERNS = [

    "PUT_HREF",
    "Task_instance",
    "Diff_diff",
    "Diff_message",
    "Hook Hook",
    "HREF_HREF",
    "application/x-www-form-urlencoded",
    "base_url",
    "https://",
    "http://",
]


def has_repetition(text: str) -> bool:

    tokens = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    if len(tokens) < 8:
        return False

    joined_tokens = " ".join(tokens)

    for i in range(len(tokens) - 3):

        chunk = tokens[i:i + 3]

        phrase = " ".join(chunk)

        count = joined_tokens.count(phrase)

        if count >= 4:
            return True

    return False


def validate_output(
    text: str,
    concept: str = "",
    task_type: str = "",
) -> Dict[str, object]:

    issues: List[str] = []

    clean = (text or "").strip()

    lower = clean.lower()

    # ==================================================
    # BASIC CHECKS
    # ==================================================

    if not clean:
        issues.append("empty_output")

    if len(clean.split()) < 5:
        issues.append("too_short")

    # ==================================================
    # BAD PATTERN DETECTION
    # ==================================================

    for bad in BAD_PATTERNS:

        if bad.lower() in lower:

            issues.append(f"bad_pattern:{bad}")

    # ==================================================
    # REPETITION DETECTION
    # ==================================================

    if has_repetition(clean):

        issues.append("repetition")

    # ==================================================
    # CONCEPT RELEVANCE
    # ==================================================

    if concept:

        concept_tokens = concept.lower().split()

        if concept_tokens:

            if concept_tokens[0] not in lower:

                issues.append("concept_name_missing")

    # ==================================================
    # TASK FORMAT VALIDATION
    # ==================================================

    if task_type == "flashcard":

        if (
            "front:" not in lower
            or "back:" not in lower
        ):
            issues.append(
                "flashcard_format_missing"
            )

    elif task_type == "debug_task":

        if (
            "buggy" not in lower
            or "fix" not in lower
        ):
            issues.append(
                "debug_format_missing"
            )

    elif task_type == "output_prediction":

        if (
            "code:" not in lower
            or "answer:" not in lower
        ):
            issues.append(
                "output_prediction_format_missing"
            )

    elif task_type == "transfer_question":

        if (
            "question:" not in lower
            or "answer:" not in lower
        ):
            issues.append(
                "transfer_question_format_missing"
            )

    elif task_type == "challenge_question":

        if (
            "challenge:" not in lower
            or "solution outline:" not in lower
        ):
            issues.append(
                "challenge_question_format_missing"
            )

    # ==================================================
    # VALIDITY DECISION
    # ==================================================

    hard_failures = [

        "empty_output",
        "repetition",
    ]

    valid = True

    for item in issues:

        for failure in hard_failures:

            if item.startswith(failure):
                valid = False

    return {

        "valid": valid,

        "issues": issues,

        "cleaned_output": clean,

        "retry_recommended": not valid,
    }