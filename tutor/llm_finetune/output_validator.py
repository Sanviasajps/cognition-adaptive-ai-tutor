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
    tokens = re.findall(r"\b\w+\b", text.lower())

    if len(tokens) < 8:
        return False

    for i in range(len(tokens) - 3):
        chunk = tokens[i:i+3]
        joined = " ".join(chunk)

        count = " ".join(tokens).count(joined)

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

    if not clean:
        issues.append("empty_output")

    if len(clean.split()) < 5:
        issues.append("too_short")

    for bad in BAD_PATTERNS:
        if bad.lower() in clean.lower():
            issues.append(f"bad_pattern:{bad}")

    if has_repetition(clean):
        issues.append("repetition")

    lower = clean.lower()

    if task_type == "flashcard":
        if "front:" not in lower or "back:" not in lower:
            issues.append("flashcard_format_missing")

    if task_type == "debug_task":
        if "buggy" not in lower or "fix" not in lower:
            issues.append("debug_format_missing")

    valid = (
        "empty_output" not in issues
        and "repetition" not in issues
    )

    return {
        "valid": valid,
        "issues": issues,
        "cleaned_output": clean,
        "retry_recommended": not valid,
    }