from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "training_data" / "llm_tutor"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_DBS = [
    ROOT / "external" / "core_data" / "python_learning.db",
    ROOT / "external" / "core_data" / "database_sql.db",
    ROOT / "external" / "core_data" / "html_web_basics.db",
    ROOT / "external" / "core_data" / "git_version_control.db",
    ROOT / "external" / "core_data" / "data_structures.db",
]

DIFFICULTIES = ["easy", "medium", "hard"]

STYLES = [
    "simple",
    "code_first",
    "analogy",
    "step_by_step",
    "question_based",
    "misconception_correction",
    "challenge_based",
    "revision_summary",
]

LEARNER_STATES = [
    "slow_learner",
    "low_confidence",
    "weak_output_prediction",
    "weak_debug",
    "weak_conceptual",
    "strong_transfer",
    "ready_for_challenge",
]

TASK_TYPES = [
    "explanation",
    "summary",
    "flashcard",
    "hint",
    "feedback",
    "debug_task",
    "output_prediction",
    "transfer_question",
    "challenge_question",
    "revision_note",
]


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def fetch_concepts_from_db(db_path: Path) -> List[Dict[str, str]]:
    if not db_path.exists():
        print(f"Missing DB: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row["name"] for row in tables}

    if "concept_resources" not in table_names:
        conn.close()
        print(f"No concept_resources table in {db_path}")
        return []

    rows = conn.execute("SELECT * FROM concept_resources").fetchall()
    conn.close()

    concepts = []
    for row in rows:
        data = dict(row)
        concepts.append({
            "concept_id": safe_text(data.get("concept_id")),
            "topic": safe_text(data.get("topic")),
            "base_content": safe_text(data.get("base_content")),
            "examples": safe_text(data.get("examples")),
            "key_points": safe_text(data.get("key_points")),
            "misconceptions": safe_text(data.get("misconceptions")),
            "real_world_use": safe_text(data.get("real_world_use")),
            "next_concept_link": safe_text(data.get("next_concept_link")),
            "source_db": db_path.name,
        })

    return concepts


def build_prompt(
    concept: Dict[str, str],
    difficulty: str,
    style: str,
    learner_state: str,
    task_type: str,
) -> str:
    return f"""
You are a tutor generation model for a cognition-adaptive AI tutor.

Task type: {task_type}
Teaching style: {style}
Difficulty: {difficulty}
Learner state: {learner_state}

Concept: {concept["topic"]}

Grounded concept resource:
Definition:
{concept["base_content"]}

Examples:
{concept["examples"]}

Key points:
{concept["key_points"]}

Misconceptions:
{concept["misconceptions"]}

Real-world use:
{concept["real_world_use"]}

Generate only the requested tutor output.
Do not add unrelated information.
""".strip()


def build_output(
    concept: Dict[str, str],
    difficulty: str,
    style: str,
    learner_state: str,
    task_type: str,
) -> str:
    topic = concept["topic"]
    definition = concept["base_content"]
    examples = concept["examples"]
    key_points = concept["key_points"]
    misconceptions = concept["misconceptions"]
    real_world = concept["real_world_use"]

    if task_type == "explanation":
        return (
            f"{topic} means: {definition}\n\n"
            f"For a {learner_state}, explain it in a {style} way at {difficulty} level.\n"
            f"Key idea: {key_points}\n"
        )

    if task_type == "summary":
        return f"Summary of {topic}: {definition}\nRemember: {key_points}"

    if task_type == "flashcard":
        return (
            f"Q: What is {topic}?\n"
            f"A: {definition}\n\n"
            f"Q: What is one important point about {topic}?\n"
            f"A: {key_points.splitlines()[0] if key_points else definition}"
        )

    if task_type == "hint":
        return f"Hint: Think about what {topic} is used for. {key_points}"

    if task_type == "feedback":
        return (
            f"Feedback: Your answer should mention the meaning of {topic}, "
            f"its purpose, and one correct example."
        )

    if task_type == "debug_task":
        return (
            "Find the mistake and fix it:\n\n"
            "name = Alice\"\n"
            "print(name)\n\n"
            "Expected fix:\n"
            "name = \"Alice\"\n"
            "print(name)"
        )

    if task_type == "output_prediction":
        return (
            "What is the output?\n\n"
            "name = \"Alice\"\n"
            "print(name)\n\n"
            "Answer: Alice"
        )

    if task_type == "transfer_question":
        return f"How can {topic} be used in a real situation? Context: {real_world}"

    if task_type == "challenge_question":
        return f"Create a new example using {topic} and explain how changing one value affects the result."

    if task_type == "revision_note":
        return (
            f"Revision note for {topic}:\n"
            f"- Meaning: {definition}\n"
            f"- Remember: {key_points}\n"
            f"- Avoid: {misconceptions}"
        )

    return definition


def build_dataset() -> List[Dict[str, str]]:
    concepts = []
    for db in SUBJECT_DBS:
        concepts.extend(fetch_concepts_from_db(db))

    print(f"Loaded concepts: {len(concepts)}")

    samples = []

    for concept in concepts:
        if not concept["topic"] or not concept["base_content"]:
            continue

        for difficulty in DIFFICULTIES:
            for style in STYLES:
                for learner_state in LEARNER_STATES:
                    for task_type in TASK_TYPES:
                        prompt = build_prompt(
                            concept, difficulty, style, learner_state, task_type
                        )
                        output = build_output(
                            concept, difficulty, style, learner_state, task_type
                        )

                        samples.append({
                            "instruction": "Generate adaptive tutor content.",
                            "input": prompt,
                            "output": output,
                            "concept": concept["topic"],
                            "difficulty": difficulty,
                            "style": style,
                            "learner_state": learner_state,
                            "task_type": task_type,
                            "source_db": concept["source_db"],
                        })

    random.shuffle(samples)
    return samples


def save_jsonl(samples: List[Dict[str, str]]) -> None:
    train_path = DATA_DIR / "tutor_train.jsonl"
    val_path = DATA_DIR / "tutor_val.jsonl"
    test_path = DATA_DIR / "tutor_test.jsonl"

    n = len(samples)
    train = samples[: int(n * 0.8)]
    val = samples[int(n * 0.8): int(n * 0.9)]
    test = samples[int(n * 0.9):]

    for path, rows in [
        (train_path, train),
        (val_path, val),
        (test_path, test),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:")
    print(train_path, len(train))
    print(val_path, len(val))
    print(test_path, len(test))


if __name__ == "__main__":
    samples = build_dataset()
    save_jsonl(samples)