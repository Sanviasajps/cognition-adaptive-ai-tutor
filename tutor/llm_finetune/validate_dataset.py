from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "training_data" / "llm_tutor"

TRAIN_FILE = DATA_DIR / "tutor_train.jsonl"
VAL_FILE = DATA_DIR / "tutor_val.jsonl"
TEST_FILE = DATA_DIR / "tutor_test.jsonl"


REQUIRED_FIELDS = [
    "instruction",
    "input",
    "output",
]

EXPECTED_DOMAINS = {
    "Python",
    "SQL",
    "HTML",
    "Git",
    "Data Structures",
}


def load_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                rows.append(json.loads(line))

    return rows


def validate_rows(rows):

    missing_field_count = 0

    concepts = set()
    domains = set()

    for row in rows:

        for field in REQUIRED_FIELDS:

            if field not in row:
                missing_field_count += 1

        input_data = row.get("input", {})

        concept = None
        domain = None

        # -------------------------------------------------
        # CASE 1 — input is dictionary
        # -------------------------------------------------

        if isinstance(input_data, dict):

            concept = input_data.get("concept_name")
            domain = input_data.get("domain")

        # -------------------------------------------------
        # CASE 2 — input is string
        # -------------------------------------------------

        elif isinstance(input_data, str):

            text = input_data.lower()

            if "python" in text:
                domain = "Python"

            elif "sql" in text:
                domain = "SQL"

            elif "html" in text:
                domain = "HTML"

            elif "git" in text:
                domain = "Git"

            elif "stack" in text or "array" in text:
                domain = "Data Structures"

            concept = input_data[:50]

        if concept:
            concepts.add(str(concept))

        if domain:
            domains.add(str(domain))

    return {
        "missing_field_count": missing_field_count,
        "concepts": concepts,
        "domains": domains,
    }


def main():

    print("\nRunning dataset validation...\n")

    missing_files = []

    for path in [TRAIN_FILE, VAL_FILE, TEST_FILE]:

        if not path.exists():
            missing_files.append(str(path))

    if missing_files:

        print("STATUS: FAIL\n")

        print("Missing files:")

        for file in missing_files:
            print(file)

        return

    train_rows = load_jsonl(TRAIN_FILE)
    val_rows = load_jsonl(VAL_FILE)
    test_rows = load_jsonl(TEST_FILE)

    total_rows = (
        len(train_rows)
        + len(val_rows)
        + len(test_rows)
    )

    combined = train_rows + val_rows + test_rows

    results = validate_rows(combined)

    concepts = results["concepts"]
    domains = results["domains"]

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Total rows: {total_rows}")

    print(f"\nUnique concepts: {len(concepts)}")
    print(f"Domains found: {len(domains)}")

    print("\nDomains:")

    for domain in sorted(domains):
        print("-", domain)

    if results["missing_field_count"] > 0:

        print("\nSTATUS: WARN")
        print(
            f"Missing fields detected: "
            f"{results['missing_field_count']}"
        )

    else:

        print("\nSTATUS: PASS")

    missing_domains = EXPECTED_DOMAINS - domains

    if missing_domains:

        print("\nMissing expected domains:")

        for domain in missing_domains:
            print("-", domain)


if __name__ == "__main__":
    main()