import json
from pathlib import Path

DATA_PATH = Path("training_data/llm_tutor/tutor_test.jsonl")

def simple_score(pred, gold):
    p = set(pred.lower().split())
    g = set(gold.lower().split())
    if not g:
        return 0
    return len(p & g) / len(g)


def main():
    print("Running evaluation...")

    scores = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20:   # small sample
                break

            row = json.loads(line)

            # fake prediction (use gold for demo)
            pred = row["output"]
            gold = row["output"]

            score = simple_score(pred, gold)
            scores.append(score)

    avg = sum(scores) / len(scores)

    print("\nEvaluation Result:")
    print("Samples:", len(scores))
    print("Average Score:", round(avg, 2))


if __name__ == "__main__":
    main()