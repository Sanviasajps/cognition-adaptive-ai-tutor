import json
from pathlib import Path

# Use generated outputs (NOT dataset)
SAMPLES_PATH = Path("outputs/samples/generated_outputs.json")

def main():
    print("Running advanced evaluation...")

    # load generated outputs
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total = len(samples)

    valid = 0
    repeated = 0
    lengths = []

    seen = set()

    for s in samples:
        text = s.strip()

        # ✔ check valid output
        if len(text) > 20:
            valid += 1

        # ✔ check repetition
        if text in seen:
            repeated += 1
        else:
            seen.add(text)

        # ✔ track length
        lengths.append(len(text))

    # metrics
    valid_percent = valid / total
    repetition_rate = repeated / total
    avg_length = sum(lengths) / total

    result = {
        "total_samples": total,
        "valid_percent": round(valid_percent, 2),
        "repetition_rate": round(repetition_rate, 2),
        "avg_length": round(avg_length, 2)
    }

    print("\nEvaluation Result:")
    print(result)

    # save result
    with open("outputs/metrics/llm_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved to outputs/metrics/llm_evaluation.json")


if __name__ == "__main__":
    main()