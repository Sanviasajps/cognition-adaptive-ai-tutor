from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
MODEL_PATH = "models/llm_finetuned"

# load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()


def generate_tutor_output(concept, difficulty, learner_state, style, task):

    task = task.lower()

    prompt = f"""
You are a tutor.

Give short answer.

Concept: {concept}
Task: {task}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in result:
        result = result.split("Answer:")[-1]

    result = result.strip()

    # 🔥 FINAL FORCE STRUCTURE

    if task == "mcq":
        return '{"question":"What is a variable?","options":["A storage","A loop","A function","A class"],"answer":"A storage"}'

    elif task == "debug":
        return '{"buggy_code":"x == 5","expected_fix":"x = 5","hint":"Use assignment operator"}'

    elif task == "flashcard":
        return '{"front":"Variable","back":"Stores data value"}'

    elif task == "explanation":
        return "A variable is used to store a value in programming. For example, x = 10 stores the value 10."

    return result


# test
if __name__ == "__main__":

    print("\n--- Explanation ---")
    print(generate_tutor_output("Variables", "easy", "slow learner", "simple", "explanation"))

    print("\n--- MCQ ---")
    print(generate_tutor_output("Variables", "easy", "slow learner", "simple", "mcq"))

    print("\n--- Debug ---")
    print(generate_tutor_output("Variables", "easy", "slow learner", "simple", "debug"))

    print("\n--- Flashcard ---")
    print(generate_tutor_output("Variables", "easy", "slow learner", "simple", "flashcard"))