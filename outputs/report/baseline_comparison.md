# Baseline vs Fine-tuned LLM Comparison

## Template Generator

* Produces fixed, rule-based outputs
* No variation in responses
* Limited adaptability to user input
* Always follows the same predefined format
* Does not require training

## Fine-tuned LLM (SmolLM2-135M + LoRA)

* Generates flexible and dynamic outputs
* Adapts responses based on concept and difficulty
* Produces more natural language explanations
* Supports multiple task types (explanation, MCQ, debug, flashcard, etc.)
* Fine-tuned using LoRA on ~2000 training samples
* Training loss decreased from ~2.6 to ~1.1
* Evaluation loss around ~1.2–1.3
* Model checkpoint successfully saved

## Comparison

| Feature           | Template Generator | Fine-tuned LLM  |
| ----------------- | ------------------ | --------------- |
| Training Required | No                 | Yes (LoRA)      |
| Flexibility       | Low                | High            |
| Diversity         | Low                | High            |
| Adaptiveness      | Low                | Medium          |
| Readability       | Medium             | Medium-High     |
| Output Structure  | Fixed              | Semi-structured |
| Learning Ability  | None               | Present         |

## Conclusion

The template-based system is simple and reliable but lacks adaptability and diversity.
The fine-tuned LLM, although smaller in size, demonstrates improved flexibility and natural language generation.

The LoRA fine-tuned model provides a better balance between performance and efficiency, making it suitable for generating adaptive tutor content in the system.
