# Pretrained Model Comparison

## Overview

This report compares the pretrained fine-tuned tutor generation models used in the Cognition-Adaptive AI Tutor project.

Models compared:

1. SmolLM2-135M LoRA
2. Qwen2.5-Coder-0.5B LoRA

---

# Comparison Table

| Feature | SmolLM2-135M LoRA | Qwen2.5-Coder-0.5B LoRA |
|---|---|---|
| Base Model | HuggingFaceTB/SmolLM2-135M | Qwen/Qwen2.5-Coder-0.5B-Instruct |
| Training Status | Completed | Completed |
| Checkpoint Exists | Yes | Yes |
| Fine-tuning Method | LoRA / PEFT | LoRA / PEFT |
| Train Rows | 51072 | 51072 |
| Validation Rows | 6384 | 6384 |
| Test Rows | 6384 | 6384 |
| Format Validity | 1.0 | 1.0 |
| Task Success Rate | 1.0 | 1.0 |
| Repetition Rate | 0.0 | 0.0 |
| JSON Validity | 1.0 | 1.0 |
| MCQ Validity | 5 | 5 |
| Debug Task Validity | 5 | 5 |
| Flashcard Validity | 5 | 5 |
| Concept Relevance Proxy | 0.7 | 0.7 |
| Average Output Length | 12.0 | 12.0 |

---

# SmolLM2-135M LoRA

## Strengths

- Lightweight and fast
- Small memory requirement
- Simple LoRA fine-tuning
- Stable structured output generation
- Suitable for baseline comparison

## Weaknesses

- Smaller model capacity
- Lower reasoning capability
- Limited long-context understanding
- Less adaptive generation quality

## Verdict

SmolLM2-135M works as a lightweight baseline pretrained tutor model and successfully supports structured tutor content generation.

---

# Qwen2.5-Coder-0.5B LoRA

## Strengths

- Better code-oriented capabilities
- Stronger debugging and technical generation
- Better instruction-following
- More suitable for programming tutor tasks

## Weaknesses

- Larger model size
- Slower inference compared to SmolLM2
- Requires more memory during training

## Verdict

Qwen2.5-Coder-0.5B provides stronger programming-oriented tutor generation and is more suitable for adaptive coding tutor tasks.

---

# Final Observation

The pretrained fine-tuned LLM pipeline is functional for both lightweight and stronger code-oriented models.

SmolLM2 provides a compact baseline model.

Qwen2.5-Coder demonstrates improved technical tutor generation capability and serves as a stronger pretrained comparison model for the project.