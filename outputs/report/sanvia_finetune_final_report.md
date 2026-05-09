# Sanvia Fine-Tuned Pretrained LLM Final Report

## 1. Project

Project Name:

Cognition-Adaptive AI Tutor

This work focuses on the pretrained fine-tuned LLM comparison pipeline for adaptive tutor content generation.

---

# 2. Environment Status

## Python

Python environment configured successfully.

## Required Libraries

Installed libraries:

- torch
- transformers
- datasets
- accelerate
- peft
- safetensors

## Runtime Status

Model loading, generation, evaluation, and reporting pipelines are functional.

---

# 3. Dataset Status

## Dataset Source

The dataset was generated using educational concept databases from:

- Python
- SQL
- HTML
- Git
- Data Structures

## Dataset Files

Generated dataset files:

- training_data/llm_tutor/tutor_train.jsonl
- training_data/llm_tutor/tutor_val.jsonl
- training_data/llm_tutor/tutor_test.jsonl

## Dataset Statistics

| Split | Rows |
|---|---|
| Train | 51072 |
| Validation | 6384 |
| Test | 6384 |

Total rows:

63840

## Dataset Validation

Dataset validation completed successfully.

Detected domains:

- Python
- SQL
- HTML
- Git
- Data Structures

Validation result:

PASS

---

# 4. SmolLM2-135M LoRA Status

## Base Model

HuggingFaceTB/SmolLM2-135M

## Fine-Tuning Method

LoRA / PEFT

## Status

Training completed successfully.

Checkpoint exists:

models/llm_finetuned/

## Capabilities

- Structured tutor generation
- Flashcards
- MCQs
- Debug tasks
- Output prediction
- Revision notes

## Evaluation Summary

| Metric | Value |
|---|---|
| Format Validity | 1.0 |
| Task Success Rate | 1.0 |
| Repetition Rate | 0.0 |
| JSON Validity | 1.0 |

---

# 5. Qwen2.5-Coder-0.5B LoRA Status

## Base Model

Qwen/Qwen2.5-Coder-0.5B-Instruct

## Fine-Tuning Method

LoRA / PEFT

## Status

Training completed successfully.

Checkpoint exists:

models/llm_finetuned/qwen_coder_05b_lora/

## Strengths

- Better programming-oriented generation
- Better instruction following
- Improved technical tutoring capability

## Evaluation Summary

| Metric | Value |
|---|---|
| Format Validity | 1.0 |
| Task Success Rate | 1.0 |
| Repetition Rate | 0.0 |
| JSON Validity | 1.0 |

---

# 6. Output Validation

Implemented validation checks:

- repetition detection
- bad pattern detection
- structured task validation
- MCQ validation
- debug task validation
- flashcard validation

Validation pipeline works successfully.

---

# 7. Human Evaluation Setup

Human evaluation preparation scripts were created.

Generated files:

- evaluation_outputs/reports/human_eval_sheet.csv
- evaluation_outputs/reports/human_eval_instructions.md
- evaluation_outputs/json/human_eval_mapping_private.json

Human evaluation analysis script also exists.

---

# 8. Generated Deliverables

## Models

- models/llm_finetuned/
- models/llm_finetuned/qwen_coder_05b_lora/

## Metrics

- outputs/metrics/smollm2_135m_metrics.json
- outputs/metrics/qwen_coder_05b_metrics.json

## Samples

- outputs/samples/smollm2_135m_samples.jsonl
- outputs/samples/qwen_coder_05b_samples.jsonl

## Reports

- outputs/report/pretrained_model_comparison.md
- outputs/report/sanvia_generation_quality_report.md
- outputs/report/baseline_comparison.md

---

# 9. Limitations

## SmolLM2

- Small model capacity
- Limited reasoning capability
- Shorter output quality

## Qwen-Coder

- Higher memory usage
- Slower inference compared to SmolLM2

## Dataset Limitation

Concept extraction validation currently detects limited unique concepts because some dataset input fields are stored as plain strings instead of structured dictionaries.

---

# 10. Final Verdict

Final Status:

PASS

The pretrained fine-tuned LLM pipeline is functional and complete.

The project successfully supports:

- dataset generation
- LoRA fine-tuning
- structured tutor generation
- evaluation
- output validation
- reporting
- human evaluation preparation

Both lightweight and stronger pretrained comparison models are available for adaptive tutor generation experiments.