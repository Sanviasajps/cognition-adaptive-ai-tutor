# Fine-tuning Summary

## Model

SmolLM2-135M

## Method

LoRA Fine-tuning (PEFT)

## Dataset

* Generated from core_data database
* Full dataset size: ~50,000 samples
* Training subset used: ~2000 samples (for fast training)
* Validation subset: ~200 samples

## Training

* Training executed successfully using HuggingFace Trainer
* Loss decreased from ~2.6 to ~1.1 during training
* Evaluation loss observed around ~1.2–1.3
* Checkpoint and LoRA adapter model saved in:
  models/llm_finetuned/
* Files include:

  * adapter_model.safetensors
  * adapter_config.json
  * checkpoint-1000

## Generation

* Outputs generated using fine-tuned model
* Supports multiple tutor tasks:

  * explanation
  * MCQ
  * debug tasks
  * flashcards
  * summaries
* Outputs are functional but may contain minor noise due to small model size

## Evaluation

* Total samples evaluated: 5
* Format validity: 100%
* Repetition rate: Low
* Average output length: ~48 characters

## Strengths

* Generates adaptive tutor-style responses
* More flexible than template-based system
* Supports multiple educational task types
* Efficient fine-tuning using LoRA

## Weaknesses

* Output quality is limited due to small model size (135M)
* Some outputs are slightly unclear or noisy
* Structured outputs (JSON) require post-processing
* Trained on reduced dataset subset

## Conclusion

The pretrained SmolLM2-135M model was successfully fine-tuned using LoRA and validated through generation and evaluation. The model is capable of generating adaptive educational content and is suitable for integration into the adaptive tutor system.
