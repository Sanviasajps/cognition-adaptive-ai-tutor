---
base_model: HuggingFaceTB/SmolLM2-135M
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-135M
- lora
- transformers
---

# SmolLM2-135M LoRA Adapter

## Model Overview

This adapter was fine-tuned using LoRA (Low-Rank Adaptation) on the HuggingFaceTB/SmolLM2-135M base model for adaptive tutor content generation tasks.

The model was trained as part of a cognition-adaptive AI tutor project focused on educational content generation and evaluation.

## Fine-Tuning Objective

The goal of this fine-tuning process was to generate structured tutor outputs including:

- explanations
- flashcards
- MCQs
- debugging tasks
- output prediction tasks
- revision notes
- challenge questions

## Base Model

- HuggingFaceTB/SmolLM2-135M

## Fine-Tuning Method

- PEFT / LoRA

## LoRA Configuration

- r = 8
- alpha = 16
- dropout = 0.05

## Training Dataset

The dataset contains adaptive tutoring examples across multiple domains:

- Python
- SQL
- HTML
- Git
- Data Structures

Dataset statistics:

- train rows: 51072
- validation rows: 6384
- test rows: 6384
- unique concepts: 38

## Training Details

Training was performed using Hugging Face Transformers and PEFT libraries.

Training included:

- tokenizer loading
- LoRA application
- causal language modeling
- evaluation during training
- adapter checkpoint saving

## Evaluation

The fine-tuned model was evaluated using:

- structured output validation
- repetition detection
- formatting checks
- manual sample inspection
- comparison against other tutor generation systems

## Intended Use

This model is intended for:

- educational experiments
- adaptive tutoring research
- prototype tutor generation systems
- academic evaluation pipelines

## Limitations

This model is a lightweight research prototype and may:

- generate repetitive outputs
- produce simplified explanations
- require additional prompt engineering
- need stronger reasoning improvements for production use

## Framework Versions

- transformers
- peft 0.19.1
- pytorch

## Final Status

PASS
### Framework versions

- PEFT 0.19.1