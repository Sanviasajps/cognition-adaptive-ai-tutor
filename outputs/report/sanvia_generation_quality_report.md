# Sanvia Pretrained LLM — Generation Quality Report

## Model

SmolLM2-135M (LoRA fine-tuned)

## Runtime Status

* Model loading: PASS
* Generation execution: PASS
* Validator integration: PASS
* Retry logic: PASS

## Improvements Applied

The following improvements were added without retraining:

* stricter tutor prompt
* deterministic decoding
* repetition penalty
* output validator
* retry mechanism for invalid generations
* output cleaning rules

## Observed Improvements

* repeated Hook/HREF patterns reduced
* validator successfully detects bad outputs
* retry logic activates correctly
* generation became slightly more stable

## Remaining Problems

The model still produces:

* unrelated API/request-like text
* incorrect task formatting
* noisy generated code
* weak concept alignment
* flashcard/debug formatting failures
* output drift across concepts

Examples observed:

* random metadata fields
* malformed code
* unrelated output prediction text
* invalid flashcard structure

## Technical Analysis

Possible reasons for weak generation quality:

1. Small base model size (135M)
2. Limited LoRA adaptation capacity
3. Noisy or inconsistent dataset formatting
4. Weak task-format alignment during training
5. Lack of grounding/RAG support
6. Mixed concept distributions in training data

## Conclusion

The pretrained fine-tuned LLM is runtime-ready and functional, but generation quality remains weaker than the project’s RAG-grounded service and CogniTutorLM from scratch.

Prompt engineering and validator-based filtering improved reliability slightly, but the model still requires cleaner task-specific datasets and possible retraining for higher-quality educational generation.

## Final Status

* generation_status: PASS
* validator_status: PASS
* evaluation_status: PASS
* quality_status: PARTIAL
* retraining_required_for_high_quality: YES
