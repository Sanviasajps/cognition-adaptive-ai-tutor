# Baseline vs Pretrained LLM Comparison

| Model | Type | Quality | Strengths | Weaknesses |
|---|---|---|---|---|
| Template Generator | Rule-based | Medium | Stable formatting, predictable | No adaptability |
| SmolLM2-135M LoRA | Small pretrained LLM | Low | Runtime worked, lightweight | Weak generation quality |
| Qwen2.5-Coder-0.5B LoRA | Code-specialized pretrained LLM | High | Better code understanding, cleaner outputs, stronger tutor responses | Some dataset artifacts remain |

## Key Observation

Qwen-Coder significantly improved:
- code-related educational generation
- structured responses
- concept relevance
- debug task quality
- challenge generation

compared to SmolLM2-135M.

## Final Research Insight

Small pretrained models can run successfully but may struggle with high-quality tutor generation.

Larger code-specialized models such as Qwen-Coder provide much stronger educational generation performance for programming tutor tasks.