# LLM-Based Framework for Patent Comparison and Bias Analysis

## Project Objective
This project aims to build a reliable and transparent framework for comparing and analyzing patents using Large Language Models (LLMs). While LLMs are useful for document-level comparisons, they often lack transparency and consistency in their judgments. To address this, we propose a structured prompting framework that guides LLMs to compare patents across three evaluative dimensions: **Functional Purpose (FP)**, **Technical Uniqueness (TU)**, and **Strategic Value (SV)**.

The framework integrates decision logic into prompt templates and applies lightweight fine-tuning (Parameter-Efficient Fine-Tuning, PEFT) to detect and mitigate an **SV-dominant bias** in the model's reasoning. This approach improves reasoning alignment and reduces overreliance on SV justifications.

## Key Contributions
The main contributions of this project are:

* **Formalization of a Multi-Dimensional Comparison Framework**: We established a three-dimensional judgment framework for LLM-based patent analysis.
* **Novel Diagnostic Metrics**: We developed diagnostic metrics (e.g., Mismatch Rate, Shake Rate) to evaluate judgment consistency, model bias, and reasoning quality.
* **Demonstration of Prompt Improvement**: We showed that prompt refinement enhances the reliability and transparency of LLM-based decision-making.
* **Bias Mitigation via PEFT**: We demonstrated that PEFT can be used to investigate and mitigate inherent biases. Specifically, fine-tuning was shown to reduce the bias towards 'Strategic Value (SV)' and strengthen reasoning on the 'Functional Purpose (FP)' and 'Technical Uniqueness (TU)' dimensions.

## File Structure
```
.
├── data
│   ├── peft                  # Data and results for PEFT fine-tuning
│   │   ├── attention         # Attention analysis results
│   │   ├── figs              # Analysis visualizations (heatmaps, KDE plots, etc.)
│   │   ├── justifications    # Reasoning outputs for base and fine-tuned models
│   │   ├── lora_output       # Fine-tuned LoRA adapter model
│   │   ├── token_probs       # Data used for token probability analysis
│   │   ├── fine_tuned_train.jsonl    # Fine-tuning training data
│   │   └── fine_tuned_valid.jsonl    # Fine-tuning validation data
│   ├── prompts               # Prompt data
│   │   ├── origin            # Original prompts
│   │   └── revised           # Revised prompts
│   ├── raw                   # Raw patent data (in JSON format)
│   └── summary               # Summarized patent data (in JSON format)
└── src
├── metrics_utils.py      # Metrics calculation utility
├── peft_analyze_attns.py # Attention analysis script
├── peft_analyze_probs.py # Token probability analysis script
├── peft_generate_outputs.py # Script for generating outputs from the fine-tuned model
├── peft_train.py         # PEFT fine-tuning training script
├── prompt_eval.py        # Prompt evaluation script
└── prompt_run.py         # Prompt execution script
```