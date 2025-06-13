# 3-Layer Patent Comparison System

This is an AI-powered patent comparison system designed to support structured, repeatable, and strategic patent evaluation. By leveraging large language models (LLMs), it enables users to identify technical advantages, strategic positioning, and implementation-level differentiators between their patents and those of competitors.

## What is this system?

The 3-Layer Patent Comparison System replaces manual and subjective analysis with structured AI-driven comparisons across three expert-designed layers:

### 1. Strategic Direction – Provides a strategic guide for how your patent should be positioned or developed further
![Strategy Result](images/1.strategic_recommend.PNG)
- Compares your patent with multiple competitors
- Aggregates relative value
- Generates a strategic recommendation based on the analysis

### 2. Technology Positioning – Determines which side has the technological and strategic upper hand.
![Technology Positioning Result](images/2.technology_positioning.PNG)
- Compares a single patent pair across:
  - Functional Purpose
  - Technical Uniqueness
  - Strategic Value
- Provides axis-level winners and overall judgment

### 3. Implementation Differentiation  – Highlights how your technical design differs in structure and approach.
![Implementation Differentiation Result](images/3.implementation_diff.PNG)
- Identifies 3–5 key implementation axes per patent pair
- Summarizes each side's solution
- Explains differences in structure, mechanism, or approach

This multi-layered insight supports more effective decision-making in R&D, IP management, and strategic investment planning.

## How to Use

This system can be used in two different ways depending on the user’s role.

### ▶ For General Users

Before using this system, it is assumed that your team has:
- Selected key patents from your company and competitors
- Filtered to focus on strategically important patent sets

#### Steps:
1. Enter your company’s target patent number
2. Enter one or more competitor patent numbers
3. Run the analysis

![Input GUI](images/input_gui.PNG)

The system will:
- Perform one-to-one comparisons
- Generate results using LLMs across the three layers
- Provide structured tables, insights, and strategy recommendations

### ▶ For Developers

If you want to run, test, or extend this system locally:

1. **Clone the repository and rename the folder**
- git clone https://github.com/deokjin-choi/multi-layer-patent-comparator.git
- mv multi-layer-patent-comparator patent_compare
- cd patent_compare

2. **Install dependencies**
- pip install -r requirements.txt

3. **Launch the local LLM using Ollama**
- ollama serve & # Start Ollama server in the background
- ollama run mistral
> The system uses **Mistral** as the default local model via Ollama.

4. **Start the Streamlit frontend**
- streamlit run main.py

5. **Project structure overview**
- `app/controllers/`  
  Core modules for summarization, positioning, strategy, etc.

- `utils/llm/`  
  LLM client interface, retry logic, model selection

- `prompts/`  
  Prompt templates (version-controlled per task)

- `data/`  
  Raw patents, summaries, and result caching

- `frontend/components/`  
  Streamlit-based user interface components

- `tests/`  
  Unit tests and evaluation scripts

## Author

Deokjin Choi  
Email: deokjin.choi@gmail.com

## License and Use Policy

This system is provided for personal, research, and portfolio use.

If you wish to apply this system in commercial, legal, or business-related settings, you must contact the author in advance for discussion.

## Acknowledgements

This project builds upon open-source LLM technologies and contributions from the AI and patent analytics community. Special thanks to the developers of Streamlit and foundational transformer models.
