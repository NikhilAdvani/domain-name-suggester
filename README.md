# Domain Name Suggestion with Fine‑Tuned LLM

A repository showcasing a complete pipeline to build, fine‑tune, evaluate, and deploy an LLM for generating business domain name suggestions.

🌟 Features

Synthetic Dataset Creation: Generate and document 1 000+ diverse business descriptions.

LoRA Fine‑Tuning: Efficiently fine‑tune Llama‑3 models with adapter tuning.

Hyperparameter Optimization: Grid‑search tuned LoRA ranks, alphas, dropouts, and learning rates.

LLM‑as‑Judge: Automated quality scoring (relevance, brevity, memorability, brandability).

Edge‑Case Analysis: Systematic failure taxonomy across 210+ challenging prompts.

Safety Guardrails: Regex‑based content filter blocking inappropriate requests.

FastAPI Deployment: Ready‑to‑use endpoint for real‑time domain suggestions.

## Repository Structure

├── data/
│   ├── synthetic_data.csv         # Initial 1 000–row synthetic dataset
│   ├── augmented_data.csv         # Paraphrased + synonym-augmented subset (1 000 rows)
│   └── augmented_data_full.csv    # Full augmented dataset (1 500 rows)
├── checkpoints/
│   ├── baseline/                  # Baseline LoRA adapter + config
│   └── final_model/               # Best‑performing LoRA adapter + config
├── notebooks/
│   └── domain_suggester.ipynb     # Jupyter notebook with experiments & analysis
├── requirements.txt               # Python package dependencies
└── README.md                      # This file

## Quick Start

1. Clone and Enter Directory

<pre> '''git clone https://github.com/yourusername/domain-suggester.git
cd domain-suggester''' </pre>

2. Create & Activate Virtual Environment

<pre> '''python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
.\.venv\Scripts\activate  # Windows PowerShell''' </pre>

3. Install Dependencies

<pre> '''pip install --upgrade pip
pip install -r requirements.txt''' </pre>

4. Configure Credentials

Create a .env file in the project root:

<pre> '''HUGGINGFACE_HUB_TOKEN=hf_xxxYOUR_TOKEN_HERExxx
OPENAI_API_KEY=sk-xxxYOUR_OPENAI_KEYxxx''' </pre>

5. Run Experiments

Open the notebook and execute all cells

6. Load & Test Model

<pre> '''
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

CKPT = "checkpoints/final_model"
cfg = PeftConfig.from_pretrained(CKPT)

tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path)
base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, CKPT, torch_dtype=torch.float16)
model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

prompt = "Suggest domain for: cozy bakery in the suburbs\nDomain:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
'''</pre>

## Results Summary

Baseline (500 rows): Val loss ↓0.875→0.493 (PPL→1.64)

Augmented (1 000 rows): Val loss ↓0.818→0.709 (PPL→2.03)

HPO best: r=8, α=8, dropout=0.0, lr=3e-4 → Val loss 0.726

Final (1 500 rows): Val loss ↓0.712 (PPL→2.04)

Edge‑case pass rate: ~33%

## Future Work

Enhance diacritics/emoji handling

Add clarification prompts for ambiguous inputs

Expand non‑English examples

Continuous failure monitoring & retraining
