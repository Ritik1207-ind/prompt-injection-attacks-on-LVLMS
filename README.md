# Image-Based Prompt Injection Attacks on LVLMs

> A synthetic dataset generation pipeline for studying prompt injection attacks on Large Vision-Language Models (LVLMs), covering 4 attack types × 4 injection goals with 4,859 labeled samples.

---

## 📌 Overview

This repository contains the complete pipeline for generating a synthetic dataset of **image-based prompt injection attacks** targeting Large Vision-Language Models (LVLMs). The dataset is designed for security research — specifically to benchmark model vulnerability and evaluate defences against visual prompt injection.

### What is Image-Based Prompt Injection?

Prompt injection attacks embed malicious instructions inside images to hijack or manipulate an AI vision system. Unlike text-based attacks, these bypass text-channel safety filters because the instruction is delivered through the visual modality.

---

## 🗂️ Dataset

**HuggingFace:** [Reet1207/image-prompt-injection](https://huggingface.co/datasets/Reet1207/image-prompt-injection)

| Split | Samples |
|---|---|
| Train | 3,887 |
| Validation | 485 |
| Test | 487 |
| **Total** | **4,859** |

### Dataset Schema

| Field | Type | Description |
|---|---|---|
| `image_path` | string | Path to generated image |
| `attack_type` | int | 1=typographic, 2=structural, 3=adversarial, 4=metadata |
| `attack_type_name` | string | Human-readable attack type |
| `goal` | string | jailbreak / exfiltration / hijacking / social_engineering |
| `malicious_instr` | string | The injected instruction (LLM-generated) |
| `benign_context` | string | Legitimate context hiding the injection |
| `regex_pattern` | string | Validation pattern for this goal |
| `regex_hit` | bool | Whether instruction matched the pattern |
| `metadata_fields` | string | EXIF fields used (T4 only) |
| `model_response` | string | LLaVA-1.5-7B raw response |
| `asr_label` | int | 1=attack succeeded, 0=failed/refused |
| `stealthiness` | string | zero / low / maximum |

---

## ⚔️ Attack Types

### Type 1 — Typographic Injection
Renders the malicious instruction as visible text on a plain white background using PIL. The vision encoder reads it as visual content; text-channel safety filters never see it.
- **Stealthiness:** Zero
- **Implementation:** PIL
- **Based on:** FigStep

### Type 2 — Structural Layout Injection
Embeds the malicious instruction inside a visually complex diagram (flowchart or mind map) where the harmful content is camouflaged among legitimate nodes.
- **Stealthiness:** Low
- **Implementation:** Graphviz + networkx
- **Based on:** Lee et al. (Electronics, 2025)

### Type 3 — Adversarial Perturbation Injection *(Core Novelty)*
Optimises pixel-level noise using PGD (Projected Gradient Descent) through CLIP's vision encoder so the image embedding encodes the attacker's instruction in latent space. The image looks completely normal to humans.
- **Stealthiness:** Maximum
- **Implementation:** PyTorch PGD on CLIP ViT-L/14
- **White-box:** Requires gradient access to vision encoder
- **Based on:** Qi et al. (2023)

### Type 4 — Metadata Injection
Embeds the malicious instruction in EXIF metadata fields (UserComment, ImageDescription) of a completely benign-looking image. Effective in agent deployments and RAG pipelines that read metadata.
- **Stealthiness:** Maximum
- **Implementation:** piexif
- **Threat surface:** Agent/pipeline deployments only

---

## 🎯 Injection Goals

| Goal | Description |
|---|---|
| **Jailbreak** | Bypass safety filters, produce restricted content |
| **Exfiltration** | Leak system prompt or context window |
| **Hijacking** | Override original task with attacker-controlled task |
| **Social Engineering** | Generate phishing content, impersonate trusted entities |

---

## 📊 Results — Attack Success Rate (ASR)

Evaluated using **LLaVA-1.5-7B** on the full dataset:

| Attack Type | Jailbreak | Exfiltration | Hijacking | Social Eng. | Total |
|---|---|---|---|---|---|
| Typographic | 2.2% | 9.9% | 2.6% | **20.8%** | 8.9% |
| Structural | 4.5% | 2.6% | 0.0% | 1.0% | 2.0% |
| **Adversarial** | **13.7%** | 0.0% | 0.0% | 0.0% | 3.4% |
| Metadata | **24.0%** | 0.0% | 0.0% | 0.0% | 6.0% |

### Key Findings
- **T3 Adversarial jailbreak: 13.7% ASR** — completely invisible to humans (core novelty)
- **T4 Metadata jailbreak: 24.0% ASR** — highest single ASR, zero visual trace
- **T1 Social engineering: 20.8% ASR** — explicit text most effective for phishing
- **T2 Structural: lowest overall (2.0%)** — camouflage reduces LVLM legibility

---

## 🏗️ Pipeline Architecture

```
LLM Orchestrator (Llama-3.1-70B via Ollama)
           ↓
  Injection Goal Sampler
  (jailbreak / exfiltration / hijacking / social_engineering)
           ↓
  ┌─────┬──────────┬────────────┬──────────┐
  T1    T2         T3           T4
  PIL   Graphviz   CLIP PGD     piexif
           ↓
  Regex Validator
           ↓
  LLaVA-1.5-7B Inference (ASR scoring)
           ↓
  HuggingFace Parquet Dataset
```

---

## 🚀 How to Run

### Prerequisites
- NVIDIA GPU (48GB VRAM recommended for T3 + LLaVA inference)
- Ubuntu 24.04
- Python 3.12
- Ollama

### Setup

```bash
# Clone the repo
git clone https://github.com/Ritik1207-ind/prompt-injection-attacks-on-LVLMS.git
cd prompt-injection-attacks-on-LVLMS

# Install system dependencies
sudo apt-get install -y graphviz python3-pip git curl

# Install Ollama and pull orchestrator model
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:70b-instruct-q4_K_M

# Install Python dependencies
pip install -r pipeline/requirements.txt --break-system-packages

# Download LLaVA
hf download llava-hf/llava-1.5-7b-hf --local-dir ./models/llava-1.5-7b
```

### Run smoke test (verify everything works)

```bash
python3 pipeline/scripts/smoke_test.py
```

### Run full pipeline

```bash
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml
```

### Run individual stages

```bash
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml --stage generate
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml --stage images
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml --stage validate
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml --stage inference
python3 pipeline/run_pipeline.py --config pipeline/configs/config.yaml --stage write
```

### Load the dataset

```python
from datasets import load_dataset
ds = load_dataset("Reet1207/image-prompt-injection")
print(ds)
```

---

## 📁 Repository Structure

```
prompt-injection-attacks-on-LVLMS/
├── pipeline/
│   ├── run_pipeline.py          # Master orchestrator
│   ├── orchestrator.py          # Llama-70B instruction generator
│   ├── validator.py             # Regex validation
│   ├── inference.py             # LLaVA inference runner
│   ├── dataset_writer.py        # HuggingFace Parquet output
│   ├── evaluate_all_models.py   # Multi-model evaluation
│   ├── requirements.txt
│   ├── configs/
│   │   └── config.yaml
│   ├── generators/
│   │   ├── t1_typographic.py
│   │   ├── t2_structural.py
│   │   ├── t3_adversarial.py
│   │   └── t4_metadata.py
│   └── scripts/
│       └── smoke_test.py
├── notebooks/
│   └── t3_adversarial_pgd.ipynb
├── results/
│   ├── pipeline.log
│   └── sample_images/
├── report/
│   └── research_report.pdf
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM Orchestrator | Llama-3.1-70B-Instruct (Ollama) |
| LVLM Target | LLaVA-1.5-7B (HuggingFace) |
| Vision Encoder (T3) | CLIP ViT-L/14 |
| T1 Generator | PIL |
| T2 Generator | Graphviz + networkx |
| T4 Generator | piexif |
| Dataset Format | HuggingFace datasets (Parquet) |
| GPU | NVIDIA RTX 6000 (48GB) |

---

## 👥 Team

| Name | GitHub |
|---|---|
| Reet | [@Reet1207](https://github.com/Ritik1207-ind) |
| Member 2 | — |
| Member 3 | — |

---

## 📄 References

1. Gong et al. "FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts" (2023)
2. Lee et al. "Prompt Injection Attacks on Vision Language Models" Electronics (2025)
3. Qi et al. "Visual Adversarial Examples Jailbreak Aligned Large Language Models" (2023)

---

## ⚠️ Disclaimer

This dataset and pipeline are created strictly for academic security research. The generated samples are synthetic and intended for use in developing defences against prompt injection attacks. Do not use for malicious purposes.
