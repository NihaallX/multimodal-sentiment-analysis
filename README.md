# Conflict-Aware Geometric Routing Network (CGRN)

> **Multimodal Sentiment Analysis with Conflict-Aware Geometric Routing**  
> A modular, explainable architecture that detects cross-modal disagreement geometrically and routes inference through sarcasm-sensitive conflict branches.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-toolkit)

---

## What is CGRN?

Most multimodal sentiment systems simply fuse text and image features together. CGRN takes a different approach: it **measures how much the two modalities geometrically disagree**, uses that as a routing signal, and sends conflicting samples (sarcasm, irony, contradiction) through a specialized cross-attention conflict branch.

**The key idea — Geometric Dissonance Score (GDS):**

$$D = \alpha \cdot (1 - \cos(S_t, S_i)) + \beta \cdot \|\ |S_t\| - \|S_i\|\ |$$

where $\alpha$, $\beta$ are **learnable** parameters, and $S_t$, $S_i$ are the L2-normalized sentiment embeddings for text and image.

If $D \geq \tau$ (learnable threshold) → **Conflict Branch** (cross-attention + sarcasm detection)  
If $D < \tau$ → **Normal Fusion Branch** (simple concat + MLP)

---

## Architecture

```
Text Input  ──► [RoBERTa-base]  ──► S_t ∈ ℝ²⁵⁶  ──┐
                                                     ├──► [GDS Module] ──► D (scalar)
Image Input ──► [MobileNetV3]   ──► S_i ∈ ℝ²⁵⁶  ──┘            │
                                                                   ▼
                                                    ┌─── [Routing Controller] ───┐
                                                    │  D < τ            D ≥ τ   │
                                                    ▼                            ▼
                                          [Normal Fusion]            [Conflict Branch]
                                           Concat + MLP          CrossAttn + Sarcasm Head
                                                    └──────────────┬─────────────┘
                                                                   ▼
                                                          Final Prediction (3-class)
                                                                   ▼
                                                      [Explainability Engine]
                                                         → Conflict Report
```

---

## Results

### MVSA-Multiple (19,600 real social media posts)

| Model | Backbone | Acc | Macro-F1 | Conflict F1 | τ learned |
|---|---|---|---|---|---|
| CGRN v1 | DistilBERT + MobileNetV3-S | 63.4% | 0.552 | 0.471 | ✅ (0.587) |
| CGRN v2 | RoBERTa-base + MobileNetV3-S | 62.9% | 0.558 | 0.477 | ✅ (0.566) |
| **CGRN v3** | **RoBERTa-base + MobileNetV3-S** | **61.8%** | **0.552** | **0.483** | ✅ **(0.735)** |

**Routing correctness (v3):** 78.1% of conflict samples routed to conflict branch; 29.9% of harmonic samples routed to normal branch (up from 5.2% in v2).  
**v3 fixes:** τ_margin 0.2→0.35, τ_weight 0.2→0.5, routing balance loss (weight=0.15), threshold_init 0.5→0.65.  
**GDS separation (v3):** GDS mean=0.886, τ=0.735 — meaningful threshold vs v2's collapsed τ=0.566.

---

## Key Technical Contributions

| # | Contribution | Details |
|---|---|---|
| 1 | **Differentiable GDS** | Learnable α, β weights; fully differentiable geometric disagreement score |
| 2 | **Soft routing during training** | `p_conflict = σ((D − τ) · T)` — both branches computed, blended; hard routing at inference |
| 3 | **Learnable τ with hinge loss** | τ hinge pushes threshold between conflict/non-conflict GDS distributions |
| 4 | **Conflict Branch** | Cross-modal attention + dedicated sarcasm detection head |
| 5 | **Auto conflict reports** | Per-inference structured report with routing decision, GDS, sarcasm probability, interpretation |
| 6 | **3-stage training** | Unimodal pretraining → routing/fusion training → end-to-end fine-tuning |

---

## Project Structure

```
NLP-Project/
├── src/
│   ├── encoders/
│   │   ├── text_encoder.py            # RoBERTa/DistilBERT → S_t ∈ ℝ²⁵⁶
│   │   └── image_encoder.py           # MobileNetV3 → S_i ∈ ℝ²⁵⁶
│   ├── modules/
│   │   ├── gds_module.py              # Geometric Dissonance Score (α, β learnable)
│   │   ├── routing_controller.py      # Soft routing, learnable τ, conflict/normal branches
│   │   └── explainability_module.py   # Auto conflict report generation
│   ├── models/
│   │   ├── cgrn_model.py              # Full CGRN end-to-end model + CGRNConfig
│   │   └── unimodal_classifiers.py    # Stage 1 unimodal wrappers
│   ├── training/
│   │   └── training_strategy.py       # 3-stage trainer, CGRNLoss (τ hinge), cosine LR
│   ├── evaluation/
│   │   ├── evaluator.py               # Model comparison framework
│   │   ├── ablation.py                # Ablation studies
│   │   └── efficiency_analysis.py     # Latency / memory profiling
│   └── utils/
│       ├── mvsa_loader.py             # MVSA-Multiple/Single dataset loader
│       ├── data_loader.py             # Generic dataset utilities
│       └── visualization.py           # Training plots
├── experiments/
│   ├── train_mvsa.py                  # Train on MVSA-Multiple / MVSA-Single
│   ├── evaluate_mvsa.py               # Evaluate on MVSA test split
│   ├── run_training.py                # Synthetic data training
│   └── run_evaluation.py             # Synthetic data evaluation
├── app.py                             # Streamlit interactive dashboard
├── demo.py                            # Quick architecture demo
├── checkpoints_mvsa/                  # Trained model checkpoints
├── results/                           # Evaluation outputs + confusion matrix
├── docs/
│   ├── problem_definition.md
│   ├── patent_draft_outline.md
│   └── architecture_diagram.md
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit demo (no training needed)

The pre-trained checkpoint (`checkpoints_mvsa/cgrn_mvsa_final.pt`) runs inference on any text + image pair.

```bash
streamlit run app.py
```

Open http://localhost:8501, load the checkpoint from the sidebar, then type any text and upload an image.

**Try a conflict example:**
- Text: *"Oh great, another Monday"* + image of empty office → GDS > τ → Conflict Branch ✅

### 3. Train on MVSA-Multiple

Download [MVSA-Multiple](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) and extract to `data/MVSA_Multiple/`.

```bash
python experiments/train_mvsa.py \
    --data_root data/MVSA_Multiple \
    --stage1_epochs 5 \
    --stage2_epochs 8 \
    --stage3_epochs 5 \
    --batch_size 32 \
    --device cuda
```

### 4. Evaluate

```bash
python experiments/evaluate_mvsa.py \
    --checkpoint checkpoints_mvsa/cgrn_mvsa_final.pt \
    --data_root  data/MVSA_Multiple \
    --device     cuda
```

Outputs: per-class F1, conflict/non-conflict subset metrics, GDS distribution, routing correctness, confusion matrix PNG.

### 5. Single inference (Python)

```python
import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
from src.models.cgrn_model import CGRNConfig

device = "cuda"
model = CGRNConfig.build().to(device)
model.load_state_dict(torch.load("checkpoints_mvsa/cgrn_mvsa_final.pt", map_location=device), strict=False)
model.eval()

tok = AutoTokenizer.from_pretrained("roberta-base")
enc = tok("This place is absolutely terrible!", return_tensors="pt",
          padding="max_length", max_length=128, truncation=True)

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
image = transform(Image.open("your_image.jpg").convert("RGB")).unsqueeze(0)

with torch.no_grad():
    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), image.to(device), return_reports=True)

labels = ["negative", "neutral", "positive"]
print(f"Prediction : {labels[out.final_logits.argmax().item()]}")
print(f"GDS        : {out.gds_output.gds.item():.4f}")
print(f"τ          : {model.routing_controller.threshold.item():.4f}")
print(f"Branch     : {'CONFLICT' if out.routing_output.routing_decisions[0] else 'NORMAL'}")
print(out.conflict_reports[0])
```

---

## Training Pipeline Details

### 3-Stage Strategy

| Stage | What trains | LR | Purpose |
|---|---|---|---|
| **Stage 1** | Text encoder + Image encoder independently | 1e-5 / 1e-4 | Learn strong unimodal representations |
| **Stage 2** | GDS module, routing controller τ, fusion branches (encoders mostly frozen, last 2 text layers unfrozen) | 5e-5 | Learn geometric routing |
| **Stage 3** | All layers end-to-end | 2e-6 | Fine-tune for final coherence |

All stages use **cosine LR schedule with linear warmup (10%)**.

### Loss Function

$$L_{total} = L_{CE} + \lambda_u (L_{text} + L_{image}) + \lambda_r L_{routing} + \lambda_\tau L_\tau$$

- $L_{CE}$: cross-entropy on final logits
- $L_{text}, L_{image}$: unimodal supervision (λ=0.3)
- $L_{routing}$: pushes GDS high for conflict samples (λ=0.1)
- $L_\tau$: **τ hinge loss** — explicitly separates threshold between conflict/non-conflict GDS (λ=0.2):

$$L_\tau = \text{ReLU}(\tau + m - D_{conflict}) + \text{ReLU}(D_{normal} + m - \tau), \quad m=0.2$$

---

## Hardware

Trained on **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (CUDA 12.8, PyTorch 2.10+cu128).  
Training time: ~60 min (DistilBERT) / ~6.5 hrs (RoBERTa-base) for 5+8+5 epochs on MVSA-Multiple (RTX 3050 6 GB).

---

## Novel Claims (Patent Basis)

1. **Differentiable geometric dissonance score** as a routing signal between multimodal branches
2. **Learnable routing threshold τ** trained via hinge loss separation
3. **Soft routing during training** via sigmoid blend; hard routing at inference — gradient flows to τ
4. **Conflict branch with cross-modal attention** specialized for sarcasm and contradictory modalities
5. **Per-inference structured conflict reports** with routing decision, GDS, sarcasm probability, and natural language interpretation

---

*CGRN — Conflict-Aware Geometric Routing Network | 2026*


> **Patent-Oriented Multimodal Sentiment Analysis System**  
> A lightweight, modular, explainable architecture that explicitly models cross-modal geometric disagreement and routes inference through conflict-specialized branches.

---

## Architecture Overview

```
Text Input → [Text Encoder]  → S_t ∈ ℝ^256 ─────────┐
                                                       ├→ [GDS Module] → D (scalar)
Image Input → [Image Encoder] → S_i ∈ ℝ^256 ─────────┘        │
                                                                 ↓
                                                    [Routing Controller]
                                                     D < τ │        │ D ≥ τ
                                              [Normal Fusion]    [Conflict Branch]
                                               Concat + MLP    CrossAttn + Sarcasm
                                                          └────┬────┘
                                                               ↓
                                                      Final Prediction
                                                               ↓
                                                   [Explainability Engine]
                                                    → Conflict Report
```

### Core Innovation: Geometric Dissonance Score (GDS)

$$D = \alpha \cdot (1 - \cos(S_t, S_i)) + \beta \cdot |\|S_t\| - \|S_i\||$$

- **α** and **β** are learnable non-negative parameters
- D quantifies geometric disagreement between modality sentiment vectors
- D is differentiable, pluggable, and loggable
- Used as a routing signal to dispatch sarcastic/contradictory samples to the Conflict Branch

---

## Project Structure

```
NLP-Project/
├── docs/
│   ├── problem_definition.md      # Phase 1: Technical problem framing
│   ├── patent_draft_outline.md    # Phase 10: Patent-oriented documentation
│   └── architecture_diagram.md   # Mermaid block diagram (auto-generated)
├── src/
│   ├── encoders/
│   │   ├── text_encoder.py        # DistilBERT/MiniLM → S_t (256-dim)
│   │   └── image_encoder.py       # MobileNetV3 → S_i (256-dim)
│   ├── modules/
│   │   ├── gds_module.py          # Geometric Dissonance Score (Phase 3)
│   │   ├── routing_controller.py  # Conflict-aware routing (Phase 4)
│   │   └── explainability_module.py  # Conflict report generation (Phase 5)
│   ├── models/
│   │   ├── cgrn_model.py          # Full end-to-end CGRN model
│   │   └── unimodal_classifiers.py  # Stage 1 unimodal classifiers
│   ├── training/
│   │   └── training_strategy.py   # 3-stage training pipeline (Phase 6)
│   ├── evaluation/
│   │   ├── evaluator.py           # Model comparison (Phase 7)
│   │   ├── ablation.py            # Ablation studies (Phase 9)
│   │   └── efficiency_analysis.py # Efficiency profiling (Phase 8)
│   └── utils/
│       ├── data_loader.py         # Dataset utilities
│       └── visualization.py       # Plots and diagrams
├── experiments/
│   ├── run_training.py            # Full training pipeline script
│   └── run_evaluation.py          # Full evaluation pipeline script
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on Synthetic Data (No dataset required)

```bash
# Full training pipeline (generates synthetic data automatically)
python experiments/run_training.py --synthetic --n_samples 500 --stage1_epochs 2 --stage2_epochs 3 --skip_stage3

# Evaluation
python experiments/run_evaluation.py --data_path data/synthetic_sentiment.csv
```

### 3. Run on Real Dataset

```bash
# Prepare dataset as JSON: [{text, image_path, label (0/1/2), conflict (0/1)}]
python experiments/run_training.py --data_path data/your_dataset.json

python experiments/run_evaluation.py \
    --model_path checkpoints/cgrn_final.pt \
    --data_path  data/your_dataset.json
```

### 4. Single Inference with Conflict Report

```python
import torch
from src.models.cgrn_model import CGRNModel
from PIL import Image
from torchvision import transforms

model = CGRNModel()
# model.load_state_dict(torch.load("checkpoints/cgrn_final.pt"))
model.eval()

# Tokenize text
encoding = model.text_encoder.tokenize(
    ["This product is absolutely wonderful!"],
    device="cpu"
)

# Load and transform image
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
image = transform(Image.new("RGB", (224, 224))).unsqueeze(0)

# Forward pass with conflict report
with torch.no_grad():
    output = model(
        encoding["input_ids"],
        encoding["attention_mask"],
        image,
        return_reports=True,
    )

# Print conflict report
print(output.conflict_reports[0])
print(f"GDS Score: {output.gds_output.gds.item():.4f}")
print(f"Routing:   {'Conflict' if output.routing_output.routing_decisions[0] else 'Normal'}")
```

---

## Phase Summary

| Phase | Description | Key File |
|---|---|---|
| 1 | Problem definition & literature | `docs/problem_definition.md` |
| 2 | Lightweight encoders | `src/encoders/` |
| 3 | Geometric Dissonance Module | `src/modules/gds_module.py` |
| 4 | Conflict-aware routing | `src/modules/routing_controller.py` |
| 5 | Explainability engine | `src/modules/explainability_module.py` |
| 6 | 3-stage training strategy | `src/training/training_strategy.py` |
| 7 | Model comparison evaluation | `src/evaluation/evaluator.py` |
| 8 | Efficiency analysis | `src/evaluation/efficiency_analysis.py` |
| 9 | Ablation studies | `src/evaluation/ablation.py` |
| 10 | Patent documentation | `docs/patent_draft_outline.md` |

---

## Novel Contributions (Patent Claims)

1. **Differentiable GDS Module** — explicit geometric dissonance score with learnable α, β weights
2. **Learnable routing threshold τ** — dynamically calibrated during training
3. **Conflict Branch with cross-attention** — specialized sarcasm-sensitive processing
4. **Auto-generated conflict reports** — structured per-inference explainability
5. **3-stage training strategy** — unimodal pretraining → routing training → end-to-end fine-tuning

---

## Design Principles

| Principle | How CGRN Achieves It |
|---|---|
| **Lightweight** | DistilBERT (66M) + MobileNetV3-Small (2.5M) = ~70M total |
| **Modular** | Each component is independently pluggable (GDM, Router, Branches) |
| **Geometric disagreement** | Explicit D score from cosine dissimilarity + magnitude difference |
| **Conflict-controlled routing** | D vs. τ determines architectural path |
| **Explainable outputs** | Per-inference ConflictReport with interpretation string |

---

*Project: CGRN | Conflict-Aware Geometric Routing Network | 2026*
