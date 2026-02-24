# Conflict-Aware Geometric Routing Network (CGRN)

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
