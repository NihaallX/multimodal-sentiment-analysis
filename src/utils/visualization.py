"""
visualization.py — Diagrams, Plots, and Architecture Visualizations
=====================================================================

Provides:
  - plot_gds_distribution(): histogram of GDS scores with routing threshold
  - plot_routing_distribution(): pie/bar chart of routing decisions
  - plot_training_history(): multi-panel training curves
  - plot_embedding_space(): 2D PCA/TSNE of sentiment embeddings
  - plot_comparison_table(): heatmap of model comparison
  - generate_architecture_diagram(): Mermaid-based block diagram text
"""

import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Safe import of matplotlib/seaborn
# =============================================================================

def _check_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not installed. Install with: pip install matplotlib")
        return None


def _check_sklearn():
    try:
        from sklearn.decomposition import PCA
        return PCA
    except ImportError:
        logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")
        return None


# =============================================================================
# GDS Distribution Plot
# =============================================================================

def plot_gds_distribution(
    gds_scores:   List[float],
    threshold:    float,
    conflict_mask: Optional[List[bool]] = None,
    output_path:  str = "logs/gds_distribution.png",
    title:        str = "GDS Score Distribution",
):
    """
    Plot histogram of GDS scores with routing threshold line.
    If conflict_mask is provided, overlay conflict vs. normal samples.
    """
    plt = _check_matplotlib()
    if plt is None:
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))

    if conflict_mask is not None:
        conflict_gds = [g for g, c in zip(gds_scores, conflict_mask) if c]
        normal_gds   = [g for g, c in zip(gds_scores, conflict_mask) if not c]
        ax.hist(normal_gds,   bins=40, alpha=0.6, color="#3498db", label="Normal samples")
        ax.hist(conflict_gds, bins=40, alpha=0.6, color="#e74c3c", label="Conflict samples")
    else:
        ax.hist(gds_scores, bins=40, alpha=0.7, color="#2ecc71", label="All samples")

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Routing threshold τ={threshold:.3f}")
    ax.axvspan(threshold, max(gds_scores) * 1.05, alpha=0.05, color="red",
               label="Conflict Branch zone")

    ax.set_xlabel("Geometric Dissonance Score (GDS)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"GDS distribution plot saved → {output_path}")


# =============================================================================
# Routing Distribution Plot
# =============================================================================

def plot_routing_distribution(
    n_normal:    int,
    n_conflict:  int,
    threshold:   float,
    output_path: str = "logs/routing_distribution.png",
):
    plt = _check_matplotlib()
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Pie chart
    ax1.pie(
        [n_normal, n_conflict],
        labels=[f"Normal Branch\n({n_normal})", f"Conflict Branch\n({n_conflict})"],
        colors=["#3498db", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax1.set_title(f"Routing Distribution\n(τ={threshold:.3f})", fontweight="bold")

    # Bar chart
    ax2.bar(
        ["Normal Branch", "Conflict Branch"],
        [n_normal, n_conflict],
        color=["#3498db", "#e74c3c"],
        edgecolor="white",
        linewidth=2,
    )
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Branch Utilization")
    ax2.grid(True, axis="y", alpha=0.3)

    for bar in ax2.patches:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{int(bar.get_height())}",
            ha="center", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Routing distribution plot saved → {output_path}")


# =============================================================================
# Training History Plot
# =============================================================================

def plot_training_history(
    history_path: str,
    output_path:  str = "logs/training_history.png",
):
    """
    Plots multi-panel training curves from a history JSON file.
    """
    plt = _check_matplotlib()
    if plt is None:
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, (key, values) in enumerate(history.items()):
        ax = axes[i]
        ax.plot(range(1, len(values) + 1), values, color=colors[i % len(colors)],
                linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.set_title(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.suptitle("CGRN Training History", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved → {output_path}")


# =============================================================================
# Embedding Space Visualization
# =============================================================================

def plot_embedding_space(
    text_embeddings,           # numpy [N, D]
    image_embeddings,          # numpy [N, D]
    labels,                    # numpy [N]
    conflict_mask=None,        # numpy [N] bool
    output_path: str = "logs/embedding_space.png",
    method: str = "pca",       # 'pca' or 'tsne'
):
    """
    2D visualization of text and image sentiment embeddings using PCA or t-SNE.
    """
    plt = _check_matplotlib()
    if plt is None:
        return

    import numpy as np

    all_embs = np.concatenate([text_embeddings, image_embeddings], axis=0)
    all_types = (["text"] * len(text_embeddings) +
                 ["image"] * len(image_embeddings))
    all_labels = np.concatenate([labels, labels])

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            reduced = TSNE(n_components=2, random_state=42).fit_transform(all_embs)
        except ImportError:
            method = "pca"

    if method == "pca":
        PCA = _check_sklearn()
        if PCA is None:
            return
        reduced = PCA(n_components=2).fit_transform(all_embs)

    N = len(text_embeddings)
    text_2d  = reduced[:N]
    image_2d = reduced[N:]

    fig, ax = plt.subplots(figsize=(10, 8))
    LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
    COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

    for lbl in [0, 1, 2]:
        mask = labels == lbl
        ax.scatter(
            text_2d[mask, 0], text_2d[mask, 1],
            c=COLORS[lbl], alpha=0.6, s=40, marker="o",
            label=f"Text-{LABEL_NAMES[lbl]}",
        )
        ax.scatter(
            image_2d[mask, 0], image_2d[mask, 1],
            c=COLORS[lbl], alpha=0.4, s=40, marker="^",
            label=f"Image-{LABEL_NAMES[lbl]}",
        )

    ax.set_xlabel(f"{method.upper()} Dim 1")
    ax.set_ylabel(f"{method.upper()} Dim 2")
    ax.set_title(
        "Sentiment Embedding Space (● Text | ▲ Image)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Embedding space plot saved → {output_path}")


# =============================================================================
# Architecture Diagram (Mermaid text)
# =============================================================================

def generate_architecture_diagram(output_path: str = "docs/architecture_diagram.md"):
    """
    Generates a Mermaid block diagram for the CGRN architecture.
    Save as Markdown for rendering in GitHub, VS Code, or Notion.
    """
    diagram = """
# CGRN Architecture Diagram

```mermaid
flowchart TD
    TEXT["📝 Text Input\\n(raw string)"] --> TE["**Text Encoder**\\n(DistilBERT/MiniLM)\\n↓ Linear Projection\\n↓ LayerNorm + L2-norm\\nS_t ∈ ℝ^256"]
    IMAGE["🖼️ Image Input\\n(224×224 RGB)"] --> IE["**Image Encoder**\\n(MobileNetV3-Small)\\n↓ Global AvgPool\\n↓ Linear Projection\\n↓ LayerNorm + L2-norm\\nS_i ∈ ℝ^256"]

    TE --> GDS["**Geometric Dissonance Module** 🔬\\nD = α·(1−cos(S_t,S_i)) + β·|‖S_t‖−‖S_i‖|\\nα, β learnable\\n→ GDS scalar D ∈ ℝ"]
    IE --> GDS

    TE --> RC
    IE --> RC
    GDS --> RC["**Routing Controller** 🔀\\nGDS < τ → Normal Branch\\nGDS ≥ τ → Conflict Branch\\n(τ learnable)"]

    RC -->|GDS < τ| NF["**Normal Fusion Branch**\\nConcat(S_t, S_i)\\n→ MLP\\n→ logits"]
    RC -->|GDS ≥ τ| CB["**Conflict Branch** ⚡\\nCross-Attention(S_t ↔ S_i)\\n+ GDS conditioning\\n→ MLP → logits\\n+ Sarcasm Head"]

    NF --> OUT["**Final Prediction**\\n(3-class: Neg/Neu/Pos)"]
    CB --> OUT

    OUT --> EX["**Explainability Engine** 📋\\nText Sentiment Strength\\nImage Sentiment Strength\\nGDS Score\\nRouting Path\\nInterpretation String\\nFinal Prediction"]

    style GDS fill:#f39c12,color:#000
    style RC fill:#3498db,color:#fff
    style CB fill:#e74c3c,color:#fff
    style NF fill:#2ecc71,color:#000
    style EX fill:#9b59b6,color:#fff
```

## Component Descriptions

| Component | Role | Patent Novelty |
|---|---|---|
| **Text Encoder** | DistilBERT backbone → normalized sentiment vector S_t | Independent modality encoding |
| **Image Encoder** | MobileNetV3 backbone → normalized sentiment vector S_i | Lightweight visual sentiment |
| **GDS Module** | Computes geometric dissonance D from S_t, S_i | **Core novel contribution** |
| **Routing Controller** | Dispatches to specialized branch based on D vs. τ | **Novel conflict-aware routing** |
| **Normal Fusion Branch** | Concat + MLP for harmonious samples | Standard path |
| **Conflict Branch** | Cross-attention refinement + sarcasm head for high-GDS samples | **Novel conflict resolution** |
| **Explainability Engine** | Auto-generates structured conflict reports | **Novel interpretability method** |
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(diagram.strip())
    logger.info(f"Architecture diagram saved → {output_path}")
