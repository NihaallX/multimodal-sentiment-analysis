# CGRN Architecture Diagram

```mermaid
flowchart TD
    TEXT["📝 Text Input\n(raw string)"] --> TE["**Text Encoder**\n(DistilBERT/MiniLM)\n↓ Linear Projection\n↓ LayerNorm + L2-norm\nS_t ∈ ℝ^256"]
    IMAGE["🖼️ Image Input\n(224×224 RGB)"] --> IE["**Image Encoder**\n(MobileNetV3-Small)\n↓ Global AvgPool\n↓ Linear Projection\n↓ LayerNorm + L2-norm\nS_i ∈ ℝ^256"]

    TE --> GDS["**Geometric Dissonance Module** 🔬\nD = α·(1−cos(S_t,S_i)) + β·|‖S_t‖−‖S_i‖|\nα, β learnable\n→ GDS scalar D ∈ ℝ"]
    IE --> GDS

    TE --> RC
    IE --> RC
    GDS --> RC["**Routing Controller** 🔀\nGDS < τ → Normal Branch\nGDS ≥ τ → Conflict Branch\n(τ learnable)"]

    RC -->|GDS < τ| NF["**Normal Fusion Branch**\nConcat(S_t, S_i)\n→ MLP\n→ logits"]
    RC -->|GDS ≥ τ| CB["**Conflict Branch** ⚡\nCross-Attention(S_t ↔ S_i)\n+ GDS conditioning\n→ MLP → logits\n+ Sarcasm Head"]

    NF --> OUT["**Final Prediction**\n(3-class: Neg/Neu/Pos)"]
    CB --> OUT

    OUT --> EX["**Explainability Engine** 📋\nText Sentiment Strength\nImage Sentiment Strength\nGDS Score\nRouting Path\nInterpretation String\nFinal Prediction"]

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