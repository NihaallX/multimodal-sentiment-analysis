# Problem Definition: Conflict-Aware Geometric Routing Network (CGRN)

## 1. Technical Field

This work addresses **multimodal sentiment analysis** — the task of determining the emotional polarity (positive, negative, neutral) of content that spans multiple modalities, specifically **text and images**. The invention introduces a patent-oriented architecture that explicitly models **cross-modal geometric disagreement** and uses it as a first-class signal to route inference through specialized computational branches.

---

## 2. Literature Summary of Existing Fusion Methods

### 2.1 Early Fusion (Feature-Level Fusion)
Early fusion concatenates raw or embedded feature vectors from each modality before any classification step.

- **Representative works**: MultiSentiment (Poria et al., 2015), Visual Sentiment (You et al., 2015)
- **Mechanism**: `[v_text || v_image] → MLP → label`
- **Limitation**: Treats all examples uniformly regardless of cross-modal agreement; dominant modality suppresses weaker signals.

### 2.2 Late Fusion (Decision-Level Fusion)
Separate unimodal classifiers produce independent predictions; final output is a weighted or voted combination.

- **Representative works**: SentiBank + LDA fusion (Borth et al., 2013)
- **Mechanism**: `p_text + λ * p_image → label`
- **Limitation**: No joint representation learning; disagreement between modalities is discarded rather than leveraged.

### 2.3 Attention-Based Fusion
Cross-modal attention mechanisms allow one modality to attend to relevant features of the other.

- **Representative works**: MISA (Hazarika et al., 2020), MAG-BERT (Rahman et al., 2020), ViLBERT (Lu et al., 2019), UNITER (Chen et al., 2020)
- **Mechanism**: `Attention(Q_text, K_image, V_image)` — bidirectional cross-modal attention
- **Limitation**: High parameter count; attention does **not** explicitly measure or exploit disagreement magnitude; sarcastic/contradictory samples are not handle distinctly.

### 2.4 Tensor Fusion Networks
Outer-product of unimodal representations captures trimodal (text × image × audio) interactions.

- **Representative works**: TFN (Zadeh et al., 2017), LMF (Liu et al., 2018)
- **Mechanism**: `Z = v_text ⊗ v_image → MLP`
- **Limitation**: Exponential parameter growth; no routing based on conflict.

### 2.5 Graph-Based Fusion
Sentiment entities and their inter-modal relations are modeled as graph nodes and edges.

- **Representative works**: MMGCN (Wei et al., 2019), UniMSE (Hu et al., 2022)
- **Mechanism**: GCN with heterogeneous modal nodes
- **Limitation**: Static graph topology; disagreement is implicit in edge weights rather than explicitly computed.

### 2.6 Transformer-Based Unified Encoders
Large pretrained models (e.g., CLIP, FLAVA, BridgeTower) jointly encode text and image into a shared space.

- **Representative works**: CLIP (Radford et al., 2021), FLAVA (Singh et al., 2022), BridgeTower (Xu et al., 2023)
- **Mechanism**: Shared tokenizer + single transformer backbone
- **Limitation**: Expensive inference (100M–1B parameters); no lightweight deployment path; disagreement between modalities is buried inside attention layers with no explicit representation or routing.

---

## 3. Identified Weaknesses in Current Approaches

| Weakness | Impact |
|---|---|
| **Static fusion** | All inputs are fused identically regardless of cross-modal agreement or disagreement; high-conflict samples receive the same treatment as harmonious ones |
| **Dominance of stronger modality** | In contradictory cases (e.g., sarcastically captioned images), one modality overwhelms the other without correction |
| **No explicit geometric disagreement modeling** | Cross-modal sentiment vectors are never compared geometrically; disagreement is latent and uninterpretable |
| **No conflict-aware routing** | No mechanism routes contradictory inputs to specialized processing branches capable of resolving or leveraging the disagreement |
| **Lack of explainability in conflict** | State-of-the-art models cannot produce structured "conflict reports" explaining why a prediction was made in a contradictory case |
| **Over-parameterization** | Unified large-model approaches are unsuitable for edge inference, embedded systems, or latency-sensitive deployments |

---

## 4. Formal Problem Statement

> **"Existing multimodal sentiment systems lack structured geometric disagreement modeling and dynamic routing based on modality conflict. Current fusion architectures treat cross-modal interactions uniformly, failing to detect, measure, or exploit the geometric dissonance between sentiment representations derived from independent modalities. This leads to degraded performance on sarcastic, ironic, and contradictory multimodal inputs — precisely the cases where sentiment understanding is most challenging and most consequential. Furthermore, the absence of conflict-aware routing prevents models from applying specialized computational paths to disagreement-heavy inputs, and the absence of an explicit dissonance score precludes interpretable conflict reporting."**

---

## 5. Proposed Solution Overview

The **Conflict-Aware Geometric Routing Network (CGRN)** addresses all identified weaknesses through four coordinated innovations:

1. **Independent Lightweight Encoders** — DistilBERT-based text encoder and MobileNetV3-based image encoder project both modalities into a shared normalized sentiment embedding space.

2. **Geometric Dissonance Score (GDS)** — A differentiable module that explicitly computes the geometric disagreement between sentiment vectors using cosine dissimilarity and magnitude difference, parameterized by learnable weights α and β:

$$D = \alpha \left(1 - \cos(S_t, S_i)\right) + \beta \left| \|S_t\| - \|S_i\| \right|$$

3. **Conflict-Aware Routing Controller** — A dynamic routing mechanism that dispatches inputs to a Normal Fusion Branch (for low-GDS inputs) or a Conflict Branch with cross-attention refinement and sarcasm sensitivity (for high-GDS inputs).

4. **Explainable Conflict Reporting Layer** — Structured per-inference output including text sentiment strength, image sentiment strength, GDS value, routing path taken, interpretation string, and final prediction.

---

## 6. Target Applications

- Social media sentiment monitoring (memes, sarcastically captioned images)
- Product review analysis (image vs. text rating disagreement)
- Clinical mental health monitoring (facial expression vs. verbal report)
- Misinformation detection (image-text semantic conflict)
- Edge-device sentiment inference (IoT, mobile)

---

## 7. Key Novel Claims (Patent Relevance)

1. **Novel**: Explicit geometric dissonance score computed from independent sentiment vectors in a shared embedding space.
2. **Novel**: Learnable α/β weighting of cosine dissimilarity and magnitude difference in a differentiable module.
3. **Novel**: Dynamic architecture routing conditioned on a scalar dissonance score.
4. **Novel**: Conflict Branch with cross-attention refinement specialized for contradictory/sarcastic inputs.
5. **Novel**: Auto-generated per-inference conflict report with structured interpretation string.

---

*Document version: 1.0 | Project: CGRN | Date: 2026-02-24*
