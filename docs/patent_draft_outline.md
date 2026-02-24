# Patent Draft Outline: Conflict-Aware Geometric Routing Network (CGRN)

## Filing Information (Draft)

| Field | Value |
|---|---|
| **Title** | System and Method for Conflict-Aware Multimodal Sentiment Analysis Using Geometric Dissonance Routing |
| **Inventors** | [To be completed] |
| **Filing Type** | Utility Patent |
| **Technology Domain** | Natural Language Processing / Computer Vision / Machine Learning |
| **CPC Classifications** | G06N 3/04 · G06F 40/30 · G06V 10/82 · G06F 40/56 |

---

## FIELD OF THE INVENTION

The present invention relates to multimodal sentiment analysis systems, and more particularly to a lightweight neural network architecture that computes a geometric dissonance score between modality-specific sentiment representations and uses this score to dynamically route multimodal inputs through specialized inference branches, producing explainable conflict reports for each prediction.

---

## BACKGROUND OF THE INVENTION

### Problem Statement

Multimodal sentiment analysis aims to infer the emotional polarity of content expressed through multiple modalities, such as text and images. Existing systems fall into three broad categories:

**1. Early Fusion Systems** concatenate feature vectors from each modality before classification. These methods treat all multimodal inputs identically, applying the same computational path regardless of whether the modalities agree or disagree in their emotional content. Consequently, when a text expresses positive sentiment while an image conveys negative sentiment — as commonly occurs in sarcastic, ironic, or figurative communication — early fusion systems conflate these contradictory signals without resolving them, producing unreliable predictions.

**2. Late Fusion Systems** produce independent predictions from each modality and combine them through voting or weighted averaging. These approaches discard the rich cross-modal interaction information necessary to resolve contradictory inputs, and lack any mechanism for detecting or leveraging the disagreement structure.

**3. Attention-Based Fusion Systems**, including transformer-based multimodal encoders, apply cross-modal attention mechanisms to model pairwise interactions between modalities. While these systems achieve strong average performance, they: (a) require hundreds of millions of parameters unsuitable for edge deployment; (b) represent cross-modal disagreement only implicitly within attention weights, without any structured quantification; and (c) apply the same computational path to all inputs, regardless of conflict level.

**Key Deficiencies in Prior Art:**
1. No existing system explicitly computes a geometric disagreement score between independently derived modality-specific sentiment vectors.
2. No existing system uses such a geometric score as a dynamic routing signal to dispatch inference to specialized architectural branches.
3. No existing system provides a structured, auto-generated per-inference conflict report with a natural-language interpretation of cross-modal disagreement.
4. No existing lightweight system is specifically optimized for performance on contradictory, sarcastic, and ambiguous multimodal inputs.

---

## SUMMARY OF THE INVENTION

The present invention provides a **Conflict-Aware Geometric Routing Network (CGRN)** comprising:

1. **A pair of lightweight independent modality encoders** that project text and image inputs into a shared normalized sentiment embedding space, producing vectors S_t (text) and S_i (image).

2. **A Geometric Dissonance Module (GDM)** that computes a scalar Geometric Dissonance Score (GDS) from the geometric relationship between S_t and S_i:

$$D = \alpha \cdot \left(1 - \cos(S_t, S_i)\right) + \beta \cdot \left|\|S_t\| - \|S_i\|\right|$$

where α and β are learnable non-negative scalar weights, cos(·,·) denotes cosine similarity, and ‖·‖ denotes the L2 norm. The GDM is fully differentiable and end-to-end trainable.

3. **A Conflict-Aware Routing Controller** that compares D against a learnable threshold τ and dispatches each input to one of two specialized branches:
   - **Normal Fusion Branch**: a concatenation-based MLP for low-dissonance inputs (D < τ)
   - **Conflict Branch**: a cross-modal attention refinement network with optional sarcasm detection head for high-dissonance inputs (D ≥ τ)

4. **An Explainability Engine** that auto-generates a structured per-inference conflict report containing: modality-specific sentiment strengths, geometric dissonance score, routing decision, routing threshold, sarcasm probability, and a natural-language interpretation string.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. System Architecture

The CGRN system comprises five principal components arranged in a sequential-then-parallel pipeline:

```
Text Input → Text Encoder → S_t ─────────┐
                                          ├→ GDS Module → D → Routing Controller
Image Input → Image Encoder → S_i ───────┘                         │
                                                          ┌─────────┴──────────┐
                                                    D < τ │                    │ D ≥ τ
                                              Normal Branch              Conflict Branch
                                                          └─────────┬──────────┘
                                                                     │
                                                           Final Prediction
                                                                     │
                                                          Explainability Engine
                                                                     │
                                                            Conflict Report
```

### 2. Text Encoder

The text encoder comprises a pre-trained transformer language model (DistilBERT or MiniLM), followed by a linear projection layer, layer normalization, and L2 normalization:

- Input: raw text string, tokenized to input_ids and attention_mask
- Backbone: DistilBERT-base-uncased (66M parameters) or MiniLM-L6 (22M parameters)
- Feature extraction: [CLS] token hidden state from final transformer layer
- Projection: Linear(hidden_dim → embed_dim), where embed_dim = 256
- Normalization: LayerNorm followed by L2 normalization
- Output: S_t ∈ ℝ^{embed_dim}, unit-normalized sentiment vector

The text encoder includes an auxiliary classification head for Stage 1 unimodal training but this head is not used during multimodal inference.

### 3. Image Encoder

The image encoder comprises a lightweight convolutional neural network (MobileNetV3-Small), followed by a global average pooling operation, linear projection, layer normalization, and L2 normalization:

- Input: RGB image, resized to 224×224
- Backbone: MobileNetV3-Small (2.5M parameters, pretrained on ImageNet)
- Feature extraction: global average-pooled feature map (576-dimensional)
- Projection: Linear(576 → embed_dim)
- Normalization: LayerNorm followed by L2 normalization
- Output: S_i ∈ ℝ^{embed_dim}, unit-normalized sentiment vector

### 4. Geometric Dissonance Module (Core Novel Contribution)

The Geometric Dissonance Module (GDM) receives S_t and S_i and computes the scalar Geometric Dissonance Score D through the following differentiable operations:

**Step 1 — Cosine Dissimilarity:**
$$C = 1 - \frac{S_t \cdot S_i}{\|S_t\| \|S_i\|}$$

This term captures directional disagreement between the two sentiment vectors. C = 0 indicates perfect directional agreement; C = 2 indicates perfect anti-correlation.

**Step 2 — Magnitude Difference:**
$$M = \left| \|S_t\| - \|S_i\| \right|$$

This term captures disagreement in sentiment intensity (confidence) between modalities. Large M indicates one modality expresses sentiment with substantially greater confidence than the other.

**Step 3 — Weighted Combination:**
$$D = \alpha \cdot C + \beta \cdot M$$

where α = exp(log_α) and β = exp(log_β) are computed via exponentiation of learnable scalar parameters log_α and log_β, ensuring non-negativity.

**Step 4 — Optional Projection Residual:**
When enabled, a third term is added:
$$D = \alpha \cdot C + \beta \cdot M + \gamma \cdot \left\| S_i - \frac{S_i \cdot \hat{S}_t}{\|\hat{S}_t\|^2}\hat{S}_t \right\|$$

where the final term measures the component of S_i orthogonal to S_t.

The GDM also computes and logs:
- Angular separation θ = arccos(cos(S_t, S_i)) in degrees
- Individual magnitudes ‖S_t‖ and ‖S_i‖
- Projection residual

### 5. Conflict-Aware Routing Controller

The Routing Controller maintains a learnable routing threshold τ (initialized to 0.5) and dispatches each sample in a batch based on the comparison D vs. τ:

**Routing Rule:**
```
if D < τ:
    output = NormalFusionBranch(S_t, S_i)
else:
    output = ConflictBranch(S_t, S_i, D)
```

**Normal Fusion Branch:**
- Input: [S_t ‖ S_i] ∈ ℝ^{2·embed_dim}
- Architecture: Linear → GELU → Dropout → Linear → GELU → Dropout → Linear
- Output: logits ∈ ℝ^{num_classes}

**Conflict Branch:**
- Cross-Modal Attention: bidirectional cross-attention between S_t and S_i
  - Text attends to image: t_refined = Attention(Q=S_t, K=S_i, V=S_i)
  - Image attends to text: i_refined = Attention(Q=S_i, K=S_t, V=S_t)
  - Residual connections and layer normalization applied to both
- GDS Conditioning: D is embedded and concatenated with refined representations
- Classifier: MLP over [t_refined ‖ i_refined ‖ embed(D)]
- Optional Sarcasm Head: Binary classifier outputting P(sarcastic | t_refined, i_refined, D)

The routing controller applies **soft routing during training** via gating, and **hard routing during inference** via threshold comparison.

### 6. Training Strategy

**Stage 1 — Unimodal Pre-training:**
Text and image encoders are trained independently on unimodal sentiment classification tasks using cross-entropy loss with label smoothing. This ensures encoders learn meaningful sentiment representations before multimodal training.

**Stage 2 — GDS and Routing Training:**
Encoder backbone weights are frozen. The GDM (α, β), Routing Controller (τ), NormalFusionBranch, and ConflictBranch are trained with the combined loss:

$$\mathcal{L} = \mathcal{L}_{main} + \lambda_u(\mathcal{L}_{text} + \mathcal{L}_{image}) + \lambda_r \mathcal{L}_{routing}$$

where $\mathcal{L}_{routing} = -\mathbb{E}[D | \text{conflict sample}]$ encourages high GDS values for known-conflict samples.

**Stage 3 — End-to-End Fine-tuning:**
All parameters are unfrozen and jointly fine-tuned with layer-wise learning rates (backbone at 10× lower rate than projection and routing layers).

### 7. Explainability Engine

For each inference, the Explainability Engine generates a `ConflictReport` dataclass containing:

| Field | Type | Description |
|---|---|---|
| text_sentiment_label | str | Predicted text sentiment class |
| text_sentiment_strength | float | Confidence of text prediction |
| image_sentiment_label | str | Predicted image sentiment class |
| image_sentiment_strength | float | Confidence of image prediction |
| cosine_similarity | float | cos(S_t, S_i) |
| angular_separation_deg | float | θ in degrees |
| gds_score | float | Computed D value |
| gds_alpha / gds_beta | float | Current learnable weights |
| routing_path | str | "Normal Fusion Branch" or "Conflict Branch" |
| routing_threshold | float | Current τ value |
| sarcasm_probability | float | P(sarcastic) from sarcasm head |
| final_prediction_label | str | Final predicted sentiment |
| final_prediction_confidence | float | Confidence of final prediction |
| interpretation | str | Auto-generated natural-language explanation |

**Example Conflict Report:**
```
Text Sentiment   : Positive (strength=0.82, conf=0.79)
Image Sentiment  : Negative (strength=0.76, conf=0.71)
Cosine Similarity: -0.34   Angular Sep: 109.9°
GDS Score        : 0.64   (α=1.00, β=1.00)
Routing Path     : Conflict Branch (threshold=0.500)
Sarcasm Prob     : 0.73
Interpretation   : High cross-modal disagreement: Text=Positive, Image=Negative.
                   GDS=0.640 ≥ τ=0.500 (opposing sentiment signals).
                   Sarcasm detector activated (p=0.73) → Possible ironic or
                   masked sentiment. Conflict branch applied cross-attention.
Final Prediction : Negative (confidence=0.68)
```

---

## CLAIMS

### Independent Claims

**Claim 1:** A computer-implemented multimodal sentiment analysis system comprising:
- a text encoder configured to generate a first normalized sentiment embedding vector S_t from a text input;
- an image encoder configured to generate a second normalized sentiment embedding vector S_i from an image input;
- a geometric dissonance module configured to compute a scalar geometric dissonance score D from S_t and S_i by computing a weighted combination of (i) a cosine dissimilarity term between S_t and S_i, and (ii) a magnitude difference term between the L2 norms of S_t and S_i, wherein the weights of said combination are learnable parameters; and
- a routing controller configured to route the inputs to a first processing branch when D is below a threshold, and to a second processing branch when D meets or exceeds said threshold.

**Claim 2:** The system of Claim 1, wherein the geometric dissonance score is computed as:

$$D = \alpha \cdot (1 - \cos(S_t, S_i)) + \beta \cdot |\|S_t\| - \|S_i\||$$

where α and β are non-negative learnable scalar parameters constrained by exponentiation.

**Claim 3:** The system of Claim 1, wherein the routing threshold is a learnable parameter updated during training.

**Claim 4:** The system of Claim 1, wherein the second processing branch comprises: a cross-modal attention mechanism that refines S_t and S_i by bidirectional attention, and a sarcasm detection head that classifies the refined representations as sarcastic or non-sarcastic.

**Claim 5:** The system of Claim 1, further comprising an explainability engine that, for each inference, automatically generates a structured conflict report containing at least: the first sentiment embedding's predicted label and confidence, the second sentiment embedding's predicted label and confidence, the geometric dissonance score, the routing path taken, and a natural-language interpretation string.

### Dependent Claims

**Claim 6:** The system of Claim 1, wherein the text encoder comprises a transformer-based language model with at most 70 million parameters, followed by a linear projection layer and L2 normalization.

**Claim 7:** The system of Claim 1, wherein the image encoder comprises a lightweight convolutional neural network with at most 10 million parameters, followed by a global average pooling layer, a linear projection layer, and L2 normalization.

**Claim 8:** The system of Claim 1, wherein the geometric dissonance module further computes an angular separation between S_t and S_i and a projection residual representing the component of S_i orthogonal to S_t.

**Claim 9:** A computer-implemented method for multimodal sentiment analysis comprising: generating independent normalized sentiment embedding vectors from text and image inputs; computing a scalar geometric dissonance score from the geometric relationship between said vectors; comparing the dissonance score against a threshold; routing inputs with dissonance below the threshold to a standard fusion branch; routing inputs with dissonance at or above the threshold to a conflict-resolution branch comprising cross-attention refinement; and generating a structured conflict report for each inference.

**Claim 10:** The method of Claim 9, wherein computing the geometric dissonance score comprises computing a weighted sum of cosine dissimilarity and L2 magnitude difference, with learnable non-negative weights.

---

## BRIEF DESCRIPTION OF THE DRAWINGS

- **FIG. 1** — Block diagram of the full CGRN architecture with all five components labeled.
- **FIG. 2** — Geometric interpretation of the GDS: two sentiment vectors in ℝ^D space showing cosine angle θ and magnitude difference.
- **FIG. 3** — Routing controller decision logic with GDS distribution and threshold τ.
- **FIG. 4** — Conflict Branch architecture: cross-modal attention mechanism diagram.
- **FIG. 5** — Example conflict report output annotated with field descriptions.
- **FIG. 6** — Comparison chart of CGRN vs. baselines on conflict-heavy subset.
- **FIG. 7** — Ablation study results showing contribution of each CGRN component.
- **FIG. 8** — GDS score distribution over test set with routing threshold overlay.
- **FIG. 9** — Parameter count and inference latency comparison chart.
- **FIG. 10** — Training pipeline: 3-stage training strategy diagram.

---

## ABSTRACT

A Conflict-Aware Geometric Routing Network (CGRN) for multimodal sentiment analysis comprising: (1) independent lightweight text and image encoders that project each modality into a shared normalized sentiment embedding space; (2) a Geometric Dissonance Module that computes a scalar geometric dissonance score from the cosine dissimilarity and magnitude difference between the two sentiment vectors, with learnable weighting parameters; (3) a Conflict-Aware Routing Controller that dynamically dispatches inference to a standard fusion branch or a conflict branch based on the dissonance score relative to a learnable threshold; (4) a conflict branch featuring cross-modal attention refinement and a sarcasm detection head, specialized for high-dissonance inputs; and (5) an explainability engine that auto-generates a structured conflict report for each inference, including a natural-language interpretation string. The system achieves measurable performance improvements on sarcastic, contradictory, and ambiguous multimodal inputs while maintaining a parameter budget suitable for edge deployment.

---

## POTENTIAL APPLICATIONS

| Domain | Use Case |
|---|---|
| **Social Media Monitoring** | Detecting sarcasm in meme-captioned images and ironic posts |
| **E-Commerce** | Identifying product reviews where image and text sentiment disagree |
| **Healthcare** | Monitoring patient sentiment when verbal reports conflict with visual affect |
| **Misinformation Detection** | Flagging image-text pairs with semantically contradictory messages |
| **Edge/IoT Deployment** | Lightweight sentiment inference on mobile and embedded devices |
| **Content Moderation** | Identifying disguised negative content through multimodal analysis |

---

*Document status: Draft v1.0 | Project: CGRN | Date: 2026-02-24*  
*This document is prepared for patent drafting purposes and constitutes confidential technical disclosure.*
