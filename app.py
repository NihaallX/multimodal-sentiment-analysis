"""
app.py — CGRN Streamlit Dashboard
==================================

Interactive demo for the Conflict-Aware Geometric Routing Network.

Run:
    streamlit run app.py

Features:
  - Upload image + enter text for real-time inference
  - Live GDS score gauge
  - Routing path visualization (Normal vs Conflict branch)
  - Per-class sentiment probabilities
  - Full conflict report
  - Batch CSV analysis tab
"""

import os
import io
import sys
import json
import time
import random
from pathlib import Path

import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.modules.clip_scorer import CLIPScorer, detect_sarcasm

# ─── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CGRN — Multimodal Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #333355; margin: 0.5rem 0;
    }
    .sentiment-positive { color: #4ade80; font-weight: 700; font-size: 1.4rem; }
    .sentiment-negative { color: #f87171; font-weight: 700; font-size: 1.4rem; }
    .sentiment-neutral  { color: #facc15; font-weight: 700; font-size: 1.4rem; }
    .routing-conflict   { color: #f97316; font-weight: 700; }
    .routing-normal     { color: #60a5fa; font-weight: 700; }
    .gds-high { color: #ef4444; }
    .gds-low  { color: #22c55e; }
    .report-box {
        background: #12121e; border-left: 3px solid #667eea;
        padding: 1rem; border-radius: 0 8px 8px 0;
        font-family: monospace; font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model Loading (cached)
# =============================================================================

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SENTIMENT_LABELS = ["Positive 😊", "Negative 😞", "Neutral 😐"]
SENTIMENT_COLORS = ["sentiment-positive", "sentiment-negative", "sentiment-neutral"]
SENTIMENT_EMOJIS = ["😊", "😞", "😐"]


@st.cache_resource(show_spinner="Loading CLIP scorer...")
def load_clip_scorer(device: str) -> CLIPScorer:
    return CLIPScorer(device=device)


@st.cache_resource(show_spinner="Loading CGRN model...")
def load_model(model_path: str, text_model: str, embed_dim: int):
    from src.models.cgrn_model import CGRNModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGRNModel(
        text_model_name=text_model,
        embed_dim=embed_dim,
        num_classes=3,
    )
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        st.sidebar.success(f"✓ Loaded: {Path(model_path).name}")
    else:
        st.sidebar.warning("⚠ No checkpoint loaded — using random weights (demo only)")
    model.to(device)
    model.eval()
    return model


def run_inference(model, text: str, image: Image.Image, tau_override: float | None = None):
    """Run CGRN inference and return structured result dict."""
    device = next(model.parameters()).device
    # Tokenise
    enc = model.text_encoder.tokenize([text], device=str(device))
    img_tensor = IMAGE_TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(device)

    # Fix D: optionally override learned τ for this inference
    _orig_tau = None
    if tau_override is not None:
        _orig_tau = model.routing_controller.threshold.data.clone()
        model.routing_controller.threshold.data.fill_(tau_override)

    with torch.no_grad():
        out = model(
            enc["input_ids"],
            enc["attention_mask"],
            img_tensor,
            return_reports=True,
        )

    # Restore original τ if we overrode it
    if _orig_tau is not None:
        model.routing_controller.threshold.data.copy_(_orig_tau)

    probs       = torch.softmax(out.final_logits, dim=-1)[0].tolist()
    pred_idx    = int(torch.argmax(out.final_logits, dim=-1).item())
    gds         = float(out.gds_output.gds[0].item())
    tau         = float(model.routing_controller.threshold.item())
    is_conflict = bool(out.routing_output.routing_decisions[0].item())
    report      = out.conflict_reports[0] if out.conflict_reports else None

    text_probs  = torch.softmax(out.text_logits,  dim=-1)[0].tolist()
    image_probs = torch.softmax(out.image_logits, dim=-1)[0].tolist()

    return {
        "pred_idx":    pred_idx,
        "probs":       probs,
        "text_probs":  text_probs,
        "image_probs": image_probs,
        "gds":         gds,
        "tau":         tau,
        "is_conflict": is_conflict,
        "report":      report,
    }


# =============================================================================
# UI Components
# =============================================================================

def render_gds_gauge(gds: float, tau: float):
    """Render a coloured GDS gauge using Streamlit progress."""
    normalized = min(gds / 2.0, 1.0)
    color = "🔴" if gds >= tau else "🟢"
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.markdown("**0**")
    with col2:
        st.progress(normalized)
    with col3:
        st.markdown("**2.0**")
    st.markdown(
        f"<div style='text-align:center; font-size:1.1rem;'>"
        f"{color} GDS = <b>{gds:.4f}</b> &nbsp;|&nbsp; τ = {tau:.4f}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_routing_path(is_conflict: bool, gds: float, tau: float):
    """Visual routing path diagram."""
    col_l, col_m, col_r = st.columns(3)
    with col_l:
        st.markdown("#### 🔤 Text Encoder")
        st.markdown("*S_t ∈ ℝ²⁵⁶*")
    with col_m:
        direction = "⬇️ Conflict Branch" if is_conflict else "⬇️ Normal Fusion"
        tag_color = "#f97316" if is_conflict else "#60a5fa"
        st.markdown("#### 📐 GDS Module")
        st.markdown(f"*D = {gds:.3f}*")
        st.markdown(
            f"<div style='background:{tag_color};color:white;padding:6px 12px;"
            f"border-radius:8px;text-align:center;font-weight:bold;'>"
            f"{direction}</div>",
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown("#### 🖼 Image Encoder")
        st.markdown("*S_i ∈ ℝ²⁵⁶*")


def render_probabilities(probs, label="CGRN"):
    labels = ["Positive", "Negative", "Neutral"]
    colors = ["#4ade80", "#f87171", "#facc15"]
    for i, (lbl, p, c) in enumerate(zip(labels, probs, colors)):
        col1, col2 = st.columns([2, 5])
        with col1:
            st.markdown(f"<span style='color:{c}'>{lbl}</span>", unsafe_allow_html=True)
        with col2:
            st.progress(p)
        st.caption(f"{p*100:.1f}%")


def render_conflict_report(report):
    if report is None:
        st.info("No conflict report available.")
        return
    st.markdown(f"""
<div class="report-box">
<b>Sample ID:</b> {getattr(report, 'sample_id', 'N/A')}<br>
<b>Prediction:</b> {getattr(report, 'final_prediction_label', 'N/A')}<br>
<b>GDS Score:</b> {getattr(report, 'gds_score', 'N/A')}<br>
<b>Conflict:</b> {getattr(report, 'is_conflict', 'N/A')}<br>
<b>Branch:</b> {getattr(report, 'routing_path', 'N/A')}<br>
<b>Confidence:</b> {getattr(report, 'confidence', 'N/A')}<br>
<b>Interpretation:</b><br>{getattr(report, 'interpretation', 'N/A')}
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Demo Examples
# =============================================================================

DEMO_EXAMPLES = [
    {
        "label": "Sarcastic tweet (conflict expected)",
        "text":  "Oh great, another Monday. Just what I always wanted 🙄",
        "note":  "Text sentiment is negative/sarcastic — would conflict with a happy image",
    },
    {
        "label": "Positive review (no conflict)",
        "text":  "This product is absolutely amazing! Best purchase ever.",
        "note":  "Clear positive sentiment — expect Normal Fusion branch",
    },
    {
        "label": "Mixed sentiment (potential conflict)",
        "text":  "The food looked beautiful but tasted terrible.",
        "note":  "Text conflicts with visual appeal — high GDS expected",
    },
    {
        "label": "Negative opinion",
        "text":  "Worst customer service I have ever experienced. Completely disappointed.",
        "note":  "Strong negative sentiment",
    },
]


# =============================================================================
# Main App
# =============================================================================

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")

        model_path = st.text_input(
            "Checkpoint path",
            value="checkpoints_mvsa/cgrn_mvsa_final.pt",
            help="Path to a trained .pt checkpoint file",
        )
        text_model = st.selectbox(
            "Text backbone",
            ["roberta-base", "distilbert-base-uncased", "bert-base-uncased"],
            index=0,
        )
        embed_dim = st.select_slider("Embedding dim", options=[128, 256, 512], value=256)

        # ── τ routing slider ─────────────────────────────────────────
        st.divider()
        st.markdown("## 🔀 Routing Threshold")
        tau_override_val = st.slider(
            "τ threshold",
            min_value=0.10, max_value=1.50, value=0.735, step=0.05,
            help="Routing cutoff. GDS ≥ τ → Conflict branch. GDS < τ → Normal branch. Model learned τ=0.735.",
        )
        st.caption("τ=0.735 = learned value. Lower → more conflict routing. Try 0.30 to force conflict branch.")

        st.divider()
        st.markdown("## 📖 About")
        st.markdown("""
**CGRN** — Conflict-Aware Geometric Routing Network

Novelties:
- 📐 Geometric Dissonance Score (GDS)
- 🔀 Learnable routing threshold τ
- 🔀 Conflict Branch with cross-attention
- 📋 Per-inference conflict reports
        """)
        st.markdown("[GitHub →](https://github.com/NihaallX/multimodal-sentiment-analysis)")

    # ── Load model ──────────────────────────────────────────────────────────
    model = load_model(model_path, text_model, embed_dim)
    actual_device = str(next(model.parameters()).device)
    st.sidebar.caption(f"🖥 Device: **{actual_device}**")

    # ── Load CLIP scorer (always on, lazy-downloads on first use) ───────────
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_scorer = load_clip_scorer(clip_device)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<p class="main-header">🧠 CGRN Multimodal Sentiment Analysis</p>',
                unsafe_allow_html=True)
    st.caption("Conflict-Aware Geometric Routing Network — Patent-Oriented Architecture")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_single, tab_batch, tab_architecture = st.tabs([
        "🔍 Single Inference", "📊 Batch Analysis", "🏗 Architecture"
    ])

    # =========================================================================
    # Tab 1: Single Inference
    # =========================================================================
    with tab_single:
        col_input, col_results = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### 📥 Input")

            # ── Demo examples ──────────────────────────────────────────────
            with st.expander("💡 Load a demo example", expanded=False):
                ex_labels = [e["label"] for e in DEMO_EXAMPLES]
                chosen = st.selectbox("Choose example", ex_labels)
                demo  = next(e for e in DEMO_EXAMPLES if e["label"] == chosen)
                st.info(demo["note"])
                load_demo = st.button("Load this example")

            # ── Text input ─────────────────────────────────────────────────
            default_text = demo["text"] if "load_demo" in st.session_state and st.session_state.load_demo else ""
            text_input = st.text_area(
                "Enter text (tweet, review, caption…)",
                value=demo["text"] if load_demo else "",
                height=120,
                placeholder="Type or paste text here…",
            )

            # ── Image input ────────────────────────────────────────────────
            st.markdown("**Upload image** (or use a random placeholder):")
            uploaded = st.file_uploader(
                "Image", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )

            use_placeholder = st.checkbox("Use placeholder image (random noise)", value=True)

            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded image", use_container_width=True)
                use_placeholder = False
            elif use_placeholder:
                # Generate simple coloured placeholder
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(arr)
                st.image(image, caption="Random placeholder image", use_container_width=True)
            else:
                image = None

            analyze = st.button("🚀 Analyze", type="primary", use_container_width=True)

        # ── Results panel ─────────────────────────────────────────────────────
        with col_results:
            st.markdown("### 📤 Results")

            if analyze and text_input.strip() and image is not None:
                # Run sarcasm detector before inference (Fix C)
                sarcasm_info = detect_sarcasm(text_input)

                with st.spinner("Running CGRN inference…"):
                    t0 = time.perf_counter()
                    result = run_inference(
                        model, text_input, image,
                        tau_override=tau_override_val,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000

                # Run CLIP in parallel (shown after CGRN results)
                clip_result = None
                if clip_scorer is not None:
                    with st.spinner("Running CLIP analysis…"):
                        try:
                            clip_alignment  = clip_scorer.score_alignment(text_input, image)
                            clip_img_probs  = clip_scorer.score_image_sentiment(image)
                            clip_result = {
                                "alignment":  clip_alignment,
                                "clip_gds":   1.0 - clip_alignment,  # semantic conflict score
                                "img_probs":  clip_img_probs,
                            }
                        except Exception as e:
                            st.warning(f"CLIP scorer error: {e}")

                pred   = result["pred_idx"]
                label  = SENTIMENT_LABELS[pred]
                color  = SENTIMENT_COLORS[pred]

                # ── Top metrics row ───────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", SENTIMENT_EMOJIS[pred] + " " +
                          ["Positive", "Negative", "Neutral"][pred])
                m2.metric("Confidence", f"{max(result['probs'])*100:.1f}%")
                m3.metric("Latency", f"{elapsed:.0f} ms")

                st.divider()

                # ── GDS gauge ─────────────────────────────────────────────
                st.markdown("#### 📐 Geometric Dissonance Score (GDS)")
                render_gds_gauge(result["gds"], result["tau"])

                # ── Routing decision ─────────────────────────────────────
                st.markdown("#### 🔀 Routing Decision")
                if result["is_conflict"]:
                    st.markdown(
                        "🟠 **CONFLICT BRANCH** — Cross-modal dissonance detected. "
                        "Routed to sarcasm-sensitive conflict branch.",
                        help="GDS ≥ τ triggered the conflict branch"
                    )
                else:
                    st.markdown(
                        "🔵 **NORMAL FUSION** — Modalities agree. "
                        "Standard concatenation + MLP fusion applied."
                    )
                render_routing_path(result["is_conflict"], result["gds"], result["tau"])

                st.divider()

                # ── Probability bars ──────────────────────────────────────
                st.markdown("#### 📊 Sentiment Probabilities")
                tabs_prob = st.tabs(["🧠 CGRN", "📝 Text-only", "🖼 Image-only"])
                with tabs_prob[0]:
                    render_probabilities(result["probs"], "CGRN")
                with tabs_prob[1]:
                    render_probabilities(result["text_probs"], "Text")
                with tabs_prob[2]:
                    render_probabilities(result["image_probs"], "Image")

                # ── Sarcasm indicator (Fix C) ─────────────────────────────
                if sarcasm_info["likely_sarcasm"]:
                    with st.expander("⚠️ Sarcasm Signals Detected", expanded=True):
                        st.warning(
                            f"**Sarcasm score: {sarcasm_info['score']}** — "
                            "rule-based detector flagged this text. "
                            "CGRN may underestimate negative sentiment for sarcastic posts."
                        )
                        for sig in sarcasm_info["signals"]:
                            st.markdown(f"  - {sig}")

                # ── CLIP augmented results ────────────────────────────────
                if clip_result:
                    with st.expander("🖼 CLIP Supplementary Analysis", expanded=True):
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**Semantic Alignment (CLIP)**")
                            alignment = clip_result["alignment"]
                            clip_gds  = clip_result["clip_gds"]
                            align_color = "🟢" if alignment > 0.25 else ("🟠" if alignment > 0.10 else "🔴")
                            st.markdown(
                                f"{align_color} CLIP similarity = **{alignment:.4f}**  "
                                f"| CLIP-GDS = **{clip_gds:.4f}**"
                            )
                            st.caption(
                                "CLIP similarity: text & image in the *same* semantic space. "
                                "< 0.10 = strong conflict, 0.10–0.25 = mild, > 0.25 = agree."
                            )
                            # Conflict override hint
                            if clip_gds > 0.80 and not result["is_conflict"]:
                                st.error(
                                    "💡 **CLIP suggests conflict** (CLIP-GDS > 0.80) but CGRN routed "
                                    "to Normal branch. Try lowering τ in the sidebar to < "
                                    f"{result['gds']:.2f} to force conflict routing."
                                )

                        with cc2:
                            st.markdown("**CLIP Zero-Shot Image Sentiment**")
                            ip = clip_result["img_probs"]
                            st.markdown(f"😊 Positive: **{ip['positive']*100:.1f}%**")
                            st.markdown(f"😞 Negative: **{ip['negative']*100:.1f}%**")
                            st.markdown(f"😐 Neutral:  **{ip['neutral']*100:.1f}%**")
                            clip_img_pred = max(ip, key=ip.get)
                            cgrn_text_pred = ["positive", "negative", "neutral"][result["pred_idx"]]
                            if clip_img_pred != cgrn_text_pred:
                                st.warning(
                                    f"CLIP image says **{clip_img_pred}** but "
                                    f"RoBERTa text says **{cgrn_text_pred}** — "
                                    "cross-modal disagreement detected."
                                )

                # ── Conflict report ───────────────────────────────────────
                with st.expander("📋 Conflict Report (Patent Feature)", expanded=True):
                    render_conflict_report(result["report"])

            elif analyze:
                st.warning("Please provide both text and an image before analyzing.")
            else:
                st.markdown("""
<div style="text-align:center; color:#555; padding: 4rem 1rem;">
    <div style="font-size: 3rem;">🧠</div>
    <p>Enter text and upload an image, then click <b>Analyze</b>.</p>
    <p style="font-size:0.85rem;">The CGRN will compute the Geometric Dissonance Score and route
    through the appropriate sentiment branch.</p>
</div>
""", unsafe_allow_html=True)

    # =========================================================================
    # Tab 2: Batch Analysis
    # =========================================================================
    with tab_batch:
        st.markdown("### 📊 Batch Text Analysis")
        st.caption("Analyze multiple texts at once (uses placeholder images).")

        batch_text = st.text_area(
            "Enter one text per line:",
            height=200,
            placeholder="This product is amazing!\nWorst experience ever.\nThe food was okay I guess.",
        )

        if st.button("🚀 Analyze Batch", type="primary"):
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            if not lines:
                st.warning("Enter at least one line of text.")
            else:
                placeholder_img = Image.fromarray(
                    np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                )
                results_data = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    r = run_inference(model, line, placeholder_img)
                    results_data.append({
                        "Text":       line[:80] + ("…" if len(line) > 80 else ""),
                        "Sentiment":  ["Positive", "Negative", "Neutral"][r["pred_idx"]],
                        "Confidence": f"{max(r['probs'])*100:.1f}%",
                        "GDS":        f"{r['gds']:.4f}",
                        "Branch":     "Conflict 🟠" if r["is_conflict"] else "Normal 🔵",
                    })
                    progress.progress((i + 1) / len(lines))

                st.success(f"Analyzed {len(results_data)} samples")
                st.dataframe(results_data, use_container_width=True)

                # Summary stats
                sentiments = [d["Sentiment"] for d in results_data]
                n_conflict = sum(1 for d in results_data if "Conflict" in d["Branch"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Positive", sentiments.count("Positive"))
                c2.metric("Negative", sentiments.count("Negative"))
                c3.metric("Neutral",  sentiments.count("Neutral"))
                c4.metric("Conflicts Detected", n_conflict)

                # Download
                import io as _io
                csv_buf = _io.StringIO()
                import csv as _csv
                writer = _csv.DictWriter(csv_buf, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_buf.getvalue(),
                    file_name="cgrn_batch_results.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # Tab 3: Architecture
    # =========================================================================
    with tab_architecture:
        st.markdown("### 🏗 CGRN Architecture")

        col_diag, col_params = st.columns([3, 2])
        with col_diag:
            st.code("""
Text Input → [RoBERTa + Proj]   → S_t ∈ ℝ²⁵⁶ ─────────────┐
                                                               │
Image Input → [MobileNetV3 + Proj] → S_i ∈ ℝ²⁵⁶ ────────────┤
                                                               ↓
                               ┌─── GDS Module ─────────────────────────┐
                               │  D = α·(1−cos(S_t,S_i)) + β·|‖S_t‖−‖S_i‖|  │
                               │  α, β: learnable non-negative params    │
                               └────────────────────────────────────────┘
                                              │ D (scalar)
                                              ↓
                               ┌─── Routing Controller ─────────────────┐
                               │  D < τ  →  Normal Fusion Branch        │
                               │  D ≥ τ  →  Conflict Branch             │
                               │  τ: learnable threshold                │
                               └────────────────────────────────────────┘
                                    /                        \\
                    ┌──────────────-┐              ┌──────────────────┐
                    │ Normal Fusion │              │ Conflict Branch  │
                    │ Concat + MLP  │              │ CrossAttn + Head │
                    │               │              │ Sarcasm-aware    │
                    └──────────────-┘              └──────────────────┘
                                    \\                        /
                                     └────────── ┬ ──────────┘
                                                 ↓
                                        Final Prediction
                                                 ↓
                                    ┌─────────────────────────┐
                                    │  Explainability Engine  │
                                    │  → ConflictReport       │
                                    └─────────────────────────┘
""", language="text")

        with col_params:
            st.markdown("#### 📊 Model Parameters")
            n_total  = sum(p.numel() for p in model.parameters())
            n_text   = sum(p.numel() for p in model.text_encoder.parameters())
            n_image  = sum(p.numel() for p in model.image_encoder.parameters())
            n_gds    = sum(p.numel() for p in model.gds_module.parameters())
            n_router = sum(p.numel() for p in model.routing_controller.parameters())

            st.metric("Total params",     f"{n_total/1e6:.1f}M")
            st.metric("Text Encoder",     f"{n_text/1e6:.1f}M  ({text_model.split('-')[0].capitalize()})")
            st.metric("Image Encoder",    f"{n_image/1e6:.1f}M  (MobileNetV3)")
            st.metric("GDS Module",       f"{n_gds}")
            st.metric("Routing + Fusion", f"{n_router/1e3:.0f}K")

            st.divider()
            st.markdown("#### 🏅 Patent Claims")
            claims = [
                "Differentiable GDS with learnable α, β",
                "Learnable routing threshold τ",
                "Conflict branch with cross-attention",
                "Auto-generated ConflictReport",
                "3-stage training strategy",
            ]
            for i, c in enumerate(claims, 1):
                st.markdown(f"{i}. {c}")

        # Benchmark table
        st.divider()
        st.markdown("#### 📈 Benchmark Results — MVSA-Multiple (2,940 test samples)")
        benchmark = {
            "Model":           ["DistilBERT v1", "RoBERTa v2", "**RoBERTa v3 (ours)**"],
            "Backbone":        ["DistilBERT + MobileNetV3", "RoBERTa-base + MobileNetV3", "**RoBERTa-base + MobileNetV3**"],
            "Accuracy":        ["63.4%", "62.9%", "**61.8%**"],
            "Macro F1":        ["0.552", "0.558", "**0.552**"],
            "Conflict F1":     ["0.471", "0.477", "**0.483**"],
            "τ learned":       ["0.587", "0.566", "**0.735**"],
            "Harmonic→Normal": ["n/a", "5.2%", "**29.9%**"],
        }
        st.table(benchmark)
        st.caption("v3 routing fix: τ_margin 0.35 + balance loss forces GDS separation. Conflict→Conflict: 78.1%, Harmonic→Normal: 29.9% (v2: 5.2%). Non-conflict Acc=65.2%.")


if __name__ == "__main__":
    main()
