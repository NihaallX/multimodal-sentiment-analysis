"""
explainability_module.py — Explainable Conflict Reporting Layer
================================================================

For every inference, this module auto-generates a structured conflict
report containing:

  - Text sentiment strength and predicted label
  - Image sentiment strength and predicted label
  - Geometric Dissonance Score (GDS)
  - Routing decision (Normal vs Conflict Branch)
  - Threshold used at routing
  - Final prediction with confidence
  - Interpretation string (human-readable explanation)
  - Optional sarcasm probability

Example Output
--------------
  ┌─────────────────────────────────────────────────────────────────┐
  │ CGRN Conflict Report                                            │
  ├─────────────────────────────────────────────────────────────────┤
  │ Text Sentiment   :  Positive (strength=0.82, conf=0.79)         │
  │ Image Sentiment  :  Negative (strength=0.76, conf=0.71)         │
  │ Cosine Similarity:  -0.34   Angular Sep: 109.9°                 │
  │ GDS Score        :  0.64   (α=1.00, β=1.00)                    │
  │ Routing Path     :  Conflict Branch  (threshold=0.50)           │
  │ Sarcasm Prob     :  0.73                                        │
  │ Interpretation   :  High cross-modal disagreement               │
  │                     → Possible sarcasm or masked sentiment.     │
  │ Final Prediction :  Negative (confidence=0.68)                  │
  └─────────────────────────────────────────────────────────────────┘

Patent Claim Supported:
  "A per-inference explainability layer that generates structured
   conflict reports including modality-specific sentiment strengths,
   geometric dissonance score, routing path, and an auto-generated
   natural-language interpretation string describing the cross-modal
   relationship."
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .gds_module import GDSOutput
from .routing_controller import RoutingOutput


# =============================================================================
# Label mappings
# =============================================================================

SENTIMENT_LABELS  = {0: "Positive", 1: "Negative", 2: "Neutral"}
SENTIMENT_POLARITY = {0: 1.0, 1: -1.0, 2: 0.0}


# =============================================================================
# Conflict Report dataclass
# =============================================================================

@dataclass
class ConflictReport:
    """
    Structured per-sample conflict report.
    """
    # --- Text modality -------------------------------------------------------
    text_sentiment_label: str
    text_sentiment_strength: float
    text_sentiment_confidence: float
    text_label_idx: int

    # --- Image modality ------------------------------------------------------
    image_sentiment_label: str
    image_sentiment_strength: float
    image_sentiment_confidence: float
    image_label_idx: int

    # --- Geometric quantities ------------------------------------------------
    cosine_similarity: float
    angular_separation_deg: float
    magnitude_text: float
    magnitude_image: float
    magnitude_difference: float
    projection_residual: float

    # --- GDS -----------------------------------------------------------------
    gds_score: float
    gds_alpha: float
    gds_beta: float

    # --- Routing -------------------------------------------------------------
    routing_path: str          # "Normal Fusion Branch" or "Conflict Branch"
    routing_threshold: float
    is_conflict: bool

    # --- Sarcasm (optional) --------------------------------------------------
    sarcasm_probability: Optional[float]

    # --- Final prediction ----------------------------------------------------
    final_prediction_label: str
    final_prediction_idx: int
    final_prediction_confidence: float

    # --- Interpretation ------------------------------------------------------
    interpretation: str

    # --- Raw tensors (optional storage) --------------------------------------
    raw: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Human-readable formatted report."""
        w = 65
        sep = "─" * w

        lines = [
            f"┌{sep}┐",
            f"│{'CGRN Conflict Report':^{w}}│",
            f"├{sep}┤",
            f"│ Text Sentiment   : {self.text_sentiment_label:8s} "
            f"(strength={self.text_sentiment_strength:.2f}, "
            f"conf={self.text_sentiment_confidence:.2f}){'':<5}│",
            f"│ Image Sentiment  : {self.image_sentiment_label:8s} "
            f"(strength={self.image_sentiment_strength:.2f}, "
            f"conf={self.image_sentiment_confidence:.2f}){'':<5}│",
            f"│ Cosine Similarity: {self.cosine_similarity:+.4f}   "
            f"Angular Sep: {self.angular_separation_deg:.1f}°{'':<16}│",
            f"│ GDS Score        : {self.gds_score:.4f}   "
            f"(α={self.gds_alpha:.2f}, β={self.gds_beta:.2f}){'':<20}│",
            f"│ Routing Path     : {self.routing_path:<30}"
            f"(τ={self.routing_threshold:.3f}){'':<1}│",
        ]
        if self.sarcasm_probability is not None:
            lines.append(
                f"│ Sarcasm Prob     : {self.sarcasm_probability:.4f}{'':<39}│"
            )
        lines += [
            f"│ Interpretation   : {self.interpretation:<{w-21}}│",
            f"│ Final Prediction : {self.final_prediction_label:8s} "
            f"(confidence={self.final_prediction_confidence:.2f}){'':<10}│",
            f"└{sep}┘",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_sentiment": self.text_sentiment_label,
            "text_strength":  self.text_sentiment_strength,
            "text_conf":      self.text_sentiment_confidence,
            "image_sentiment": self.image_sentiment_label,
            "image_strength":  self.image_sentiment_strength,
            "image_conf":      self.image_sentiment_confidence,
            "cosine_similarity": self.cosine_similarity,
            "angular_sep_deg":   self.angular_separation_deg,
            "gds_score":         self.gds_score,
            "routing_path":      self.routing_path,
            "routing_threshold": self.routing_threshold,
            "is_conflict":       self.is_conflict,
            "sarcasm_prob":      self.sarcasm_probability,
            "final_prediction":  self.final_prediction_label,
            "final_confidence":  self.final_prediction_confidence,
            "interpretation":    self.interpretation,
        }


# =============================================================================
# Interpretation Engine
# =============================================================================

def _generate_interpretation(
    text_label: int,
    image_label: int,
    final_label: int,
    gds: float,
    is_conflict: bool,
    routing_threshold: float,
    sarcasm_prob: Optional[float] = None,
) -> str:
    """
    Auto-generates a natural-language interpretation string.
    """
    t_name = SENTIMENT_LABELS[text_label]
    i_name = SENTIMENT_LABELS[image_label]
    f_name = SENTIMENT_LABELS[final_label]
    agree  = (text_label == image_label)

    if agree and not is_conflict:
        return (
            f"Text and image both signal {t_name.lower()} sentiment. "
            f"GDS={gds:.3f} < τ={routing_threshold:.3f} — low dissonance, "
            f"routed through normal fusion. Final: {f_name}."
        )

    if agree and is_conflict:
        sarcasm_note = ""
        if sarcasm_prob is not None and sarcasm_prob > 0.5:
            sarcasm_note = (
                f" Sarcasm detector fired (p={sarcasm_prob:.2f}) — "
                f"text may be ironic despite surface {t_name.lower()} signal."
            )
        changed = f_name != t_name
        override_note = (
            f" Cross-modal attention overrode unimodal agreement → Final: {f_name}."
            if changed else
            f" Final prediction confirms: {f_name}."
        )
        return (
            f"Text encoder: {t_name} | Image encoder: {i_name}. "
            f"GDS={gds:.3f} ≥ τ={routing_threshold:.3f} — geometric dissonance "
            f"detected despite matching unimodal labels; routed to conflict branch."
            + sarcasm_note + override_note
        )

    # Unimodal labels disagree
    sarcasm_note = ""
    if sarcasm_prob is not None and sarcasm_prob > 0.5:
        sarcasm_note = (
            f" Sarcasm detector fired (p={sarcasm_prob:.2f}) — "
            f"possible ironic or masked sentiment."
        )
    polarity_t = SENTIMENT_POLARITY[text_label]
    polarity_i = SENTIMENT_POLARITY[image_label]
    direction = "opposing" if polarity_t * polarity_i < 0 else "divergent"

    return (
        f"Cross-modal conflict: Text={t_name}, Image={i_name} ({direction} signals). "
        f"GDS={gds:.3f} ≥ τ={routing_threshold:.3f} — routed to conflict branch. "
        f"Cross-attention resolved final prediction: {f_name}."
        + sarcasm_note
    )




# =============================================================================
# Main Explainability Module
# =============================================================================

class ExplainabilityModule:
    """
    Auto-generates per-inference structured conflict reports.

    Not a nn.Module — operates purely on detached tensors/values
    after forward pass. Designed to be called post-inference.

    Usage
    -----
    >>> explainer = ExplainabilityModule()
    >>> reports = explainer.generate_reports(
    ...     text_logits, image_logits, gds_output,
    ...     routing_output, final_logits
    ... )
    >>> for r in reports:
    ...     print(r)
    """

    def __init__(
        self,
        sentiment_labels: dict = None,
        conflict_threshold_label: float = 0.5,
    ):
        self.labels = sentiment_labels or SENTIMENT_LABELS
        self.conflict_threshold_label = conflict_threshold_label

    # -------------------------------------------------------------------------
    def _sentiment_strength(self, probs: torch.Tensor, label_idx: int) -> float:
        """
        Sentiment strength = max probability (confidence of the dominant class).
        Additionally penalizes neutral predictions.
        """
        return probs[label_idx].item()

    # -------------------------------------------------------------------------
    def generate_reports(
        self,
        text_logits: torch.Tensor,          # [B, num_classes]
        image_logits: torch.Tensor,          # [B, num_classes]
        gds_output: GDSOutput,
        routing_output: RoutingOutput,
        final_logits: torch.Tensor,          # [B, num_classes]
        sarcasm_logits: Optional[torch.Tensor] = None,  # [B, 2] or None
    ) -> List[ConflictReport]:
        """
        Generate one ConflictReport per sample in the batch.

        Parameters
        ----------
        text_logits   : [B, C] — unimodal text classifier logits
        image_logits  : [B, C] — unimodal image classifier logits
        gds_output    : GDSOutput from GeometricDissonanceModule
        routing_output: RoutingOutput from RoutingController
        final_logits  : [B, C] — final prediction logits
        sarcasm_logits: [B, 2] aligned to full batch (or None)

        Returns
        -------
        List[ConflictReport] — one per sample
        """
        B = final_logits.shape[0]

        # Convert to probabilities
        text_probs  = F.softmax(text_logits.detach().cpu().float(), dim=-1)
        image_probs = F.softmax(image_logits.detach().cpu().float(), dim=-1)
        final_probs = F.softmax(final_logits.detach().cpu().float(), dim=-1)

        text_preds  = text_probs.argmax(dim=-1)    # [B]
        image_preds = image_probs.argmax(dim=-1)   # [B]
        final_preds = final_probs.argmax(dim=-1)   # [B]

        # Sarcasm probs (batch-aligned)
        sarc_probs = None
        if sarcasm_logits is not None:
            sarc_p = F.softmax(sarcasm_logits.detach().cpu().float(), dim=-1)
            sarc_probs = sarc_p[:, 1]   # prob of "sarcastic"

        reports = []
        for idx in range(B):
            t_idx   = text_preds[idx].item()
            i_idx   = image_preds[idx].item()
            f_idx   = final_preds[idx].item()
            is_conf = routing_output.routing_decisions[idx].item()
            gds_val = gds_output.gds[idx].item()
            tau     = routing_output.threshold

            sarcasm_p = sarc_probs[idx].item() if sarc_probs is not None else None

            interp = _generate_interpretation(
                text_label=t_idx,
                image_label=i_idx,
                final_label=f_idx,
                gds=gds_val,
                is_conflict=bool(is_conf),
                routing_threshold=tau,
                sarcasm_prob=sarcasm_p,
            )

            report = ConflictReport(
                # Text
                text_sentiment_label=self.labels[t_idx],
                text_sentiment_strength=self._sentiment_strength(text_probs[idx], t_idx),
                text_sentiment_confidence=text_probs[idx].max().item(),
                text_label_idx=t_idx,
                # Image
                image_sentiment_label=self.labels[i_idx],
                image_sentiment_strength=self._sentiment_strength(image_probs[idx], i_idx),
                image_sentiment_confidence=image_probs[idx].max().item(),
                image_label_idx=i_idx,
                # Geometric
                cosine_similarity=gds_output.cosine_similarity[idx].item(),
                angular_separation_deg=gds_output.angular_separation_deg[idx].item(),
                magnitude_text=gds_output.magnitude_text[idx].item(),
                magnitude_image=gds_output.magnitude_image[idx].item(),
                magnitude_difference=gds_output.magnitude_difference[idx].item(),
                projection_residual=gds_output.projection_residual[idx].item(),
                # GDS
                gds_score=gds_val,
                gds_alpha=gds_output.alpha,
                gds_beta=gds_output.beta,
                # Routing
                routing_path="Conflict Branch" if is_conf else "Normal Fusion Branch",
                routing_threshold=tau,
                is_conflict=bool(is_conf),
                # Sarcasm
                sarcasm_probability=sarcasm_p,
                # Final
                final_prediction_label=self.labels[f_idx],
                final_prediction_idx=f_idx,
                final_prediction_confidence=final_probs[idx].max().item(),
                # Interpretation
                interpretation=interp,
            )
            reports.append(report)

        return reports

    # -------------------------------------------------------------------------
    def print_reports(self, reports: List[ConflictReport]):
        for i, rpt in enumerate(reports):
            print(f"\n[Sample {i+1}]")
            print(rpt)

    # -------------------------------------------------------------------------
    def batch_summary(self, reports: List[ConflictReport]) -> dict:
        """Aggregate statistics over a list of reports."""
        n = len(reports)
        if n == 0:
            return {}
        n_conflict  = sum(r.is_conflict for r in reports)
        gds_vals    = [r.gds_score for r in reports]
        sarc_probs  = [r.sarcasm_probability for r in reports
                       if r.sarcasm_probability is not None]
        agree_ratio = sum(r.text_label_idx == r.image_label_idx
                          for r in reports) / n
        return {
            "n_samples":             n,
            "n_conflict_routed":     n_conflict,
            "conflict_ratio":        n_conflict / n,
            "gds_mean":              sum(gds_vals) / n,
            "gds_max":               max(gds_vals),
            "gds_min":               min(gds_vals),
            "modality_agree_ratio":  agree_ratio,
            "mean_sarcasm_prob":     sum(sarc_probs) / len(sarc_probs)
                                     if sarc_probs else None,
        }
