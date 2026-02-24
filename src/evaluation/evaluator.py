"""
evaluator.py — Full Evaluation Framework (Phase 7)
====================================================

Compares:
  - Text-only baseline
  - Image-only baseline
  - Static Fusion baseline (concat + MLP, no routing)
  - CGRN (proposed)

Evaluates on:
  - Full test set
  - Contradictory/conflict-only subset
  - (Estimated) sarcastic samples
  - Ambiguous-sentiment samples
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..training.training_strategy import compute_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Static Fusion Baseline (no routing)
# =============================================================================

class StaticFusionBaseline(nn.Module):
    """
    Simple baseline: concatenate text and image embeddings → MLP.
    No GDS, no routing — represents standard static fusion.
    """

    def __init__(self, text_encoder, image_encoder, embed_dim=256, num_classes=3):
        super().__init__()
        self.text_encoder  = text_encoder
        self.image_encoder = image_encoder
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        s_t, _ = self.text_encoder(input_ids, attention_mask)
        s_i, _ = self.image_encoder(images)
        fused  = torch.cat([s_t, s_i], dim=-1)
        return self.mlp(fused)


# =============================================================================
# Evaluation Results
# =============================================================================

@dataclass
class EvaluationResult:
    model_name:      str
    accuracy:        float
    macro_f1:        float
    per_class_f1:    List[float]
    conflict_acc:    Optional[float] = None
    conflict_f1:     Optional[float] = None
    sarcasm_f1:      Optional[float] = None
    n_samples:       int = 0
    n_conflict:      int = 0

    def __str__(self):
        lines = [
            f"Model: {self.model_name}",
            f"  Full  | Acc={self.accuracy:.4f} | F1={self.macro_f1:.4f}",
        ]
        if self.conflict_f1 is not None:
            lines.append(
                f"  Conflict Subset | Acc={self.conflict_acc:.4f} | F1={self.conflict_f1:.4f}"
            )
        if self.sarcasm_f1 is not None:
            lines.append(f"  Sarcasm F1 ≈ {self.sarcasm_f1:.4f}")
        return "\n".join(lines)


# =============================================================================
# Evaluator
# =============================================================================

class CGRNEvaluator:
    """
    Evaluates one or more models and produces formatted comparison tables.

    Usage
    -----
    >>> evaluator = CGRNEvaluator(device="cuda")
    >>> results = evaluator.evaluate_all(
    ...     models={
    ...         "CGRN": cgrn_model,
    ...         "Static Fusion": static_model,
    ...         "Text Only": text_cls,
    ...         "Image Only": img_cls,
    ...     },
    ...     test_loader=test_loader,
    ... )
    >>> evaluator.print_comparison_table(results)
    """

    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

    # -------------------------------------------------------------------------
    def evaluate_cgrn(
        self,
        model,
        loader: DataLoader,
        model_name: str = "CGRN",
    ) -> EvaluationResult:
        model.eval().to(self.device)
        all_preds, all_labels, all_conflict = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images         = batch["images"].to(self.device)
                labels         = batch["labels"]
                conflict       = batch.get("conflict", torch.zeros_like(labels))

                output = model(input_ids, attention_mask, images)
                preds  = output.final_logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_conflict.extend(conflict.tolist())

        metrics = compute_metrics(all_preds, all_labels, all_conflict)
        return EvaluationResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics["per_class_f1"],
            conflict_acc=metrics.get("conflict_accuracy"),
            conflict_f1=metrics.get("conflict_f1"),
            n_samples=len(all_labels),
            n_conflict=sum(all_conflict),
        )

    # -------------------------------------------------------------------------
    def evaluate_text_only(
        self,
        text_classifier,
        loader: DataLoader,
        model_name: str = "Text Only",
    ) -> EvaluationResult:
        text_classifier.eval().to(self.device)
        all_preds, all_labels, all_conflict = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"]
                conflict       = batch.get("conflict", torch.zeros_like(labels))

                logits = text_classifier(input_ids, attention_mask)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_conflict.extend(conflict.tolist())

        metrics = compute_metrics(all_preds, all_labels, all_conflict)
        return EvaluationResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics["per_class_f1"],
            conflict_acc=metrics.get("conflict_accuracy"),
            conflict_f1=metrics.get("conflict_f1"),
            n_samples=len(all_labels),
            n_conflict=sum(all_conflict),
        )

    # -------------------------------------------------------------------------
    def evaluate_image_only(
        self,
        image_classifier,
        loader: DataLoader,
        model_name: str = "Image Only",
    ) -> EvaluationResult:
        image_classifier.eval().to(self.device)
        all_preds, all_labels, all_conflict = [], [], []

        with torch.no_grad():
            for batch in loader:
                images   = batch["images"].to(self.device)
                labels   = batch["labels"]
                conflict = batch.get("conflict", torch.zeros_like(labels))

                logits = image_classifier(images)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_conflict.extend(conflict.tolist())

        metrics = compute_metrics(all_preds, all_labels, all_conflict)
        return EvaluationResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics["per_class_f1"],
            conflict_acc=metrics.get("conflict_accuracy"),
            conflict_f1=metrics.get("conflict_f1"),
            n_samples=len(all_labels),
            n_conflict=sum(all_conflict),
        )

    # -------------------------------------------------------------------------
    def evaluate_static_fusion(
        self,
        static_model,
        loader: DataLoader,
        model_name: str = "Static Fusion",
    ) -> EvaluationResult:
        static_model.eval().to(self.device)
        all_preds, all_labels, all_conflict = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images         = batch["images"].to(self.device)
                labels         = batch["labels"]
                conflict       = batch.get("conflict", torch.zeros_like(labels))

                logits = static_model(input_ids, attention_mask, images)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_conflict.extend(conflict.tolist())

        metrics = compute_metrics(all_preds, all_labels, all_conflict)
        return EvaluationResult(
            model_name=model_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics["per_class_f1"],
            conflict_acc=metrics.get("conflict_accuracy"),
            conflict_f1=metrics.get("conflict_f1"),
            n_samples=len(all_labels),
            n_conflict=sum(all_conflict),
        )

    # -------------------------------------------------------------------------
    def evaluate_all(
        self,
        models: Dict,
        test_loader: DataLoader,
    ) -> List[EvaluationResult]:
        """
        Evaluate all models in models dict.

        models dict format:
          {
            "CGRN":          (cgrn_model,   "cgrn"),
            "Static Fusion": (static_model, "static"),
            "Text Only":     (text_cls,     "text"),
            "Image Only":    (img_cls,      "image"),
          }
        """
        results = []
        for name, (model, mode) in models.items():
            logger.info(f"Evaluating: {name}")
            if mode == "cgrn":
                r = self.evaluate_cgrn(model, test_loader, name)
            elif mode == "text":
                r = self.evaluate_text_only(model, test_loader, name)
            elif mode == "image":
                r = self.evaluate_image_only(model, test_loader, name)
            elif mode == "static":
                r = self.evaluate_static_fusion(model, test_loader, name)
            else:
                logger.warning(f"Unknown mode '{mode}' for model '{name}'")
                continue
            results.append(r)
            logger.info(str(r))

        return results

    # -------------------------------------------------------------------------
    def print_comparison_table(self, results: List[EvaluationResult]):
        """Print formatted Markdown comparison table."""
        header = (
            f"| {'Model':<22} | {'Acc':>6} | {'F1':>6} | "
            f"{'Conflict Acc':>12} | {'Conflict F1':>11} |"
        )
        sep = "|" + "-" * 24 + "|" + "-" * 8 + "|" + "-" * 8 + \
              "|" + "-" * 14 + "|" + "-" * 13 + "|"

        print("\n" + "=" * 75)
        print("CGRN Evaluation Comparison Table")
        print("=" * 75)
        print(header)
        print(sep)

        for r in results:
            c_acc = f"{r.conflict_acc:.4f}" if r.conflict_acc is not None else "  N/A  "
            c_f1  = f"{r.conflict_f1:.4f}"  if r.conflict_f1  is not None else "  N/A  "
            print(
                f"| {r.model_name:<22} | {r.accuracy:>6.4f} | {r.macro_f1:>6.4f} | "
                f"{c_acc:>12} | {c_f1:>11} |"
            )

        print("=" * 75 + "\n")

    # -------------------------------------------------------------------------
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save results to JSON."""
        data = []
        for r in results:
            data.append({
                "model_name":   r.model_name,
                "accuracy":     r.accuracy,
                "macro_f1":     r.macro_f1,
                "per_class_f1": r.per_class_f1,
                "conflict_acc": r.conflict_acc,
                "conflict_f1":  r.conflict_f1,
                "n_samples":    r.n_samples,
                "n_conflict":   r.n_conflict,
            })
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved → {output_path}")
