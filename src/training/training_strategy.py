"""
training_strategy.py — 3-Stage CGRN Training Pipeline
=======================================================

Stage 1: Train unimodal classifiers independently
  - Train TextEncoder + classifier head
  - Train ImageEncoder + classifier head
  - Evaluate each unimodal model

Stage 2: Freeze encoders, train GDS + routing + fusion layers
  - Fix text/image encoder weights
  - Train GeometricDissonanceModule (α, β learnable)
  - Train RoutingController (threshold τ learnable)
  - Train NormalFusionBranch and ConflictBranch

Stage 3: (Optional) End-to-end fine-tuning
  - Unfreeze all layers
  - Fine-tune with lower learning rate
  - Apply optional auxiliary routing loss

Loss Functions
--------------
  - Primary:   CrossEntropyLoss on final_logits
  - Auxiliary: CrossEntropyLoss on text_logits + image_logits (Stage 1)
  - Routing:   Optional auxiliary loss to encourage correct routing
               (maximize GDS for known-conflict samples)
"""

import os
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    # Paths
    output_dir:         str   = "checkpoints"
    log_dir:            str   = "logs"

    # Stage 1
    stage1_epochs:      int   = 5
    stage1_lr_text:     float = 2e-5
    stage1_lr_image:    float = 1e-4
    stage1_batch_size:  int   = 32

    # Stage 2
    stage2_epochs:      int   = 10
    stage2_lr:          float = 1e-4
    stage2_batch_size:  int   = 32
    aux_loss_weight:    float = 0.1    # weight for routing auxiliary loss

    # Stage 3
    stage3_epochs:      int   = 5
    stage3_lr:          float = 5e-6
    stage3_batch_size:  int   = 16
    unfreeze_all:       bool  = True

    # General
    max_grad_norm:        float = 1.0
    warmup_steps:         int   = 100
    eval_every_n_steps:   int   = 500
    save_every_n_epochs:  int   = 1
    device:               str   = "auto"
    seed:                 int   = 42
    num_workers:          int   = 0
    use_cosine_schedule:  bool  = True
    warmup_ratio:         float = 0.1
    partial_unfreeze_text_layers: int = 2   # unfreeze last N text encoder layers in Stage 2

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# =============================================================================
# Loss Functions
# =============================================================================

class CGRNLoss(nn.Module):
    """
    Combined training loss for CGRN.

    L_total = L_main + λ_uni * (L_text + L_image) + λ_routing * L_routing + λ_tau * L_tau

    where:
      L_main    = CrossEntropy(final_logits, labels)
      L_text    = CrossEntropy(text_logits, labels)   [unimodal supervision]
      L_image   = CrossEntropy(image_logits, labels)  [unimodal supervision]
      L_routing = -GDS mean for known-conflict samples  [encourages high GDS for conflicts]
      L_tau     = hinge loss that separates τ between conflict/non-conflict GDS values
                  For conflict:     max(0, τ + margin - gds)  → push gds > τ
                  For non-conflict: max(0, gds + margin - τ)  → push gds < τ
    """

    def __init__(
        self,
        unimodal_weight: float = 0.3,
        routing_weight:  float = 0.1,
        tau_weight:      float = 0.2,
        tau_margin:      float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.unimodal_weight = unimodal_weight
        self.routing_weight  = routing_weight
        self.tau_weight      = tau_weight
        self.tau_margin      = tau_margin
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        final_logits:  torch.Tensor,    # [B, C]
        text_logits:   torch.Tensor,    # [B, C]
        image_logits:  torch.Tensor,    # [B, C]
        labels:        torch.Tensor,    # [B]
        gds_scores:    torch.Tensor,    # [B]
        conflict_mask: Optional[torch.Tensor] = None,  # [B] bool
        tau:           Optional[torch.Tensor] = None,  # scalar learnable threshold
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with individual loss components and total.
        """
        L_main  = self.ce(final_logits,  labels)
        L_text  = self.ce(text_logits,   labels)
        L_image = self.ce(image_logits,  labels)

        L_routing = torch.tensor(0.0, device=final_logits.device)
        L_tau     = torch.tensor(0.0, device=final_logits.device)

        if conflict_mask is not None and conflict_mask.any():
            # Routing loss: push GDS high for conflict samples
            L_routing = -gds_scores[conflict_mask].mean()

            # τ hinge loss: separate threshold from both GDS distributions
            if tau is not None:
                # Conflict samples should have gds > tau + margin
                conflict_hinge = torch.relu(
                    tau + self.tau_margin - gds_scores[conflict_mask]
                ).mean()
                L_tau = conflict_hinge

                non_conflict_mask = ~conflict_mask
                if non_conflict_mask.any():
                    # Non-conflict samples should have gds < tau - margin
                    non_conflict_hinge = torch.relu(
                        gds_scores[non_conflict_mask] + self.tau_margin - tau
                    ).mean()
                    L_tau = L_tau + non_conflict_hinge

        total = (
            L_main
            + self.unimodal_weight * (L_text + L_image)
            + self.routing_weight  * L_routing
            + self.tau_weight      * L_tau
        )

        return {
            "total":     total,
            "main":      L_main,
            "text_uni":  L_text,
            "image_uni": L_image,
            "routing":   L_routing,
            "tau":       L_tau,
        }


class UnimodalLoss(nn.Module):
    """Simpler loss for Stage 1 unimodal training."""
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        return self.ce(logits, labels)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    all_preds:   List[int],
    all_labels:  List[int],
    all_conflict: List[int] = None,
    num_classes: int = 3,
) -> Dict:
    """Compute accuracy, macro F1, per-class F1, and conflict-subset F1."""
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report
    )

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0
    ).tolist()

    result = {
        "accuracy": acc,
        "macro_f1": f1,
        "per_class_f1": per_class_f1,
    }

    # Conflict-subset metrics
    if all_conflict is not None:
        conflict_idx = [i for i, c in enumerate(all_conflict) if c == 1]
        if conflict_idx:
            c_preds  = [all_preds[i] for i in conflict_idx]
            c_labels = [all_labels[i] for i in conflict_idx]
            result["conflict_accuracy"] = accuracy_score(c_labels, c_preds)
            result["conflict_f1"]       = f1_score(
                c_labels, c_preds, average="macro", zero_division=0
            )

    return result


# =============================================================================
# Stage 1 Trainer — Unimodal
# =============================================================================

class UnimodalTrainer:
    """
    Stage 1: Train the text encoder and image encoder independently.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

    def train_text_encoder(
        self,
        text_classifier,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """Train the text encoder + classification head."""
        logger.info("=== Stage 1: Training Text Encoder ===")
        text_classifier = text_classifier.to(self.device)
        loss_fn  = UnimodalLoss()
        optimizer = optim.AdamW(
            text_classifier.parameters(),
            lr=self.config.stage1_lr_text,
            weight_decay=0.01,
        )
        return self._run_training_loop(
            model=text_classifier,
            forward_fn=lambda batch: text_classifier.encoder(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )[1],   # returns logits
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_epochs=self.config.stage1_epochs,
            name="text_encoder",
        )

    def train_image_encoder(
        self,
        image_classifier,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """Train the image encoder + classification head."""
        logger.info("=== Stage 1: Training Image Encoder ===")
        image_classifier = image_classifier.to(self.device)
        loss_fn   = UnimodalLoss()
        optimizer = optim.AdamW(
            image_classifier.parameters(),
            lr=self.config.stage1_lr_image,
            weight_decay=0.01,
        )
        return self._run_training_loop(
            model=image_classifier,
            forward_fn=lambda batch: image_classifier.encoder(
                batch["images"].to(self.device)
            )[1],   # returns logits
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            n_epochs=self.config.stage1_epochs,
            name="image_encoder",
        )

    def _run_training_loop(
        self,
        model,
        forward_fn:    Callable,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        optimizer:     optim.Optimizer,
        loss_fn:       nn.Module,
        n_epochs:      int,
        name:          str,
    ) -> Dict:
        history = {"train_loss": [], "val_acc": [], "val_f1": []}
        best_f1 = 0.0

        # Cosine LR schedule with linear warmup
        scheduler = None
        if self.config.use_cosine_schedule:
            total_steps  = n_epochs * len(train_loader)
            warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                logits = forward_fn(batch)
                loss   = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)
            val_metrics = self._evaluate_unimodal(model, forward_fn, val_loader)

            logger.info(
                f"[{name}] Epoch {epoch}/{n_epochs} | "
                f"Loss={avg_loss:.4f} | "
                f"Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val F1={val_metrics['macro_f1']:.4f}"
            )

            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["macro_f1"])

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                ckpt_path = os.path.join(
                    self.config.output_dir, f"best_{name}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"  ✓ Saved best model → {ckpt_path}")

        return {"history": history, "best_val_f1": best_f1}

    def _evaluate_unimodal(self, model, forward_fn, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"]
                logits = forward_fn(batch)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        return compute_metrics(all_preds, all_labels)


# =============================================================================
# Stage 2 & 3 Trainer — Full CGRN
# =============================================================================

class CGRNTrainer:
    """
    Trains the full CGRN model through Stage 2 (and optionally Stage 3).
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
    ):
        self.model  = model
        self.config = config
        self.device = torch.device(config.device)
        self.loss_fn = CGRNLoss(
            unimodal_weight=0.3,
            routing_weight=config.aux_loss_weight,
        )
        self.gds_logger_cls = None
        try:
            from ..modules.gds_module import GDSStatisticsLogger
            self.gds_logger_cls = GDSStatisticsLogger
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    def _freeze_encoders(self):
        for param in self.model.text_encoder.backbone.parameters():
            param.requires_grad = False
        for param in self.model.image_encoder.backbone.parameters():
            param.requires_grad = False
        logger.info("  Encoders frozen (backbone weights fixed).")

    def _partial_unfreeze_text_encoder(self, n_layers: int = 2):
        """Unfreeze the last n transformer layers of the text backbone."""
        backbone = self.model.text_encoder.backbone
        layers = None
        # RoBERTa / BERT style
        if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
            layers = list(backbone.encoder.layer)
        # DistilBERT style
        elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
            layers = list(backbone.transformer.layer)
        if layers is None:
            logger.info("  Could not identify transformer layers — skipping partial unfreeze.")
            return
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(f"  Unfroze last {n_layers} text encoder layers for Stage 2.")

    def _unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("  All parameters unfrozen for end-to-end fine-tuning.")

    # -------------------------------------------------------------------------
    def _build_optimizer_stage2(self):
        """Stage 2: only projection heads, GDS, routing layers."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        return optim.AdamW(trainable_params, lr=self.config.stage2_lr, weight_decay=0.01)

    def _build_optimizer_stage3(self):
        """Stage 3: layerwise LR — lower for encoders."""
        backbone_params = (
            list(self.model.text_encoder.backbone.parameters()) +
            list(self.model.image_encoder.backbone.parameters())
        )
        other_params = [
            p for p in self.model.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        return optim.AdamW(
            [
                {"params": backbone_params, "lr": self.config.stage3_lr * 0.1},
                {"params": other_params,    "lr": self.config.stage3_lr},
            ],
            weight_decay=0.01,
        )

    # -------------------------------------------------------------------------
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        logger.info("=== Stage 2: Training GDS + Routing + Fusion (Encoders Frozen) ===")
        self.model = self.model.to(self.device)
        self._freeze_encoders()
        if self.config.partial_unfreeze_text_layers > 0:
            self._partial_unfreeze_text_encoder(self.config.partial_unfreeze_text_layers)
        optimizer = self._build_optimizer_stage2()
        return self._run_cgrn_loop(
            train_loader, val_loader, optimizer,
            self.config.stage2_epochs, stage="stage2"
        )

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        logger.info("=== Stage 3: End-to-End Fine-Tuning ===")
        self._unfreeze_all()
        optimizer = self._build_optimizer_stage3()
        return self._run_cgrn_loop(
            train_loader, val_loader, optimizer,
            self.config.stage3_epochs, stage="stage3"
        )

    # -------------------------------------------------------------------------
    def _run_cgrn_loop(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        optimizer:    optim.Optimizer,
        n_epochs:     int,
        stage:        str,
    ) -> Dict:
        history = {
            "train_loss": [], "val_acc": [], "val_f1": [],
            "gds_mean": [], "conflict_ratio": [], "threshold": [],
        }
        best_f1 = 0.0
        gds_stats = self.gds_logger_cls() if self.gds_logger_cls else None

        # Cosine LR schedule with linear warmup
        scheduler = None
        if self.config.use_cosine_schedule:
            total_steps  = n_epochs * len(train_loader)
            warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            if gds_stats:
                gds_stats.reset()

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images         = batch["images"].to(self.device)
                labels         = batch["labels"].to(self.device)
                conflict_mask  = batch["conflict"].bool().to(self.device)

                optimizer.zero_grad()
                output = self.model(input_ids, attention_mask, images)

                losses = self.loss_fn(
                    final_logits=output.final_logits,
                    text_logits=output.text_logits,
                    image_logits=output.image_logits,
                    labels=labels,
                    gds_scores=output.gds_output.gds,
                    conflict_mask=conflict_mask,
                    tau=self.model.routing_controller.threshold,
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss += losses["total"].item()

                if gds_stats:
                    gds_stats.update(
                        output.gds_output,
                        output.routing_output.routing_decisions,
                    )

            avg_loss       = epoch_loss / max(len(train_loader), 1)
            val_metrics    = self._evaluate_cgrn(val_loader)
            gds_summary    = gds_stats.summary() if gds_stats else {}
            tau_val        = self.model.routing_controller.threshold.item()

            # Track per-epoch tau loss (last batch value as proxy)
            tau_loss_val = losses.get("tau", torch.tensor(0.0)).item()

            logger.info(
                f"[{stage}] Epoch {epoch}/{n_epochs} | "
                f"Loss={avg_loss:.4f} | "
                f"Val Acc={val_metrics.get('accuracy', 0):.4f} | "
                f"Val F1={val_metrics.get('macro_f1', 0):.4f} | "
                f"GDS_mean={gds_summary.get('gds_mean', 0):.4f} | "
                f"ConflictRatio={gds_summary.get('conflict_ratio', 0):.3f} | "
                f"τ={tau_val:.4f} | "
                f"τ_loss={tau_loss_val:.4f}"
            )

            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_metrics.get("accuracy", 0))
            history["val_f1"].append(val_metrics.get("macro_f1", 0))
            history["gds_mean"].append(gds_summary.get("gds_mean", 0))
            history["conflict_ratio"].append(gds_summary.get("conflict_ratio", 0))
            history["threshold"].append(tau_val)

            if val_metrics.get("macro_f1", 0) > best_f1:
                best_f1 = val_metrics["macro_f1"]
                ckpt_path = os.path.join(
                    self.config.output_dir, f"best_cgrn_{stage}.pt"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"  ✓ Saved best CGRN model → {ckpt_path}")

        # Save training history
        hist_path = os.path.join(self.config.log_dir, f"{stage}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "best_val_f1": best_f1}

    # -------------------------------------------------------------------------
    def _evaluate_cgrn(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds, all_labels, all_conflict = [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images         = batch["images"].to(self.device)
                labels         = batch["labels"]
                conflict       = batch.get("conflict", torch.zeros_like(labels))

                output = self.model(input_ids, attention_mask, images)
                preds  = output.final_logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
                all_conflict.extend(conflict.tolist())

        return compute_metrics(all_preds, all_labels, all_conflict)
