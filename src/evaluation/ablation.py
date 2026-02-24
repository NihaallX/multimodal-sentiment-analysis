"""
ablation.py — Ablation Studies (Phase 9)
==========================================

Ablation configurations tested:
  A: Full CGRN (proposed)
  B: Remove GDS → static fusion (no routing)
  C: Disable routing → always use normal branch
  D: Fix threshold (τ=0.5, not learnable)
  E: Remove magnitude term (β=0, cosine-only GDS)
  F: Remove cosine term  (α=0, magnitude-only GDS)

Each ablation measures the performance drop vs. full CGRN.
"""

import copy
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Ablation Configurations
# =============================================================================

@dataclass
class AblationResult:
    config_name:    str
    description:    str
    accuracy:       float
    macro_f1:       float
    conflict_f1:    float = None
    param_count:    int   = 0
    delta_vs_full:  float = None    # Δ F1 vs. full CGRN

    def __str__(self):
        delta = f"  Δ={self.delta_vs_full:+.4f}" if self.delta_vs_full is not None else ""
        return (
            f"[{self.config_name}] {self.description}\n"
            f"  Acc={self.accuracy:.4f} | F1={self.macro_f1:.4f}{delta}"
        )


class AblationStudy:
    """
    Runs ablation experiments by modifying the CGRN model configuration.

    The study creates modified versions of a trained CGRN model and
    evaluates each variant on the test set.
    """

    CONFIGS = {
        "A_full": {
            "description": "Full CGRN (GDS + routing + conflict branch)",
            "modifications": None,
        },
        "B_no_gds": {
            "description": "No GDS → static concatenation fusion",
            "modifications": "disable_gds",
        },
        "C_no_routing": {
            "description": "Disable routing → always use normal branch",
            "modifications": "disable_routing",
        },
        "D_fixed_threshold": {
            "description": "Fix threshold τ=0.5 (non-learnable)",
            "modifications": "fix_threshold",
        },
        "E_no_magnitude": {
            "description": "Remove magnitude term (β=0, cosine-only GDS)",
            "modifications": "zero_beta",
        },
        "F_no_cosine": {
            "description": "Remove cosine term (α=0, magnitude-only GDS)",
            "modifications": "zero_alpha",
        },
    }

    def __init__(self, base_model, evaluator, test_loader, device="auto"):
        self.base_model  = base_model
        self.evaluator   = evaluator
        self.test_loader = test_loader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

    # -------------------------------------------------------------------------
    def _apply_modification(self, model, modification: str):
        """
        Non-destructive: returns a modified copy of the model.
        """
        m = copy.deepcopy(model)

        if modification == "disable_gds":
            # Replace GDS module with zero-output → routing always uses normal branch
            class ZeroGDS(nn.Module):
                def forward(self, s_t, s_i):
                    from ..modules.gds_module import GDSOutput
                    B = s_t.shape[0]
                    z = torch.zeros(B, device=s_t.device)
                    return GDSOutput(
                        gds=z, cosine_similarity=z.clone(),
                        cosine_dissimilarity=z.clone(),
                        angular_separation_deg=z.clone(),
                        magnitude_text=z.clone(),
                        magnitude_image=z.clone(),
                        magnitude_difference=z.clone(),
                        projection_residual=z.clone(),
                        alpha=0.0, beta=0.0,
                    )
            m.gds_module = ZeroGDS()

        elif modification == "disable_routing":
            # Force threshold to very high value → always normal branch
            m.routing_controller.learn_threshold = False
            m.routing_controller.register_buffer(
                "_fixed_threshold", torch.tensor(1e6)
            )

        elif modification == "fix_threshold":
            # Freeze threshold at 0.5
            m.routing_controller.learn_threshold = False
            threshold_val = m.routing_controller.threshold.item()
            m.routing_controller.register_buffer(
                "_fixed_threshold", torch.tensor(threshold_val)
            )

        elif modification == "zero_beta":
            # Remove magnitude term: β → 0
            with torch.no_grad():
                m.gds_module.log_beta.fill_(-10.0)   # exp(-10) ≈ 0
            m.gds_module.log_beta.requires_grad_(False)

        elif modification == "zero_alpha":
            # Remove cosine term: α → 0
            with torch.no_grad():
                m.gds_module.log_alpha.fill_(-10.0)  # exp(-10) ≈ 0
            m.gds_module.log_alpha.requires_grad_(False)

        return m

    # -------------------------------------------------------------------------
    def run(self) -> List[AblationResult]:
        """Run all ablation experiments and return results."""
        results = []
        full_f1 = None

        for config_name, config_info in self.CONFIGS.items():
            logger.info(f"\nAblation: {config_name} — {config_info['description']}")

            if config_info["modifications"] is None:
                model = self.base_model
            else:
                model = self._apply_modification(
                    self.base_model, config_info["modifications"]
                )

            # Evaluate
            eval_result = self.evaluator.evaluate_cgrn(
                model, self.test_loader, model_name=config_name
            )

            param_count = sum(p.numel() for p in model.parameters())

            result = AblationResult(
                config_name=config_name,
                description=config_info["description"],
                accuracy=eval_result.accuracy,
                macro_f1=eval_result.macro_f1,
                conflict_f1=eval_result.conflict_f1,
                param_count=param_count,
            )

            if config_name == "A_full":
                full_f1 = eval_result.macro_f1
            elif full_f1 is not None:
                result.delta_vs_full = eval_result.macro_f1 - full_f1

            results.append(result)
            logger.info(str(result))

        return results

    # -------------------------------------------------------------------------
    def print_ablation_table(self, results: List[AblationResult]):
        """Print formatted ablation comparison table."""
        print("\n" + "=" * 80)
        print("CGRN Ablation Study Results")
        print("=" * 80)
        header = (
            f"| {'Config':<20} | {'Acc':>6} | {'F1':>6} | "
            f"{'Conflict F1':>11} | {'Δ F1':>8} |"
        )
        sep = "|" + "-" * 22 + "|" + "-" * 8 + "|" + "-" * 8 + \
              "|" + "-" * 13 + "|" + "-" * 10 + "|"
        print(header)
        print(sep)

        for r in results:
            c_f1   = f"{r.conflict_f1:.4f}" if r.conflict_f1 else "  N/A  "
            delta  = f"{r.delta_vs_full:+.4f}" if r.delta_vs_full is not None else "baseline"
            print(
                f"| {r.config_name:<20} | {r.accuracy:>6.4f} | {r.macro_f1:>6.4f} | "
                f"{c_f1:>11} | {delta:>8} |"
            )

        print("=" * 80)
        print("Negative Δ = performance drop relative to full CGRN (A_full)\n")

    # -------------------------------------------------------------------------
    def save_results(self, results: List[AblationResult], output_path: str):
        data = [
            {
                "config": r.config_name,
                "description": r.description,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "conflict_f1": r.conflict_f1,
                "param_count": r.param_count,
                "delta_vs_full": r.delta_vs_full,
            }
            for r in results
        ]
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Ablation results saved → {output_path}")
