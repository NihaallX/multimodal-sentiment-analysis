"""
evaluate_mvsa.py
================
Evaluate a trained CGRN checkpoint on the MVSA-Multiple (or Single) test split.

Usage
-----
python experiments/evaluate_mvsa.py \
    --checkpoint checkpoints_mvsa/cgrn_mvsa_final.pt \
    --data_root   data/MVSA_Multiple \
    --variant     multiple \
    --batch_size  64 \
    --device      cuda

Output
------
- Per-class precision / recall / F1
- Overall accuracy + macro-F1
- Conflict-subset accuracy + F1
- Non-conflict-subset accuracy + F1
- GDS distribution statistics
- Learned τ (threshold) value
- Confusion matrix saved to results/mvsa_confusion.png
- Full results JSON saved to results/mvsa_eval_results.json
"""

import argparse
import json
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from transformers import DistilBertTokenizer

from src.models.cgrn_model import CGRNConfig
from src.utils.mvsa_loader import build_mvsa_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
LABEL_NAMES = ["negative", "neutral", "positive"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CGRN on MVSA test set")
    p.add_argument(
        "--checkpoint",
        default="checkpoints_mvsa/cgrn_mvsa_final.pt",
        help="Path to trained .pt checkpoint",
    )
    p.add_argument(
        "--data_root",
        default="data/MVSA_Multiple",
        help="Root directory of the MVSA dataset",
    )
    p.add_argument(
        "--variant",
        default="multiple",
        choices=["multiple", "single"],
        help="MVSA variant to evaluate on",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output_dir",
        default="results",
        help="Directory to save evaluation outputs",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  CGRN Evaluation on MVSA-{args.variant.capitalize()}")
    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Device     : {device}")

    # ── Build dataloaders ────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loaders = build_mvsa_dataloaders(
        data_root=args.data_root,
        tokenizer=tokenizer,
        variant=args.variant,
        batch_size=args.batch_size,
        num_workers=0,
    )
    test_loader = loaders["test"]
    print(f"  Test batches: {len(test_loader)}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = CGRNConfig.build().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # Strip DataParallel or module. prefix if present
    state = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    tau_val = model.routing_controller.threshold.item()
    print(f"  Learned τ  : {tau_val:.4f}")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds, all_labels, all_conflict = [], [], []
    all_gds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images         = batch["images"].to(device)
            labels         = batch["labels"]
            conflict       = batch["conflict"].bool()

            output = model(input_ids, attention_mask, images)
            preds = output.final_logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_conflict.extend(conflict.tolist())
            all_gds.extend(output.gds_output.gds.cpu().tolist())

    all_preds    = np.array(all_preds)
    all_labels   = np.array(all_labels)
    all_conflict = np.array(all_conflict, dtype=bool)
    all_gds      = np.array(all_gds)

    # ── Overall metrics ──────────────────────────────────────────────────────
    overall_acc  = accuracy_score(all_labels, all_preds)
    overall_f1   = f1_score(all_labels, all_preds, average="macro")
    report       = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, digits=4
    )

    print(f"\n{'─'*60}")
    print("  OVERALL METRICS")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {overall_acc:.4f}")
    print(f"  Macro-F1  : {overall_f1:.4f}")
    print(f"\n{report}")

    # ── Conflict-subset metrics ───────────────────────────────────────────────
    results = {
        "checkpoint": args.checkpoint,
        "tau": tau_val,
        "overall": {
            "accuracy": overall_acc,
            "macro_f1": overall_f1,
        },
    }

    for subset_name, mask in [
        ("conflict",     all_conflict),
        ("non_conflict", ~all_conflict),
    ]:
        if mask.sum() == 0:
            continue
        sub_acc = accuracy_score(all_labels[mask], all_preds[mask])
        sub_f1  = f1_score(all_labels[mask], all_preds[mask], average="macro")
        sub_gds = all_gds[mask]
        print(f"  {subset_name.upper():15s} | n={mask.sum():5d} | "
              f"Acc={sub_acc:.4f} | F1={sub_f1:.4f} | "
              f"GDS mean={sub_gds.mean():.4f} std={sub_gds.std():.4f}")
        results[subset_name] = {
            "n": int(mask.sum()),
            "accuracy": sub_acc,
            "macro_f1": sub_f1,
            "gds_mean": float(sub_gds.mean()),
            "gds_std":  float(sub_gds.std()),
        }

    # ── GDS distribution ─────────────────────────────────────────────────────
    print(f"\n  GDS Distribution (all):")
    print(f"    min={all_gds.min():.4f}  max={all_gds.max():.4f}  "
          f"mean={all_gds.mean():.4f}  std={all_gds.std():.4f}")

    # Routing accuracy: did conflict samples route via conflict branch?
    routed_conflict  = (all_gds[all_conflict]  >= tau_val).mean() if all_conflict.any() else 0.0
    routed_normal    = (all_gds[~all_conflict] <  tau_val).mean() if (~all_conflict).any() else 0.0
    print(f"\n  Routing correctness:")
    print(f"    Conflict  samples → conflict branch : {routed_conflict:.2%}")
    print(f"    Harmonic  samples → normal  branch  : {routed_normal:.2%}")
    results["routing"] = {
        "conflict_routed_correctly":  float(routed_conflict),
        "harmonic_routed_correctly":  float(routed_normal),
    }

    # ── Confusion matrix ─────────────────────────────────────────────────────
    _save_confusion_matrix(all_labels, all_preds, args.output_dir)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "mvsa_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*60}\n")


def _save_confusion_matrix(labels, preds, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(LABEL_NAMES)),
            yticks=range(len(LABEL_NAMES)),
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            xlabel="Predicted",
            ylabel="True",
            title="CGRN Confusion Matrix (MVSA)",
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.tight_layout()
        path = os.path.join(output_dir, "mvsa_confusion.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Confusion matrix → {path}")
    except Exception as exc:
        print(f"  [warn] Could not save confusion matrix: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation(parse_args())
