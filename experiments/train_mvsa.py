"""
train_mvsa.py — Train CGRN on MVSA-Multiple / MVSA-Single
==========================================================

Usage
-----
# First download MVSA-Multiple → data/MVSA_Multiple/
python experiments/train_mvsa.py \\
    --data_root    data/MVSA_Multiple \\
    --variant      multiple \\
    --stage1_epochs  3 \\
    --stage2_epochs  5 \\
    --stage3_epochs  3 \\
    --batch_size     32 \\
    --output_dir   checkpoints_mvsa

# Quick test with fewer samples:
python experiments/train_mvsa.py \\
    --data_root  data/MVSA_Multiple \\
    --max_samples 1000 \\
    --stage1_epochs 1 --stage2_epochs 2 --skip_stage3
"""

import sys
import os
import argparse
import logging
import random

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.cgrn_model import CGRNModel
from src.models.unimodal_classifiers import UnimodalTextClassifier, UnimodalImageClassifier
from src.training.training_strategy import (
    TrainingConfig, UnimodalTrainer, CGRNTrainer,
)
from src.utils.mvsa_loader import build_mvsa_dataloaders, print_download_instructions
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train CGRN on MVSA dataset")

    # Dataset
    p.add_argument("--data_root",   type=str, default="data/MVSA_Multiple",
                   help="Path to extracted MVSA_Multiple or MVSA_Single folder")
    p.add_argument("--variant",     type=str, default="multiple",
                   choices=["multiple", "single"])
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size for quick testing")

    # Model
    p.add_argument("--embed_dim",      type=int, default=256)
    p.add_argument("--text_model",     type=str, default="roberta-base")
    p.add_argument("--image_backbone", type=str, default="mobilenet_v3_small")

    # Training stages
    p.add_argument("--stage1_epochs",  type=int, default=3)
    p.add_argument("--stage2_epochs",  type=int, default=5)
    p.add_argument("--stage3_epochs",  type=int, default=3)
    p.add_argument("--skip_stage1",    action="store_true")
    p.add_argument("--skip_stage3",    action="store_true")
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--max_length",     type=int, default=128)

    # LR
    p.add_argument("--lr_text",   type=float, default=1e-5,
                   help="Text encoder LR (1e-5 recommended for roberta-base)")
    p.add_argument("--lr_image",  type=float, default=1e-4)
    p.add_argument("--lr_stage2", type=float, default=5e-5)
    p.add_argument("--lr_stage3", type=float, default=2e-6,
                   help="End-to-end LR (lower for roberta-base)")

    # Output
    p.add_argument("--output_dir", type=str, default="checkpoints_mvsa")
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--num_workers",type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/mvsa_training.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # ── Device / seed ─────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Check dataset exists ──────────────────────────────────────────────────
    if not os.path.isdir(args.data_root):
        logger.error(f"Dataset directory not found: {args.data_root}")
        print_download_instructions()
        sys.exit(1)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.text_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info(f"Building MVSA-{args.variant} dataloaders from: {args.data_root}")
    loaders = build_mvsa_dataloaders(
        data_root=args.data_root,
        tokenizer=tokenizer,
        variant=args.variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )

    # Optional: cap to max_samples for quick testing
    if args.max_samples:
        logger.info(f"Capping to {args.max_samples} samples per split.")
        from torch.utils.data import Subset
        for split in ["train", "val", "test"]:
            ds = loaders[split].dataset
            cap = min(args.max_samples, len(ds))
            subset = Subset(ds, list(range(cap)))
            loaders[split] = torch.utils.data.DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                num_workers=args.num_workers,
            )

    logger.info(
        f"  Train: {len(loaders['train'].dataset)} | "
        f"Val: {len(loaders['val'].dataset)} | "
        f"Test: {len(loaders['test'].dataset)}"
    )

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building CGRN model...")
    model = CGRNModel(
        text_model_name=args.text_model,
        image_backbone=args.image_backbone,
        embed_dim=args.embed_dim,
        num_classes=3,
    ).to(device)

    n_total  = sum(p.numel() for p in model.parameters())
    n_text   = sum(p.numel() for p in model.text_encoder.parameters())
    n_image  = sum(p.numel() for p in model.image_encoder.parameters())
    logger.info(f"Model parameters:\n  Total: {n_total:,} | Text: {n_text:,} | Image: {n_image:,}")

    # ── Training config ───────────────────────────────────────────────────────
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage1_lr_text=args.lr_text,
        stage1_lr_image=args.lr_image,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.lr_stage2,
        stage3_epochs=args.stage3_epochs,
        stage3_lr=args.lr_stage3,
        device=str(device),
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train_config.device = str(device)

    # ── Stage 1: Unimodal pre-training ────────────────────────────────────────
    if not args.skip_stage1:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Unimodal Pre-training on MVSA")
        logger.info("=" * 60)

        text_cls  = UnimodalTextClassifier(model.text_encoder)
        image_cls = UnimodalImageClassifier(model.image_encoder)
        uni_trainer = UnimodalTrainer(train_config)

        text_results  = uni_trainer.train_text_encoder(
            text_cls, loaders["train"], loaders["val"]
        )
        image_results = uni_trainer.train_image_encoder(
            image_cls, loaders["train"], loaders["val"]
        )
        logger.info(
            f"Stage 1 complete | Text F1={text_results['best_val_f1']:.4f} | "
            f"Image F1={image_results['best_val_f1']:.4f}"
        )

        # Re-load best unimodal weights into CGRN
        for ckpt_name, module in [("best_text_encoder.pt",  model.text_encoder),
                                   ("best_image_encoder.pt", model.image_encoder)]:
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                encoder_state = {k[len("encoder."):]: v
                                 for k, v in ckpt.items() if k.startswith("encoder.")}
                module.load_state_dict(encoder_state)
                logger.info(f"  Loaded {ckpt_name} → CGRN encoder.")
    else:
        logger.info("Skipping Stage 1.")

    # ── Stage 2: GDS + Routing Training ───────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: GDS + Routing Training (Encoders Frozen)")
    logger.info("=" * 60)

    cgrn_trainer = CGRNTrainer(model, train_config)
    s2 = cgrn_trainer.train_stage2(loaders["train"], loaders["val"])
    logger.info(f"Stage 2 complete | Best Val F1={s2['best_val_f1']:.4f}")

    # ── Stage 3: End-to-end fine-tuning ───────────────────────────────────────
    if not args.skip_stage3:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: End-to-End Fine-Tuning")
        logger.info("=" * 60)
        s3 = cgrn_trainer.train_stage3(loaders["train"], loaders["val"])
        logger.info(f"Stage 3 complete | Best Val F1={s3['best_val_f1']:.4f}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "cgrn_mvsa_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n✓ Final model saved → {final_path}")
    logger.info("MVSA training pipeline complete.")


if __name__ == "__main__":
    main()
