"""
run_training.py — Full CGRN 3-Stage Training Pipeline
=======================================================

Usage
-----
  # With synthetic data (for quick testing):
  python experiments/run_training.py --synthetic --n_samples 500

  # With real dataset:
  python experiments/run_training.py --data_path data/sentiment_dataset.json

  # Skip Stage 1 (unimodal pretraining):
  python experiments/run_training.py --data_path ... --skip_stage1

Arguments
---------
  --data_path     : path to JSON/CSV dataset file
  --synthetic     : use synthetic data (overrides data_path)
  --n_samples     : number of synthetic samples (default 500)
  --output_dir    : checkpoint output directory (default checkpoints/)
  --log_dir       : logs directory (default logs/)
  --stage1_epochs : number of Stage 1 epochs (default 5)
  --stage2_epochs : number of Stage 2 epochs (default 10)
  --stage3_epochs : number of Stage 3 epochs (default 5)
  --embed_dim     : embedding dimension (default 256)
  --batch_size    : training batch size (default 16)
  --skip_stage1   : skip unimodal pretraining
  --skip_stage3   : skip end-to-end fine-tuning
  --device        : 'cpu', 'cuda', or 'auto' (default auto)
  --seed          : random seed (default 42)
"""

import sys
import os
import argparse
import logging
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure logs directory exists before configuring file handler
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

from src.models.cgrn_model import CGRNModel, CGRNConfig
from src.models.unimodal_classifiers import UnimodalTextClassifier, UnimodalImageClassifier
from src.training.training_strategy import (
    TrainingConfig, UnimodalTrainer, CGRNTrainer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CGRN Training Pipeline")
    p.add_argument("--data_path",      type=str, default=None)
    p.add_argument("--synthetic",      action="store_true")
    p.add_argument("--n_samples",      type=int, default=500)
    p.add_argument("--conflict_ratio", type=float, default=0.3)
    p.add_argument("--output_dir",     type=str, default="checkpoints")
    p.add_argument("--log_dir",        type=str, default="logs")
    p.add_argument("--stage1_epochs",  type=int, default=3)
    p.add_argument("--stage2_epochs",  type=int, default=5)
    p.add_argument("--stage3_epochs",  type=int, default=3)
    p.add_argument("--embed_dim",      type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--skip_stage1",    action="store_true")
    p.add_argument("--skip_stage3",    action="store_true")
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--text_model",     type=str,
                   default="distilbert-base-uncased")
    p.add_argument("--image_backbone", type=str,
                   default="mobilenet_v3_small")
    return p.parse_args()


# =============================================================================
# Data Setup
# =============================================================================

def setup_data(args):
    """Returns train/val/test DataLoaders."""
    from torch.utils.data import DataLoader

    if args.synthetic:
        logger.info(f"Generating synthetic dataset ({args.n_samples} samples)...")
        from src.utils.data_loader import generate_synthetic_csv
        os.makedirs("data", exist_ok=True)
        data_path = generate_synthetic_csv(
            out_path="data/synthetic_sentiment.csv",
            n_samples=args.n_samples,
            conflict_ratio=args.conflict_ratio,
            seed=args.seed,
        )
    else:
        data_path = args.data_path
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found: {data_path}. "
                f"Use --synthetic to generate test data."
            )

    from src.utils.data_loader import build_dataloaders
    logger.info(f"Building dataloaders from: {data_path}")
    loaders = build_dataloaders(
        data_path=data_path,
        tokenizer_name=args.text_model,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
    )
    logger.info(
        f"  Train: {len(loaders['train'].dataset)} | "
        f"Val: {len(loaders['val'].dataset)} | "
        f"Test: {len(loaders['test'].dataset)}"
    )
    return loaders


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    loaders = setup_data(args)

    # -------------------------------------------------------------------------
    # Build CGRN Model
    # -------------------------------------------------------------------------
    logger.info("Building CGRN model...")
    model = CGRNModel(
        text_model_name=args.text_model,
        image_backbone=args.image_backbone,
        embed_dim=args.embed_dim,
        num_classes=3,
        gds_alpha_init=1.0,
        gds_beta_init=1.0,
        routing_threshold=0.5,
        learn_threshold=True,
        use_sarcasm_head=True,
        generate_reports=False,
    )

    counts = model.param_count()
    logger.info(
        f"Model parameters:\n"
        f"  Total        : {counts['full_model']['total']:,}\n"
        f"  Text Encoder : {counts['text_encoder']['total']:,}\n"
        f"  Image Encoder: {counts['image_encoder']['total']:,}\n"
        f"  GDS Module   : {counts['gds_module']['total']:,}\n"
        f"  Routing Ctrl : {counts['routing_controller']['total']:,}"
    )

    # -------------------------------------------------------------------------
    # Training Config
    # -------------------------------------------------------------------------
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_batch_size=args.batch_size,
        stage2_batch_size=args.batch_size,
        stage3_batch_size=args.batch_size,
        device=str(device),
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # Stage 1: Unimodal Training
    # -------------------------------------------------------------------------
    if not args.skip_stage1:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Unimodal Pre-training")
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
            f"Stage 1 complete | "
            f"Text best F1={text_results['best_val_f1']:.4f} | "
            f"Image best F1={image_results['best_val_f1']:.4f}"
        )

        # Load best unimodal weights back into CGRN encoders.
        # Checkpoints were saved from UnimodalTextClassifier / UnimodalImageClassifier
        # which wrap the encoder under self.encoder, so keys have an 'encoder.' prefix
        # that must be stripped before loading into the bare encoder module.
        text_ckpt  = os.path.join(args.output_dir, "best_text_encoder.pt")
        image_ckpt = os.path.join(args.output_dir, "best_image_encoder.pt")
        if os.path.exists(text_ckpt):
            ckpt = torch.load(text_ckpt, map_location=device)
            encoder_state = {k[len("encoder."):]: v
                             for k, v in ckpt.items() if k.startswith("encoder.")}
            model.text_encoder.load_state_dict(encoder_state)
            logger.info("  Loaded best text encoder weights into CGRN.")
        if os.path.exists(image_ckpt):
            ckpt = torch.load(image_ckpt, map_location=device)
            encoder_state = {k[len("encoder."):]: v
                             for k, v in ckpt.items() if k.startswith("encoder.")}
            model.image_encoder.load_state_dict(encoder_state)
            logger.info("  Loaded best image encoder weights into CGRN.")
    else:
        logger.info("Skipping Stage 1 (--skip_stage1 flag set).")

    # -------------------------------------------------------------------------
    # Stage 2: GDS + Routing + Fusion Training (Encoders Frozen)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: GDS + Routing Training (Encoders Frozen)")
    logger.info("=" * 60)

    cgrn_trainer = CGRNTrainer(model, train_config)
    s2_results   = cgrn_trainer.train_stage2(loaders["train"], loaders["val"])
    logger.info(f"Stage 2 complete | Best Val F1={s2_results['best_val_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Stage 3: End-to-End Fine-Tuning (optional)
    # -------------------------------------------------------------------------
    if not args.skip_stage3:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: End-to-End Fine-Tuning")
        logger.info("=" * 60)

        s3_results = cgrn_trainer.train_stage3(loaders["train"], loaders["val"])
        logger.info(f"Stage 3 complete | Best Val F1={s3_results['best_val_f1']:.4f}")
    else:
        logger.info("Skipping Stage 3 (--skip_stage3 flag set).")

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "cgrn_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n✓ Final model saved → {final_path}")
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
