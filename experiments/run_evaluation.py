"""
run_evaluation.py — Full CGRN Evaluation Pipeline
===================================================

Runs:
  - Model comparison  (text-only, image-only, static fusion, CGRN)
  - Ablation studies
  - Efficiency analysis
  - Generates all plots

Usage
-----
  python experiments/run_evaluation.py \\
      --model_path checkpoints/cgrn_final.pt \\
      --data_path  data/synthetic_sentiment.csv \\
      --output_dir results/
"""

import sys
import os
import argparse
import json
import logging
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.cgrn_model import CGRNModel
from src.models.unimodal_classifiers import (
    UnimodalTextClassifier, UnimodalImageClassifier
)
from src.evaluation.evaluator import CGRNEvaluator, StaticFusionBaseline
from src.evaluation.ablation import AblationStudy
from src.evaluation.efficiency_analysis import EfficiencyAnalyzer
from src.utils.visualization import (
    plot_gds_distribution, plot_routing_distribution,
    plot_training_history, generate_architecture_diagram,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CGRN Evaluation Pipeline")
    p.add_argument("--model_path",   type=str, default="checkpoints/cgrn_final.pt")
    p.add_argument("--data_path",    type=str, default="data/synthetic_sentiment.csv")
    p.add_argument("--output_dir",   type=str, default="results")
    p.add_argument("--embed_dim",    type=int, default=256)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--skip_ablation", action="store_true")
    p.add_argument("--skip_efficiency", action="store_true")
    p.add_argument("--text_model",   type=str, default="distilbert-base-uncased")
    p.add_argument("--image_backbone", type=str, default="mobilenet_v3_small")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    logger.info(f"Loading CGRN model from: {args.model_path}")
    model = CGRNModel(
        text_model_name=args.text_model,
        image_backbone=args.image_backbone,
        embed_dim=args.embed_dim,
        generate_reports=False,
    )

    if os.path.exists(args.model_path):
        model.load_state_dict(
            torch.load(args.model_path, map_location=device)
        )
        logger.info("  Checkpoint loaded.")
    else:
        logger.warning(
            f"No checkpoint found at {args.model_path}. "
            f"Using randomly initialized weights (for demo only)."
        )

    model.eval()

    # -------------------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------------------
    if not os.path.exists(args.data_path):
        logger.warning(
            f"Data not found at {args.data_path}. "
            f"Generating synthetic data..."
        )
        from src.utils.data_loader import generate_synthetic_csv
        os.makedirs("data", exist_ok=True)
        args.data_path = generate_synthetic_csv(
            out_path="data/synthetic_sentiment.csv",
            n_samples=300,
        )

    from src.utils.data_loader import build_dataloaders
    loaders = build_dataloaders(
        data_path=args.data_path,
        tokenizer_name=args.text_model,
        batch_size=args.batch_size,
        num_workers=0,
    )
    test_loader = loaders["test"]
    logger.info(f"Test set size: {len(test_loader.dataset)}")

    # -------------------------------------------------------------------------
    # Model Comparison (Phase 7)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7: Model Comparison Evaluation")
    logger.info("=" * 60)

    text_cls  = UnimodalTextClassifier(model.text_encoder)
    image_cls = UnimodalImageClassifier(model.image_encoder)
    static    = StaticFusionBaseline(
        model.text_encoder, model.image_encoder, args.embed_dim
    )

    eval_models = {
        "Text Only":     (text_cls, "text"),
        "Image Only":    (image_cls, "image"),
        "Static Fusion": (static, "static"),
        "CGRN":          (model, "cgrn"),
    }

    evaluator = CGRNEvaluator(device=str(device))
    comp_results = evaluator.evaluate_all(eval_models, test_loader)
    evaluator.print_comparison_table(comp_results)
    evaluator.save_results(
        comp_results,
        os.path.join(args.output_dir, "model_comparison.json")
    )

    # -------------------------------------------------------------------------
    # Ablation Studies (Phase 9)
    # -------------------------------------------------------------------------
    if not args.skip_ablation:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 9: Ablation Studies")
        logger.info("=" * 60)

        ablation = AblationStudy(
            base_model=model,
            evaluator=evaluator,
            test_loader=test_loader,
            device=str(device),
        )
        ablation_results = ablation.run()
        ablation.print_ablation_table(ablation_results)
        ablation.save_results(
            ablation_results,
            os.path.join(args.output_dir, "ablation_results.json")
        )

    # -------------------------------------------------------------------------
    # Efficiency Analysis (Phase 8)
    # -------------------------------------------------------------------------
    if not args.skip_efficiency:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 8: Efficiency Analysis")
        logger.info("=" * 60)

        # Build sample batch for profiling
        sample_batch = next(iter(test_loader))
        analyzer = EfficiencyAnalyzer(n_warmup=3, n_repeat=10)

        eff_results = [
            analyzer.profile_model(model, "CGRN", sample_batch),
            analyzer.profile_model(static, "Static Fusion", sample_batch),
        ]
        analyzer.print_efficiency_table(eff_results)
        analyzer.save_results(
            eff_results,
            os.path.join(args.output_dir, "efficiency_results.json")
        )

    # -------------------------------------------------------------------------
    # Generate Architecture Diagram
    # -------------------------------------------------------------------------
    generate_architecture_diagram("docs/architecture_diagram.md")

    # -------------------------------------------------------------------------
    # Collect GDS statistics and plot
    # -------------------------------------------------------------------------
    logger.info("\nCollecting GDS statistics for visualization...")
    gds_scores_list   = []
    conflict_flags    = []
    routing_decisions = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images         = batch["images"].to(device)
            conflict       = batch.get("conflict", torch.zeros(input_ids.shape[0]))

            output = model(input_ids, attention_mask, images)
            gds_scores_list.extend(output.gds_output.gds.cpu().tolist())
            conflict_flags.extend(conflict.tolist())
            routing_decisions.extend(
                output.routing_output.routing_decisions.cpu().tolist()
            )

    threshold = model.routing_controller.threshold.item()

    plot_gds_distribution(
        gds_scores=gds_scores_list,
        threshold=threshold,
        conflict_mask=[bool(c) for c in conflict_flags],
        output_path=os.path.join(args.output_dir, "gds_distribution.png"),
    )

    n_conflict = sum(routing_decisions)
    n_normal   = len(routing_decisions) - n_conflict
    plot_routing_distribution(
        n_normal=n_normal,
        n_conflict=n_conflict,
        threshold=threshold,
        output_path=os.path.join(args.output_dir, "routing_distribution.png"),
    )

    # Training history (if available)
    hist_path = "logs/stage2_history.json"
    if os.path.exists(hist_path):
        plot_training_history(
            history_path=hist_path,
            output_path=os.path.join(args.output_dir, "training_history.png"),
        )

    logger.info(f"\n✓ Evaluation complete. Results saved to: {args.output_dir}/")
    logger.info("  model_comparison.json")
    logger.info("  ablation_results.json    (if not skipped)")
    logger.info("  efficiency_results.json  (if not skipped)")
    logger.info("  gds_distribution.png")
    logger.info("  routing_distribution.png")
    logger.info("  docs/architecture_diagram.md")


if __name__ == "__main__":
    main()
