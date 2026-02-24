"""
demo.py — Quick CGRN Demo (No Dataset Required)
================================================

Demonstrates:
  1. Building the CGRN model
  2. Running a forward pass with synthetic inputs
  3. Printing a conflict report
  4. Showing GDS computation
  5. Routing decision visualization

Run: python experiments/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F


def main():
    print("=" * 65)
    print(" CGRN — Conflict-Aware Geometric Routing Network Demo")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 1. Build model
    # -------------------------------------------------------------------------
    print("\n[1] Building CGRN model...")
    from src.models.cgrn_model import CGRNModel

    model = CGRNModel(
        text_model_name="distilbert-base-uncased",
        image_backbone="mobilenet_v3_small",
        embed_dim=256,
        num_classes=3,
        use_sarcasm_head=True,
        generate_reports=False,
    )
    counts = model.param_count()
    print(f"    Total parameters : {counts['full_model']['total']:,}")
    print(f"    Text Encoder     : {counts['text_encoder']['total']:,}")
    print(f"    Image Encoder    : {counts['image_encoder']['total']:,}")
    print(f"    GDS Module       : {counts['gds_module']['total']:,}")
    print(f"    Routing Ctrl     : {counts['routing_controller']['total']:,}")

    # -------------------------------------------------------------------------
    # 2. Prepare 3 synthetic examples:
    #    - Example A: harmonious (agree)
    #    - Example B: contradictory (sarcasm candidate)
    #    - Example C: ambiguous
    # -------------------------------------------------------------------------
    print("\n[2] Preparing synthetic examples...")
    texts = [
        "This product is absolutely wonderful! Best purchase ever.",  # A: positive
        "Oh sure, this is just GREAT. Totally not broken at all.",     # B: sarcastic
        "The product arrived. It is a product.",                        # C: neutral/ambiguous
    ]

    tokenizer = model.text_encoder.tokenizer
    encoding  = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Random synthetic images (in practice these would be real images)
    images = torch.randn(3, 3, 224, 224)

    # -------------------------------------------------------------------------
    # 3. Forward pass
    # -------------------------------------------------------------------------
    print("\n[3] Running CGRN forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask,
            images,
            return_reports=True,
        )

    # -------------------------------------------------------------------------
    # 4. Print GDS results
    # -------------------------------------------------------------------------
    LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
    gds    = output.gds_output
    routes = output.routing_output

    print("\n[4] Geometric Dissonance Scores:")
    print(f"    {'Example':<12} {'GDS':>8} {'Cos Sim':>10} {'Angle°':>9} "
          f"{'Branch':<24}")
    print("    " + "-" * 70)
    for i, name in enumerate(["A (harmonious)", "B (sarcastic)", "C (ambiguous)"]):
        branch = "Conflict Branch" if routes.routing_decisions[i] else "Normal Fusion"
        print(
            f"    {name:<22} {gds.gds[i].item():>8.4f} "
            f"{gds.cosine_similarity[i].item():>10.4f} "
            f"{gds.angular_separation_deg[i].item():>9.1f}°  "
            f"{branch}"
        )

    print(f"\n    GDS α = {gds.alpha:.4f}  |  β = {gds.beta:.4f}")
    print(f"    Routing threshold τ = {routes.threshold:.4f}")

    # -------------------------------------------------------------------------
    # 5. Print conflict reports
    # -------------------------------------------------------------------------
    print("\n[5] Conflict Reports:")
    if output.conflict_reports:
        for i, (name, report) in enumerate(
            zip(["A (harmonious)", "B (sarcastic)", "C (ambiguous)"],
                output.conflict_reports)
        ):
            print(f"\n  Example {name}:")
            print(f"    Text   → {report.text_sentiment_label:8s} "
                  f"(conf={report.text_sentiment_confidence:.2f})")
            print(f"    Image  → {report.image_sentiment_label:8s} "
                  f"(conf={report.image_sentiment_confidence:.2f})")
            print(f"    GDS    = {report.gds_score:.4f}")
            print(f"    Route  : {report.routing_path}")
            if report.sarcasm_probability is not None:
                print(f"    Sarcasm: {report.sarcasm_probability:.4f}")
            print(f"    Interpretation: {report.interpretation[:80]}...")
            print(f"    Final  → {report.final_prediction_label:8s} "
                  f"(conf={report.final_prediction_confidence:.2f})")

    # -------------------------------------------------------------------------
    # 6. Demonstrate standalone GDS module
    # -------------------------------------------------------------------------
    print("\n[6] Standalone GDS Module Demo:")
    from src.modules.gds_module import GeometricDissonanceModule

    gds_module = GeometricDissonanceModule(alpha_init=1.0, beta_init=1.0)

    # Construct vectors with known properties
    s_t_agree  = torch.tensor([[0.8, 0.6, 0.0]])   # positive direction
    s_i_agree  = torch.tensor([[0.85, 0.52, 0.1]])  # similar direction
    result_agree  = gds_module(s_t_agree, s_i_agree)

    s_t_oppose = torch.tensor([[0.8, 0.6, 0.0]])
    s_i_oppose = torch.tensor([[-0.8, -0.6, 0.0]])  # opposite direction
    result_oppose = gds_module(s_t_oppose, s_i_oppose)

    print(f"    Agreeing  vectors: GDS={result_agree.gds.item():.4f}  "
          f"(cos={result_agree.cosine_similarity.item():+.4f}, "
          f"θ={result_agree.angular_separation_deg.item():.1f}°)")
    print(f"    Opposing  vectors: GDS={result_oppose.gds.item():.4f}  "
          f"(cos={result_oppose.cosine_similarity.item():+.4f}, "
          f"θ={result_oppose.angular_separation_deg.item():.1f}°)")

    print("\n✓ Demo complete.")
    print("  Run 'python experiments/run_training.py --synthetic' for full training.")
    print("  Run 'python experiments/run_evaluation.py' for evaluation.")


if __name__ == "__main__":
    main()
