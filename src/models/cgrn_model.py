"""
cgrn_model.py — Conflict-Aware Geometric Routing Network (CGRN)
================================================================

Full end-to-end model integrating:
  1. TextEncoder   (DistilBERT/MiniLM  →  S_t ∈ R^D)
  2. ImageEncoder  (MobileNetV3        →  S_i ∈ R^D)
  3. GeometricDissonanceModule         →  GDS scalar D ∈ R
  4. RoutingController                 →  dynamic branch dispatch
  5. ExplainabilityModule              →  per-inference conflict report

Forward pass data flow:
  texts, images
      │
  ┌───┴───────────────────┐
  TextEncoder          ImageEncoder
  (S_t, text_logits)   (S_i, img_logits)
  └───────────┬──────────┘
              │
  GeometricDissonanceModule(S_t, S_i)
              │  GDSOutput
  RoutingController(S_t, S_i, GDS)
              │  RoutingOutput
         final_logits
              │
     ExplainabilityModule
              │
      ConflictReport[]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..encoders.text_encoder import TextEncoder
from ..encoders.image_encoder import ImageEncoder
from ..modules.gds_module import GeometricDissonanceModule, GDSOutput
from ..modules.routing_controller import RoutingController, RoutingOutput
from ..modules.explainability_module import ExplainabilityModule, ConflictReport


# =============================================================================
# CGRN output container
# =============================================================================

@dataclass
class CGRNOutput:
    """Complete output of a CGRN forward pass."""
    final_logits: torch.Tensor           # [B, num_classes]
    text_logits: torch.Tensor            # [B, num_classes]  — unimodal
    image_logits: torch.Tensor           # [B, num_classes]  — unimodal
    text_embeddings: torch.Tensor        # [B, D]
    image_embeddings: torch.Tensor       # [B, D]
    gds_output: GDSOutput
    routing_output: RoutingOutput
    conflict_reports: Optional[List[ConflictReport]] = None


# =============================================================================
# CGRN Model
# =============================================================================

class CGRNModel(nn.Module):
    """
    Conflict-Aware Geometric Routing Network.

    Parameters
    ----------
    text_model_name : str
        HuggingFace model id for text encoder.
    image_backbone  : str
        Backbone name for image encoder.
    embed_dim       : int
        Shared embedding dimension for both modalities.
    num_classes     : int
        Number of sentiment classes (default 3).
    freeze_text_backbone  : bool
        Freeze text encoder backbone weights.
    freeze_image_backbone : bool
        Freeze image encoder backbone weights.
    gds_alpha_init  : float
        Initial α for GDS (cosine dissimilarity weight).
    gds_beta_init   : float
        Initial β for GDS (magnitude difference weight).
    use_projection_residual : bool
        Include optional projection residual term in GDS.
    routing_threshold : float
        Initial routing threshold τ.
    learn_threshold   : bool
        Whether τ is a learnable parameter.
    use_sarcasm_head  : bool
        Include sarcasm detection head in ConflictBranch.
    generate_reports  : bool
        Whether to auto-generate ConflictReports during forward pass.
    num_attn_heads    : int
        Number of attention heads in CrossModalAttention.
    dropout           : float
        Dropout rate.
    """

    def __init__(
        self,
        text_model_name:         str   = "distilbert-base-uncased",
        image_backbone:          str   = "mobilenet_v3_small",
        embed_dim:               int   = 256,
        num_classes:             int   = 3,
        freeze_text_backbone:    bool  = False,
        freeze_image_backbone:   bool  = False,
        gds_alpha_init:          float = 1.0,
        gds_beta_init:           float = 1.0,
        use_projection_residual: bool  = False,
        routing_threshold:       float = 0.5,
        learn_threshold:         bool  = True,
        use_sarcasm_head:        bool  = True,
        generate_reports:        bool  = False,
        num_attn_heads:          int   = 4,
        dropout:                 float = 0.2,
    ):
        super().__init__()
        self.embed_dim       = embed_dim
        self.num_classes     = num_classes
        self.generate_reports = generate_reports

        # --- Encoders --------------------------------------------------------
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
            freeze_backbone=freeze_text_backbone,
            dropout=dropout,
        )
        self.image_encoder = ImageEncoder(
            backbone_name=image_backbone,
            embed_dim=embed_dim,
            freeze_backbone=freeze_image_backbone,
            dropout=dropout,
        )

        # --- Geometric Dissonance Module ------------------------------------
        self.gds_module = GeometricDissonanceModule(
            alpha_init=gds_alpha_init,
            beta_init=gds_beta_init,
            use_projection_residual=use_projection_residual,
        )

        # --- Routing Controller ---------------------------------------------
        self.routing_controller = RoutingController(
            embed_dim=embed_dim,
            num_classes=num_classes,
            threshold_init=routing_threshold,
            learn_threshold=learn_threshold,
            use_sarcasm_head=use_sarcasm_head,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
        )

        # --- Explainability (non-trainable) ---------------------------------
        self.explainer = ExplainabilityModule()

    # -------------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,           # [B, seq_len]
        attention_mask: torch.Tensor,           # [B, seq_len]
        images:         torch.Tensor,           # [B, 3, H, W]
        return_reports: bool = False,
    ) -> CGRNOutput:
        """
        Full CGRN forward pass.

        Parameters
        ----------
        input_ids      : tokenized text ids [B, seq_len]
        attention_mask : attention mask [B, seq_len]
        images         : image tensor [B, 3, H, W]
        return_reports : if True, auto-generate ConflictReports

        Returns
        -------
        CGRNOutput
        """
        # Stage 1: Encode both modalities independently
        s_t, text_logits  = self.text_encoder(input_ids, attention_mask)
        s_i, image_logits = self.image_encoder(images)

        # Stage 2: Compute Geometric Dissonance Score
        gds_output = self.gds_module(s_t, s_i)

        # Stage 3: Dynamic routing and final prediction
        routing_output = self.routing_controller(s_t, s_i, gds_output.gds)

        # Stage 4: Optional conflict reports
        reports = None
        if return_reports or self.generate_reports:
            # Align sarcasm logits to full batch (None for normal-routed samples)
            batch_sarc = self._align_sarcasm_logits(
                routing_output, s_t.shape[0]
            )
            reports = self.explainer.generate_reports(
                text_logits=text_logits,
                image_logits=image_logits,
                gds_output=gds_output,
                routing_output=routing_output,
                final_logits=routing_output.logits,
                sarcasm_logits=batch_sarc,
            )

        return CGRNOutput(
            final_logits=routing_output.logits,
            text_logits=text_logits,
            image_logits=image_logits,
            text_embeddings=s_t,
            image_embeddings=s_i,
            gds_output=gds_output,
            routing_output=routing_output,
            conflict_reports=reports,
        )

    # -------------------------------------------------------------------------
    def _align_sarcasm_logits(
        self,
        routing_output: RoutingOutput,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Builds a [B, 2] sarcasm logits tensor aligned to the full batch.
        Samples routed to Normal branch get zero-logits (indicating no sarcasm).
        """
        if routing_output.sarcasm_logits is None:
            return None

        device = routing_output.logits.device
        sarc_logits = torch.zeros(batch_size, 2, device=device)
        conflict_mask = routing_output.routing_decisions
        sarc_logits[conflict_mask] = routing_output.sarcasm_logits
        return sarc_logits

    # -------------------------------------------------------------------------
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Standalone text embedding (for retrieval / unimodal inference)."""
        return self.text_encoder.encode(input_ids, attention_mask, normalize)

    def encode_image(
        self,
        images: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Standalone image embedding (for retrieval / unimodal inference)."""
        return self.image_encoder.encode(images, normalize)

    # -------------------------------------------------------------------------
    def compute_gds(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ) -> GDSOutput:
        """Standalone GDS computation."""
        return self.gds_module(s_t, s_i)

    # -------------------------------------------------------------------------
    def param_count(self) -> Dict[str, Any]:
        """Detailed parameter count per component."""
        def _count(m):
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        text_cnt  = _count(self.text_encoder)
        img_cnt   = _count(self.image_encoder)
        gds_cnt   = _count(self.gds_module)
        route_cnt = _count(self.routing_controller)
        total     = _count(self)

        return {
            "text_encoder":       text_cnt,
            "image_encoder":      img_cnt,
            "gds_module":         gds_cnt,
            "routing_controller": route_cnt,
            "full_model":         total,
        }

    def extra_repr(self) -> str:
        counts = self.param_count()
        total  = counts["full_model"]["total"]
        return f"total_params={total:,}"


# =============================================================================
# Configuration helper
# =============================================================================

class CGRNConfig:
    """Serializable configuration for CGRN."""

    DEFAULT = dict(
        text_model_name         = "distilbert-base-uncased",
        image_backbone          = "mobilenet_v3_small",
        embed_dim               = 256,
        num_classes             = 3,
        freeze_text_backbone    = False,
        freeze_image_backbone   = False,
        gds_alpha_init          = 1.0,
        gds_beta_init           = 1.0,
        use_projection_residual = False,
        routing_threshold       = 0.5,
        learn_threshold         = True,
        use_sarcasm_head        = True,
        generate_reports        = False,
        num_attn_heads          = 4,
        dropout                 = 0.2,
    )

    LIGHTWEIGHT = dict(
        text_model_name         = "sentence-transformers/all-MiniLM-L6-v2",
        image_backbone          = "mobilenet_v3_small",
        embed_dim               = 128,
        num_classes             = 3,
        freeze_text_backbone    = False,
        freeze_image_backbone   = False,
        gds_alpha_init          = 1.0,
        gds_beta_init           = 0.5,
        use_projection_residual = False,
        routing_threshold       = 0.4,
        learn_threshold         = True,
        use_sarcasm_head        = False,
        generate_reports        = False,
        num_attn_heads          = 2,
        dropout                 = 0.1,
    )

    @staticmethod
    def build(config: dict = None) -> "CGRNModel":
        cfg = CGRNConfig.DEFAULT.copy()
        if config:
            cfg.update(config)
        return CGRNModel(**cfg)
