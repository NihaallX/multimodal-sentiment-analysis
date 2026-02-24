"""
routing_controller.py — Conflict-Aware Routing Controller
===========================================================

This module implements the dynamic routing mechanism that dispatches
multimodal feature pairs to one of two specialized branches:

  GDS < threshold  →  Normal Fusion Branch  (concatenation + MLP)
  GDS ≥ threshold  →  Conflict Branch        (cross-attention + optional sarcasm head)

Patent Claim Supported:
  "A conflict-aware routing controller that uses a scalar geometric
   dissonance score to dynamically dispatch multimodal inputs to
   specialized processing branches, wherein a conflict branch applies
   cross-attention refinement and a sarcasm detection head when
   cross-modal disagreement exceeds a learnable threshold."

Architecture Details
--------------------
                        GDS score
                           │
              ┌────────────▼────────────┐
              │   RoutingController     │
              │   threshold (learnable) │
              └────────┬────────────────┘
                       │
           ┌───────────┴───────────────┐
      GDS < τ                     GDS ≥ τ
           │                           │
  NormalFusionBranch          ConflictBranch
   [concat → MLP]          [CrossAttention → MLP]
           │                    │ optional SarcasmHead
           └─────────┬──────────┘
                     ▼
              Final logits [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class RoutingOutput:
    """Output from the routing controller."""
    logits: torch.Tensor                      # [B, num_classes]  — final predictions
    routing_decisions: torch.Tensor           # [B] bool — True = conflict branch
    gds_scores: torch.Tensor                  # [B]  — GDS values
    threshold: float                          # current routing threshold value
    normal_branch_logits: Optional[torch.Tensor] = None   # [B_n, num_classes]
    conflict_branch_logits: Optional[torch.Tensor] = None # [B_c, num_classes]
    sarcasm_logits: Optional[torch.Tensor] = None         # [B_c, 2]


# =============================================================================
# Normal Fusion Branch
# =============================================================================

class NormalFusionBranch(nn.Module):
    """
    Low-dissonance branch: simple feature concatenation followed by MLP.

    Input : s_t [B_n, D], s_i [B_n, D]
    Output: logits [B_n, num_classes]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([s_t, s_i], dim=-1)   # [B_n, 2*D]
        return self.mlp(fused)                   # [B_n, num_classes]


# =============================================================================
# Cross-Attention Refinement  (used in Conflict Branch)
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between text and image embeddings.
    Designed for high-dissonance samples where modalities must be reconciled.

    Applies:
      text_refined  = Attention(Q=s_t, K=s_i, V=s_i)
      image_refined = Attention(Q=s_i, K=s_t, V=s_t)

    Then combines refined representations for conflict resolution.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_proj_t  = nn.Linear(embed_dim, embed_dim)
        self.q_proj_i  = nn.Linear(embed_dim, embed_dim)
        self.kv_proj_t = nn.Linear(embed_dim, embed_dim * 2)
        self.kv_proj_i = nn.Linear(embed_dim, embed_dim * 2)
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.out_proj_t = nn.Linear(embed_dim, embed_dim)
        self.out_proj_i = nn.Linear(embed_dim, embed_dim)
        self.norm_t     = nn.LayerNorm(embed_dim)
        self.norm_i     = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

    def _attention(
        self,
        q: torch.Tensor,
        kv_source: torch.Tensor,
        kv_proj: nn.Linear,
    ) -> torch.Tensor:
        """
        Single-direction cross-attention.
        q          : [B, D] — query from the primary modality
        kv_source  : [B, D] — key/value source from the other modality
        """
        B, D = q.shape
        H, dh = self.num_heads, self.head_dim

        # Expand to sequence-of-1 for MHA compatibility
        q_seq  = q.unsqueeze(1)           # [B, 1, D]
        kv_seq = kv_source.unsqueeze(1)   # [B, 1, D]

        kv = kv_proj(kv_seq)              # [B, 1, 2*D]
        k, v = kv.chunk(2, dim=-1)        # [B, 1, D] each

        # Reshape for multi-head: [B, H, 1, dh]
        q_r = q_seq.view(B, 1, H, dh).transpose(1, 2)
        k_r = k.view(B, 1, H, dh).transpose(1, 2)
        v_r = v.view(B, 1, H, dh).transpose(1, 2)

        scale = dh ** -0.5
        attn_w = (q_r @ k_r.transpose(-2, -1)) * scale   # [B, H, 1, 1]
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.dropout(attn_w)

        out = (attn_w @ v_r)              # [B, H, 1, dh]
        out = out.transpose(1, 2).contiguous().view(B, 1, D).squeeze(1)  # [B, D]
        return out

    def forward(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ):
        """
        Returns
        -------
        t_refined : Tensor [B, D]
        i_refined : Tensor [B, D]
        """
        # Text attends to image
        q_t = self.q_proj_t(s_t)
        t_attn_out = self._attention(q_t, s_i, self.kv_proj_i)
        t_out = self.out_proj_t(t_attn_out)
        t_refined = self.norm_t(s_t + t_out)   # residual + norm

        # Image attends to text
        q_i = self.q_proj_i(s_i)
        i_attn_out = self._attention(q_i, s_t, self.kv_proj_t)
        i_out = self.out_proj_i(i_attn_out)
        i_refined = self.norm_i(s_i + i_out)   # residual + norm

        return t_refined, i_refined


# =============================================================================
# Sarcasm Detection Head  (optional auxiliary head in conflict branch)
# =============================================================================

class SarcasmDetectionHead(nn.Module):
    """
    Binary classifier detecting potential sarcasm/irony from conflict features.
    Input: concatenated [t_refined, i_refined, gds_expanded]
    Output: sarcasm logits [B_c, 2]
    """

    def __init__(self, embed_dim: int = 256, gds_embed_dim: int = 32):
        super().__init__()
        in_dim = embed_dim * 2 + gds_embed_dim
        self.gds_embed = nn.Linear(1, gds_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        t_refined: torch.Tensor,
        i_refined: torch.Tensor,
        gds: torch.Tensor,
    ) -> torch.Tensor:
        gds_feat = self.gds_embed(gds.unsqueeze(-1))    # [B_c, gds_embed_dim]
        x = torch.cat([t_refined, i_refined, gds_feat], dim=-1)
        return self.head(x)                              # [B_c, 2]


# =============================================================================
# Conflict Branch
# =============================================================================

class ConflictBranch(nn.Module):
    """
    High-dissonance branch with cross-attention refinement.

    Input : s_t [B_c, D], s_i [B_c, D], gds [B_c]
    Output: logits [B_c, num_classes], sarcasm_logits [B_c, 2] (optional)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_sarcasm_head: bool = True,
        gds_embed_dim: int = 32,
    ):
        super().__init__()
        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.use_sarcasm_head = use_sarcasm_head
        if use_sarcasm_head:
            self.sarcasm_head = SarcasmDetectionHead(embed_dim, gds_embed_dim)

        # GDS is embedded and injected as conditioning signal
        self.gds_embed = nn.Linear(1, gds_embed_dim)

        in_dim = embed_dim * 2 + gds_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
        gds: torch.Tensor,
    ):
        """
        Returns
        -------
        logits         : Tensor [B_c, num_classes]
        sarcasm_logits : Tensor [B_c, 2] or None
        """
        t_ref, i_ref = self.cross_attention(s_t, s_i)   # refined representations
        gds_feat = self.gds_embed(gds.unsqueeze(-1))     # [B_c, gds_embed_dim]

        fused = torch.cat([t_ref, i_ref, gds_feat], dim=-1)   # [B_c, 2D+gds_dim]
        logits = self.classifier(fused)

        sarcasm_logits = None
        if self.use_sarcasm_head:
            sarcasm_logits = self.sarcasm_head(t_ref, i_ref, gds)

        return logits, sarcasm_logits


# =============================================================================
# Routing Controller  (main exported class)
# =============================================================================

class RoutingController(nn.Module):
    """
    Conflict-Aware Routing Controller.

    Dispatches each sample in a batch to either the NormalFusionBranch
    or the ConflictBranch based on a comparison of the GDS score against
    a learnable threshold τ.

    Parameters
    ----------
    embed_dim      : int    — shared embedding dimension.
    num_classes    : int    — number of sentiment classes (default 3).
    threshold_init : float  — initial routing threshold τ (default 0.5).
    learn_threshold: bool   — if True, τ is a learnable parameter.
    use_sarcasm_head: bool  — include sarcasm detection head in ConflictBranch.
    normal_hidden  : int    — hidden dim for NormalFusionBranch MLP.
    conflict_hidden: int    — hidden dim for ConflictBranch MLP.
    num_attn_heads : int    — number of attention heads in ConflictBranch.
    dropout        : float  — dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        threshold_init: float = 0.5,
        learn_threshold: bool = True,
        use_sarcasm_head: bool = True,
        normal_hidden: int = 256,
        conflict_hidden: int = 256,
        num_attn_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_classes = num_classes

        # --- Routing threshold ------------------------------------------------
        if learn_threshold:
            # Learnable τ parameter, clamped to [0.01, 2.0] to stay meaningful
            self.log_threshold = nn.Parameter(torch.tensor(threshold_init))
            self.learn_threshold = True
        else:
            self.register_buffer("_fixed_threshold",
                                  torch.tensor(threshold_init))
            self.learn_threshold = False

        # Temperature for soft routing (higher = sharper routing boundary)
        # Anneals from soft → hard during training naturally as τ learns
        self.register_buffer("temperature", torch.tensor(10.0))

        # --- Specialized branches ---------------------------------------------
        self.normal_branch = NormalFusionBranch(
            embed_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=normal_hidden,
            dropout=dropout,
        )
        self.conflict_branch = ConflictBranch(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_attn_heads,
            hidden_dim=conflict_hidden,
            dropout=dropout,
            use_sarcasm_head=use_sarcasm_head,
        )

    # -------------------------------------------------------------------------
    @property
    def threshold(self) -> torch.Tensor:
        if self.learn_threshold:
            return self.log_threshold.clamp(min=0.01, max=2.0)
        return self._fixed_threshold

    # -------------------------------------------------------------------------
    def forward(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
        gds: torch.Tensor,
    ) -> RoutingOutput:
        """
        Parameters
        ----------
        s_t : Tensor [B, D]  — text sentiment embedding
        s_i : Tensor [B, D]  — image sentiment embedding
        gds : Tensor [B]     — scalar GDS scores

        Returns
        -------
        RoutingOutput  — logits, routing decisions, and branch-specific outputs
        """
        tau = self.threshold   # scalar tensor

        # Hard routing decisions (used for stats and inference)
        routing_decisions = (gds >= tau).detach()   # [B] bool — True = conflict

        # --- Always compute both branches so gradients flow to τ --------------
        # Soft routing weight via sigmoid: p_conflict ∈ (0,1), differentiable in τ
        p_conflict = torch.sigmoid((gds - tau) * self.temperature)  # [B]
        p_normal   = 1.0 - p_conflict                                # [B]

        # Normal Fusion: all samples
        normal_logits  = self.normal_branch(s_t, s_i)               # [B, C]
        # Conflict Branch: all samples
        conflict_logits, sarcasm_logits_out = self.conflict_branch(s_t, s_i, gds)  # [B, C]

        if self.training:
            # Soft blend — τ gets gradients through p_conflict
            logits = (p_normal.unsqueeze(-1) * normal_logits +
                      p_conflict.unsqueeze(-1) * conflict_logits)    # [B, C]
        else:
            # Hard routing at inference — no ambiguity
            logits = torch.where(
                routing_decisions.unsqueeze(-1).expand_as(normal_logits),
                conflict_logits,
                normal_logits,
            )

        return RoutingOutput(
            logits=logits,
            routing_decisions=routing_decisions,
            gds_scores=gds,
            threshold=tau.item(),
            normal_branch_logits=normal_logits,
            conflict_branch_logits=conflict_logits,
            sarcasm_logits=sarcasm_logits_out,
        )

    # -------------------------------------------------------------------------
    def routing_stats(self, routing_decisions: torch.Tensor) -> dict:
        """Log routing distribution for a batch."""
        n_normal   = (~routing_decisions).sum().item()
        n_conflict = routing_decisions.sum().item()
        n_total    = routing_decisions.shape[0]
        return {
            "n_normal":       n_normal,
            "n_conflict":     n_conflict,
            "conflict_ratio": n_conflict / max(n_total, 1),
            "threshold":      self.threshold.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold.item():.4f}, "
            f"learn_threshold={self.learn_threshold}, "
            f"embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}"
        )
