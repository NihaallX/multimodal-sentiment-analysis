"""
gds_module.py — Geometric Dissonance Score (GDS) Module
=========================================================

Core Novel Module of CGRN.

Given independent sentiment embedding vectors from text (S_t) and image (S_i),
the GDS module computes a scalar dissimilarity score:

    D = α · (1 - cos(S_t, S_i))  +  β · |‖S_t‖ - ‖S_i‖|

Where:
  - cos(S_t, S_i) : cosine similarity between the two vectors
  - ‖·‖           : L2 norm (magnitude)
  - α, β          : learnable non-negative scalar weights

Properties:
  - Differentiable end-to-end
  - Pluggable: works on any pair of same-dim embeddings
  - Loggable: all intermediate quantities are returned for explainability
  - D ∈ [0, α + β·max_mag_diff] — bounded and interpretable

Additional geometric quantities logged:
  - Angular separation θ = arccos(cos_sim)
  - Projection residual: component of S_i orthogonal to S_t

Patent Claim Supported:
  "A differentiable geometric dissonance module that computes a structured
   disagreement score from independent modality-specific sentiment vectors
   using learnable weights applied to cosine dissimilarity and magnitude
   difference, used as a routing signal in a multimodal inference pipeline."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# GDS Detail Container  (for explainability pipeline)
# =============================================================================

@dataclass
class GDSOutput:
    """
    All geometric quantities computed by the GDS module.
    Returned alongside the scalar GDS for logging and explainability.
    """
    gds: torch.Tensor                      # [B] — scalar dissonance score
    cosine_similarity: torch.Tensor        # [B] — cos(S_t, S_i)
    cosine_dissimilarity: torch.Tensor     # [B] — 1 - cos(S_t, S_i)
    angular_separation_deg: torch.Tensor   # [B] — θ in degrees
    magnitude_text: torch.Tensor           # [B] — ‖S_t‖
    magnitude_image: torch.Tensor          # [B] — ‖S_i‖
    magnitude_difference: torch.Tensor     # [B] — |‖S_t‖ - ‖S_i‖|
    projection_residual: torch.Tensor      # [B] — ‖S_i - proj_{S_t}(S_i)‖
    alpha: float                           # current α weight
    beta: float                            # current β weight


# =============================================================================
# Core GDS Module
# =============================================================================

class GeometricDissonanceModule(nn.Module):
    """
    Geometric Dissonance Score Module.

    Parameters
    ----------
    alpha_init : float
        Initial value for learnable α weight (cosine dissimilarity term).
    beta_init : float
        Initial value for learnable β weight (magnitude difference term).
    eps : float
        Numerical stability epsilon for normalization and arccos.
    use_projection_residual : bool
        If True, adds a third term: γ · projection_residual.
    gamma_init : float
        Initial value for optional γ weight (projection residual term).
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        eps: float = 1e-8,
        use_projection_residual: bool = False,
        gamma_init: float = 0.5,
    ):
        super().__init__()
        self.eps = eps
        self.use_projection_residual = use_projection_residual

        # Learnable parameters — stored as log to enforce positivity
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.log_beta  = nn.Parameter(torch.tensor(math.log(beta_init)))

        if use_projection_residual:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma_init)))

    # -------------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        """α — always positive via exp."""
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        """β — always positive via exp."""
        return self.log_beta.exp()

    @property
    def gamma(self) -> Optional[torch.Tensor]:
        """γ — always positive via exp (only if projection residual enabled)."""
        if self.use_projection_residual:
            return self.log_gamma.exp()
        return None

    # -------------------------------------------------------------------------
    def _cosine_similarity(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Numerically stable cosine similarity.
        s_t, s_i : [B, D]  — need not be pre-normalized
        returns   : [B]
        """
        s_t_norm = F.normalize(s_t, p=2, dim=-1, eps=self.eps)
        s_i_norm = F.normalize(s_i, p=2, dim=-1, eps=self.eps)
        return (s_t_norm * s_i_norm).sum(dim=-1)   # [B]

    # -------------------------------------------------------------------------
    def _angular_separation(self, cos_sim: torch.Tensor) -> torch.Tensor:
        """
        Computes angular separation θ in degrees from cosine similarity.
        Clamps to [-1+eps, 1-eps] to avoid arccos domain errors.
        """
        cos_clamped = cos_sim.clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta_rad = torch.acos(cos_clamped)      # [B]
        return torch.rad2deg(theta_rad)           # [B]

    # -------------------------------------------------------------------------
    def _projection_residual(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the L2 norm of the component of S_i orthogonal to S_t.
        Quantifies how much of S_i lies "outside" the direction of S_t.

        proj_{S_t}(S_i) = (S_i · ŝ_t) · ŝ_t
        residual        = S_i - proj_{S_t}(S_i)
        """
        s_t_hat = F.normalize(s_t, p=2, dim=-1, eps=self.eps)  # [B, D]
        scalar_proj = (s_i * s_t_hat).sum(dim=-1, keepdim=True)  # [B, 1]
        projection  = scalar_proj * s_t_hat                       # [B, D]
        residual    = s_i - projection                            # [B, D]
        return residual.norm(p=2, dim=-1)                         # [B]

    # -------------------------------------------------------------------------
    def forward(
        self,
        s_t: torch.Tensor,
        s_i: torch.Tensor,
    ) -> GDSOutput:
        """
        Compute Geometric Dissonance Score.

        Parameters
        ----------
        s_t : Tensor [B, D]  — text sentiment embedding (pre-normalized or raw)
        s_i : Tensor [B, D]  — image sentiment embedding (pre-normalized or raw)

        Returns
        -------
        GDSOutput dataclass with:
          .gds : Tensor [B]  — scalar dissonance score per sample
          (all intermediate geometric quantities included for explainability)
        """
        # --- Magnitudes -------------------------------------------------------
        mag_t = s_t.norm(p=2, dim=-1)   # [B]
        mag_i = s_i.norm(p=2, dim=-1)   # [B]
        mag_diff = (mag_t - mag_i).abs()  # [B]

        # --- Angular quantities -----------------------------------------------
        cos_sim  = self._cosine_similarity(s_t, s_i)       # [B]
        cos_diss = 1.0 - cos_sim                            # [B]  ∈ [0, 2]
        angle_deg = self._angular_separation(cos_sim)       # [B]

        # --- Projection residual (optional) -----------------------------------
        proj_res = self._projection_residual(s_t, s_i)     # [B]

        # --- GDS computation --------------------------------------------------
        #   D = α · (1 - cos(S_t, S_i))  +  β · |‖S_t‖ - ‖S_i‖|
        gds = self.alpha * cos_diss + self.beta * mag_diff

        if self.use_projection_residual:
            gds = gds + self.gamma * proj_res

        return GDSOutput(
            gds=gds,
            cosine_similarity=cos_sim,
            cosine_dissimilarity=cos_diss,
            angular_separation_deg=angle_deg,
            magnitude_text=mag_t,
            magnitude_image=mag_i,
            magnitude_difference=mag_diff,
            projection_residual=proj_res,
            alpha=self.alpha.item(),
            beta=self.beta.item(),
        )

    # -------------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha.item():.4f}, beta={self.beta.item():.4f}, "
            f"use_projection_residual={self.use_projection_residual}"
        )


# =============================================================================
# GDS Statistics Logger  (for training monitoring and Phase 6 logging)
# =============================================================================

class GDSStatisticsLogger:
    """
    Accumulates per-batch GDS statistics for analysis during training.
    Call .update() each batch, .summary() at epoch end.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._gds_vals       = []
        self._cos_sim_vals   = []
        self._angle_vals     = []
        self._routing_normal = 0
        self._routing_conflict = 0
        self._n = 0

    def update(self, gds_output: GDSOutput, routing_decisions: torch.Tensor):
        """
        Parameters
        ----------
        gds_output        : GDSOutput from the module.
        routing_decisions : Tensor [B] bool — True=conflict branch, False=normal.
        """
        self._gds_vals.append(gds_output.gds.detach().cpu())
        self._cos_sim_vals.append(gds_output.cosine_similarity.detach().cpu())
        self._angle_vals.append(gds_output.angular_separation_deg.detach().cpu())
        self._routing_normal   += (~routing_decisions).sum().item()
        self._routing_conflict += routing_decisions.sum().item()
        self._n += routing_decisions.shape[0]

    def summary(self) -> dict:
        if self._n == 0:
            return {}
        all_gds   = torch.cat(self._gds_vals)
        all_cos   = torch.cat(self._cos_sim_vals)
        all_angle = torch.cat(self._angle_vals)
        return {
            "gds_mean":          all_gds.mean().item(),
            "gds_std":           all_gds.std().item(),
            "gds_min":           all_gds.min().item(),
            "gds_max":           all_gds.max().item(),
            "cos_sim_mean":      all_cos.mean().item(),
            "angle_mean_deg":    all_angle.mean().item(),
            "routing_normal":    self._routing_normal,
            "routing_conflict":  self._routing_conflict,
            "conflict_ratio":    self._routing_conflict / max(self._n, 1),
        }
