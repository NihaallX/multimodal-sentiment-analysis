"""
CGRN — Text Encoder
====================
Wraps DistilBERT (default) or MiniLM to produce a fixed-size,
L2-normalized sentiment embedding vector.

Architecture
------------
  DistilBERT / MiniLM
       ↓  [CLS] hidden state  (768 or 384 dim)
  Linear projection  →  embed_dim  (default 256)
  LayerNorm
  L2 Normalization
       ↓
  S_t  ∈ R^{embed_dim}   (unit-normalized sentiment vector)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """
    Parameters
    ----------
    model_name : str
        HuggingFace model id.
        Recommended lightweight options:
          - 'distilbert-base-uncased'          (66M params)
          - 'sentence-transformers/all-MiniLM-L6-v2'  (22M params)
    embed_dim : int
        Output sentiment embedding dimension (default 256).
    freeze_backbone : bool
        If True, backbone weights are frozen; only projection head trains.
    dropout : float
        Dropout applied before projection.
    """

    SUPPORTED_MODELS = {
        "distilbert-base-uncased": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-MiniLM-L3-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim

        # --- Backbone ---------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Infer hidden size
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # --- Projection head --------------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # --- Unimodal classification head (optional, used during Stage 1) ----
        self.classifier = nn.Linear(embed_dim, 3)   # 3-class: neg/neu/pos

        self._init_weights()

    # -------------------------------------------------------------------------
    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # -------------------------------------------------------------------------
    def tokenize(self, texts, max_length: int = 128, device=None):
        """
        Tokenize a list of strings and return tensors on `device`.
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        return encoding

    # -------------------------------------------------------------------------
    def encode(self, input_ids, attention_mask, normalize: bool = True):
        """
        Forward-pass through backbone + projection.

        Returns
        -------
        embedding : Tensor  [B, embed_dim]
            L2-normalized sentiment vector if normalize=True.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token hidden state
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # [B, hidden]
        x = self.dropout(cls_hidden)
        x = self.proj(x)                                   # [B, embed_dim]
        x = self.norm(x)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    # -------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, normalize: bool = True):
        """
        Returns
        -------
        embedding : Tensor [B, embed_dim]   — sentiment vector S_t
        logits    : Tensor [B, 3]           — unimodal classification logits
        """
        emb = self.encode(input_ids, attention_mask, normalize=normalize)
        logits = self.classifier(emb)
        return emb, logits

    # -------------------------------------------------------------------------
    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        proj_params = sum(p.numel() for p in self.proj.parameters()) + \
                      sum(p.numel() for p in self.norm.parameters()) + \
                      sum(p.numel() for p in self.classifier.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_params,
            "projection_head": proj_params,
        }


# =============================================================================
# Convenience factory
# =============================================================================

def build_text_encoder(
    model_name: str = "distilbert-base-uncased",
    embed_dim: int = 256,
    freeze_backbone: bool = False,
) -> TextEncoder:
    """Factory function for TextEncoder."""
    return TextEncoder(
        model_name=model_name,
        embed_dim=embed_dim,
        freeze_backbone=freeze_backbone,
    )
