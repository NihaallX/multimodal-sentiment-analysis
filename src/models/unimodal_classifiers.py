"""
unimodal_classifiers.py — Stage 1: Independent Unimodal Classifiers
====================================================================

Wraps the text and image encoders as standalone sentiment classifiers
for Stage 1 training. Also provides evaluation helpers.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class UnimodalTextClassifier(nn.Module):
    """Standalone text sentiment classifier using TextEncoder."""

    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        emb, logits = self.encoder(input_ids, attention_mask)
        return logits

    def get_embedding(self, input_ids, attention_mask):
        emb, _ = self.encoder(input_ids, attention_mask)
        return emb


class UnimodalImageClassifier(nn.Module):
    """Standalone image sentiment classifier using ImageEncoder."""

    def __init__(self, image_encoder):
        super().__init__()
        self.encoder = image_encoder

    def forward(self, images):
        emb, logits = self.encoder(images)
        return logits

    def get_embedding(self, images):
        emb, _ = self.encoder(images)
        return emb
