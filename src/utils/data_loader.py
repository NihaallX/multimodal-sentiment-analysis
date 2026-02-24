"""
data_loader.py — Multimodal Sentiment Dataset Utilities
=========================================================

Provides:
  - MultimodalSentimentDataset: Torch Dataset for (text, image, label) triples
  - SyntheticConflictDataset: Synthetic dataset with known sarcasm/conflict samples
  - build_dataloaders(): factory for train/val/test splits
  - CollatorFn: custom collation with tokenization

Supported Dataset Formats
--------------------------
The dataset expects a CSV/JSON file with columns:
  'text'    : str  — raw text
  'image'   : str  — path to image file
  'label'   : int  — 0=Negative, 1=Neutral, 2=Positive
  'conflict': int  — optional, 1=known contradictory/sarcastic sample
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Callable, List, Dict, Tuple
from transformers import AutoTokenizer


# =============================================================================
# Main Dataset
# =============================================================================

class MultimodalSentimentDataset(Dataset):
    """
    Multimodal Sentiment Dataset.

    Parameters
    ----------
    data        : List[dict] — list of {text, image_path, label, [conflict]} dicts
    tokenizer   : HuggingFace tokenizer
    image_transform : torchvision transforms
    max_length  : int — max text token length
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        image_transform: Callable,
        max_length: int = 128,
    ):
        self.data       = data
        self.tokenizer  = tokenizer
        self.transform  = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        text       = item["text"]
        label      = int(item["label"])
        conflict   = int(item.get("conflict", 0))

        # --- Text tokenization -----------------------------------------------
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # --- Image loading ---------------------------------------------------
        image_path = item.get("image_path", None)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Fallback: random noise image (for testing / synthetic mode)
            image = Image.fromarray(
                (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            )

        if self.transform:
            image = self.transform(image)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "images":         image,
            "labels":         torch.tensor(label, dtype=torch.long),
            "conflict":       torch.tensor(conflict, dtype=torch.long),
            "text":           text,
        }


# =============================================================================
# Synthetic Conflict Dataset (for testing and demo)
# =============================================================================

class SyntheticConflictDataset(Dataset):
    """
    Generates synthetic (text, image, label) triples with controllable
    conflict ratio for testing the GDS module and routing controller.

    Each sample embeds sentiment directly as vectors (bypasses actual
    text/image content — useful for unit tests and ablation studies).
    """

    def __init__(
        self,
        n_samples:      int   = 1000,
        embed_dim:      int   = 256,
        conflict_ratio: float = 0.3,
        seed:           int   = 42,
    ):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)

        self.n         = n_samples
        self.embed_dim = embed_dim
        self.labels    = torch.randint(0, 3, (n_samples,))
        self.conflict  = (torch.rand(n_samples) < conflict_ratio).long()

        # Generate embedding pairs
        self.text_embs  = torch.randn(n_samples, embed_dim)
        self.image_embs = torch.randn(n_samples, embed_dim)

        # For conflict samples: make image embedding anti-correlated with text
        for i in range(n_samples):
            if self.conflict[i]:
                self.image_embs[i] = -self.text_embs[i] + 0.3 * torch.randn(embed_dim)

        # Normalize embeddings
        import torch.nn.functional as F
        self.text_embs  = F.normalize(self.text_embs, p=2, dim=-1)
        self.image_embs = F.normalize(self.image_embs, p=2, dim=-1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "text_emb":   self.text_embs[idx],
            "image_emb":  self.image_embs[idx],
            "labels":     self.labels[idx],
            "conflict":   self.conflict[idx],
        }


# =============================================================================
# Dataset factory from JSON/CSV
# =============================================================================

def load_dataset_from_json(json_path: str) -> List[Dict]:
    """Load dataset from a JSON file (list of dicts)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_dataset_from_csv(csv_path: str) -> List[Dict]:
    """Load dataset from a CSV file."""
    import csv
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            if "conflict" in row:
                row["conflict"] = int(row["conflict"])
            data.append(row)
    return data


def train_val_test_split(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List, List, List]:
    """Split data into train/val/test."""
    random.seed(seed)
    data_copy = list(data)
    random.shuffle(data_copy)
    n     = len(data_copy)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        data_copy[:n_tr],
        data_copy[n_tr: n_tr + n_val],
        data_copy[n_tr + n_val:],
    )


def build_dataloaders(
    data_path:       str,
    tokenizer_name:  str   = "distilbert-base-uncased",
    image_transform_train = None,
    image_transform_val   = None,
    batch_size:      int   = 32,
    num_workers:     int   = 0,
    max_length:      int   = 128,
    train_ratio:     float = 0.7,
    val_ratio:       float = 0.15,
    seed:            int   = 42,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from a JSON or CSV file.

    Returns
    -------
    dict with keys: 'train', 'val', 'test'
    """
    from torchvision import transforms
    from ..encoders.image_encoder import ImageEncoder

    # Load raw data
    if data_path.endswith(".json"):
        raw = load_dataset_from_json(data_path)
    elif data_path.endswith(".csv"):
        raw = load_dataset_from_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    train_data, val_data, test_data = train_val_test_split(
        raw, train_ratio, val_ratio, seed
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if image_transform_train is None:
        image_transform_train = ImageEncoder.get_transforms("train")
    if image_transform_val is None:
        image_transform_val   = ImageEncoder.get_transforms("val")

    datasets = {
        "train": MultimodalSentimentDataset(
            train_data, tokenizer, image_transform_train, max_length
        ),
        "val": MultimodalSentimentDataset(
            val_data, tokenizer, image_transform_val, max_length
        ),
        "test": MultimodalSentimentDataset(
            test_data, tokenizer, image_transform_val, max_length
        ),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


# =============================================================================
# Synthetic data generation (for demo / unit tests)
# =============================================================================

def generate_synthetic_csv(
    out_path:       str   = "data/synthetic_sentiment.csv",
    n_samples:      int   = 500,
    conflict_ratio: float = 0.3,
    seed:           int   = 42,
):
    """
    Generates a synthetic CSV dataset for testing.
    Images are saved as random noise PNGs.
    """
    import csv
    import os
    from PIL import Image as PILImage
    random.seed(seed)
    np.random.seed(seed)

    TEMPLATES = {
        0: ["This is terrible.", "I hated every moment.", "Absolutely disappointing.",
            "Dreadful experience.", "Would not recommend."],
        1: ["It was okay.", "Nothing special.", "Average at best.",
            "Mediocre performance.", "Neither good nor bad."],
        2: ["I loved this!", "Absolutely amazing!", "Fantastic experience!",
            "Highly recommended.", "Superb quality!"],
    }

    # Sarcastic / conflict templates
    SARCASTIC = [
        ("Oh great, another broken product.", 2, 0),   # text=positive-sounding, true=negative
        ("Yeah sure, best thing ever.", 2, 0),
        ("Wonderful, just wonderful.", 2, 0),
        ("Oh this is just perfect.", 2, 0),
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img_dir = os.path.join(os.path.dirname(out_path) or ".", "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(n_samples):
        is_conflict = (random.random() < conflict_ratio)
        if is_conflict and SARCASTIC:
            text, text_label, true_label = random.choice(SARCASTIC)
            label     = true_label
            conflict  = 1
        else:
            label     = random.randint(0, 2)
            text      = random.choice(TEMPLATES[label])
            conflict  = 0

        # Generate random image
        img_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        img_path  = os.path.join(img_dir, f"sample_{i:04d}.png")
        PILImage.fromarray(img_array).save(img_path)

        rows.append({
            "text":       text,
            "image_path": img_path,
            "label":      label,
            "conflict":   conflict,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "image_path", "label", "conflict"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n_samples} samples → {out_path}")
    print(f"  Conflict/sarcasm samples: {sum(r['conflict'] for r in rows)}")
    return out_path
