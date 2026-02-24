"""
mvsa_loader.py — MVSA-Multiple Dataset Loader
==============================================

Handles the MVSA-Multiple (~19k samples) and MVSA-Single (~5k) datasets.

Dataset structure (after downloading and extracting):
----------------------------------------------------
MVSA_Multiple/
├── data/
│   ├── 1/
│   │   ├── 1.jpg
│   │   └── 1.txt
│   ├── 2/
│   │   ├── 2.jpg
│   │   └── 2.txt
│   └── ...
└── labelResultAll.txt   ← multiple annotators per sample

MVSA_Single/
├── data/
│   ├── 1/ ...
└── labelResult.txt      ← single consensus label per sample

Label format in annotation file
--------------------------------
MVSA-Multiple:  id, [text_label, img_label] x3 annotators
MVSA-Single:    id, text_label, img_label

Label encoding:  positive=1 → 0,  negative=2 → 1,  neutral=3 → 2
Conflict flag:   text_label ≠ img_label after majority vote

Download MVSA-Multiple:
  http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
  or via: https://Drive.google.com/...  (search "MVSA dataset")

Usage
-----
  from src.utils.mvsa_loader import build_mvsa_dataloaders

  loaders = build_mvsa_dataloaders(
      data_root="data/MVSA_Multiple",
      variant="multiple",   # or "single"
      batch_size=32,
  )
  train_loader = loaders["train"]
"""

import os
import re
import csv
import random
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ─── Label mappings ────────────────────────────────────────────────────────────
RAW_TO_IDX = {"positive": 0, "negative": 1, "neutral": 2,
               "1": 0, "2": 1, "3": 2,
               1: 0, 2: 1, 3: 2}
IDX_TO_NAME = {0: "positive", 1: "negative", 2: "neutral"}
NUM_CLASSES  = 3

# ─── Image transforms ──────────────────────────────────────────────────────────
def get_image_transforms(split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def clean_tweet(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols, extra whitespace from tweets."""
    text = re.sub(r"http\S+|www\S+", "", text)      # URLs
    text = re.sub(r"@\w+", "", text)                 # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)            # #hashtag → hashtag
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# Annotation Parsing
# =============================================================================

def _parse_label(raw) -> int:
    """Convert raw annotation value to class index 0/1/2."""
    key = str(raw).strip().lower()
    if key in RAW_TO_IDX:
        return RAW_TO_IDX[key]
    raise ValueError(f"Unknown label: {raw!r}")


def _majority(labels: List[int]) -> int:
    """Return majority vote; break ties toward neutral (2)."""
    c = Counter(labels)
    if not c:
        return 2
    max_count = max(c.values())
    candidates = [l for l, cnt in c.items() if cnt == max_count]
    if 2 in candidates:
        return 2      # neutral as tie-breaker
    return sorted(candidates)[0]


def parse_mvsa_multiple_annotations(label_file: str) -> Dict[str, dict]:
    """
    Parse MVSA-Multiple annotation file.

    Expected format per line (CSV, no header):
        id, text1, img1, text2, img2, text3, img3
    where labels are: positive / negative / neutral  (or 1/2/3)

    Returns dict: {sample_id → {text_label, image_label, conflict, valid}}
    """
    samples = {}
    with open(label_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row]
            if not row or row[0].lower() in ("id", ""):
                continue  # skip header / blank
            try:
                sid = row[0]
                text_labels  = [_parse_label(row[i]) for i in [1, 3, 5] if i < len(row)]
                image_labels = [_parse_label(row[i]) for i in [2, 4, 6] if i < len(row)]

                text_maj  = _majority(text_labels)
                image_maj = _majority(image_labels)
                conflict  = int(text_maj != image_maj)

                # Skip samples where any annotator marked invalid ("invalid")
                invalid_flags = [str(r).strip().lower() for r in row[1:]]
                valid = not any(f in ("invalid", "-1", "") for f in invalid_flags)

                samples[sid] = {
                    "text_label":  text_maj,
                    "image_label": image_maj,
                    "final_label": text_maj,   # use text majority as ground truth (common practice)
                    "conflict":    conflict,
                    "valid":       valid,
                }
            except (IndexError, ValueError) as e:
                logger.debug(f"Skipping malformed row: {row} ({e})")
    return samples


def parse_mvsa_single_annotations(label_file: str) -> Dict[str, dict]:
    """
    Parse MVSA-Single annotation file.

    Expected format per line:
        id, text_label, img_label
    """
    samples = {}
    with open(label_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row]
            if not row or row[0].lower() in ("id", ""):
                continue
            try:
                sid       = row[0]
                text_lbl  = _parse_label(row[1])
                image_lbl = _parse_label(row[2]) if len(row) > 2 else text_lbl
                conflict  = int(text_lbl != image_lbl)
                samples[sid] = {
                    "text_label":  text_lbl,
                    "image_label": image_lbl,
                    "final_label": text_lbl,
                    "conflict":    conflict,
                    "valid":       True,
                }
            except (IndexError, ValueError) as e:
                logger.debug(f"Skipping malformed row: {row} ({e})")
    return samples


# =============================================================================
# Dataset Class
# =============================================================================

class MVSADataset(Dataset):
    """
    PyTorch Dataset for MVSA-Single and MVSA-Multiple.

    Parameters
    ----------
    data_root : str
        Path to MVSA_Multiple/ or MVSA_Single/ directory.
    variant : str
        "multiple" or "single".
    tokenizer : transformers tokenizer
        Pre-initialised tokenizer (e.g. DistilBertTokenizer).
    split : str
        "train", "val", or "test" (controls image augmentation).
    max_length : int
        Max token length for text.
    sample_ids : list, optional
        Subset of sample IDs to include.
    """

    def __init__(
        self,
        data_root:  str,
        variant:    str,
        tokenizer,
        split:      str  = "train",
        max_length: int  = 128,
        sample_ids: Optional[List[str]] = None,
    ):
        self.data_root  = data_root
        self.variant    = variant.lower()
        self.tokenizer  = tokenizer
        self.split      = split
        self.max_length = max_length
        self.transform  = get_image_transforms(split)

        # Load annotations
        label_file = os.path.join(
            data_root,
            "labelResultAll.txt" if self.variant == "multiple" else "labelResult.txt",
        )
        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Annotation file not found: {label_file}\n"
                f"Please download MVSA-{variant.capitalize()} and place it at: {data_root}"
            )

        parse_fn = (parse_mvsa_multiple_annotations
                    if self.variant == "multiple"
                    else parse_mvsa_single_annotations)
        all_annotations = parse_fn(label_file)

        # Filter to valid samples that have both image and text on disk
        data_dir = os.path.join(data_root, "data")
        self.samples = []
        skipped = 0
        for sid, ann in all_annotations.items():
            if not ann["valid"]:
                skipped += 1
                continue
            sample_dir = os.path.join(data_dir, sid)
            img_path   = os.path.join(sample_dir, f"{sid}.jpg")
            txt_path   = os.path.join(sample_dir, f"{sid}.txt")
            # Some datasets use .jpeg or .png
            if not os.path.exists(img_path):
                for ext in [".jpeg", ".png", ".gif"]:
                    alt = os.path.join(sample_dir, f"{sid}{ext}")
                    if os.path.exists(alt):
                        img_path = alt
                        break
            if not (os.path.exists(img_path) and os.path.exists(txt_path)):
                skipped += 1
                continue
            if sample_ids is not None and sid not in sample_ids:
                continue
            self.samples.append({
                "id":          sid,
                "img_path":    img_path,
                "txt_path":    txt_path,
                "label":       ann["final_label"],
                "conflict":    ann["conflict"],
                "text_label":  ann["text_label"],
                "image_label": ann["image_label"],
            })

        logger.info(
            f"MVSADataset [{variant}/{split}]: {len(self.samples)} samples loaded "
            f"({skipped} skipped — missing files or invalid annotations)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ── Text ──────────────────────────────────────────────────────────────
        with open(s["txt_path"], "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read().strip()
        text = clean_tweet(raw_text) or raw_text[:512]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Image ─────────────────────────────────────────────────────────────
        try:
            image = Image.open(s["img_path"]).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "images":         image,
            "labels":         torch.tensor(s["label"],    dtype=torch.long),
            "conflict_flags": torch.tensor(s["conflict"], dtype=torch.float),
            "sample_id":      s["id"],
        }


# =============================================================================
# DataLoader Builder
# =============================================================================

def build_mvsa_dataloaders(
    data_root:   str,
    tokenizer,
    variant:     str   = "multiple",
    batch_size:  int   = 32,
    num_workers: int   = 0,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
    max_length:  int   = 128,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for MVSA.

    Returns
    -------
    dict with keys "train", "val", "test", "class_weights"
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Load full dataset (with train augmentation — will override transforms per-split)
    full_ds = MVSADataset(
        data_root=data_root,
        variant=variant,
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
    )
    n      = len(full_ds)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    # Override transforms for val/test subsets
    # (random_split wraps in Subset, so we patch via the underlying dataset's transform
    #  for the actual indices — instead, create separate Dataset objects for val/test)
    all_ids = [full_ds.samples[i]["id"] for i in range(n)]
    train_ids = [all_ids[i] for i in train_ds.indices]
    val_ids   = [all_ids[i] for i in val_ds.indices]
    test_ids  = [all_ids[i] for i in test_ds.indices]

    train_dataset = MVSADataset(data_root, variant, tokenizer, "train",   max_length, train_ids)
    val_dataset   = MVSADataset(data_root, variant, tokenizer, "val",     max_length, val_ids)
    test_dataset  = MVSADataset(data_root, variant, tokenizer, "test",    max_length, test_ids)

    # Compute class weights for imbalanced labels
    label_counts = Counter(s["label"] for s in full_ds.samples)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float,
    )

    n_conflict = sum(1 for s in full_ds.samples if s["conflict"])
    logger.info(
        f"MVSA-{variant.capitalize()} split: "
        f"Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)} | "
        f"Conflict samples={n_conflict} ({100*n_conflict/n:.1f}%)"
    )
    logger.info(f"Label distribution: { {IDX_TO_NAME[k]: v for k, v in sorted(label_counts.items())} }")
    logger.info(f"Class weights: {class_weights.tolist()}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return {
        "train":         make_loader(train_dataset, shuffle=True),
        "val":           make_loader(val_dataset,   shuffle=False),
        "test":          make_loader(test_dataset,  shuffle=False),
        "class_weights": class_weights,
        "n_conflict":    n_conflict,
        "n_total":       n,
    }


# =============================================================================
# Dataset Download Helper
# =============================================================================

def print_download_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              MVSA Dataset Download Instructions                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  MVSA-Multiple (~19,600 samples):                                        ║
║    http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-    ║
║    social-data/                                                          ║
║                                                                          ║
║  Alternative mirror:                                                     ║
║    https://github.com/headacheboy/data-of-multimodal-sentiment-analysis  ║
║                                                                          ║
║  After downloading, extract to:                                          ║
║    data/MVSA_Multiple/                                                   ║
║                                                                          ║
║  Expected structure:                                                     ║
║    data/MVSA_Multiple/                                                   ║
║    ├── data/                                                             ║
║    │   ├── 1/  (1.jpg + 1.txt)                                          ║
║    │   ├── 2/  ...                                                       ║
║    │   └── ...                                                           ║
║    └── labelResultAll.txt                                                ║
║                                                                          ║
║  Then run:                                                               ║
║    python experiments/train_mvsa.py --data_root data/MVSA_Multiple      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
