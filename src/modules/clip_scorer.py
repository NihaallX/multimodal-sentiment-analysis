"""
clip_scorer.py — CLIP-based supplementary text-image semantic scorer
=====================================================================

Provides three things without requiring any retraining:

1. CLIPScorer
   - score_image_sentiment(image)  → {positive, negative, neutral} probs via zero-shot
   - score_alignment(text, image)  → cosine similarity in CLIP's joint embedding space
                                      (High = text+image agree, Low = they conflict)

2. preprocess_text(text)           → Fix A: replace sentiment/sarcasm emojis with tokens

3. detect_sarcasm(text)            → Fix C: rule-based sarcasm signal detector
                                      returns {score, signals, likely_sarcasm}

Design notes
------------
- CLIPScorer lazy-loads on first call (no startup cost if never used)
- CLIP model is cached via @st.cache_resource when called from Streamlit
- Uses openai/clip-vit-base-patch32 (~150MB, always CPU-safe)
- Text is truncated to 77 tokens (CLIP hard limit)
"""

from __future__ import annotations
import re
import torch
import torch.nn.functional as F
from typing import Optional
from PIL import Image


# ─── Sentiment prompts for CLIP zero-shot image classification ──────────────
_IMAGE_SENTIMENT_PROMPTS = {
    "positive": "a photo that expresses happy, joyful, cheerful, or positive sentiment",
    "negative": "a photo that expresses sad, angry, stressed, frustrated, or negative sentiment",
    "neutral":  "a neutral photo without strong emotional content",
}

# ─── Emoji → text token map (Fix A) ─────────────────────────────────────────
_EMOJI_MAP = {
    # Sarcasm / irony markers
    "🙃": " [sarcasm_face] ",
    "😒": " [annoyed] ",
    "🙄": " [eyeroll] ",
    "😑": " [expressionless] ",
    "😬": " [grimace] ",
    "😏": " [smirk] ",
    "🤷": " [shrug] ",
    "🤦": " [facepalm] ",
    # Strong negative
    "😡": " [angry] ",
    "🤬": " [furious] ",
    "😤": " [frustrated] ",
    "😭": " [crying_hard] ",
    "💀": " [dead_from_shock] ",
    "🤮": " [disgusted] ",
    # Positive
    "😂": " [laughing_hard] ",
    "😍": " [love_struck] ",
    "🥰": " [loving] ",
    "🎉": " [celebrating] ",
    "❤️": " [love] ",
    "💔": " [heartbreak] ",
    "👏": " [clapping] ",
    "🔥": " [fire_hot] ",
    # Neutral/ambiguous
    "🤔": " [thinking] ",
    "😐": " [neutral_face] ",
}


def preprocess_text(text: str) -> str:
    """Fix A: Replace emoji with semantic text tokens before tokenization.

    This lets RoBERTa 'see' sentiment signals that were previously mapped to
    unknown/rare subwords. The replacement tokens are in-vocabulary words that
    carry the intended sentiment meaning.

    Parameters
    ----------
    text : str
        Raw input text (may contain emoji).

    Returns
    -------
    str
        Cleaned text with emoji replaced by descriptive tokens.
    """
    for emoji, token in _EMOJI_MAP.items():
        text = text.replace(emoji, token)
    return text.strip()


# ─── Sarcasm heuristics (Fix C) ─────────────────────────────────────────────

_SARCASM_EMOJIS = {"🙃", "😒", "🙄", "😑", "😬", "😏", "🤷", "🤦"}
_POSITIVE_CAPS_WORDS = {
    "LOVE", "AMAZING", "GREAT", "AWESOME", "FANTASTIC", "WONDERFUL",
    "PERFECT", "BRILLIANT", "OUTSTANDING", "EXCELLENT", "BEST",
}


def detect_sarcasm(text: str) -> dict:
    """Fix C: Rule-based sarcasm signal detector.

    Detects:
    1. Known sarcasm emojis (🙃🙄😒 etc.)
    2. Positive words written in ALL-CAPS (e.g. LOVE, AMAZING)
    3. Excessive punctuation sequences (!!!, ???)
    4. Classic sarcastic openers (Oh wow, Just love, So great, etc.)

    Returns
    -------
    dict with keys:
        score         : int — total signal count (0=clean, ≥2=likely sarcasm)
        signals       : list[str] — human-readable list of detected signals
        likely_sarcasm: bool — True if score ≥ 2
    """
    score = 0
    signals = []

    # 1. Sarcasm emojis
    found_emojis = [e for e in _SARCASM_EMOJIS if e in text]
    if found_emojis:
        score += len(found_emojis)
        signals.append(f"sarcasm emoji(s): {'  '.join(found_emojis)}")

    # 2. Positive CAPS words
    caps_hits = [w for w in _POSITIVE_CAPS_WORDS if re.search(rf'\b{w}\b', text)]
    if caps_hits:
        score += len(caps_hits)
        signals.append(f"positive word(s) in CAPS: {', '.join(caps_hits)}")

    # 3. Excessive punctuation
    if re.search(r'[!?]{3,}', text):
        score += 1
        signals.append("excessive punctuation (!!! or ???)")

    # 4. Classic sarcastic phrases
    sarcasm_phrases = [
        r'\boh\s+wow\b', r'\bjust\s+love\b', r'\bso\s+great\b',
        r'\bso\s+fun\b', r'\bso\s+helpful\b', r'\bcan\'t\s+wait\b',
        r'\btotally\s+worth\b', r'\bjust\s+what\s+i\s+needed\b',
    ]
    for pattern in sarcasm_phrases:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
            signals.append(f"sarcastic phrase detected: '{pattern.replace(r'\\b','').replace(r'\\s+', ' ')}'")

    return {
        "score": score,
        "signals": signals,
        "likely_sarcasm": score >= 2,
    }


# ─── CLIP Scorer ─────────────────────────────────────────────────────────────

class CLIPScorer:
    """Lazy-loading CLIP scorer.

    Usage
    -----
    scorer = CLIPScorer(device="cuda")   # nothing loaded yet
    probs  = scorer.score_image_sentiment(pil_image)   # loads on first call
    sim    = scorer.score_alignment("happy text", pil_image)
    """

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
            self._model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device)
            self._model.eval()

    @torch.no_grad()
    def score_image_sentiment(self, image: Image.Image) -> dict:
        """Zero-shot classify image sentiment using CLIP.

        Returns probability distribution over [positive, negative, neutral]
        derived from cosine similarity between the image embedding and the
        three sentiment prompt embeddings.
        """
        self._load()
        labels  = list(_IMAGE_SENTIMENT_PROMPTS.keys())
        prompts = list(_IMAGE_SENTIMENT_PROMPTS.values())

        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs  = self._model(**inputs)
        # logits_per_image : [1, 3]
        probs = torch.softmax(outputs.logits_per_image, dim=-1)[0].cpu().tolist()
        return dict(zip(labels, probs))

    @torch.no_grad()
    def score_alignment(self, text: str, image: Image.Image) -> float:
        """Compute cosine similarity between CLIP text and image embeddings.

        Returns
        -------
        float in roughly [-1, 1], typically [0, 1] for related content.
        Higher = modalities semantically agree.
        Lower  = modalities semantically disagree (potential conflict).
        """
        self._load()
        # CLIP text encoder has a hard 77-token limit
        inputs = self._processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_feat  = self._model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_feat = self._model.get_image_features(
            pixel_values=inputs["pixel_values"],
        )
        text_feat  = F.normalize(text_feat,  p=2, dim=-1)
        image_feat = F.normalize(image_feat, p=2, dim=-1)

        return float((text_feat * image_feat).sum(dim=-1).item())
