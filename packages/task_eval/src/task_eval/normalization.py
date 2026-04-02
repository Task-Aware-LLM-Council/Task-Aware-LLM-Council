from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_normalized(text: str) -> list[str]:
    return normalize_answer(text).split()


def normalize_fever_label(text: str) -> str:
    label_map = {
        "supported": "SUPPORTED",
        "supports": "SUPPORTED",
        "support": "SUPPORTED",
        "true": "SUPPORTED",
        "confirmed": "SUPPORTED",
        "verified": "SUPPORTED",
        "correct": "SUPPORTED",
        "refuted": "REFUTED",
        "refutes": "REFUTED",
        "refute": "REFUTED",
        "false": "REFUTED",
        "contradicted": "REFUTED",
        "incorrect": "REFUTED",
        "not supported": "REFUTED",
        "nei": "NEI",
        "not enough information": "NEI",
        "unknown": "NEI",
        "cannot determine": "NEI",
        "cannot be determined": "NEI",
        "insufficient": "NEI",
        "not enough evidence": "NEI",
    }
    normalized = (text or "").strip().lower()
    return label_map.get(normalized, normalized.upper())
