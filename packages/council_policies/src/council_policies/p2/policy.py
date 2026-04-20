from __future__ import annotations

import re


def _parse_vote(text: str) -> str:
    """Extract A, B, or C from a vote response. Defaults to A."""
    match = re.search(r"\b([ABC])\b", text.strip().upper())
    return match.group(1) if match else "A"


def _majority(votes: dict[str, str]) -> str:
    """Return the label with the most votes. Alphabetical tiebreak (A beats B beats C)."""
    tally: dict[str, int] = {}
    for label in votes.values():
        tally[label] = tally.get(label, 0) + 1
    return sorted(tally, key=lambda lbl: (-tally[lbl], lbl))[0]
