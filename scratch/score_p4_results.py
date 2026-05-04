"""Score a P4 benchmark JSONL using P3's compute_metrics logic.

Mirrors the P3 `council_policies.cli:compute_metrics` scoring path as
described in the session handoff:
  - _extract_answer priority: (1) `Final Answer:` regex, (2) last non-empty
    line for musique|quality|fever, (3) full stripped response.
  - FEVER       : label_accuracy (whole-word SUPPORT/REFUTE/NOT ENOUGH INFO).
  - HARDMATH    : `\\boxed{}` extraction on both pred AND gold, then EM.
  - MuSiQue     : exact_match_multi + token_f1_multi (best-of-golds), plus
    abstention credit: if pred == 'NOT_FOUND' and the row is unanswerable,
    score EM=1, F1=1 (mirrors P3's bonus). The answerable flag is recovered
    by joining router_dataset[index].original_id → bdsaglam/musique.id.
  - QuALITY     : exact_match_multi + token_f1_multi.

HumanEvalPlus pass@1 is scored separately via score_humaneval_pass1.py.

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/score_p4_results.py \\
        --results p4_gemma_lora_v2.jsonl
"""
import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

from task_eval.extraction import extract_fever_label, extract_math_answer
from task_eval.scoring import (
    exact_match_multi,
    label_accuracy,
    math_exact_match,
    token_f1_multi,
)


# Math notation normalizer — recovers ~7/60 HARDMath rows the strict
# math_exact_match misses on pure notation drift. Conservative: only
# rewrites between LaTeX synonyms that are mathematically identical
# (display vs inline fractions, set-membership prefix, currency, brace
# shorthand). No semantic rearrangement.
_PREFIX_IN_RE = re.compile(r"^[a-zA-Z]\s*\\in\s*")
_PREFIX_EQ_RE = re.compile(r"^[a-zA-Z]\s*=\s*")
_WS_RE = re.compile(r"\s+")


def _normalize_math(s: str) -> str:
    s = (s or "").strip()
    # \dfrac (display) and \tfrac (text) are typesetting variants of \frac.
    s = re.sub(r"\\(d|t)frac\b", r"\\frac", s)
    # \left / \right are pure typesetting.
    s = s.replace("\\left", "").replace("\\right", "")
    # Currency prefix: \$36 ≡ 36 for math comparison.
    s = s.replace("\\$", "")
    # Set-membership / equality prefix: "x \in [..]" ≡ "[..]", "x = ..." ≡ "...".
    s = _PREFIX_IN_RE.sub("", s)
    s = _PREFIX_EQ_RE.sub("", s)
    # Brace shorthand: \frac{5}{9} ≡ \frac 59 once whitespace is gone.
    s = s.replace("{", "").replace("}", "")
    # All whitespace and dollars.
    s = _WS_RE.sub("", s)
    s = s.replace("$", "")
    return s


def _math_em_loose(pred: str, gold: str) -> float:
    """Strict math_exact_match first; if that misses, retry under
    notation-only normalization. Returns 1.0 / 0.0 like math_exact_match."""
    if math_exact_match(pred, gold):
        return 1.0
    n_pred, n_gold = _normalize_math(pred), _normalize_math(gold)
    if n_pred and n_pred == n_gold:
        return 1.0
    return 0.0


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# FEVER removed from this set — its output depends on the specialist's
# style (gemma: verdict-last, Qwen: verdict-first) and neither maps cleanly
# to the last-non-empty-line heuristic. FEVER now uses a dedicated scan
# over the full response; see _fever_extract below.
_LAST_LINE_SOURCES = {"MuSiQue", "QuALITY"}

_FEVER_LABEL_RE = re.compile(
    r"\b(SUPPORTS|REFUTES|NOT ENOUGH INFO)\b",
    re.IGNORECASE,
)


def _fever_extract(response: str) -> str:
    """Scan the full response for a whole-word FEVER label (case-insensitive)
    and return the first match, uppercased. This is deliberately permissive
    — Qwen-14B puts the verdict on line 1 then explains; gemma-9B does the
    opposite. Picking the first label matches both styles (the specialist's
    initial commit is typically the verdict, and the explanation that
    follows rarely mentions alternative labels except to deny them)."""
    m = _FEVER_LABEL_RE.search(response or "")
    return m.group(1).upper() if m else ""

_ORIGINAL_ID_KEYS = ("original_id", "id", "musique_id")
_ANSWERABLE_KEYS = ("answerable", "is_answerable")

# Abstention sentinels the MuSiQue-style template may produce. The P4
# `_MUSIQUE_CONSTRAINED_PROMPT` instructs the model to say
# "NOT PRESENT IN CONTEXT" rather than "NOT_FOUND", so matching only the
# latter would never credit abstention even on a perfectly-behaved model.
_ABSTAIN_SENTINELS = frozenset({
    "NOT_FOUND",
    "NOT FOUND",
    "NOT PRESENT IN CONTEXT",
    "NOT PRESENT",
    "UNANSWERABLE",
    "CANNOT BE ANSWERED",
    "CANNOT ANSWER",
    "NO ANSWER",
    "I DON'T KNOW",
})

# Hedge phrases that a well-behaved model might emit when context is
# insufficient but doesn't use the exact sentinel. Only checked on the
# extracted final answer (not the reasoning) to avoid crediting rows
# where the model hedged mid-reasoning but ultimately hallucinated an
# answer. Case-insensitive substring match.
_ABSTAIN_HEDGES = (
    "not mentioned",
    "not specified",
    "not provided",
    "not stated",
    "not given",
    "not available",
    "not explicitly",
    "no information",
    "no mention",
    "cannot be determined",
    "cannot determine",
    "cannot find",
    "insufficient information",
    "doesn't mention",
    "does not mention",
    "doesn't specify",
    "does not specify",
    "context doesn't",
    "context does not",
    "isn't present",
    "is not present",
    "isn't mentioned",
    "is not mentioned",
    "isn't specified",
    "is not specified",
)


def _strip_markdown(text: str) -> str:
    """Remove leading/trailing markdown emphasis markers (**, *, __, _)
    that models often wrap short labels in (e.g. '**Answer:** REFUTES'
    → captured answer is '** REFUTES', which breaks whole-word matchers)."""
    s = (text or "").strip()
    # Strip leading * or _ runs
    while s and s[0] in "*_":
        s = s[1:]
    while s and s[-1] in "*_":
        s = s[:-1]
    return s.strip()


# MuSiQue subject-phrase trimmer. Specialists frequently wrap the answer
# in a sentence ("X has Y", "X was born on Y", "X signed Y") which makes
# token-F1 collapse from precision penalty even when the answer is fully
# present.
#
# Two-pass strip:
#   1. _SUBJECT_TRIM_RE removes the leading "<subject> <single linking verb>"
#      so "Gustave Courbet was born on 10 June 1819" → "born on 10 June 1819".
#   2. _LEADING_VERB_RE then strips the verb phrase if it leads the
#      remainder, so "born on 10 June 1819" → "10 June 1819".
# Conservative bounds: only fires on multi-token predictions, and falls
# back to the input if a trim produces <3 chars.
_SUBJECT_TRIM_RE = re.compile(
    r"^.+?\s(?:is|was|are|were|has|have|had|became|signed|made|wrote|"
    r"built|constructed|established|founded|located|placed|published|"
    r"released)\s+(.+)$",
    re.IGNORECASE,
)
_LEADING_VERB_RE = re.compile(
    r"^(?:born\s+(?:on|in)|set\s+(?:on|in)|located\s+in|placed\s+in|"
    r"constructed\s+in|built\s+in|established\s+in|founded\s+in|"
    r"published\s+in|released\s+in)\s+",
    re.IGNORECASE,
)


def _trim_subject_prefix(text: str) -> str:
    """Strip "<subject> <linking verb>" then any residual leading verb
    phrase. No-op if the input is already short (≤4 tokens) or the trim
    would leave <3 chars."""
    s = (text or "").strip()
    if not s or len(s.split()) <= 4:
        return s
    # Pass 1: subject + verb
    m = _SUBJECT_TRIM_RE.match(s)
    if m:
        candidate = m.group(1).strip().rstrip(".,;:")
        if candidate and len(candidate) >= 3:
            s = candidate
    # Pass 2: leading verb phrase remaining ("born on", "constructed in")
    m = _LEADING_VERB_RE.match(s)
    if m:
        candidate = s[m.end():].strip().rstrip(".,;:")
        if candidate and len(candidate) >= 3:
            s = candidate
    return s


def _is_abstention(text: str) -> bool:
    """Case-insensitive abstention match — canonical sentinel OR hedge
    phrase appearing in the final answer."""
    cleaned = (text or "").strip().rstrip(".").upper()
    if cleaned in _ABSTAIN_SENTINELS:
        return True
    lower = (text or "").lower()
    return any(h in lower for h in _ABSTAIN_HEDGES)


def _extract_answer(response: str, source: str) -> str:
    """P3-style priority extraction. Preserves NOT_FOUND sentinel.

    Markdown emphasis markers (**, *) are stripped from the extracted
    answer — without this, '**Answer:** REFUTES' captures '** REFUTES'
    and whole-word label matchers miss it."""
    response = (response or "").strip()
    if not response:
        return ""
    if response == "NOT_FOUND":
        return "NOT_FOUND"

    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        return _strip_markdown(matches[-1].strip().rstrip("."))

    if source in _LAST_LINE_SOURCES:
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped:
                return _strip_markdown(stripped.rstrip("."))

    return response


def _coerce_gold_item(item) -> str | None:
    """Extract a reference string from a gold entry. Handles dicts
    ({'text': ..., 'tokens': ...}) and plain strings."""
    if not item:
        return None
    if isinstance(item, dict):
        text = item.get("text")
        return str(text) if text else None
    return str(item)


def _gold_as_list(gold):
    """Normalize a gold_answer field to list[str].

    Shapes seen across the benchmark datasets:
      - str: most sources (MuSiQue, FEVER, HARDMATH) — wrap in a list.
      - str that is a Python repr of list[dict]: QuALITY on router_dataset-2.
        Each entry is {"text": "...", "tokens": [...]}. The dataset
        serialized the list via str() instead of json; detect by the
        leading "[{" and parse with ast.literal_eval, then extract text.
      - list[str] / list[dict]: defensive — older router_dataset used
        proper lists; handle both here too so the same scorer works
        across dataset versions.
    """
    if gold is None:
        return []

    # list form (older datasets or future proper-JSON versions)
    if isinstance(gold, list):
        return [s for s in (_coerce_gold_item(g) for g in gold) if s]

    if isinstance(gold, dict):
        text = _coerce_gold_item(gold)
        return [text] if text else []

    # string form — may be a plain answer, or a stringified list of dicts.
    s = str(gold).strip()
    if s.startswith("[{") or s.startswith("[\""):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list) and parsed:
            return [t for t in (_coerce_gold_item(g) for g in parsed) if t]
    return [s]


def _first_present(mapping, keys):
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def build_musique_answerable_map(
    router_source: str, router_split: str,
    musique_source: str, musique_split: str,
) -> dict[int, bool]:
    """Map router_dataset row index -> answerable bool.

    P3's MuSiQue abstention credit requires the upstream `answerable` flag
    that router_dataset strips. We rejoin: router_dataset[i].original_id
    matches the id on bdsaglam/musique, whose rows carry `answerable`.
    Returns {} and prints a warning if the join cannot be built (lets
    scoring proceed without the bonus instead of failing).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[abstention] datasets package unavailable; skipping.")
        return {}
    try:
        router = load_dataset(router_source, split=router_split)
        musique = load_dataset(musique_source, split=musique_split)
    except Exception as exc:
        print(f"[abstention] dataset load failed ({exc}); skipping.")
        return {}

    by_mid: dict[str, bool] = {}
    for row in musique:
        mid = _first_present(row, _ORIGINAL_ID_KEYS)
        ans = _first_present(row, _ANSWERABLE_KEYS)
        if mid is None or ans is None:
            continue
        by_mid[str(mid)] = bool(ans)

    if not by_mid:
        print(
            f"[abstention] {musique_source}:{musique_split} has no "
            f"answerable flags on the expected keys {_ANSWERABLE_KEYS}; "
            f"skipping."
        )
        return {}

    out: dict[int, bool] = {}
    for i, row in enumerate(router):
        if (row.get("source_dataset") or "") != "MuSiQue":
            continue
        rid = _first_present(row, _ORIGINAL_ID_KEYS)
        if rid is None:
            continue
        rid_s = str(rid)
        if rid_s in by_mid:
            out[i] = by_mid[rid_s]

    print(
        f"[abstention] built answerable map for {len(out)} MuSiQue rows "
        f"(unanswerable: {sum(1 for v in out.values() if not v)})."
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--router-source", default="task-aware-llm-council/router_dataset",
    )
    p.add_argument("--router-split", default="validation")
    p.add_argument("--musique-source", default="bdsaglam/musique")
    p.add_argument("--musique-split", default="validation")
    p.add_argument(
        "--no-abstention", action="store_true",
        help="Skip MuSiQue abstention credit (default: enabled).",
    )
    args = p.parse_args()

    per_source = defaultdict(
        lambda: {"em": [], "f1": [], "acc": [], "n": 0, "errors": 0,
                 "abstain_credited": 0}
    )

    answerable_map: dict[int, bool] = {}
    if not args.no_abstention:
        answerable_map = build_musique_answerable_map(
            args.router_source, args.router_split,
            args.musique_source, args.musique_split,
        )

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row.get("source_dataset") or "UNKNOWN"
            if source == "HumanEvalPlus":
                # Scored separately via score_humaneval_pass1.py (execution).
                continue
            per_source[source]["n"] += 1
            if row.get("error"):
                per_source[source]["errors"] += 1
                continue

            pred_raw = row.get("p4_answer") or ""

            if source == "HARDMATH":
                pred = extract_math_answer(pred_raw)
            elif source == "FEVER":
                # Use task_eval's FEVER extractor — same one P3 uses. It
                # accepts synonyms ('supported', 'true', 'yes', 'insufficient',
                # 'neutral', etc.) and checks NEI-patterns first so 'insufficient
                # evidence' resolves to NOT ENOUGH INFO instead of being flagged
                # as wrong. Matches P3's scoring methodology so the comparison
                # is apples-to-apples.
                pred = extract_fever_label(pred_raw)
                if not pred:
                    pred = _fever_extract(pred_raw) or _extract_answer(pred_raw, source)
            else:
                pred = _extract_answer(pred_raw, source)
                if source == "MuSiQue":
                    # Strip "<subject> <verb>" prefix when the model wraps
                    # the answer in a sentence — recovers ~8/60 mid_f1 rows
                    # from precision-penalty under token-F1 (e.g.
                    # "Gustave Courbet was born on 10 June 1819" → "10 June 1819").
                    pred = _trim_subject_prefix(pred)

            gold_field = "gold_label" if source == "FEVER" else "gold_answer"
            gold = _gold_as_list(row.get(gold_field))
            if not gold:
                continue

            if source == "FEVER":
                per_source[source]["acc"].append(label_accuracy(pred, gold[0]))
            elif source == "HARDMATH":
                gold_extracted = extract_math_answer(gold[0])
                per_source[source]["acc"].append(
                    _math_em_loose(pred, gold_extracted)
                )
            else:
                # P3 MuSiQue abstention bonus: correct NOT_FOUND on an
                # unanswerable row scores EM=1, F1=1.
                if (
                    source == "MuSiQue"
                    and _is_abstention(pred)
                    and answerable_map.get(row.get("index")) is False
                ):
                    per_source[source]["em"].append(1.0)
                    per_source[source]["f1"].append(1.0)
                    per_source[source]["abstain_credited"] += 1
                else:
                    per_source[source]["em"].append(exact_match_multi(pred, gold))
                    per_source[source]["f1"].append(token_f1_multi(pred, gold))

    print(f"{'source':<20} {'n':>4} {'err':>4} {'EM':>8} {'F1':>8} {'Acc':>8} {'abst':>5}")
    print("-" * 66)
    for source, stats in sorted(per_source.items()):
        n = stats["n"]
        em = sum(stats["em"]) / len(stats["em"]) if stats["em"] else None
        f1 = sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else None
        acc = sum(stats["acc"]) / len(stats["acc"]) if stats["acc"] else None
        em_s = f"{em:>8.3f}" if em is not None else f"{'-':>8}"
        f1_s = f"{f1:>8.3f}" if f1 is not None else f"{'-':>8}"
        acc_s = f"{acc:>8.3f}" if acc is not None else f"{'-':>8}"
        abst = stats["abstain_credited"]
        print(f"{source:<20} {n:>4} {stats['errors']:>4} {em_s} {f1_s} {acc_s} {abst:>5}")


if __name__ == "__main__":
    main()
