from __future__ import annotations

import re

# For code extraction
FENCED_CODE_RE = re.compile(
    r"```(?:python|py|python3)?\s*\n(?P<code>[\s\S]*?)\n```",
    re.IGNORECASE,
)

OPEN_FENCE_RE = re.compile(
    r"```(?:python|py|python3)?\s*\n(?P<code>[\s\S]*)",
    re.IGNORECASE,
)

DEF_OR_CLASS_RE = re.compile(
    r"(?ms)^(?:from\s+\S+\s+import\s+.*\n|import\s+\S+(?:\s+as\s+\S+)?\n|#.*\n|\s*)*"
    r"(?P<code>(?:def|class)\s+\w+[\s\S]*)$"
)


def extract_qa_answer(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    match = re.search(
        r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\.|$)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    preamble = (
        "let me",
        "based on",
        "according to",
        "looking at",
        "from the",
        "to answer",
        "the passage",
        "in the",
        "first,",
        "step",
    )
    for line in response.splitlines():
        candidate = line.strip()
        if candidate and not candidate.lower().startswith(preamble):
            return candidate.rstrip(".") if len(candidate.split()) <= 10 else candidate

    return response.splitlines()[0].strip()


def extract_qa_answer_musique(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    match = re.search(
        r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE,
    )
    if match:
        ans = match.group(1).strip()
        # Strip out trailing XML tags if the model forgets to put them on a new line
        return re.sub(r"</?scratchpad>", "", ans, flags=re.IGNORECASE).strip()

    preamble = (
        "let me",
        "based on",
        "according to",
        "looking at",
        "from the",
        "to answer",
        "the passage",
        "in the",
        "first,",
        "step",
    )
    for line in response.splitlines():
        candidate = line.strip()
        if candidate and not candidate.lower().startswith(preamble):
            return candidate.rstrip(".") if len(candidate.split()) <= 10 else candidate

    return response.splitlines()[0].strip()


def extract_mcq_answer(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    match = re.match(r"^\(?([A-Da-d])\)?[\s\.\,\:]", response)
    if match:
        return match.group(1).upper()
    if re.match(r"^[A-Da-d]$", response):
        return response.upper()
    match = re.search(
        r"(?:answer is|answer:)\s*\(?([A-Da-d])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1)
    return response[:1].upper()


def extract_fever_label(text: str) -> str:
    """
    Robustly extracts a FEVER classification label from verbose LLM output.
    Maps various synonyms and acronyms to the standard dataset labels.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Define robust keyword clusters.
    nei_patterns = ["not enough info", "nei",
                    "neutral", "neither", "insufficient"]
    supports_patterns = ["supports", "supported",
                         "true", "yes", "entailment", "correct"]
    refutes_patterns = ["refutes", "refuted",
                        "false", "no", "contradiction", "incorrect"]

    # Use regex word boundaries (\b) so we don't accidentally match substrings
    # (e.g., matching "no" inside "enough")
    for pattern in nei_patterns:
        if re.search(rf"\b{pattern}\b", text):
            return "NOT ENOUGH INFO"

    for pattern in supports_patterns:
        if re.search(rf"\b{pattern}\b", text):
            return "SUPPORTS"

    for pattern in refutes_patterns:
        if re.search(rf"\b{pattern}\b", text):
            return "REFUTES"

    # Fallback: return the raw text if no pattern matches,
    return text.strip()


# def extract_math_answer(response: str) -> str:
#     response = (response or "").strip()
#     if not response:
#         return ""

#     for pattern in (
#         r"(?:ANSWER|Final Answer)\s*[:=]\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)",
#         r"\\boxed\{([^}]+)\}",
#         r"(?:the answer is|equals|result is|approximately)\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)",
#     ):
#         match = re.search(pattern, response, re.IGNORECASE)
#         if match:
#             return match.group(1).strip()

#     numbers = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", response)
#     return numbers[-1] if numbers else ""


def extract_math_answer(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    # 1. Primary Strategy: Extract the last \boxed{} using brace counting
    match_iter = list(re.finditer(r"\\boxed\{", response))
    if match_iter:
        last_match = match_iter[-1]
        start_idx = last_match.end()
        brace_count = 1
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response[start_idx:i].strip()

    # 2. Secondary Strategy: Standard Regex Fallbacks (for non-boxed prompts)
    for pattern in (
        r"(?:ANSWER|Final Answer)\s*[:=]\s*(.+)$",
        r"(?:the answer is|equals|result is|approximately)\s*(.+)$",
    ):
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            # Return the matched string, stripping trailing periods
            return match.group(1).strip().rstrip('.')

    # 3. Tertiary Strategy: Fallback to your original numeric extraction
    numbers = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", response)
    return numbers[-1] if numbers else response

# def extract_code_answer(response: str) -> str:
#     response = (response or "").strip()
#     if not response:
#         return ""

#     match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
#     if match:
#         return match.group(1).strip()

#     def_match = re.search(
#         r"((?:from .+\n|import .+\n)*def .+)", response, re.DOTALL)
#     if def_match:
#         return def_match.group(1).strip()

#     response = re.sub(r"^```\s*\n?", "", response)
#     response = re.sub(r"\n?```\s*$", "", response)
#     return response.strip()


def _clean_extracted_code(code: str) -> str:
    lines = code.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Remove example usage / prints
        if stripped.startswith("print("):
            continue

        # Remove "if __name__ == '__main__'"
        if stripped.startswith("if __name__"):
            break

        # Optional: remove comments like "# Example usage"
        if stripped.lower().startswith("# example"):
            continue

        cleaned.append(line)

    code = "\n".join(cleaned)

    return code.strip()

def extract_code_answer(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    code = ""

    # 1) Perfect fenced code block
    m = FENCED_CODE_RE.search(response)
    if m:
        code = m.group("code")

    # 2) Open fence without closing backticks
    elif (m := OPEN_FENCE_RE.search(response)):
        code = m.group("code")

    # 3) Code-ish response
    elif (m := DEF_OR_CLASS_RE.search(response)):
        code = m.group("code")

    else:
        code = re.sub(r"^\s*```\w*\s*\n?", "", response)
        code = re.sub(r"\n?\s*```\s*$", "", code)

    code = code.strip()

    # cleanup step
    code = _clean_extracted_code(code)
    return code


def extract_answer(response: str, dataset_name: str) -> str:
    extractors = {
        "musique": extract_qa_answer,
        "quality": extract_qa_answer,
        "fever": extract_fever_label,
        "hardmath": extract_math_answer,
        "humaneval_plus": extract_code_answer,
    }
    try:
        extractor = extractors[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset_name}") from exc
    return extractor(response)
