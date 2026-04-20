from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

from benchmarking_pipeline import BenchmarkExample

from task_eval.extraction import (
    extract_code_answer,
    extract_fever_label,
    extract_math_answer,
    extract_mcq_answer,
    extract_qa_answer,
    extract_qa_answer_musique
)
from task_eval.interfaces import DatasetProfile
from task_eval.models import EvaluationCase, MetricResult, PredictionRecord
from task_eval.scoring import (
    exact_match_multi,
    label_accuracy,
    numeric_accuracy,
    pass_at_1,
    token_f1_multi,
    math_exact_match
)


def _first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


@dataclass(slots=True)
class BaseDatasetProfile(ABC):
    name: str
    metric_names: tuple[str, ...]
    primary_metric: str
    dataset_name: str
    split: str = "validation"
    config_name: str | None = None
    streaming: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_cases(self):
        dataset = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=self.streaming,
        )
        for index, row in enumerate(dataset):
            yield self.row_to_case(dict(row), index)

    @abstractmethod
    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        ...

    @abstractmethod
    def score(
        self,
        *,
        case: EvaluationCase,
        prediction: PredictionRecord,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> MetricResult:
        ...

    def _prediction_text(self, prediction: PredictionRecord) -> str:
        value = prediction.get("response_text")
        return value if isinstance(value, str) else ""


@dataclass(slots=True)
class MusiqueProfile(BaseDatasetProfile):
    name: str = "musique"
    metric_names: tuple[str, ...] = ("exact_match", "token_f1")
    primary_metric: str = "token_f1"
    dataset_name: str = "bdsaglam/musique"
    NOT_PRESENT_IN_CONTEXT = "NOT PRESENT IN CONTEXT" 

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        question = str(_first_present(row, "question",
                       "query", "prompt", default=""))
        raw_context = _first_present(row, "paragraphs", "context")

        # Extract ONLY the supporting paragraphs (Oracle Context)
        best_paragraphs = []
        
        if isinstance(raw_context, list):
            for item in raw_context:
                if isinstance(item, dict):
                    # Musique flags the gold paragraphs needed for the hops!
                    if item.get("is_supporting") is True:
                        text_chunk = item.get("paragraph_text", item.get("text", ""))
                        if text_chunk.strip(): 
                            best_paragraphs.append(text_chunk.strip())
            
            # Fallback just in case a split doesn't have the flag
            if not best_paragraphs:
                for item in raw_context[:4]: # Just grab the first 4
                    if isinstance(item, dict):
                        text_chunk = item.get("paragraph_text", item.get("text", ""))
                        if text_chunk.strip(): best_paragraphs.append(text_chunk.strip())

        elif isinstance(raw_context, str):
            best_paragraphs = [p.strip() for p in raw_context.split('\n\n') if p.strip()][:4]

        # 3Assemble the Shrunk Oracle Context
        context_str = "\n\n".join(best_paragraphs)

        answers = _as_list(_first_present(
            row, "answers", "answer", "gold_answers"))
        example_id = str(_first_present(
            row, "id", "example_id", default=index))
        
        # The Strict Chain-of-Thought Prompt
        constrained_question = (
            "You are a strict reading comprehension assistant. You must analyze the context and think step-by-step out loud before answering.\n\n"
            "RULES:\n"
            "1. You must ONLY use the information provided in the Context. Do NOT use general knowledge.\n"
            "2. The answer is ALWAYS hidden somewhere in the text. You must search carefully.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Write your step-by-step reasoning inside <scratchpad> tags. "
            "After you are done thinking, conclude your response on a new line with the exact format: 'Final Answer: <exact entity name>'. "
            f"If the answer is completely missing, output 'Final Answer: {self.NOT_PRESENT_IN_CONTEXT}'."
        )

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=constrained_question,
                metadata={"profile": self.name},
            ),
            reference={"answers": [str(answer) for answer in answers]},
            metadata={"raw_row": row, "answerable": bool(row.get("answerable", False))},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        answerable = bool(case.metadata.get("answerable"))
        extracted = extract_qa_answer_musique(self._prediction_text(prediction))
        answers = [str(answer) for answer in case.reference.get("answers", [])]
        false_negative = 0
        if not answerable:
            answers = [self.NOT_PRESENT_IN_CONTEXT]
        elif extracted == self.NOT_PRESENT_IN_CONTEXT:
            false_negative = 1
        return MetricResult(
            values={
                "exact_match": exact_match_multi(extracted, answers),
                "token_f1": token_f1_multi(extracted, answers),
                "false_negative": false_negative,
                "answerable" : answerable
            },
            metadata={"extracted_answer": extracted},
        )


@dataclass(slots=True)
class QualityProfile(BaseDatasetProfile):
    name: str = "quality"
    metric_names: tuple[str, ...] = ("exact_match", "token_f1")
    primary_metric: str = "token_f1"
    dataset_name: str = "narrativeqa"

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        question = str(_first_present(row, "question", "query", default=""))

        document = row.get("document", {})
        summary_dict = document.get("summary", {})
        article = summary_dict.get("text", "")

        if not article:
            full_text = document.get("text", "")
            article = full_text[:120000] if full_text else ""

        options = _as_list(_first_present(
            row, "options", "choices", default=[]))
        example_id = str(_first_present(
            row, "id", "example_id", default=index))

        # NEW FIX: Correctly unpack the 'text' key from the answer dictionaries
        answers_data = row.get("answers", [])
        references = []

        if isinstance(answers_data, list):
            for ans in answers_data:
                # If it's a dictionary containing 'text', grab just the text
                if isinstance(ans, dict) and "text" in ans:
                    references.append(str(ans["text"]))
                else:
                    references.append(str(ans))

        constrained_question = question + \
            "\n\nAnswer the question concisely with just the answer."

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=constrained_question,
                context=article if article else None,
                metadata={"options": options, "profile": self.name},
            ),
            reference={"answers": references},
            metadata={"raw_row": row},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        # NEW FIX: Swapped extract_mcq_answer for extract_qa_answer
        extracted = extract_qa_answer(self._prediction_text(prediction))
        answers = [str(answer) for answer in case.reference.get("answers", [])]

        return MetricResult(
            values={
                "exact_match": exact_match_multi(extracted, answers),
                "token_f1": token_f1_multi(extracted, answers),
            },
            metadata={"extracted_answer": extracted},
        )


@dataclass(slots=True)
class FeverProfile(BaseDatasetProfile):
    name: str = "fever"
    metric_names: tuple[str, ...] = ("label_accuracy",)
    primary_metric: str = "label_accuracy"
    dataset_name: str = "copenlu/fever_gold_evidence"

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        claim = str(_first_present(
            row, "claim", "question", "query", default=""))

        # FIX 1: Robustly unpack the FEVER evidence
        raw_evidence = _first_present(row, "evidence", "context", "passage")
        evidence_str = ""

        if isinstance(raw_evidence, str):
            evidence_str = raw_evidence
        elif isinstance(raw_evidence, list):
            # FEVER evidence can be a list of strings, lists, or dicts depending on the specific split
            for item in raw_evidence:
                if isinstance(item, str):
                    evidence_str += f"{item} "
                elif isinstance(item, list):
                    # Sometimes evidence is a list of [DocID, SentenceID, Text]
                    evidence_str += " ".join(str(i) for i in item) + " "
                elif isinstance(item, dict):
                    evidence_str += str(item.get("text", "")) + " "

        evidence_str = evidence_str.strip()

        label = str(_first_present(
            row, "label", "answer", "gold_label", default=""))
        example_id = str(_first_present(
            row, "id", "example_id", default=index))

        # FIX 2: Constrain the prompt to force exact label matches
        constrained_question = (
            f"Claim: {claim}\n\n"
            "Based on the provided context, verify the claim. "
            "Answer strictly with one of these three labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
        )

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=constrained_question,
                context=evidence_str if evidence_str else None,
                metadata={"profile": self.name},
            ),
            reference={"label": label},
            metadata={"raw_row": row},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        extracted = extract_fever_label(self._prediction_text(prediction))
        reference_label = str(case.reference.get("label", ""))
        return MetricResult(
            values={"label_accuracy": label_accuracy(extracted, reference_label)},
            metadata={"extracted_answer": extracted},
        )


@dataclass(slots=True)
class HardMathProfile(BaseDatasetProfile):
    name: str = "hardmath"
    metric_names: tuple[str, ...] = ("math_exact_match",)
    primary_metric: str = "math_exact_match"
    dataset_name: str = "math-ai/MATH-500"

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        question = str(_first_present(row, "question", "problem", "prompt", default=""))
        answer = str(_first_present(row, "answer", "solution", "gold", default=""))
        example_id = str(_first_present(row, "id", "example_id", default=index))
        
        # Instruct the model to use \boxed{} so we can find the final answer reliably
        constrained_question = question + "\n\nPlease put your final answer enclosed in \\boxed{}."

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=constrained_question,
                metadata={"profile": self.name},
            ),
            reference={"answer": answer},
            metadata={"raw_row": row},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        extracted = extract_math_answer(self._prediction_text(prediction))
        reference_answer = str(case.reference.get("answer", ""))
        return MetricResult(
            values={
                "math_exact_match": math_exact_match(extracted, reference_answer)
            },
            metadata={"extracted_answer": extracted},
        )

@dataclass(slots=True)
class HumanEvalPlusProfile(BaseDatasetProfile):
    name: str = "humaneval_plus"
    metric_names: tuple[str, ...] = ("pass_at_1",)
    primary_metric: str = "pass_at_1"
    dataset_name: str = "openai/openai_humaneval"
    timeout_seconds: int = 10

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        prompt = str(_first_present(row, "prompt", "question", default=""))
        test_code = str(_first_present(row, "test", "test_code", default=""))
        entry_point = str(_first_present(row, "entry_point", default=""))
        example_id = str(_first_present(
            row, "task_id", "id", "example_id", default=index))

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=prompt,
                metadata={"entry_point": entry_point, "profile": self.name},
            ),
            reference={"test_code": test_code, "entry_point": entry_point},
            metadata={"raw_row": row},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        extracted = extract_code_answer(self._prediction_text(prediction))
        return MetricResult(
            values={
                "pass_at_1": pass_at_1(
                    extracted,
                    test_code=str(case.reference.get("test_code", "")),
                    entry_point=str(case.reference.get("entry_point", "")),
                    timeout_seconds=self.timeout_seconds,
                )
            },
            metadata={"extracted_answer": extracted},
        )


_MUSIQUE_NOT_PRESENT = "NOT PRESENT IN CONTEXT"


@dataclass(slots=True)
class RouterDatasetProfile(BaseDatasetProfile):
    """
    Profile for task-aware-llm-council/router_dataset on HuggingFace.
    A mixed dataset covering QA, fact-verification, math, and code.
    Each example gets the same constrained prompt as its source P1 dataset,
    and is scored with the matching P1 metric.
    """

    name: str = "router_dataset"
    metric_names: tuple[str, ...] = ("token_f1", "exact_match", "label_accuracy", "math_exact_match", "pass_at_1")
    primary_metric: str = "token_f1"
    dataset_name: str = "task-aware-llm-council/router_dataset"
    split: str = "validation"

    def row_to_case(self, row: dict[str, Any], index: int) -> EvaluationCase:
        example_id = str(row.get("id", index))
        raw_question = str(row.get("question", ""))
        context = row.get("context") or None
        skill_tags: list[str] = row.get("skill_tags") or []
        source_dataset = str(row.get("source_dataset", "")).lower()

        question, context, metric, reference = _build_prompt_and_reference(
            raw_question, context, skill_tags, source_dataset, row
        )

        return EvaluationCase(
            example=BenchmarkExample(
                example_id=example_id,
                dataset_name=self.name,
                question=question,
                context=context,
                metadata={
                    "source_dataset": source_dataset,
                    "skill_tags": skill_tags,
                    "profile": self.name,
                },
            ),
            reference={"metric": metric, **reference},
            metadata={"raw_row": row},
        )

    def score(self, *, case: EvaluationCase, prediction: PredictionRecord, dataset_metadata=None) -> MetricResult:
        text = self._prediction_text(prediction)
        metric = str(case.reference.get("metric", "token_f1"))

        if metric == "musique_token_f1":
            extracted = extract_qa_answer_musique(text)
            answers = [str(a) for a in case.reference.get("answers", [])]
            return MetricResult(
                values={
                    "exact_match": exact_match_multi(extracted, answers),
                    "token_f1": token_f1_multi(extracted, answers),
                },
                metadata={"extracted_answer": extracted, "metric": metric},
            )

        if metric == "token_f1":
            extracted = extract_qa_answer(text)
            answers = [str(a) for a in case.reference.get("answers", [])]
            return MetricResult(
                values={
                    "exact_match": exact_match_multi(extracted, answers),
                    "token_f1": token_f1_multi(extracted, answers),
                },
                metadata={"extracted_answer": extracted, "metric": metric},
            )

        if metric == "label_accuracy":
            extracted = extract_fever_label(text)
            reference = str(case.reference.get("label", ""))
            return MetricResult(
                values={"label_accuracy": label_accuracy(extracted, reference)},
                metadata={"extracted_answer": extracted, "metric": metric},
            )

        if metric == "math_exact_match":
            extracted = extract_math_answer(text)
            reference = str(case.reference.get("answer", ""))
            return MetricResult(
                values={"math_exact_match": math_exact_match(extracted, reference)},
                metadata={"extracted_answer": extracted, "metric": metric},
            )

        if metric == "pass_at_1":
            extracted = extract_code_answer(text)
            return MetricResult(
                values={"pass_at_1": pass_at_1(
                    extracted,
                    test_code=str(case.reference.get("test_code", "")),
                    entry_point=str(case.reference.get("entry_point", "")),
                )},
                metadata={"extracted_answer": extracted, "metric": metric},
            )

        # Fallback
        extracted = extract_qa_answer(text)
        return MetricResult(
            values={"token_f1": token_f1_multi(extracted, [])},
            metadata={"extracted_answer": extracted, "metric": metric},
        )


def _build_prompt_and_reference(
    question: str,
    context: str | None,
    skill_tags: list[str],
    source_dataset: str,
    row: dict[str, Any],
) -> tuple[str, str | None, str, dict[str, Any]]:
    """
    Returns (constrained_question, context, metric_key, reference_dict)
    matching the P1 prompt format for each source dataset.
    """
    gold_answer = str(row.get("gold_answer") or "")

    if "musique" in source_dataset:
        # MuSiQue: chain-of-thought, context embedded in the question
        constrained = (
            "You are a strict reading comprehension assistant. You must analyze the context and think step-by-step out loud before answering.\n\n"
            "RULES:\n"
            "1. You must ONLY use the information provided in the Context. Do NOT use general knowledge.\n"
            "2. The answer is ALWAYS hidden somewhere in the text. You must search carefully.\n\n"
            f"Context:\n{context or ''}\n\n"
            f"Question: {question}\n\n"
            "Write your step-by-step reasoning inside <scratchpad> tags. "
            "After you are done thinking, conclude your response on a new line with the exact format: 'Final Answer: <exact entity name>'. "
            f"If the answer is completely missing, output 'Final Answer: {_MUSIQUE_NOT_PRESENT}'."
        )
        return constrained, None, "musique_token_f1", {"answers": [gold_answer]}

    if "fever" in source_dataset or "fact-verification" in skill_tags:
        gold_label = str(row.get("gold_label") or "")
        constrained = (
            f"Claim: {question}\n\n"
            "Based on the provided context, verify the claim. "
            "Answer strictly with one of these three labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
        )
        return constrained, context, "label_accuracy", {"label": gold_label}

    if "math" in source_dataset or "math" in skill_tags:
        constrained = question + "\n\nPlease put your final answer enclosed in \\boxed{}."
        return constrained, context, "math_exact_match", {"answer": gold_answer}

    if "humaneval" in source_dataset or "code" in skill_tags:
        unit_tests = str(row.get("unit_tests") or "")
        entry_point = str(row.get("entry_point") or "")
        return question, context, "pass_at_1", {"test_code": unit_tests, "entry_point": entry_point}

    # narrativeqa / quality — concise answer format
    constrained = question + "\n\nAnswer the question concisely with just the answer."
    return constrained, context, "token_f1", {"answers": [gold_answer]}
