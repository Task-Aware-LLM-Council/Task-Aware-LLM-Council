"""
P1 — Single best model baseline.

Sends every query to the one model with the highest average rank across
all 5 P1 benchmark datasets (the "best overall" model from specialists.json).

This is the simplest possible policy and establishes the cost-accuracy
baseline that every other policy is compared against.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from inference.aggregation import aggregate
from inference.extraction import extract_answer, score_answer
from inference.gateway import gateway
from inference.registry import registry
from inference.schemas import PolicyResult, Query, TaskTag

logger = logging.getLogger(__name__)


class P1Policy:
    """
    Policy 1: single best overall model, no routing, no council.

    The model is selected at init time from the specialist registry.
    Override with model_id=... to pin a specific model.
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or registry.best_overall()
        logger.info("P1 initialised with model_id=%s", self.model_id)

    def _build_prompt(self, query: Query) -> str:
        if query.context:
            return f"Context:\n{query.context}\n\nQuestion: {query.text}"
        return query.text

    async def run_query(self, query: Query) -> PolicyResult:
        """Run a single query through P1 and return a PolicyResult."""
        tag = query.task_tag or TaskTag.from_dataset_name(query.dataset)
        prompt = self._build_prompt(query)

        t0 = time.perf_counter()
        response = await gateway.call(
            model_id=self.model_id,
            prompt=prompt,
            query_id=query.id,
            role=tag.value,
        )
        wall_ms = (time.perf_counter() - t0) * 1000

        extracted = extract_answer(response.raw_text, tag)
        response.extracted_answer = extracted
        metric = score_answer(extracted, query.gold_answers, tag)

        return PolicyResult(
            policy="P1",
            query_id=query.id,
            dataset=query.dataset,
            task_tag=tag,
            predicted_tag=None,
            final_answer=extracted,
            gold_answers=query.gold_answers,
            primary_metric=metric,
            models_called=[self.model_id],
            total_input_tokens=response.input_tokens,
            total_output_tokens=response.output_tokens,
            wall_latency_ms=wall_ms,
            routed_correctly=None,
            model_responses=[response],
        )

    async def run_dataset(self, queries: list[Query]) -> list[PolicyResult]:
        """Run all queries sequentially (avoids overwhelming a single vLLM instance)."""
        results = []
        for i, q in enumerate(queries):
            logger.debug("P1 query %d/%d  id=%s", i + 1, len(queries), q.id)
            results.append(await self.run_query(q))
        return results
