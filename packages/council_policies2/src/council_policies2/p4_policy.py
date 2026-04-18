from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator

from council_policies2.models import TASK_TO_ROLE, CouncilResponse, TaskType

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_CENTROID_SEEDS: dict[TaskType, list[str]] = {
    TaskType.MATH: [
        "Solve the equation x^2 - 5x + 6 = 0.",
        "What is the integral of x^2?",
        "Prove that sqrt(2) is irrational.",
        "Calculate the probability of rolling two sixes.",
    ],
    TaskType.CODE: [
        "Write a Python function to reverse a linked list.",
        "Implement binary search in JavaScript.",
        "Debug this code: for i in range(10): print(i",
        "What does this SQL query return? SELECT * FROM users WHERE age > 18;",
    ],
    TaskType.REASONING: [
        "Why did the Roman Empire fall? Analyze the key factors.",
        "Compare and contrast utilitarianism and deontological ethics.",
        "What are the implications of quantum computing for cryptography?",
        "Evaluate the argument: all humans are mortal, Socrates is human, therefore...",
    ],
    TaskType.FEVER: [
        "True or false: The Great Wall of China is visible from space.",
        "Verify the claim that vaccines cause autism.",
        "Is it accurate that humans only use 10% of their brains?",
        "Fact-check: Einstein failed mathematics in school.",
    ],
    TaskType.QA: [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "When did World War II end?",
        "What is the boiling point of water?",
    ],
    TaskType.GENERAL: [
        "How's life treating you?",
        "Tell me something interesting.",
        "What do you think about the future of AI?",
        "Give me some advice on staying motivated.",
    ],
}


class LearnedRouterPolicy:
    """
    P4: Learned routing using sentence-transformer embeddings.

    Embeds the incoming prompt and finds the nearest task centroid
    (built from seed examples) to route to the appropriate specialist.

    Parameters
    ----------
    orchestrator:
        A fully-configured ModelOrchestrator.
    encoder_model:
        HuggingFace model name for the sentence-transformer encoder.
    fallback_role:
        Role to use when the classified role is missing from the orchestrator.
        Defaults to ``"general"``.

    Notes
    -----
    Requires the p4-training extra: pip install council-policies2[p4-training]

    FIX (bug 4): Validates ``fallback_role`` at construction time and guards
    the fallback call so a missing fallback produces a clear RuntimeError.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        encoder_model: str = "all-MiniLM-L6-v2",
        fallback_role: str = "general",
    ) -> None:
        self.orchestrator = orchestrator
        self.fallback_role = fallback_role
        self._encoder_model = encoder_model
        self._encoder: Any = None
        self._centroids: dict[TaskType, Any] = {}
        # Validate at construction time so an unconfigured fallback is caught early.
        self._validate_fallback_role()

    def _validate_fallback_role(self) -> None:
        try:
            self.orchestrator.get_client(self.fallback_role)
        except KeyError as exc:
            raise ValueError(
                f"LearnedRouterPolicy: fallback_role {self.fallback_role!r} is not "
                f"registered in the orchestrator. Register a model with that role or "
                f"alias, or pass a different fallback_role."
            ) from exc

    def load(self, *, custom_seeds: dict[TaskType, list[str]] | None = None) -> None:
        """Load the encoder and build task centroids from seed prompts."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "LearnedRouterPolicy requires sentence-transformers. "
                "Install with: pip install council-policies2[p4-training]"
            ) from exc

        import numpy as np

        self._encoder = SentenceTransformer(self._encoder_model)
        seeds = custom_seeds or _CENTROID_SEEDS

        for task_type, prompts in seeds.items():
            embeddings = self._encoder.encode(prompts, convert_to_numpy=True)
            self._centroids[task_type] = embeddings.mean(axis=0)

        logger.info("LearnedRouterPolicy loaded with %d task centroids", len(self._centroids))

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None and bool(self._centroids)

    def classify(self, prompt: str) -> TaskType:
        if not self.is_loaded:
            raise RuntimeError("Call load() before using LearnedRouterPolicy")

        import numpy as np

        embedding = self._encoder.encode([prompt], convert_to_numpy=True)[0]

        best_type = TaskType.GENERAL
        best_sim = -float("inf")
        for task_type, centroid in self._centroids.items():
            sim = float(np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-9
            ))
            if sim > best_sim:
                best_sim = sim
                best_type = task_type

        logger.debug("Classified %r → %s (cosine=%.3f)", prompt[:60], best_type.value, best_sim)
        return best_type

    async def run(
        self,
        request: PromptRequest,
        *,
        task_type: TaskType | None = None,
    ) -> CouncilResponse:
        if task_type is None:
            task_type = self.classify(request.user_prompt or "")

        role = TASK_TO_ROLE[task_type]
        try:
            response = await self.orchestrator.get_client(role).get_response(request)
        except KeyError:
            logger.warning(
                "Role %r not configured; falling back to %r", role, self.fallback_role
            )
            # FIX (bug 4): guard the fallback call — emit a clear RuntimeError.
            try:
                response = await self.orchestrator.get_client(self.fallback_role).get_response(request)
            except KeyError as exc:
                raise RuntimeError(
                    f"Primary role {role!r} and fallback role {self.fallback_role!r} "
                    f"are both missing from the orchestrator."
                ) from exc

        return CouncilResponse(
            winner=response,
            policy="p4",
            task_type=task_type,
            candidates=(response,),
            metadata={"routed_role": role, "encoder": self._encoder_model},
        )
