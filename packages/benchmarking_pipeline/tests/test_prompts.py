from pathlib import Path

from benchmarking_pipeline import BenchmarkExample, BenchmarkRunConfig, build_prompt_request
from llm_gateway import Message, Provider, ProviderConfig


def _config() -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        provider_config=ProviderConfig(provider=Provider.OPENROUTER),
        models=("openai/gpt-4o-mini",),
        output_root=Path("results"),
        prompt_version="prompt_v1",
        temperature=0.2,
        max_tokens=128,
        stop_sequences=("STOP",),
        provider_params={"top_p": 0.9},
    )


def test_build_prompt_request_from_question_and_context() -> None:
    example = BenchmarkExample(
        example_id="example-1",
        dataset_name="musique",
        question="Answer the question.",
        context="Context block.",
        system_prompt="You are helpful.",
        metadata={"difficulty": "hard"},
    )

    request = build_prompt_request(example, model="openai/gpt-4o-mini", config=_config())

    assert request.model == "openai/gpt-4o-mini"
    assert request.system_prompt == "You are helpful."
    assert request.user_prompt == "Answer the question."
    assert request.context == "Context block."
    assert request.temperature == 0.2
    assert request.max_tokens == 128
    assert request.stop_sequences == ("STOP",)
    assert request.provider_params == {"top_p": 0.9}
    assert request.metadata["dataset_name"] == "musique"
    assert request.metadata["example_id"] == "example-1"
    assert request.metadata["prompt_version"] == "prompt_v1"
    assert request.metadata["difficulty"] == "hard"


def test_build_prompt_request_prefers_explicit_messages() -> None:
    example = BenchmarkExample(
        example_id="example-2",
        dataset_name="quality",
        messages=(Message(role="user", content="Hello"),),
    )

    request = build_prompt_request(example, model="model-a", config=_config())

    assert request.messages == (Message(role="user", content="Hello"),)
    assert request.user_prompt is None
    assert request.context is None
