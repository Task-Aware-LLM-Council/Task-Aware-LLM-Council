from model_orchestration.client import OrchestratedModelClient
from model_orchestration.defaults import (
    API_DEFAULT_QA_MODEL,
    API_DEFAULT_REASONING_MODEL,
    API_DEFAULT_GENERAL_MODEL,
    vLLM_DEFAULT_QA_MODEL,
    vLLM_DEFAULT_REASONING_MODEL,
    vLLM_DEFAULT_GENERAL_MODEL,
    DEFAULT_LOCAL_VLLM_BIND,
    build_default_orchestrator_config,
    build_default_local_vllm_orchestrator_config,
)
from model_orchestration.models import (
    JSONLRecordingConfig,
    LocalVLLMPresetConfig,
    ModelRole,
    ModelSpec,
    OrchestratorCallRecord,
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResponse,
)
from model_orchestration.orchestrator import ModelOrchestrator
from model_orchestration.recorders import (
    BaseRecorder,
    InMemoryRecorder,
    JSONLRecorder,
    NoOpRecorder,
)

__all__ = [
    "BaseRecorder",
    "API_DEFAULT_QA_MODEL",
    "API_DEFAULT_REASONING_MODEL",
    "API_DEFAULT_GENERAL_MODEL",
    "vLLM_DEFAULT_QA_MODEL",
    "vLLM_DEFAULT_REASONING_MODEL",
    "vLLM_DEFAULT_GENERAL_MODEL",
    "DEFAULT_LOCAL_VLLM_BIND",
    "InMemoryRecorder",
    "JSONLRecorder",
    "JSONLRecordingConfig",
    "LocalVLLMPresetConfig",
    "ModelOrchestrator",
    "ModelRole",
    "ModelSpec",
    "NoOpRecorder",
    "OrchestratedModelClient",
    "OrchestratorCallRecord",
    "OrchestratorConfig",
    "OrchestratorRequest",
    "OrchestratorResponse",
    "build_default_orchestrator_config",
    "build_default_local_vllm_orchestrator_config",
]


def main() -> None:
    print("model-orchestration provides a library API; import ModelOrchestrator.")
