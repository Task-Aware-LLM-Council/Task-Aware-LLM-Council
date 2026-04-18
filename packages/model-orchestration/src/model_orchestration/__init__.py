from model_orchestration.client import OrchestratedModelClient
from model_orchestration.defaults import (
    DEFAULT_GENERAL_MODEL,
    DEFAULT_LOCAL_VLLM_BIND,
    DEFAULT_QA_MODEL,
    DEFAULT_REASONING_MODEL,
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
    "DEFAULT_GENERAL_MODEL",
    "DEFAULT_LOCAL_VLLM_BIND",
    "DEFAULT_QA_MODEL",
    "DEFAULT_REASONING_MODEL",
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
