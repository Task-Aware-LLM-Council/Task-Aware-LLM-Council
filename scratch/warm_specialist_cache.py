"""Pre-download the three quantized specialist models + the synthesizer to
the local HF cache so the first vLLM server launch on gpu doesn't pay I/O
time on top of its CUDA-init cost."""
from huggingface_hub import snapshot_download

MODELS = [
    "task-aware-llm-council/gemma-2-9b-it-GPTQ",
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2",
]

for m in MODELS:
    print(f"--- downloading {m} ---")
    path = snapshot_download(m)
    print(f"    -> {path}")

print("done.")
