import argparse
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path

from huggingface_hub import HfApi, login
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoConfig

from common import get_current_user

load_dotenv()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize an LLM to AWQ 4-bit using llm-compressor and push to Hugging Face.")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Source HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        required=True, 
        help="Destination HuggingFace repo ID (e.g., 'your-username/Qwen2.5-7B-Instruct-AWQ')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="open_platypus",
        help="Dataset used for AWQ calibration (default: open_platypus)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=256, 
        help="Number of calibration samples to use (default: 512)"
    )
    parser.add_argument(
        "--seq-len", 
        type=int, 
        default=1024, 
        help="Max sequence length for calibration (default: 2048)"
    )
    parser.add_argument(
        "--temp-dir", 
        type=str, 
        default=f"/scratch1/{get_current_user()}/quantized_temp", 
        help="Temporary local directory to store quantized weights before upload"
    )

    parser.add_argument(
        "--bits", 
        type=int, 
        choices=[4, 8], 
        default=4, 
        help="Quantization precision: 4 for W4A16 (AWQ) or 8 for W8A16 (GPTQ)"
    )
    
    parser.add_argument(
        "--skip-quantize", 
        action="store_true", 
        help="Skip the quantization step and only run the upload step (requires existing quantized files in temp_dir)."
    )

    parser.add_argument(
        "--skip-upload", 
        action="store_true", 
        help="Skip pushing to Hugging Face (useful for just generating local files or avoiding OOM crashes)."
    )
    
    return parser

def _authenticate_hf(hf_token:str):
    """Authenticates with Hugging Face using env variables."""
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face.")

def main():
    parser = _build_parser()
    args = parser.parse_args()
    
    
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
    
    
    if not hf_token:
        raise ValueError(                                                               
            "Authentication token not found! Please set the HUGGINGFACE_API_KEY "
        )

    # 1. Authenticate
    _authenticate_hf(hf_token)

    safe_model_name = args.model.replace('/', '_')
    temp_dir = f"{args.temp_dir}/{safe_model_name}"
    print(f"--- Starting AWQ Quantization for {args.model} using llm-compressor, temp dir {temp_dir} ---")


    # 2. Dynamically Define the Recipe based on CLI input
    if args.bits == 4:
        print(f"--- Starting 4-bit (AWQ) Quantization for {args.model} ---")
        recipe = [
            AWQModifier(
                targets=["Linear"], 
                scheme="W4A16_ASYM", 
                ignore=["lm_head"] 
            )
        ]
    elif args.bits == 8:
        print(f"--- Starting 8-bit (GPTQ) Quantization for {args.model} ---")
        recipe = [
            GPTQModifier(
                targets=["Linear"], 
                scheme="W8A16", 
                ignore=["lm_head"] 
            )
        ]

    # Dynamically determine the sequential target from the HF config
    print(f"Detecting architecture for {args.model}...")
    config = AutoConfig.from_pretrained(args.model)
    architecture_name = config.architectures[0] if config.architectures else ""
    
    if "ForCausalLM" in architecture_name:
        # Translates 'Gemma2ForCausalLM' -> 'Gemma2DecoderLayer'
        layer_class = architecture_name.replace("ForCausalLM", "DecoderLayer")
        seq_targets = [layer_class]
        print(f"--- Dynamically mapped {architecture_name} to target layer: {layer_class} ---")
    else:
        print(f"--- Warning: Non-standard architecture '{architecture_name}'. Proceeding without sequential targets. ---")
        seq_targets = None    


    # ==========================================
    # PHASE 1: QUANTIZATION
    # ==========================================
    if not args.skip_quantize:
        # 3. Apply Oneshot Compression
        # This automatically loads the model, applies calibration, and saves to disk.
        print(f"--- Running oneshot quantization with {args.samples} samples from '{args.dataset}' ---")
        oneshot(
        model=args.model,
        dataset=args.dataset,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        output_dir=temp_dir,
        sequential_targets=seq_targets
        )

        print(f"--- Quantization done!!! ---")
    else:
        print(f"--- Skipping Quantization (Looking for existing files in {args.temp_dir}) ---") 


    
    # ==========================================
    # PHASE 2: UPLOAD
    # ==========================================
    if not args.skip_upload:

        # 4. Push to Hugging Face
        print(f"--- Pushing quantized model to Hugging Face Hub: {args.repo_id} ---")
        api = HfApi(token=hf_token)
        
        # Create the repo if it doesn't exist
        api.create_repo(repo_id=args.repo_id, exist_ok=True, private=False)
        
        # Upload the entire folder generated by llm-compressor
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=args.repo_id,
            commit_message=f"Upload AWQ {args.bits}-bit quantization of {args.model} via llm-compressor"
        )
    else:
        print("--- Skipping Upload to Hugging Face ---")

    # 5. Cleanup
    print("--- Cleaning up local files ---")
    #if os.path.exists(temp_dir):
    #    shutil.rmtree(temp_dir)
    
    print(f"Success! Model is available at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
