from __future__ import annotations

import asyncio
import json
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from llm_gateway.factory import create_client
from llm_gateway.models import ProviderConfig, PromptRequest

from council.orchestrator import council_generate


# -------------------------
# Load provider configs from ENV
# -------------------------
def load_provider_configs() -> List[ProviderConfig]:
    """
    Expect JSON in env:

    COUNCIL_PROVIDERS='[
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "openai-compatible", "api_base": "...", "api_key_env": "GROQ_API_KEY", "model": "..."}
    ]'
    """

    raw = os.getenv("COUNCIL_PROVIDERS")

    if not raw:
        raise ValueError("❌ COUNCIL_PROVIDERS env variable not set")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("❌ COUNCIL_PROVIDERS must be valid JSON")

    configs: List[ProviderConfig] = []

    for item in data:
        configs.append(
            ProviderConfig(
                provider=item["provider"],
                api_base=item.get("api_base"),
                api_key_env=item.get("api_key_env"),
                default_model=item.get("model"),
            )
        )

    return configs


# -------------------------
# Create clients safely
# -------------------------
def create_clients(configs: List[ProviderConfig]):
    clients = []

    for cfg in configs:
        try:
            client = create_client(cfg)
            clients.append(client)
        except Exception as e:
            print(f"[CLIENT ERROR] {cfg.provider}: {e}")

    return clients


# -------------------------
# MAIN
# -------------------------
async def main():
    # -------------------------
    # Load configs dynamically
    # -------------------------
    configs = load_provider_configs()

    clients = create_clients(configs)

    if not clients:
        print("❌ No valid clients created")
        return

    print(f"✅ Loaded {len(clients)} client(s)")

    # -------------------------
    # Input prompt (from ENV or CLI)
    # -------------------------
    prompt = os.getenv("COUNCIL_PROMPT") or "What is the capital of France?"

    request = PromptRequest(user_prompt=prompt)

    # -------------------------
    # Run council
    # -------------------------
    result = await council_generate(
        clients,
        request,
        debug=os.getenv("DEBUG", "false").lower() == "true",
    )

    # -------------------------
    # Output
    # -------------------------
    print("\n" + "=" * 50)
    print("FINAL ANSWER:", result)
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())