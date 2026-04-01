import asyncio
from llm_gateway.base import BaseLLMClient
from llm_gateway.models import PromptRequest, PromptResponse

from packages.llm_gateway.src.llm_gateway.base import ClientInfo


# -------------------------
# Fake client (no API needed)
# -------------------------
class MockClient(BaseLLMClient):
    def __init__(self, name, response, fail=False):
        super().__init__(
            info=ClientInfo(provider=name, default_model=name)
        )
        self.response = response
        self.fail = fail

    async def generate(self, request: PromptRequest) -> PromptResponse:
        if self.fail:
            raise Exception("Simulated failure")

        return PromptResponse(
            model=self.info.default_model or "mock",
            text=self.response,
        )


# -------------------------
# Test council
# -------------------------
from council.orchestrator import council_generate


async def main():
    # clients = [
    #     MockClient("model1", "4"),
    #     MockClient("model2", "4"),
    #     MockClient("model3", "5"),
    #     MockClient("model4", "4"),
    #     MockClient("model5", "six"),
    # ]
    # clients = [
    # MockClient("model1", "Paris"),
    # MockClient("model2", "London"),
    # MockClient("model3", "Berlin"),
    # MockClient("model4", "Paris"),
    # MockClient("model5", "London"),
    # ]
#     clients = [
#     MockClient("model1", "4"),
#     MockClient("model2", "4"),
#     MockClient("model3", "5"),
#     MockClient("model4", "4", fail=True),   # ❌ fails
#     MockClient("model5", "6", fail=True),   # ❌ fails
# ]
    clients = [
    MockClient("m1", "A"),
    MockClient("m2", "B"),
    MockClient("m3", ""),                # empty
    MockClient("m4", "C", fail=True),    # fail
    MockClient("m5", "B"),
]
    

    request = PromptRequest(user_prompt="What is 2+2?")

    result = await council_generate(clients, request, debug=True)

    print("\nFINAL RESULT:", result)


if __name__ == "__main__":
    asyncio.run(main())