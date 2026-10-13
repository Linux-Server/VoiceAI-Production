"""Call the Gemma 4 endpoint exactly like the OpenAI API.

    export GEMMA_BASE_URL=https://<workspace>--gemma4-serve-eu-server.eu-west.modal.direct/v1
    export GEMMA_API_KEY=sk-...           # your vLLM API key
    uv run python llm/openai_client.py

Any OpenAI-compatible client works the same way: point `base_url` at the endpoint and pass the key.
"""

import os

import httpx
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["GEMMA_BASE_URL"],
    api_key=os.environ["GEMMA_API_KEY"],
    # HTTP/2 reuses one connection across requests (~200 ms TTFT instead of ~750 ms per new connection).
    http_client=httpx.Client(http2=True, timeout=httpx.Timeout(600, connect=30)),
)
MODEL = "gemma-4-26b-a4b-w4a16"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

# 1. Regular request: wait for the full answer.
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the UAE?"},
    ],
    max_tokens=100,
    extra_body=NO_THINKING,
)
print(response.choices[0].message.content)
print(f"(tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out)\n")

# 2. Streaming request: print tokens as they arrive (what a voice app would use).
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Give me three quick tips for better sleep."}],
    max_tokens=200,
    stream=True,
    extra_body=NO_THINKING,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# 3. List the models the endpoint serves.
print("\nModels:", [m.id for m in client.models.list().data])
