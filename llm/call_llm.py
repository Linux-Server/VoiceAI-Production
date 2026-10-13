"""Minimal OpenAI-compatible streaming call to the Gemma 4 endpoint, with time to first token.

    export GEMMA_BASE_URL=https://<workspace>--gemma4-serve-eu-server.eu-west.modal.direct/v1
    export GEMMA_API_KEY=sk-...
    uv run python llm/call_llm.py
"""

import os
import time

import httpx
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["GEMMA_BASE_URL"],
    api_key=os.environ["GEMMA_API_KEY"],
    http_client=httpx.Client(http2=True),  # reuses one connection: ~200 ms TTFT instead of ~750 ms
)

start = time.perf_counter()
ttft = None

stream = client.chat.completions.create(
    model="gemma-4-26b-a4b-w4a16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What can you help me with?"},
    ],
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # keep thinking off for voice
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        if ttft is None:
            ttft = (time.perf_counter() - start) * 1000
        print(chunk.choices[0].delta.content, end="", flush=True)

total = (time.perf_counter() - start) * 1000
print(f"\n\nTTFT: {ttft:.0f} ms | total: {total:.0f} ms")
