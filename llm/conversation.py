"""Simulate a 20-turn voice conversation over one open connection and measure each turn.

Opens the connection first (warm-up), then streams every turn over that same HTTP/2 connection,
keeping the full history like a real call. Prints TTFT, ITL and end-to-end time per turn plus a summary.

    export GEMMA_BASE_URL=https://<workspace>--gemma4-serve-eu-server.eu-west.modal.direct/v1
    export GEMMA_API_KEY=sk-...
    uv run python llm/conversation.py
"""

import os
import time

import httpx
import openai
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["GEMMA_BASE_URL"],
    api_key=os.environ["GEMMA_API_KEY"],
    http_client=httpx.Client(http2=True, timeout=httpx.Timeout(600, connect=30)),
)
MODEL = "gemma-4-26b-a4b-w4a16"

TURNS = [
    "Hi there, can you hear me okay?",
    "Great. I'm planning a weekend trip to Abu Dhabi.",
    "What are two places I shouldn't miss?",
    "How far is Abu Dhabi from Dubai by car?",
    "Is it better to rent a car or take a taxi?",
    "What's the weather like there in December?",
    "Do I need to dress conservatively at the Grand Mosque?",
    "What time does the mosque open for visitors?",
    "Can you suggest a good place for dinner?",
    "Something with seafood would be nice.",
    "Roughly how much should I budget for dinner for two?",
    "Okay. What about something fun for the evening?",
    "Is Yas Island good for families?",
    "How long should I plan to spend at Ferrari World?",
    "Are tickets cheaper online?",
    "What's one thing people usually forget to pack?",
    "Good point. Do I need cash or are cards fine?",
    "Can you summarize my plan in one sentence?",
    "Perfect, thanks for the help.",
    "Bye for now!",
]

# Open the connection before measuring, like a voice app would at the start of a call.
# If the endpoint scaled to zero it answers 503 until a container is up, so wait for that too.
t = time.perf_counter()
while True:
    try:
        client.models.list()
        break
    except openai.APIStatusError as e:
        if e.status_code != 503 or time.perf_counter() - t > 15 * 60:
            raise
        time.sleep(3)
print(f"Endpoint ready and connection open after {time.perf_counter() - t:.1f} s\n")

messages = [{"role": "system", "content": "You are a friendly voice assistant. Answer in one or two short sentences."}]
results = []

for i, user in enumerate(TURNS, 1):
    messages.append({"role": "user", "content": user})
    start = time.perf_counter()
    token_times, parts = [], []
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=100,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token_times.append(time.perf_counter())
            parts.append(chunk.choices[0].delta.content)
    e2e = (time.perf_counter() - start) * 1000
    ttft = (token_times[0] - start) * 1000
    itl = (token_times[-1] - token_times[0]) / max(1, len(token_times) - 1) * 1000
    reply = "".join(parts)
    messages.append({"role": "assistant", "content": reply})
    results.append((ttft, itl, e2e))
    print(f"turn {i:2d}  TTFT {ttft:5.0f} ms  ITL {itl:4.1f} ms  e2e {e2e:5.0f} ms  | {reply[:70]!r}")


def summary(label, values):
    v = sorted(values)
    p50, p90 = v[len(v) // 2], v[min(len(v) - 1, int(0.9 * len(v)))]
    print(f"{label:5s} p50 {p50:6.1f} ms   p90 {p90:6.1f} ms   min {v[0]:6.1f} ms   max {v[-1]:6.1f} ms")


print()
summary("TTFT", [r[0] for r in results])
summary("ITL", [r[1] for r in results])
summary("e2e", [r[2] for r in results])
