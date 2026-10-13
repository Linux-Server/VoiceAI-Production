"""Simulate a voice call: one long-lived HTTP/2 connection, several short streamed turns over it.
Reports the protocol and, per turn, time to first token and total time.

    uv run python llm/voice_turns.py
"""

import json
import time

import httpx
import modal

MY_APP = "gemma4-serve-eu"
MODEL = "gemma-4-26b-a4b-w4a16"
TURNS = [
    "Hi, can you hear me?",
    "What's a good way to stay focused while working from home?",
    "And how long should my breaks be?",
    "Can you remind me to drink water?",
    "What's the weather usually like in Dubai in March?",
    "Thanks. Recommend one short book on productivity.",
    "Why that one?",
    "Okay, summarize our chat in one sentence.",
    "Great, that's all.",
    "Bye!",
]


def main():
    url = modal.Server.from_name(MY_APP, "Server").get_url()
    key = modal.Function.from_name(MY_APP, "_api_key").remote()
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=url, http2=True, timeout=httpx.Timeout(600, connect=30)) as client:
        t = time.perf_counter()
        while client.get("/health").status_code != 200:
            time.sleep(3)
        print(f"Ready after {time.perf_counter() - t:.0f} s", flush=True)

        messages = [{"role": "system", "content": "You are a friendly voice assistant. Answer in one or two short sentences."}]
        for i, turn in enumerate(TURNS, 1):
            messages.append({"role": "user", "content": turn})
            body = {
                "model": MODEL, "messages": messages, "max_tokens": 60, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            start = time.perf_counter()
            ttft, text = None, []
            with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
                proto = resp.http_version
                for line in resp.iter_lines():
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content") if json.loads(line[6:]).get("choices") else None
                    if delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        text.append(delta)
            total = (time.perf_counter() - start) * 1000
            reply = "".join(text)
            messages.append({"role": "assistant", "content": reply})
            print(f"turn {i:2d} [{proto}] TTFT {ttft:6.0f} ms  total {total:6.0f} ms  | {reply[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
