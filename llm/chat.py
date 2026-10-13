"""Chat with the Gemma 4 26B-A4B endpoint on Modal (OpenAI-compatible, streaming).

    uv run python llm/chat.py "What's a good way to stay focused?"
    uv run python llm/chat.py            # interactive chat
    uv run python llm/chat.py --bench 10 # latency test: TTFT, ITL and end-to-end over 10 requests

The endpoint URL is read from GEMMA_BASE_URL and the API key from GEMMA_API_KEY; if unset, they are looked up
from the deployed Modal app (its URL and the `vllm-api-key` secret).
"""

import os
import sys
import time

import httpx
import openai
from openai import OpenAI

MODEL = "gemma-4-26b-a4b-w4a16"
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers concise."


def base_url() -> str:
    if url := os.environ.get("GEMMA_BASE_URL"):
        return url
    import modal

    return modal.Server.from_name("gemma4-serve-eu", "Server").get_url() + "/v1"


def api_key() -> str:
    if key := os.environ.get("GEMMA_API_KEY"):
        return key
    import modal

    return modal.Function.from_name("gemma4-serve-eu", "_api_key").remote()


def ask(client: OpenAI, messages: list[dict], quiet: bool = False) -> tuple[str, dict]:
    """Stream one reply, retrying while the endpoint cold-starts (it returns 503 until a container is up)."""
    deadline = time.time() + 15 * 60
    while True:
        try:
            start = time.perf_counter()
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=512,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            token_times, parts = [], []
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    token_times.append(time.perf_counter())
                    parts.append(delta)
                    if not quiet:
                        print(delta, end="", flush=True)
            e2e = time.perf_counter() - start
            stats = {
                "ttft_ms": (token_times[0] - start) * 1000,
                # mean gap between streamed tokens after the first one
                "itl_ms": (token_times[-1] - token_times[0]) / max(1, len(token_times) - 1) * 1000,
                "e2e_ms": e2e * 1000,
                "tokens": len(token_times),
            }
            if not quiet:
                print(f"\n[TTFT {stats['ttft_ms']:.0f} ms | ITL {stats['itl_ms']:.1f} ms | "
                      f"e2e {stats['e2e_ms']:.0f} ms | {stats['tokens']} tokens]")
            return "".join(parts), stats
        except openai.APIStatusError as e:
            if e.status_code == 503 and time.time() < deadline:
                print("[endpoint is starting up, retrying in 5 s...]", flush=True)
                time.sleep(5)
                continue
            raise


def bench(client: OpenAI, n: int):
    """Send n identical voice-style requests over one client and summarize latency."""
    messages = [
        {"role": "system", "content": "You are a friendly voice assistant. Answer in one or two short sentences."},
        {"role": "user", "content": "What's a good way to stay focused while working from home?"},
    ]
    ask(client, messages, quiet=True)  # warm-up: opens the connection, wakes the container if needed
    runs = []
    for i in range(n):
        _, st = ask(client, messages, quiet=True)
        runs.append(st)
        print(f"  #{i + 1:2d}  TTFT {st['ttft_ms']:6.0f} ms   ITL {st['itl_ms']:5.1f} ms   "
              f"e2e {st['e2e_ms']:6.0f} ms   {st['tokens']} tokens")
    for key, label in [("ttft_ms", "TTFT"), ("itl_ms", "ITL"), ("e2e_ms", "e2e")]:
        vals = sorted(r[key] for r in runs)
        p50, p90 = vals[len(vals) // 2], vals[min(len(vals) - 1, int(0.9 * len(vals)))]
        print(f"{label:5s} p50 {p50:7.1f} ms   p90 {p90:7.1f} ms   min {vals[0]:7.1f} ms")


def main():
    # HTTP/2 lets every request reuse one open connection. Over HTTP/1.1 the proxy keeps each streamed
    # response open, so the SDK has to open a new TCP + TLS connection per request (~400 ms to Ireland).
    client = OpenAI(
        base_url=base_url(),
        api_key=api_key(),
        max_retries=0,
        http_client=httpx.Client(http2=True, timeout=httpx.Timeout(600, connect=30)),
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if len(sys.argv) > 2 and sys.argv[1] == "--bench":
        bench(client, int(sys.argv[2]))
        return

    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        ask(client, messages)
        return

    print("Chat with Gemma 4 (Ctrl+C or empty line to quit)")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        print("gemma> ", end="", flush=True)
        reply, _ = ask(client, messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
