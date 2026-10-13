"""Time a cold start of the deployed Gemma endpoint: first request until it answers a chat request.

    uv run python llm/coldstart.py
"""

import json
import time
import urllib.error
import urllib.request

import modal

MY_APP = "gemma4-serve-eu"


def main():
    url = modal.Server.from_name(MY_APP, "Server").get_url()
    key = modal.Function.from_name(MY_APP, "_api_key").remote()
    print(f"Endpoint: {url}", flush=True)

    start = time.perf_counter()
    while True:
        try:
            if urllib.request.urlopen(f"{url}/health", timeout=30).status == 200:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    ready = time.perf_counter() - start

    body = json.dumps({
        "model": "gemma-4-26b-a4b-w4a16",
        "messages": [{"role": "user", "content": "Say hello in five words."}],
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t = time.perf_counter()
    answer = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    first = time.perf_counter() - t

    print(f"Healthy after {ready:.1f} s; first chat reply in {first:.2f} s: {answer!r}")


if __name__ == "__main__":
    main()
