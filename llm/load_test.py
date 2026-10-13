"""Latency under load for the deployed Gemma endpoint: TTFT, inter-token latency and end-to-end latency at
several concurrency levels, measured from this machine. Every request streams exactly MAX_TOKENS tokens.

    uv run python llm/load_test.py
    uv run python llm/load_test.py --levels 1,8 --requests 16
"""

import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

import modal

MY_APP = "gemma4-serve-eu"
MODEL = "gemma-4-26b-a4b-w4a16"
MAX_TOKENS = 128


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def one_request(host, key, prompt):
    # Fresh connection per request: the modal.direct proxy leaves a streamed response open after
    # [DONE], so a kept-alive connection can't be reused. Connect (TCP + TLS) before the timer starts,
    # so TTFT matches what a client with a warm connection sees; setup time is reported separately.
    conn = http.client.HTTPSConnection(host, timeout=600)
    t0 = time.perf_counter()
    conn.connect()
    connect = time.perf_counter() - t0
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
    token_times, tokens = [], 0
    for line in resp:
        line = line.decode().strip()
        if line == "data: [DONE]":
            break  # don't wait for the connection to close; the modal.direct proxy keeps it open
        if not line.startswith("data: "):
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            tokens = chunk["usage"]["completion_tokens"]
        if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
            token_times.append(time.perf_counter())
    total = time.perf_counter() - start
    conn.close()
    ttft = token_times[0] - start
    # Mean gap between tokens after the first one (decode time / remaining tokens).
    itl = (token_times[-1] - token_times[0]) / max(1, tokens - 1)
    return {"ttft": ttft * 1000, "itl": itl * 1000, "e2e": total * 1000, "tokens": tokens, "connect": connect * 1000}


def run_level(host, key, concurrency, n_requests):
    results, errors, lock = [], [], threading.Lock()
    queue = list(range(n_requests))

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                i = queue.pop()
            try:
                r = one_request(host, key, f"Write a short paragraph about the number {i}.")
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001 - report and keep going
                with lock:
                    errors.append(str(e))


    t = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(errors),
        "ttft_p50": round(pct([r["ttft"] for r in results], .5)),
        "ttft_p90": round(pct([r["ttft"] for r in results], .9)),
        "itl_p50": round(pct([r["itl"] for r in results], .5), 1),
        "itl_p90": round(pct([r["itl"] for r in results], .9), 1),
        "e2e_p50": round(pct([r["e2e"] for r in results], .5)),
        "e2e_p90": round(pct([r["e2e"] for r in results], .9)),
        "connect_p50": round(pct([r["connect"] for r in results], .5)),
        # Steady-state aggregate: concurrency x mean (tokens / request latency). Wall time would also
        # count each request's connection setup, which isn't serving time.
        "throughput": round(concurrency * statistics.mean(r["tokens"] / (r["e2e"] / 1000) for r in results)),
        "per_request_tps": round(statistics.mean(r["tokens"] / ((r["e2e"] - r["ttft"]) / 1000) for r in results)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--requests", type=int, default=0, help="requests per level (default: max(20, 3x concurrency))")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    url = modal.Server.from_name(MY_APP, "Server").get_url()
    key = modal.Function.from_name(MY_APP, "_api_key").remote()
    host = urllib.parse.urlparse(url).netloc
    print(f"Endpoint: {url}", flush=True)

    t = time.perf_counter()
    while True:
        try:
            c = http.client.HTTPSConnection(host, timeout=30)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                break
        except OSError:
            pass
        time.sleep(2)
    print(f"Ready after {time.perf_counter() - t:.0f} s", flush=True)
    one_request(host, key, "Hi")  # warm-up

    rows = []
    for level in [int(x) for x in args.levels.split(",")]:
        n = args.requests or max(20, 3 * level)
        r = run_level(host, key, level, n)
        rows.append(r)
        print(json.dumps(r), flush=True)

    if args.out:
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
