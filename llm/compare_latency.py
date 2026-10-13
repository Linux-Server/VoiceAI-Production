"""Compare latency of two deployed vLLM endpoints from this machine (read-only: only sends requests).

    uv run python llm/compare_latency.py
"""

import http.client
import json
import statistics
import threading
import time
import urllib.parse

import modal

MY_APP = "gemma4-serve-eu"

N_SEQ = 20  # sequential requests
CONCURRENCY = 8  # parallel requests for the throughput test
MAX_TOKENS = 128
PROMPTS = [f"Write a short paragraph about the number {i}." for i in range(N_SEQ)]


def endpoints():
    mine_key = modal.Function.from_name(MY_APP, "_api_key").remote()
    return [
        {
            "name": "Gemma4-26B-A4B W4A16 (mine, L40S/A100-40GB eu-west)",
            "url": modal.Server.from_name(MY_APP, "Server").get_url(),
            "model": "gemma-4-26b-a4b-w4a16",
            "headers": {"Authorization": f"Bearer {mine_key}"},
        },
        {
            "name": "HireStella-Gemma-12B (L40S/A100-40GB eu-west)",
            "url": modal.Server.from_name("HireStella-Gemma-12B", "Server").get_url(),
            "model": "llm",
            "headers": {},
        },
    ]


def connect(url):
    return http.client.HTTPSConnection(urllib.parse.urlparse(url).netloc, timeout=600)


def wait_ready(ep):
    start = time.perf_counter()
    while True:
        try:
            conn = connect(ep["url"])
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return time.perf_counter() - start
        except Exception:
            pass
        time.sleep(2)


def stream(conn, ep, prompt):
    body = json.dumps({
        "model": ep["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,  # always generate exactly MAX_TOKENS so totals are comparable
        "temperature": 1.0, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    headers = {"Content-Type": "application/json", **ep["headers"]}
    start = time.perf_counter()
    ttft, tokens = None, 0
    conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            tokens = chunk["usage"]["completion_tokens"]
        delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
        if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
            ttft = time.perf_counter() - start
    resp.read()
    total = time.perf_counter() - start
    return {"ttft": ttft * 1000, "total": total * 1000, "tps": tokens / (total - ttft), "tokens": tokens}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def bench(ep):
    r = {"cold_start_s": wait_ready(ep)}
    conn = connect(ep["url"])

    rtts = []
    for _ in range(10):
        t = time.perf_counter()
        conn.request("GET", "/health")
        conn.getresponse().read()
        rtts.append((time.perf_counter() - t) * 1000)
    r["rtt"] = rtts

    stream(conn, ep, "Hi")  # warm-up
    r["seq"] = [stream(conn, ep, p) for p in PROMPTS]

    results, lock = [], threading.Lock()

    def worker(i):
        c = connect(ep["url"])
        out = stream(c, ep, PROMPTS[i % len(PROMPTS)])
        with lock:
            results.append(out)

    t = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t
    r["conc"] = results
    r["conc_throughput"] = sum(x["tokens"] for x in results) / wall
    return r


def main():
    eps = endpoints()
    out = {}
    # Wake both at once so cold starts overlap, then benchmark one at a time.
    ready = {}
    threads = [threading.Thread(target=lambda e=e: ready.__setitem__(e["name"], wait_ready(e))) for e in eps]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for ep in eps:
        print(f"benchmarking {ep['name']} ...", flush=True)
        out[ep["name"]] = bench(ep)
        out[ep["name"]]["cold_start_s"] = ready[ep["name"]]

    rows = [
        ("Ready after (cold start if idle)", lambda r: f"{r['cold_start_s']:.0f} s"),
        ("Request RTT /health p50", lambda r: f"{pct(r['rtt'], .5):.0f} ms"),
        ("TTFT p50 / p90", lambda r: f"{pct([x['ttft'] for x in r['seq']], .5):.0f} / {pct([x['ttft'] for x in r['seq']], .9):.0f} ms"),
        (f"Total {MAX_TOKENS} tok p50 / p90", lambda r: f"{pct([x['total'] for x in r['seq']], .5):.0f} / {pct([x['total'] for x in r['seq']], .9):.0f} ms"),
        ("Decode tok/s (single stream)", lambda r: f"{statistics.mean(x['tps'] for x in r['seq']):.0f}"),
        (f"TTFT p50 @ {CONCURRENCY} concurrent", lambda r: f"{pct([x['ttft'] for x in r['conc']], .5):.0f} ms"),
        (f"Per-stream tok/s @ {CONCURRENCY}", lambda r: f"{statistics.mean(x['tps'] for x in r['conc']):.0f}"),
        (f"Aggregate tok/s @ {CONCURRENCY}", lambda r: f"{r['conc_throughput']:.0f}"),
    ]
    names = list(out)
    w = 34
    print("\n" + "".ljust(w) + " | ".join(n.ljust(46) for n in names))
    for label, fmt in rows:
        print(label.ljust(w) + " | ".join(fmt(out[n]).ljust(46) for n in names))


if __name__ == "__main__":
    main()
