"""Server-side latency of the deployed Gemma endpoint, read from vLLM's own /metrics (excludes the network),
for voice-style turns at a given concurrency. Also prints the client-side TTFT for reference.

    uv run python llm/server_latency.py
    uv run python llm/server_latency.py --concurrency 1,8 --turns 20
    uv run python llm/server_latency.py --prompt-tokens 2000 --max-tokens 150
    uv run python llm/server_latency.py --prompt-tokens 10000 --max-tokens 200 --prefix-cache
"""

import argparse
import json
import re
import statistics
import threading
import time
import uuid

import httpx
import modal

MY_APP = "gemma4-serve-eu"
MODEL = "gemma-4-26b-a4b-w4a16"
MESSAGES = [
    {"role": "system", "content": "You are a friendly voice assistant. Answer in one or two short sentences."},
    {"role": "user", "content": "What's a good way to stay focused while working from home?"},
]
METRICS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "queue": "vllm:request_queue_time_seconds",
}
# Follow-up questions for --prefix-cache conversations.
FOLLOW_UPS = ["Can you give me another one?", "How long should I try that for?", "What if that doesn't work?",
              "Any tips for the afternoon slump?", "How do I handle noisy neighbours?"]


def read_metrics(client, headers):
    text = client.get("/metrics", headers=headers).text
    out = {}
    for key, name in METRICS.items():
        s = re.search(rf"^{name}_sum\{{[^}}]*\}} ([0-9.e+-]+)", text, re.M)
        n = re.search(rf"^{name}_count\{{[^}}]*\}} ([0-9.e+-]+)", text, re.M)
        out[key] = (float(s.group(1)), float(n.group(1))) if s and n else (0.0, 0.0)
    for key in ("prefix_cache_queries", "prefix_cache_hits"):  # counted in tokens
        m = re.search(rf"^vllm:{key}_total\{{[^}}]*\}} ([0-9.e+-]+)", text, re.M)
        out[key] = float(m.group(1)) if m else 0.0
    return out


FILLER = ("Background notes: the user works remotely, has a standing desk, two monitors, a noisy street outside, "
          "and meetings scattered across three time zones. ")


def build_messages(prompt_tokens, client, headers):
    """Pad the system prompt with filler until the prompt is ~prompt_tokens long (0 = the default short prompt)."""
    if not prompt_tokens:
        return MESSAGES

    def count(n):
        msgs = [{"role": "system", "content": MESSAGES[0]["content"] + "\n\n" + FILLER * n}, MESSAGES[1]]
        body = {"model": MODEL, "messages": msgs, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}
        return client.post("/v1/chat/completions", json=body, headers=headers).json()["usage"]["prompt_tokens"]

    base, per = count(0), count(10) / 10 - count(0) / 10
    return [{"role": "system", "content": MESSAGES[0]["content"] + "\n\n" + FILLER * round((prompt_tokens - base) / per)},
            MESSAGES[1]]


def turn(client, headers, messages=MESSAGES, max_tokens=60, session=None):
    """Stream one turn and return (client TTFT in ms, reply text).

    Without a session, a unique prefix per request defeats vLLM's prefix cache so every turn pays the full prefill.
    With one, the prefix stays the same across a conversation's turns, as in real traffic.
    """
    messages = [{**messages[0], "content": f"[session {session or uuid.uuid4()}] " + messages[0]["content"]},
                *messages[1:]]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    start = time.perf_counter()
    ttft, reply = None, []
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
        for line in resp.iter_lines():  # read to the end so no request keeps running on the server
            if line == "data: [DONE]":
                break
            if line.startswith("data: ") and '"content"' in line:
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                choices = json.loads(line[6:])["choices"]
                reply.append(choices[0]["delta"].get("content") or "" if choices else "")
    return ttft, "".join(reply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=MY_APP, help="deployed app to benchmark, e.g. gemma4-serve-eu-spec")
    ap.add_argument("--concurrency", default="1,8")
    ap.add_argument("--turns", type=int, default=20, help="requests per client")
    ap.add_argument("--prompt-tokens", type=int, default=0, help="pad the prompt to ~this many tokens (0 = short prompt)")
    ap.add_argument("--max-tokens", type=int, default=60, help="tokens generated per turn")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="each caller holds one growing conversation, so turns after the first reuse the cached prefix")
    args = ap.parse_args()

    url = modal.Server.from_name(args.app, "Server").get_url()
    key = modal.Function.from_name(args.app, "_api_key").remote()
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=url, http2=True, timeout=httpx.Timeout(600, connect=30)) as client:
        t = time.perf_counter()
        while (r := client.get("/health")).status_code != 200:
            if time.perf_counter() - t > 20 * 60:
                raise SystemExit(f"Server not ready after 20 min (last status {r.status_code})")
            time.sleep(3)
        print(f"Ready after {time.perf_counter() - t:.0f} s over {r.http_version}", flush=True)
        messages = build_messages(args.prompt_tokens, client, headers)
        if args.prompt_tokens:
            print(f"Prompt padded to ~{args.prompt_tokens} tokens, {args.max_tokens} generated per turn", flush=True)
        if args.prefix_cache:
            print("Prefix caching: each caller holds one growing conversation", flush=True)
        for _ in range(3):
            turn(client, headers, messages, args.max_tokens)  # warm-up

        for c in [int(x) for x in args.concurrency.split(",")]:
            before = read_metrics(client, headers)
            client_ttfts, lock = [], threading.Lock()

            def worker():
                # Each simulated caller has its own long-lived HTTP/2 connection.
                session = uuid.uuid4() if args.prefix_cache else None
                convo = list(messages)
                with httpx.Client(base_url=url, http2=True, timeout=httpx.Timeout(600, connect=30)) as own:
                    for i in range(args.turns):
                        v, reply = turn(own, headers, convo, args.max_tokens, session)
                        with lock:
                            client_ttfts.append(v)
                        if args.prefix_cache:
                            convo += [{"role": "assistant", "content": reply},
                                      {"role": "user", "content": FOLLOW_UPS[i % len(FOLLOW_UPS)]}]

            threads = [threading.Thread(target=worker) for _ in range(c)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            time.sleep(2)  # let the server finish requests whose streams we stopped reading
            after = read_metrics(client, headers)
            row = {"concurrency": c}
            for k in METRICS:
                ds, dn = after[k][0] - before[k][0], after[k][1] - before[k][1]
                row[f"server_{k}_ms"] = round(1000 * ds / dn, 1) if dn else None
            row["client_ttft_p50_ms"] = round(statistics.median(v for v in client_ttfts if v))
            queries = after["prefix_cache_queries"] - before["prefix_cache_queries"]
            hits = after["prefix_cache_hits"] - before["prefix_cache_hits"]
            row["prefix_cache_hit_pct"] = round(100 * hits / queries, 1) if queries else None
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
