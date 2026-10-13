"""Compare answer quality of two deployed vLLM endpoints with lm-evaluation-harness (read-only: only sends requests).

    uv run python llm/quality_eval.py
    uv run python llm/quality_eval.py --tasks gsm8k --limit 50   # quick check

Results land in results/quality/<endpoint>/ and a summary table is printed at the end.
"""

import argparse
import glob
import json
import os
import subprocess
import threading
from pathlib import Path

import modal

MY_APP = "gemma4-serve-eu"
OUT = Path(__file__).parent / "results" / "quality"

# task -> (lm-eval args, metric keys to report)
TASKS = {
    "gsm8k": (["--num_fewshot", "5", "--fewshot_as_multiturn"], ["exact_match,strict-match", "exact_match,flexible-extract"]),
    "ifeval": ([], ["prompt_level_strict_acc,none", "inst_level_strict_acc,none", "prompt_level_loose_acc,none"]),
    "mmlu_pro": (["--num_fewshot", "5", "--fewshot_as_multiturn"], ["exact_match,custom-extract"]),
}
# Extra generation settings per task. gsm8k's default 256-token cap cut off ~150 of the 26B model's
# answers mid-working (it writes longer step-by-step solutions), so it scored verbosity, not math.
GEN_OVERRIDES = {"gsm8k": ["max_gen_toks=1024"]}
# Per-task sample limits (None = full set). mmlu_pro's limit applies to each of its 14 subjects.
LIMITS = {"gsm8k": None, "ifeval": None, "mmlu_pro": 25}


def endpoints():
    key = modal.Function.from_name(MY_APP, "_api_key").remote()
    return {
        "gemma4-26b-a4b": {
            "url": modal.Server.from_name(MY_APP, "Server").get_url(),
            "model": "gemma-4-26b-a4b-w4a16",
            "key": key,
        },
        "hirestella-12b": {
            "url": modal.Server.from_name("HireStella-Gemma-12B", "Server").get_url(),
            "model": "llm",
            "key": "",
        },
    }


def run(name, ep, task, limit, concurrency):
    extra, _ = TASKS[task]
    out = OUT / name
    cmd = [
        "lm_eval", "--model", "local-chat-completions",
        "--model_args", ",".join([
            f"model={ep['model']}",
            f"base_url={ep['url']}/v1/chat/completions",
            f"num_concurrent={concurrency}",
            "max_retries=5",
            "tokenized_requests=False",
            "timeout=600",
        ]),
        "--tasks", task,
        "--apply_chat_template",
        # Greedy decoding, and thinking explicitly off on every request.
        "--gen_kwargs", "temperature=0", "chat_template_kwargs={'enable_thinking': False}",
        *GEN_OVERRIDES.get(task, []),
        "--output_path", str(out),
        "--log_samples",
        *extra,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    env = {**os.environ, "OPENAI_API_KEY": ep["key"]}
    log = OUT / f"{name}_{task}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[{name}] {task} finished (exit {rc}), log: {log}", flush=True)


def latest_results(name, task):
    files = sorted(glob.glob(str(OUT / name / "**" / "results_*.json"), recursive=True), key=os.path.getmtime)
    for f in reversed(files):
        data = json.load(open(f))
        if task in data.get("results", {}):
            return data["results"][task]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="override every task's sample limit")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    tasks = args.tasks.split(",")

    eps = endpoints()

    def worker(name, ep):
        for task in tasks:
            run(name, ep, task, args.limit if args.limit is not None else LIMITS[task], args.concurrency)

    threads = [threading.Thread(target=worker, args=(n, e)) for n, e in eps.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n{'task / metric':48s}" + "".join(f"{n:>18s}" for n in eps))
    for task in tasks:
        for metric in TASKS[task][1]:
            vals = []
            for n in eps:
                r = latest_results(n, task)
                v = r.get(metric) if r else None
                vals.append(f"{v * 100:17.1f}%" if isinstance(v, (int, float)) else f"{'n/a':>18s}")
            print(f"{task + ' / ' + metric:48s}" + "".join(vals))


if __name__ == "__main__":
    main()
