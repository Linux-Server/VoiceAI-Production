"""Serve the W4A16 Gemma 4 26B-A4B as an OpenAI-compatible vLLM API on L40S/A100-40GB in eu-west (Ireland).

Routing matches where the GPU runs: requests enter Modal in eu-west, so they don't cross from another
region first. ap-south (Mumbai) is allowed as a fallback, but as of 2026-10-12 it had no L40S/A100-40GB.

Startup: vLLM boots once, warms up, goes to sleep, and Modal takes a GPU memory snapshot. Later cold starts
restore that snapshot and wake vLLM instead of redoing imports, torch.compile and CUDA graph capture.
Modal needs 2-3 snapshots per GPU type, so the first few cold starts per GPU type are still slow.

Deploy:
    uv run modal deploy llm/serve_gemma4.py
Deploy a separate copy with speculative decoding (the W4A16 assistant drafter from gemma4_assistant_compress.py):
    SPEC_DECODE=1 uv run modal deploy --name gemma4-serve-eu-spec llm/serve_gemma4.py
Benchmark (latency) and evaluate (quality) from this machine:
    uv run python llm/compare_latency.py
    uv run python llm/quality_eval.py
"""

import json
import os
import subprocess
import time

import modal

MODEL_DIR = "/models/gemma-4-26B-A4B-it-qat-w4a16-ct"
SERVED_NAME = "gemma-4-26b-a4b-w4a16"
ROUTING_REGION = "eu-west"  # where requests enter Modal; fixed at a Server's first deploy
COMPUTE_REGION = ["eu-west", "ap-south"]  # allowed regions
GPUS = ["L40S", "A100-40GB"]  # tried in order
PORT = 8000
MINUTES = 60

MAX_MODEL_LEN = 32768  # context window per request
MAX_NUM_SEQS = 32  # concurrent requests; CUDA graphs are captured only up to this batch size
GPU_MEMORY_UTILIZATION = 0.94
# Store the KV cache in FP8 (half the bytes of bf16), roughly doubling how many tokens fit.
# vLLM's Triton attention needs native FP8 (SM89+, e.g. L40S); the A100 (SM80) falls back to bf16.
FP8_KV_MIN_COMPUTE_CAP = 8.9
# Schedule the next step while the GPU runs the current one, hiding CPU scheduling time.
ASYNC_SCHEDULING = True
# Text only: skips loading the vision tower and its profiling, leaving that memory to the KV cache.
# Set to False to accept images again.
LANGUAGE_MODEL_ONLY = True
# OpenAI-style tool calling (`tools` / `tool_choice` in requests). The model's own chat template formats tools;
# vLLM's `gemma4` parser turns the model's tool-call tokens back into `tool_calls` in the response.
TOOL_CALLING = True
# Moves the model's <|channel>thought ... <channel|> block into `reasoning_content`. Without it, replies after a
# tool result can start with an empty thinking block in `content`, even with enable_thinking off.
REASONING_PARSER = "gemma4"
# Log each request's prompt and its final output (with finish_reason), to debug replies that arrive empty.
# Prompts and replies end up in Modal logs; MAX_LOG_LEN caps how many characters/tokens of each are printed.
LOG_REQUESTS = True
MAX_LOG_LEN = 2000
# Speculative decoding with Google's MTP assistant drafter, compressed to W4A16 like the target. Opt-in per deploy
# with SPEC_DECODE=1; the setting reaches the container through a secret built at deploy time.
DRAFT_DIR = "/models/gemma-4-26B-A4B-it-qat-w4a16-ct-assistant"
NUM_SPECULATIVE_TOKENS = 4  # Google's recommendation for 26B-A4B
spec_secret = modal.Secret.from_dict({"SPEC_DECODE": os.environ.get("SPEC_DECODE", "")})

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Modal needs `python` on PATH
    .env(
        {
            "VLLM_SERVER_DEV_MODE": "1",  # exposes /sleep and /wake_up, used around the snapshot
            "TORCH_CPP_LOG_LEVEL": "FATAL",
        }
    )
)

models_vol = modal.Volume.from_name("llm-compressed-models")
# No compile/JIT cache volumes: a GPU snapshot that saw cache files being created in a Volume fails to
# restore ("failed to walk .../0.7.0.post1/89"), even after committing. Compiled kernels live in the
# snapshot instead; only snapshot-creating boots (2-3 per GPU type) pay for compilation.
api_key = modal.Secret.from_name("vllm-api-key")  # VLLM_API_KEY, read by `vllm serve`

app = modal.App("gemma4-serve-eu")

with vllm_image.imports():
    import requests


def _auth():
    return {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}


def _wait_ready(process: subprocess.Popen, timeout: int = 30 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=process.args)
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, requests.exceptions.Timeout):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready within {timeout} s")


def _kv_cache_dtype() -> str:
    cap = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).split()[0]
    return "fp8" if float(cap) >= FP8_KV_MIN_COMPUTE_CAP else "auto"


def _warmup():
    """Run a few short and one longer request so JIT kernels are compiled before the snapshot."""
    prompts = ["Hello, how are you?", "Summarize the plot of Hamlet. " * 200]
    for prompt in prompts:
        for _ in range(2):
            requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                headers=_auth(),
                json={
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            ).raise_for_status()


@app.server(
    image=vllm_image,
    gpu=GPUS,
    compute_region=COMPUTE_REGION,
    routing_region=ROUTING_REGION,
    volumes={"/models": models_vol},
    secrets=[api_key, spec_secret],
    port=PORT,
    unauthenticated=True,  # Modal's proxy is open; vLLM itself requires VLLM_API_KEY on /v1 routes
    target_concurrency=MAX_NUM_SEQS,
    scaledown_window=20 * MINUTES,  # scale to zero after 20 idle minutes
    startup_timeout=45 * MINUTES,  # the first boot per GPU type compiles and creates the snapshot
    exit_grace_period=5,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Server:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_DIR,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-cudagraph-capture-size", str(MAX_NUM_SEQS),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--kv-cache-dtype", _kv_cache_dtype(),
            "--enable-sleep-mode",
        ]
        if ASYNC_SCHEDULING:
            cmd.append("--async-scheduling")
        if LANGUAGE_MODEL_ONLY:
            cmd.append("--language-model-only")
        if TOOL_CALLING:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "gemma4"]
        if REASONING_PARSER:
            cmd += ["--reasoning-parser", REASONING_PARSER]
        if LOG_REQUESTS:
            cmd += ["--enable-log-requests", "--enable-log-outputs", "--max-log-len", str(MAX_LOG_LEN)]
        if os.environ.get("SPEC_DECODE") == "1":
            spec = {"method": "mtp", "model": DRAFT_DIR, "num_speculative_tokens": NUM_SPECULATIVE_TOKENS}
            cmd += ["--speculative-config", json.dumps(spec)]
        print(*cmd, flush=True)
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()
        # Level 1 offloads weights to CPU and drops the (empty) KV cache, so the snapshot stays small.
        requests.post(f"http://127.0.0.1:{PORT}/sleep?level=1", headers=_auth(), timeout=300).raise_for_status()

    @modal.enter(snap=False)
    def wake(self):
        requests.post(f"http://127.0.0.1:{PORT}/wake_up", headers=_auth(), timeout=300).raise_for_status()
        _wait_ready(self.process)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.function(secrets=[api_key])
def _api_key() -> str:
    return os.environ["VLLM_API_KEY"]
