"""Compress google/gemma-4-26B-A4B-it-qat-q4_0-unquantized to W4A16 (compressed-tensors) for vLLM.

The checkpoint was trained with QAT for Q4_0 (symmetric int4, group size 32), so plain
round-to-nearest with the same grid recovers the QAT quality - no calibration data needed.
The scheme mirrors Google's own google/gemma-4-31B-it-qat-w4a16-ct release, which doesn't
exist for 26B-A4B. The MoE router is also kept in bf16.

Usage:
    uv run modal run llm/gemma4_compress.py::main        # compress, then test with vLLM
    uv run modal run llm/gemma4_compress.py::test        # vLLM test only
    uv run modal volume get llm-compressed-models gemma-4-26B-A4B-it-qat-w4a16-ct ./models/
    HF_PUSH_TOKEN=hf_... uv run modal run llm/gemma4_compress.py::upload --repo-id <user>/<name> [--card-only]
"""

import os
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
SAVE_DIR = "gemma-4-26B-A4B-it-qat-w4a16-ct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llmcompressor==0.14.0", "torchvision")  # torchvision: Gemma4Processor
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Modal needs `python` on PATH
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("llm-compressed-models", create_if_missing=True)

app = modal.App("gemma4-compress")


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=128 * 1024,  # bf16 checkpoint is ~52 GB
    timeout=2 * 60 * 60,
    volumes={"/cache": hf_cache, "/output": output_vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def compress():
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from llmcompressor.modifiers.quantization import QuantizationModifier

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Unfuse the 3D expert tensors into per-expert Linear layers so they get quantized.
    with load_quantizable_moe(AutoModelForImageTextToText):
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
    hf_cache.commit()

    q4_0 = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=32,
        observer="memoryless_minmax",
    )
    recipe = QuantizationModifier(
        config_groups={"group_0": QuantizationScheme(targets=["Linear"], weights=q4_0)},
        ignore=["lm_head", "re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*router.*"],
    )
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(f"/output/{SAVE_DIR}", save_compressed=True)
    processor.save_pretrained(f"/output/{SAVE_DIR}")
    output_vol.commit()


@app.function(image=vllm_image, gpu="L40S", timeout=30 * 60, volumes={"/output": output_vol})
def vllm_generate(prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=f"/output/{SAVE_DIR}", max_model_len=8192)
    params = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=300)
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    return [o.outputs[0].text for o in outputs]


def run_test():
    prompts = [
        "Explain quantization-aware training in three sentences.",
        "Write a Python function that checks whether a string is a palindrome.",
        "What is 17 * 23? Answer with just the number.",
    ]
    for prompt, answer in zip(prompts, vllm_generate.remote(prompts)):
        print(f"\n>>> {prompt}\n{answer}")


@app.local_entrypoint()
def test():
    run_test()


# Token comes from the local shell for this run only; nothing is stored in Modal.
push_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_PUSH_TOKEN", "")})


@app.function(image=image, timeout=60 * 60, volumes={"/output": output_vol}, secrets=[push_secret])
def push(repo_id: str, private: bool, card: str, card_only: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    if not card_only:
        api.upload_folder(folder_path=f"/output/{SAVE_DIR}", repo_id=repo_id, ignore_patterns=[".tmp*"])
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def upload(repo_id: str, public: bool = False, card_only: bool = False):
    """HF_PUSH_TOKEN=hf_... uv run modal run llm/gemma4_compress.py::upload --repo-id user/name [--card-only]"""
    template = (Path(__file__).parent / "gemma4_model_card.md").read_text()
    card = template.format(name=repo_id.split("/")[-1], repo=repo_id)
    print(push.remote(repo_id, private=not public, card=card, card_only=card_only))


@app.local_entrypoint()
def main():
    compress.remote()
    print(f"Saved to volume 'llm-compressed-models' at /{SAVE_DIR}")
    run_test()
