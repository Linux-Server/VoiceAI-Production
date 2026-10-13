"""Compress google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant (the MTP drafter) to W4A16 for vLLM.

Same scheme as gemma4_compress.py: the drafter was also trained with QAT for Q4_0, so round-to-nearest
on the same grid (symmetric int4, group size 32) needs no calibration data. Google's model card asks for
the assistant to match the target's precision, and the target is W4A16. The centroid head that picks
which part of the vocabulary to score, and the tied lm_head, stay in bf16 (vLLM loads them unquantized).

Usage:
    uv run modal run llm/gemma4_assistant_compress.py::main   # compress, then test spec decoding with vLLM
    uv run modal run llm/gemma4_assistant_compress.py::test   # vLLM test only
    HF_PUSH_TOKEN=hf_... uv run modal run llm/gemma4_assistant_compress.py::upload --repo-id <user>/<name> [--card-only]
"""

import os
from pathlib import Path

import modal

MODEL_ID = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
SAVE_DIR = "gemma-4-26B-A4B-it-qat-w4a16-ct-assistant"
TARGET_DIR = "gemma-4-26B-A4B-it-qat-w4a16-ct"  # made by gemma4_compress.py
TARGET_REPO = "sachin6624/gemma-4-26B-A4B-it-qat-w4a16-ct"  # where gemma4_compress.py::upload pushed it

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llmcompressor==0.14.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Modal needs `python` on PATH
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("llm-compressed-models", create_if_missing=True)

app = modal.App("gemma4-assistant-compress")


@app.function(
    image=image,
    gpu="L4",  # the drafter is ~0.4B params; any GPU will do
    timeout=60 * 60,
    volumes={"/cache": hf_cache, "/output": output_vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def compress():
    import transformers
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print("transformers", transformers.__version__)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    hf_cache.commit()
    print(model)

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
        ignore=["lm_head", "re:.*centroids.*"],
    )
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(f"/output/{SAVE_DIR}", save_compressed=True)
    tokenizer.save_pretrained(f"/output/{SAVE_DIR}")
    output_vol.commit()


@app.function(image=vllm_image, gpu="L40S", timeout=30 * 60, volumes={"/output": output_vol})
def vllm_spec_generate(prompts: list[str], num_speculative_tokens: int = 4) -> tuple[list[str], dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=f"/output/{TARGET_DIR}",
        max_model_len=8192,
        language_model_only=True,
        disable_log_stats=False,  # needed for llm.get_metrics()
        speculative_config={
            "method": "mtp",
            "model": f"/output/{SAVE_DIR}",
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    params = SamplingParams(temperature=0.0, max_tokens=300)
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params,
                       chat_template_kwargs={"enable_thinking": False})
    stats = {m.name: getattr(m, "value", None) or getattr(m, "values", None)
             for m in llm.get_metrics() if "spec_decode" in m.name}
    return [o.outputs[0].text for o in outputs], stats


def run_test():
    prompts = [
        "Explain quantization-aware training in three sentences.",
        "Write a Python function that checks whether a string is a palindrome.",
        "What's a good way to stay focused while working from home?",
    ]
    answers, stats = vllm_spec_generate.remote(prompts)
    for prompt, answer in zip(prompts, answers):
        print(f"\n>>> {prompt}\n{answer}")
    print("\nSpec decode metrics:")
    for name, value in stats.items():
        print(f"  {name}: {value}")
    drafts, accepted = stats.get("vllm:spec_decode_num_draft_tokens"), stats.get("vllm:spec_decode_num_accepted_tokens")
    if drafts:
        print(f"  acceptance rate: {accepted / drafts:.1%}")


@app.local_entrypoint()
def test():
    run_test()


# Token comes from the local shell for this run only; nothing is stored in Modal.
push_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_PUSH_TOKEN", "")})


@app.function(image=image, timeout=30 * 60, volumes={"/output": output_vol}, secrets=[push_secret])
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
    """HF_PUSH_TOKEN=hf_... uv run modal run llm/gemma4_assistant_compress.py::upload --repo-id user/name [--card-only]"""
    template = (Path(__file__).parent / "gemma4_assistant_model_card.md").read_text()
    card = template.format(name=repo_id.split("/")[-1], repo=repo_id, target_repo=TARGET_REPO)
    print(push.remote(repo_id, private=not public, card=card, card_only=card_only))


@app.local_entrypoint()
def main():
    compress.remote()
    print(f"Saved to volume 'llm-compressed-models' at /{SAVE_DIR}")
    run_test()
