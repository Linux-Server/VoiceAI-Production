"""Compress an LLM with llm-compressor on a Modal GPU.

Usage:
    uv run modal run llm/compress.py
    uv run modal run llm/compress.py --model-id meta-llama/Llama-3.1-8B-Instruct --scheme W4A16
    uv run modal run llm/compress.py --model-id Qwen/Qwen2.5-7B-Instruct --scheme FP8_DYNAMIC

Test with vLLM:
    uv run modal run llm/compress.py::test --save-dir <save-dir>

Download the result:
    uv run modal volume get llm-compressed-models <save-dir> ./models/
"""

import modal

GPU = "A10G"  # use "A100-80GB" or "H100" for models above ~8B

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llmcompressor==0.14.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("llm-compressed-models", create_if_missing=True)

app = modal.App("llm-compress", image=image)


@app.function(
    gpu=GPU,
    timeout=4 * 60 * 60,
    volumes={"/cache": hf_cache, "/output": output_vol},
    secrets=[modal.Secret.from_name("hf-token")],
)
def compress(model_id: str, scheme: str, num_samples: int, max_seq_len: int) -> str:
    from compressed_tensors.offload import dispatch_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if scheme == "FP8_DYNAMIC":
        # Data-free: no calibration set needed.
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
        oneshot(model=model, recipe=recipe)
    else:
        # W4A16 / W8A8 etc: GPTQ with a calibration set.
        from llmcompressor.modifiers.gptq import GPTQModifier

        recipe = GPTQModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
        oneshot(
            model=model,
            dataset="perfectblend",
            splits=f"train[:{num_samples}]",
            recipe=recipe,
            max_seq_length=max_seq_len,
            num_calibration_samples=num_samples,
        )

    print("========== SAMPLE GENERATION ==========")
    dispatch_model(model)
    sample = tokenizer("Hello my name is", return_tensors="pt").to(model.device)
    output = model.generate(**sample, max_new_tokens=50)
    print(tokenizer.decode(output[0]))
    print("=======================================")

    save_dir = f"{model_id.split('/')[-1]}-{scheme}"
    model.save_pretrained(f"/output/{save_dir}", save_compressed=True)
    tokenizer.save_pretrained(f"/output/{save_dir}")
    output_vol.commit()
    return save_dir


# Official vLLM image: includes the CUDA toolkit (nvcc) that FlashInfer needs for JIT kernels.
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.31.0")
    .entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/bin/python")  # Modal needs `python` on PATH
)


@app.function(image=vllm_image, gpu=GPU, timeout=30 * 60, volumes={"/output": output_vol})
def vllm_generate(save_dir: str, prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=f"/output/{save_dir}", max_model_len=4096)
    params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def test(save_dir: str = "Qwen2.5-0.5B-Instruct-W4A16"):
    """uv run modal run llm/compress.py::test --save-dir <save-dir>"""
    prompts = ["Explain quantization of neural networks in two sentences.", "What is the capital of France?"]
    for prompt, answer in zip(prompts, vllm_generate.remote(save_dir, prompts)):
        print(f"\n>>> {prompt}\n{answer}")


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    scheme: str = "W4A16",
    num_samples: int = 512,
    max_seq_len: int = 2048,
):
    save_dir = compress.remote(model_id, scheme, num_samples, max_seq_len)
    print(f"Saved to volume 'llm-compressed-models' at /{save_dir}")
    print(f"Download: modal volume get llm-compressed-models {save_dir} ./models/")
