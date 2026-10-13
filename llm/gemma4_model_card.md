---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: vllm
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
- gemma4
- moe
- quantized
- w4a16
- int4
- qat
- compressed-tensors
- llm-compressor
- vllm
---

# {name}

A **4-bit weight (W4A16) [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)** build of
[google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized),
ready to serve with [vLLM](https://github.com/vllm-project/vllm).

Google ships vLLM-ready QAT `w4a16-ct` checkpoints for Gemma 4 E2B, E4B, 12B and 31B, but not for the
26B-A4B mixture-of-experts model. This repo fills that gap using the same quantization scheme as Google's
[gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct).

| | |
|---|---|
| Architecture | Gemma 4 MoE, 26B total / ~4B active parameters, 128 experts (top-8), 30 layers |
| Modalities | Text + image input, text output |
| Weights | int4, symmetric, group size 32 |
| Activations | bf16 (unquantized) |
| Checkpoint size | **15.3 GB** (vs ~52 GB in bf16) |
| Format | compressed-tensors `pack-quantized` |

## Why this quantization works well

The base checkpoint was produced with **quantization-aware training (QAT) targeting Q4_0**: symmetric int4 with
one scale per group of 32 weights. The model's weights are already adapted to that grid, so quantizing with the
*exact same* grid using plain round-to-nearest recovers the QAT quality. No calibration data, GPTQ or AWQ is
needed. Using a different scheme (for example the common W4A16 preset with group size 128) would throw away
the benefit of QAT.

### What was quantized

- **Quantized to int4:** all attention projections, the dense MLP, and **all 128 experts in every layer**
  (11,725 linear layers in total). Experts are stored fused in the original checkpoint. They were split into
  per-expert linear layers for quantization and re-fused on save in the layout vLLM expects.
- **Kept in bf16:** vision tower, `embed_vision`, MoE router, `lm_head` / embeddings.

## Usage

### vLLM server (OpenAI-compatible)

```bash
vllm serve {repo} --max-model-len 8192
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="{repo}",
    messages=[{{"role": "user", "content": "Explain mixture-of-experts in two sentences."}}],
    temperature=1.0, top_p=0.95,
)
print(resp.choices[0].message.content)
```

### vLLM offline

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{repo}", max_model_len=8192)
params = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=512)
out = llm.chat([{{"role": "user", "content": "Write a haiku about quantization."}}], params)
print(out[0].outputs[0].text)
```

Recommended sampling (from Google's Gemma 4 defaults): `temperature=1.0`, `top_p=0.95`, `top_k=64`.

## Validation

Tested with **vLLM 0.31.0 on a single NVIDIA L40S (48 GB)**:

- Model loaded in 23 s using **15.9 GiB** of GPU memory, leaving 22.7 GiB for KV cache at `max_model_len=8192`
  and the default `gpu_memory_utilization`.
- Sanity-check prompts all answered correctly: an accurate three-sentence explanation of QAT, a correct
  `is_palindrome` implementation with tests, and `17 * 23 = 391`.

> **Note:** these are smoke tests, not a benchmark. No accuracy evaluation (MMLU, GSM8K, etc.) against the
> bf16 base model has been run yet.

## Reproduction

Made with [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.14.0 (transformers 5.x) on an
A100-80GB:

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"

processor = AutoProcessor.from_pretrained(MODEL_ID)
with load_quantizable_moe(AutoModelForImageTextToText):  # unfuse experts so they get quantized
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")

q4_0 = QuantizationArgs(
    num_bits=4, type="int", symmetric=True,
    strategy="group", group_size=32, observer="memoryless_minmax",
)
recipe = QuantizationModifier(
    config_groups={{"group_0": QuantizationScheme(targets=["Linear"], weights=q4_0)}},
    ignore=["lm_head", "re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*router.*"],
)
oneshot(model=model, recipe=recipe)

model.save_pretrained("{name}", save_compressed=True)
processor.save_pretrained("{name}")
```

## License

Released under the same terms as the base model: [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
All credit for the model and the QAT checkpoint goes to Google DeepMind.
