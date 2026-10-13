---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
library_name: vllm
base_model:
- google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
base_model_relation: quantized
tags:
- gemma4
- speculative-decoding
- mtp
- draft-model
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
[google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant),
the multi-token-prediction (MTP) drafter for Gemma 4 26B-A4B, for speculative decoding in
[vLLM](https://github.com/vllm-project/vllm).

It is meant to be paired with the W4A16 target [{target_repo}](https://huggingface.co/{target_repo}). Google's
model card asks for the assistant to be a QAT checkpoint at the same precision as the QAT target it drafts for.

| | |
|---|---|
| Architecture | Gemma 4 assistant: 4 decoder layers (Q-only attention, reads the target's KV cache), hidden size 1024 |
| Weights | int4, symmetric, group size 32 |
| Activations | bf16 (unquantized) |
| Checkpoint size | **622 MB** (vs 839 MB in bf16; most of what remains is the bf16 262k-token embedding) |
| Format | compressed-tensors `pack-quantized` |

## What was quantized

The drafter was trained with QAT for Q4_0, so round-to-nearest on the same grid (symmetric int4, group size 32)
needs no calibration data.

- **Quantized to int4 (22 linear layers):** the attention `q_proj`/`o_proj` and MLP of all 4 decoder layers, plus
  `pre_projection` and `post_projection` (the layers that map between the target's and the drafter's hidden size).
- **Kept in bf16:** `lm_head` / embeddings (tied).

## Usage

```bash
vllm serve {target_repo} \
    --max-model-len 32768 \
    --speculative-config '{{"method": "mtp", "model": "{repo}", "num_speculative_tokens": 4}}'
```

Use `"method": "mtp"`: vLLM maps the `gemma4_assistant` checkpoint to its Gemma 4 MTP drafter, which shares the
target's KV cache. Requires a vLLM release with Gemma 4 MTP support (tested on 0.31.0).

## Validation

Tested with **vLLM 0.31.0 on a single NVIDIA L40S**, paired with [{target_repo}](https://huggingface.co/{target_repo}):
the drafter loads in under a second, attaches to the target's KV cache (draft layers 0-2 to target layer 28,
layer 3 to layer 29), and generates correctly with `num_speculative_tokens=4`.

On three prompts (an explanation, a coding task and an advice question; greedy decoding, 300 tokens each):

| | |
|---|---|
| Draft acceptance rate | **59.2%** (481 of 812 draft tokens) |
| Mean tokens per target forward pass | **3.37** (vs 1 without speculative decoding) |
| Acceptance by draft position | 85% / 66% / 50% / 37% |

> These are greedy-decoding numbers on a small prompt set. Acceptance with sampling (`temperature=1.0`) and on
> other workloads will be lower.

## Reproduction

Made with [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.14.0 (transformers 5.17):

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

q4_0 = QuantizationArgs(
    num_bits=4, type="int", symmetric=True,
    strategy="group", group_size=32, observer="memoryless_minmax",
)
recipe = QuantizationModifier(
    config_groups={{"group_0": QuantizationScheme(targets=["Linear"], weights=q4_0)}},
    ignore=["lm_head", "re:.*centroids.*"],
)
oneshot(model=model, recipe=recipe)

model.save_pretrained("{name}", save_compressed=True)
tokenizer.save_pretrained("{name}")
```

## License

Released under the same terms as the base model: [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
All credit for the model and the QAT checkpoint goes to Google DeepMind.
