## Vietnamese legal Q&A model on NPU

### Overview
Estimate speed up of speculative decoding on Qwen3-0.6B / Qwen3-1.7B of Vietnamese legal article retrieval (Zalo Legal 2021, 640 test questions), quantized to W4A16 and profiled on Snapdragon devices through Qualcomm AI Hub Workbench.

### 1. Material
**Dataset:** Zalo Legal 2021, from [another-symato/VMTEB-Zalo-legel-retrieval](https://huggingface.co/datasets/another-symato/VMTEB-Zalo-legel-retrieval)
- Corpus: Vietnamese law articles (title + text)
- Questions: 2556 train, 640 test, each with one or more gold articles

**Models**
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): draft model for speculative decoding
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B): target model

**Devices** (Qualcomm AI Hub Workbench, W4A16, context 2048)
- Galaxy S26 (Snapdragon 8 Elite Gen 5)
- Galaxy S25 (Snapdragon 8 Elite)
- Snapdragon X Elite CRD
- Snapdragon X Plus 8-Core CRD

**Tools:** `qai-hub-models` 0.67.0, QAIRT 2.51, GenieX runtime

### 2. Method
**2.1. Retrieval-augmented generation (RAG) and Low-Rank Adaptation (LoRA)** ([train.py](train.py))
- Retrieval: BM25 over the corpus (syllables + syllable pairs) gives the top 5 articles per question. They are shuffled and each is cut to 256 tokens.
- LoRA: r=16, alpha=32, all linear layers, 2 epochs, lr 2e-4. Trained on the train questions whose top 5 contains a gold article, 200 of them held out as dev.

**2.2. Quantize** ([quant.py](quant.py), [aihub.py](aihub.py))
- Merge the LoRA adapter into the base weights.
- Quantize to W4A16 (4-bit weights, 16-bit activations) with the AI Hub recipes.

**2.3. Estimate speed up of speculative decoding** ([specdec.py](specdec.py))

Qwen3-0.6B drafts k tokens, Qwen3-1.7B verifies them in one call. No draft-and-verify loop runs on a device; the speed is projected from two measurements:
1. Acceptance: on the GPU (W4A16 simulation), greedy decoding of the 640 test prompts gives the mean tokens per cycle E(k) for k = 1..6.
2. Cost: decode step of each model and the target's verify call, profiled on AI Hub.

```
before = 1000 / T_target                       tok/s
after  = E(k) / (k × T_draft + T_verify) × 1000 tok/s
```
- `T_verify`: target's 128-token call (a), or 4-token call (b, X Plus only).
- Latencies are scaled so the target alone matches Qualcomm's published tok/s.
- Greedy speculative decoding gives the same answer as the target, so accuracy is the W4A16 one.

### 3. Result
Qwen3-1.7B W4A16 with speculative decoding (Qwen3-0.6B draft), context 2048. Projected, an upper bound.

| Device | Tok/s | TTFT (s) | Peak mem (MB) | Time per question (s) |
|---|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 51.5 | 0.61 | 282 | 3.03 |
| Galaxy S25 (8 Elite) | 42.8 | 0.76 | 275 | 3.68 |
| X Elite CRD | 32.8 | 1.02 | 224 | 4.82 |
| X Plus 8-Core CRD | 38.6 | 1.01 | 224 | 4.25 |

- Tok/s: decode speed.
- TTFT: time to first token for the ~1.4k-token RAG prompt, target and draft both reading the prompt.
- Peak mem: decode graphs of Qwen3-1.7B + Qwen3-0.6B.
- Time per question: BM25 + TTFT + 124.8 answer tokens.

More details at [result.md](results/result.md)