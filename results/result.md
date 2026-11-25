
## Accuracy: RAG and LoRA (BF16)

| # | Condition | Model | acc@1 | MRR |
|---|---|---|---|---|
| 0 | Before: BM25 alone | – | 0.5094 | 0.6187 |
| 1 | + RAG (BM25 top 5) | Qwen3-0.6B | 0.1422 | 0.3456 |
| 2 | After: + RAG + LoRA | Qwen3-0.6B | **0.6359** | **0.7004** |
| 1 | + RAG (BM25 top 5) | Qwen3-1.7B | 0.1391 | 0.3457 |
| 2 | After: + RAG + LoRA | Qwen3-1.7B | **0.6344** | **0.6983** |

BM25 recall@5: **0.7844**.

- **#:** condition number (0 = BM25 only, 1 = base Qwen3 picks among BM25's top 5, 2 = LoRA-tuned Qwen3 picks among the same 5).
- **Condition:** what produces the answer.
- **Model:** the Qwen3 model that picks the article (– for BM25 alone).
- **acc@1:** share of the 640 test questions where the picked article is one of the gold articles. 1.0 = always right.
- **MRR:** mean reciprocal rank of the first gold article (BM25's order for condition 0, the logits of `1`–`5` for conditions 1 and 2). 1.0 = gold always first, 0.5 = second on average.
- **BM25 recall@5:** share of questions with a gold article in BM25's top 5. Conditions 1 and 2 pick from those 5, so this is their maximum acc@1.

## Quantization: BF16 vs W4A16

### Quality and size

| Model | Precision | acc@1 | MRR | Answer ppl | Top-1 agreement | KL | Size (MB) |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | BF16 | 0.6375 | 0.7008 | 1.0045 | 100% | 0 | 1136.9 |
| Qwen3-0.6B | W4A16 | 0.6156 | 0.6874 | 1.0058 | 99.89% | 0.00157 | 686.7 |
| Qwen3-1.7B | BF16 | 0.6344 | 0.6981 | 1.0046 | 100% | 0 | 3281.7 |
| Qwen3-1.7B | W4A16 | 0.6031 | 0.6795 | 1.0073 | 99.84% | 0.00297 | 1604.2 |

- **Model:** the merged LoRA model (LoRA adapter folded into the base weights).
- **Precision:** BF16 = merged model unquantized; W4A16 = 4-bit weights, 16-bit activations, run on the GPU with AIMET simulation (same encodings as the NPU).
- **acc@1, MRR:** as in the accuracy table, on the 640 test questions.
- **Answer ppl:** perplexity of the gold answer tokens (article number + quote), teacher-forced. 1.0 = the model is certain of every token.
- **Top-1 agreement:** share of answer positions where W4A16 predicts the same next token as BF16.
- **KL:** mean KL divergence from the BF16 to the W4A16 next-token distribution. 0 = identical.
- **Size (MB):** BF16 = parameter bytes; W4A16 = compiled AI Hub asset of one export bundle.

### Speed on devices (AI Hub Workbench)

| Model | Device | Ctx | Decode step (ms) | Tok/s | 128-token call (ms) | TTFT @ 2048 (s) | Peak mem, decode (MB) |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | Galaxy S26 (8 Elite Gen 5) | 2048 | 12.72 | 78.6 | 19.55 | 0.313 | 132.9 |
| Qwen3-0.6B | Galaxy S25 (8 Elite) | 2048 | 14.99 | 66.7 | 25.02 | 0.400 | 124.0 |
| Qwen3-0.6B | Galaxy S25 (8 Elite) | 1024 | 10.31 | 97.0 | 18.53 | 0.296 | 53.1 |
| Qwen3-0.6B | X Elite CRD | 2048 | 24.67 | 40.5 | 42.68 | 0.683 | 111.8 |
| Qwen3-0.6B | X Plus 8-Core CRD | 2048 | 23.47 | 42.6 | 42.13 | 0.674 | 111.8 |
| Qwen3-1.7B | Galaxy S26 (8 Elite Gen 5) | 2048 | 20.71 | 48.3 | 30.15 | 0.482 | 149.1 |
| Qwen3-1.7B | Galaxy S25 (8 Elite) | 2048 | 22.59 | 44.3 | 36.25 | 0.580 | 150.6 |
| Qwen3-1.7B | Galaxy S25 (8 Elite) | 1024 | 18.47 | 54.1 | 28.43 | 0.455 | 48.7 |
| Qwen3-1.7B | X Elite CRD | 2048 | 32.88 | 30.4 | 61.68 | 0.987 | 112.3 |
| Qwen3-1.7B | X Plus 8-Core CRD | 2048 | 32.00 | 31.3 | 58.21 | 0.931 | 112.4 |

- **Model, Device:** the W4A16 model and the AI Hub device it was profiled on. All graphs ran 100% on the NPU.
- **Ctx:** context length (KV cache size) the graphs were exported with.
- **Decode step (ms):** latency of one token-generator (AR-1) call, summed over the model's parts (2 for 0.6B, 4 for 1.7B).
- **Tok/s:** decode speed, `1000 / decode step`.
- **128-token call (ms):** latency of one prompt-processor (AR-128) call, summed over parts.
- **TTFT @ 2048 (s):** time to first token for a full 2048-token prompt, `16 × 128-token call`.
- **Peak mem, decode (MB):** peak inference memory reported by the AI Hub profile of the decode graph, summed over parts. Mostly the 8-bit KV cache (~112 MiB at 2048); weights are not included.

## Speculative decoding (Qwen3-0.6B draft, Qwen3-1.7B target)

### Acceptance

| Precision | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | α |
|---|---|---|---|---|---|---|---|
| W4A16 (deployed) | 1.990 | 2.957 | 3.958 | 4.866 | 5.744 | 6.640 | 0.9975 |
| BF16 | 1.991 | 2.960 | 3.964 | 4.875 | 5.756 | 6.657 | 0.9980 |

- **Precision:** numerics of both models (W4A16 simulation = what runs on the device).
- **k=1 … k=6:** mean tokens produced per draft-and-verify cycle when the draft proposes k tokens, over the 640 test prompts with greedy decoding. Maximum is k+1 (all k drafts accepted plus one target token). Mean answer length: 124.8 tokens (of 128 max).
- **α:** per-token acceptance rate, the share of positions where the draft's token equals the target's.

### Draft vs target cost

| Device | Target step (ms) | Draft step (ms) | c | AR-128 verify / target step | AR-4 verify / target step | Break-even α (AR-128) | Prior c | Prior break-even α |
|---|---|---|---|---|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 20.71 | 12.72 | **0.61** | 1.46 | – | 0.873 | 0.48 | 0.837 |
| Galaxy S25 (8 Elite) | 22.59 | 14.99 | **0.66** | 1.60 | – | 0.920 | 0.51 | 0.870 |
| X Elite CRD | 32.88 | 24.67 | **0.75** | 1.88 | – | 0.969 | 0.54 | 0.891 |
| X Plus 8-Core CRD | 32.00 | 23.47 | **0.73** | 1.82 | **1.00** | 0.961 | – | – |

- **Target step, Draft step (ms):** decode step of Qwen3-1.7B and Qwen3-0.6B at 2048 (from the speed table).
- **c:** `draft step / target step`, how expensive a drafted token is compared to a target token. Lower = more room for speedup.
- **AR-128 / AR-4 verify / target step:** cost of one verify call in target decode steps. 1.0 = verifying is as cheap as decoding one token. AR-4 at 2048 exists only on X Plus.
- **Break-even α (AR-128):** lowest acceptance rate at which speculative decoding with the AR-128 verify is faster than the target alone.
- **Prior c, Prior break-even α:** the same from Qualcomm's published numbers at context 1024 (`results/prior.csv`; none for X Plus).

### Decode speed

| Device | Before (1.7B alone) | After, verify (a) AR-128 | After, verify (b) AR-4 | Speedup | Best k | Break-even α (a) |
|---|---|---|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 39.9 | 51.5 (k=6) | – | **1.29×** | 6 | 0.873 |
| Galaxy S25 (8 Elite) | 36.0 | 42.8 (k=6) | – | **1.19×** | 6 | 0.920 |
| X Elite CRD | 31.5 | 32.8 (k=6) | – | **1.04×** | 6 | 0.969 |
| X Plus 8-Core CRD | 31.3 | 33.4 (k=6) | **38.6 (k=3)** | **1.24×** | 3 | 0.961 |

- **Before (1.7B alone):** target decode tok/s. Latencies are scaled per device so this matches the published number (×1.230 S25, ×1.210 S26, ×0.966 X Elite; X Plus raw). The before/after ratio does not depend on the scale.
- **After, verify (a) AR-128:** projected tok/s, `tokens per cycle / (k × T_draft + T_verify)`, with the target's 128-token graph as verify, at the best k.
- **After, verify (b) AR-4:** same with a 4-token target graph (k ≤ 3). – = no AR-4 graph at 2048 on that device.
- **Speedup:** best "after" / "before".
- **Best k:** draft length of the best option.
- **Break-even α (a):** as in the draft vs target cost table.

### Speed per k

| k | Galaxy S26 | Galaxy S25 | X Elite | X Plus, AR-128 | X Plus, AR-4 |
|---|---|---|---|---|---|
| 1 | 38.4 (0.96×) | 31.6 (0.88×) | 23.9 (0.76×) | 24.4 (0.78×) | 35.8 (1.15×) |
| 2 | 44.0 (1.10×) | 36.3 (1.01×) | 27.6 (0.88×) | 28.1 (0.90×) | 37.4 (1.20×) |
| 3 | 47.9 (1.20×) | 39.6 (1.10×) | 30.2 (0.96×) | 30.8 (0.98×) | **38.6 (1.24×)** |
| 4 | 49.6 (1.24×) | 41.1 (1.14×) | 31.4 (1.00×) | 32.0 (1.02×) | – |
| 5 | 50.6 (1.27×) | 42.0 (1.17×) | 32.2 (1.02×) | 32.7 (1.05×) | – |
| 6 | **51.5 (1.29×)** | **42.8 (1.19×)** | **32.8 (1.04×)** | **33.4 (1.07×)** | – |

- **k:** number of tokens the draft guesses per cycle.
- **Device columns:** projected tok/s and, in brackets, speedup vs the target alone. Verify with the target's 128-token graph (AR-128), or its 4-token graph (AR-4, k ≤ 3, X Plus only).
- **Calibrated draft step + verify call:** 15.39 + 36.49 ms (S26), 18.44 + 44.58 ms (S25), 23.82 + 59.55 ms (X Elite), 23.47 + 58.21 or 32.09 ms (X Plus).

### Time per question

| Device | BM25 (ms, host) | TTFT before (s) | TTFT after (s) | Total before (s) | Total after (s) | Change (− = faster) |
|---|---|---|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 1.33 | 0.368 | 0.606 | 3.497 | 3.029 | **−13.4%** |
| Galaxy S25 (8 Elite) | 1.33 | 0.449 | 0.760 | 3.918 | 3.678 | **−6.1%** |
| X Elite CRD | 1.33 | 0.600 | 1.016 | 4.564 | 4.823 | **+5.7%** |
| X Plus 8-Core CRD | 1.33 | 0.587 | 1.012 | 4.582 | 4.245 | **−7.4%** |

- **BM25 (ms, host):** mean time to retrieve the top 5 articles, on the host CPU as a proxy for the device CPU.
- **TTFT before / after (s):** time to first token for the ~1.4k-token RAG prompt, `10.1 × 128-token call`. "After" adds the draft's own prompt pass: `10.1 × (T_PP_target + T_PP_draft)`.
- **Total before / after (s):** `BM25 + TTFT + 124.8 tokens / tok/s`, using the best verify option (AR-4 on X Plus, AR-128 elsewhere).
- **Change:** `(total after − total before) / total before`. Negative = faster.

### Prompt cost of the draft

| Device | Target 128-token call (ms) | Draft 128-token call (ms) | Draft / target |
|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 30.15 | 19.55 | 0.65 |
| Galaxy S25 (8 Elite) | 36.25 | 25.02 | 0.69 |
| X Elite CRD | 61.68 | 42.68 | 0.69 |
| X Plus 8-Core CRD | 58.21 | 42.13 | 0.72 |

- **Target / Draft 128-token call (ms):** raw latency of one AR-128 prompt call of each model (from the speed table).
- **Draft / target:** how much the draft's prompt pass adds to TTFT, relative to the target's.

### Decode gain vs TTFT cost

| Device | Decode before → after (s) | TTFT before → after (s) | Net per question (s) | Saved per answer token (ms) | Pays off for answers longer than |
|---|---|---|---|---|---|
| Galaxy S26 (8 Elite Gen 5) | 3.13 → 2.42 (**−0.71**) | 0.37 → 0.61 (**+0.24**) | **−0.47** | 5.7 | ~42 tokens |
| Galaxy S25 (8 Elite) | 3.47 → 2.92 (**−0.55**) | 0.45 → 0.76 (**+0.31**) | **−0.24** | 4.4 | ~71 tokens |
| X Elite CRD | 3.96 → 3.81 (**−0.16**) | 0.60 → 1.02 (**+0.42**) | **+0.26** | 1.2 | ~333 tokens |
| X Plus, AR-128 | 3.99 → 3.74 (**−0.25**) | 0.59 → 1.01 (**+0.43**) | **+0.17** | 2.0 | ~210 tokens |
| X Plus, AR-4 | 3.99 → 3.23 (**−0.76**) | 0.59 → 1.01 (**+0.43**) | **−0.34** | 6.1 | ~70 tokens |

- **Decode before → after (s):** time to write the 124.8-token answer, `124.8 / tok/s`.
- **TTFT before → after (s):** time to read the prompt, from the time-per-question table.
- **Net per question (s):** decode change + TTFT change. Negative = faster.
- **Saved per answer token (ms):** `1000 / tok/s before − 1000 / tok/s after`.
- **Pays off for answers longer than:** `extra TTFT / saved per token`, the answer length at which the decode gain equals the extra TTFT.
