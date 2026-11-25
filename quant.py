import argparse, importlib, math, os, shutil, subprocess, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import (LORA, MODELS, RESULTS, choice_ids, encode, has_gold, hit_rr, load_model, name,
                      rank_by_logits, read_csv, split_rows, summary, write_csv)

RECIPES = {"Qwen/Qwen3-0.6B": "qwen3_0_6b", "Qwen/Qwen3-1.7B": "qwen3_1_7b"}
CONTEXT = 2048
SEQ_LENS = [1, 128]
MERGED = "build/merged"
W4A16 = "build/w4a16"
QUALITY = f"{RESULTS}/quality.csv"


def merge(base, lora_dir, out_dir, dtype="bfloat16"):
    # base model id, LoRA adapter folder -> plain Qwen3 checkpoint with the adapter folded in, saved to out_dir
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, lora_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(lora_dir).save_pretrained(out_dir)
    print(f"saved {out_dir}")


def quantize(model_id):
    # merged checkpoint -> W4A16 AIMET checkpoint (model*.onnx + model.encodings), with config and tokenizer copied
    src, out = f"{MERGED}/{name(model_id)}", f"{W4A16}/{name(model_id)}"
    subprocess.run([sys.executable, "-m", f"qai_hub_models.models.{RECIPES[model_id]}.quantize",
                    "--checkpoint", src, "--output-dir", out, "--precision", "w4a16"], check=True)
    for f in os.listdir(src):
        if not f.startswith("model"):
            shutil.copy(os.path.join(src, f), out)


def load_sim(model_id, device):
    # model id, torch device -> HF-style generator over the W4A16 host simulation (AR-128 prefill, AR-1 decode)
    from qai_hub_models.models.templates.llm.generator_factory import make_generator
    recipe = importlib.import_module(f"qai_hub_models.models.{RECIPES[model_id]}")
    model = recipe.QuantizedSplitModelWrapper.from_pretrained(
        checkpoint=f"{W4A16}/{name(model_id)}", host_device=device)
    model = model.to(device).eval()
    return make_generator(model, sequence_length=SEQ_LENS, context_length=CONTEXT, device=device)


def second_gpu():
    # -> cuda:1 if there is one, else cuda:0
    import torch
    return torch.device(f"cuda:{min(1, torch.cuda.device_count() - 1)}")


def evaluate(model_id):
    bf16, tok = load_model(f"{MERGED}/{name(model_id)}", "cuda:0")
    size_mb = sum(p.numel() * p.element_size() for p in bf16.parameters()) / 2**20
    sim = load_sim(model_id, second_gpu())
    cids = choice_ids(tok)
    hits = {"bf16": [], "w4a16": []}
    nll = {"bf16": 0.0, "w4a16": 0.0}
    agree = kl = n_tok = 0
    with torch.no_grad():
        for row in tqdm(split_rows("test"), desc="eval"):
            ids, n = encode(tok, row, has_gold(row))
            x = torch.tensor([ids])
            logits = {"bf16": bf16(x.to(bf16.device)).logits[0].float(),
                      "w4a16": sim(x.to(sim.device)).logits[0].float().to(bf16.device)}
            for p, lg in logits.items():
                hits[p].append(hit_rr(rank_by_logits(row, lg[n - 1, cids].tolist()), row["gold"]))
            if len(ids) == n:
                continue
            # positions n-1 .. end-1 predict the gold answer tokens ids[n:]
            y = torch.tensor(ids[n:], device=bf16.device)
            lp = {p: F.log_softmax(lg[n - 1:-1], -1) for p, lg in logits.items()}
            for p in lp:
                nll[p] -= lp[p].gather(1, y[:, None]).sum().item()
            agree += (lp["bf16"].argmax(-1) == lp["w4a16"].argmax(-1)).sum().item()
            kl += (lp["bf16"].exp() * (lp["bf16"] - lp["w4a16"])).sum().item()
            n_tok += len(y)

    rows = [{"model": name(model_id), "precision": "bf16", **summary(hits["bf16"]),
             "answer_ppl": round(math.exp(nll["bf16"] / n_tok), 4), "top1_agree": 1.0, "kl": 0.0,
             "weights_mb": round(size_mb, 1)},
            {"model": name(model_id), "precision": "w4a16", **summary(hits["w4a16"]),
             "answer_ppl": round(math.exp(nll["w4a16"] / n_tok), 4), "top1_agree": round(agree / n_tok, 4),
             "kl": round(kl / n_tok, 6), "weights_mb": ""}]
    for r in rows:
        print(r)
    print("W4A16 size: asset_mb in results/latency.csv")
    train = [r for r in read_csv(f"{RESULTS}/train.csv") if r["cond"] == "2" and r["model"] == name(model_id)]
    if train:
        print(f"merge check: BF16 acc@1 {rows[0]['acc1']} vs train condition 2 {train[0]['acc1']}")
    old = [r for r in read_csv(QUALITY) if r["model"] != name(model_id)]
    write_csv(QUALITY, old + rows)


def main():
    ap = argparse.ArgumentParser(description="Merge, quantize, BF16 vs W4A16 quality")
    ap.add_argument("cmd", choices=["merge", "quantize", "eval"])
    ap.add_argument("--model", required=True, choices=MODELS)
    args = ap.parse_args()
    if args.cmd == "merge":
        merge(args.model, f"{LORA}/{name(args.model)}", f"{MERGED}/{name(args.model)}")
    elif args.cmd == "quantize":
        quantize(args.model)
    else:
        evaluate(args.model)


if __name__ == "__main__":
    main()
