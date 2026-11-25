import argparse, json, math, os, statistics

from aihub import LATENCY, PP
from train import END, RESULTS, encode, load_jsonl, load_model, name, read_csv, split_rows, write_csv
from quant import CONTEXT, MERGED, load_sim, second_gpu

TARGET, DRAFT = "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"
KS = range(1, 7)
MAX_NEW = 128
ALPHAS = [0.7, 0.8, 0.9]
PRIOR_CSV = f"{RESULTS}/prior.csv"
ACCEPT = f"{RESULTS}/acceptance.jsonl"
TPC = f"{RESULTS}/tokens_per_cycle.csv"
SPEEDUP = f"{RESULTS}/speedup.csv"
# device -> (Qwen3-1.7B tok/s, Qwen3-0.6B tok/s, Qwen3-1.7B 128-token call ms)
PRIOR = {"Galaxy S26 (8 Elite Gen 5)": (41.7, 87.8, 41.1),
         "Galaxy S25 (8 Elite)": (41.6, 80.9, 43.3),
         "Snapdragon X Elite CRD": (36.1, 67.4, 52.5)}


def e_tokens(a, k):
    # per-token acceptance a, draft length k -> expected tokens per cycle (Section 8.4)
    return k + 1.0 if a >= 1 else (1 - a ** (k + 1)) / (1 - a)


def speed(e, k, t_draft, t_verify):
    # tokens per cycle, draft length k, draft step and verify cost (ms) -> decode tokens/s
    return 1000 * e / (k * t_draft + t_verify)


def break_even(t_target, t_draft, t_verify):
    # step costs (ms) -> smallest alpha (0.001 steps) where the best k beats the target alone, or ""
    for i in range(1001):
        a = i / 1000
        if max(speed(e_tokens(a, k), k, t_draft, t_verify) for k in KS) >= 1000 / t_target:
            return a
    return ""


def prior_rows():
    # Section 6 table -> c, best-k speedup with an ideal verify graph, break-even alpha with AR-128 verify
    rows = []
    for device, (tps_t, tps_d, verify) in PRIOR.items():
        t, d = 1000 / tps_t, 1000 / tps_d
        row = {"device": device, "target_tok_s": tps_t, "draft_tok_s": tps_d, "c": round(d / t, 3),
               "verify_ms": verify}
        for a in ALPHAS:
            row[f"ideal_speedup_a{a}"] = round(max(speed(e_tokens(a, k), k, d, t) for k in KS) * t / 1000, 3)
        row["break_even_ar128"] = break_even(t, d, verify)
        rows.append(row)
    return rows


def prior():
    rows = prior_rows()
    for r in rows:
        print(r)
    write_csv(PRIOR_CSV, rows)


def greedy(model, prompt_ids, eos):
    # model (HF or simulation generator), prompt ids -> greedy answer ids, up to MAX_NEW, ending with eos if reached
    import torch
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False,
                                               eos_token_id=eos, pad_token_id=eos)
    x = torch.tensor([prompt_ids], device=model.device)
    return model.generate(x, attention_mask=torch.ones_like(x))[0, len(prompt_ids):].tolist()


def draft_match(model, prompt_ids, answer_ids):
    # draft model, prompt ids, target answer ids -> per answer token, 1 if the draft's greedy guess
    # given the target's prefix equals it (one teacher-forced pass)
    import torch
    x = torch.tensor([prompt_ids + answer_ids], device=model.device)
    guess = model(x).logits[0, len(prompt_ids) - 1:-1].argmax(-1).tolist()
    return [int(g == y) for g, y in zip(guess, answer_ids)]


def cycles(match, k):
    # per-token draft match, draft length k -> draft-and-verify cycles needed for all tokens
    i = c = 0
    while i < len(match):
        n = 0
        while n < k and i + n < len(match) and match[i + n]:
            n += 1
        i, c = i + n + 1, c + 1
    return c


def tokens_per_cycle():
    # results/acceptance.jsonl -> mean tokens per cycle per k, alpha and mean answer length per precision
    rows = []
    for prec in ["w4a16", "bf16"]:
        acc = [r for r in load_jsonl(ACCEPT) if r["prec"] == prec]
        if not acc:
            continue
        n = sum(r["n"] for r in acc)
        row = {"prec": prec, "prompts": len(acc), "mean_len": round(n / len(acc), 2),
               "alpha": round(sum(sum(r["match"]) for r in acc) / n, 4)}
        for k in KS:
            row[f"k{k}"] = round(n / sum(cycles(r["match"], k) for r in acc), 4)
        print(row)
        rows.append(row)
    write_csv(TPC, rows)


def accept(bf16, limit):
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    prec = "bf16" if bf16 else "w4a16"
    done = {r["id"] for r in load_jsonl(ACCEPT) if r["prec"] == prec}
    todo = [r for r in split_rows("test")[:limit] if r["id"] not in done]
    if todo:
        if bf16:
            target = load_model(f"{MERGED}/{name(TARGET)}", "cuda:0")[0]
            draft = load_model(f"{MERGED}/{name(DRAFT)}", second_gpu())[0]
        else:
            target, draft = load_sim(TARGET, torch.device("cuda:0")), load_sim(DRAFT, second_gpu())
        tok = AutoTokenizer.from_pretrained(TARGET)
        eos = tok.convert_tokens_to_ids(END)
        os.makedirs(RESULTS, exist_ok=True)
        with open(ACCEPT, "a", encoding="utf-8") as f, torch.no_grad():
            for row in tqdm(todo, desc=f"accept {prec}"):
                p = encode(tok, row)[0]
                y = greedy(target, p, eos)
                f.write(json.dumps({"id": row["id"], "prec": prec, "n": len(y),
                                    "match": draft_match(draft, p, y)}) + "\n")
                f.flush()
    tokens_per_cycle()


def project():
    lat = [r for r in read_csv(LATENCY) if int(r["context"]) == CONTEXT]
    tpc = {r["prec"]: r for r in read_csv(TPC)}.get("w4a16")
    if not lat or not tpc:
        raise SystemExit(f"need {LATENCY} (aihub.py collect) and W4A16 rows in {TPC} (specdec.py accept)")
    test = split_rows("test")
    bm25_ms = statistics.mean(r["bm25_ms"] for r in test)
    pp_calls = statistics.mean(math.ceil(r["prompt_tokens"] / PP) for r in test)
    n = float(tpc["mean_len"])
    e = {k: float(tpc[f"k{k}"]) for k in KS}
    tgt, drf = name(TARGET), name(DRAFT)
    rows = []
    for device in sorted({r["device"] for r in lat}):
        here = [r for r in lat if r["device"] == device]
        ms = {(r["model"], r["graph"]): float(r["ms"]) for r in here}
        need = [(tgt, "token_ar1"), (drf, "token_ar1"), (tgt, f"prompt_ar{PP}"), (drf, f"prompt_ar{PP}")]
        if any(x not in ms for x in need):
            print(f"skip {device}: missing {[x for x in need if x not in ms]}")
            continue
        # calibration: scale every latency so the target's decode step matches the published tok/s
        pub = next(r["published_tok_s"] for r in here if (r["model"], r["graph"]) == need[0])
        scale = 1000 / ms[need[0]] / float(pub) if pub else 1.0
        t, d, pp_t, pp_d = (ms[x] * scale for x in need)
        # verify (b): smallest exported target graph with k+1 <= seq_len < PP
        ars = sorted(int(g[len("prompt_ar"):]) for m, g in ms if m == tgt and g.startswith("prompt_ar"))

        def verify_b(k):
            s = [a for a in ars if k + 1 <= a < PP]
            return ms[(tgt, f"prompt_ar{s[0]}")] * scale if s else None

        before = 1000 / t
        after_a, k_a = max((speed(e[k], k, d, pp_t), k) for k in KS)
        after_b, k_b = max(((speed(e[k], k, d, verify_b(k)), k) for k in KS if verify_b(k)), default=("", ""))
        after, best_k = max((after_a, k_a), (after_b or 0, k_b))
        ttft_before, ttft_after = pp_calls * pp_t / 1000, pp_calls * (pp_t + pp_d) / 1000
        row = {"device": device, "calibration": round(scale, 4), "before_tok_s": round(before, 2),
               "after_a_tok_s": round(after_a, 2), "k_a": k_a,
               "after_b_tok_s": round(after_b, 2) if after_b else "", "k_b": k_b,
               "speedup": round(after / before, 3), "best_k": best_k, "break_even_a": break_even(t, d, pp_t),
               "bm25_ms": round(bm25_ms, 2), "ttft_before_s": round(ttft_before, 3),
               "ttft_after_s": round(ttft_after, 3),
               "total_before_s": round(bm25_ms / 1000 + ttft_before + n / before, 3),
               "total_after_s": round(bm25_ms / 1000 + ttft_after + n / after, 3)}
        print(row)
        rows.append(row)
    if not rows:
        raise SystemExit("no device has all four latencies")
    write_csv(SPEEDUP, rows)


def main():
    ap = argparse.ArgumentParser(description="acceptance, speedup and time-per-question projection")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prior", help="Section 6 estimate")
    p = sub.add_parser("accept", help="step 1, resumable")
    p.add_argument("--bf16", action="store_true", help="BF16 merged models instead of the W4A16 simulation")
    p.add_argument("--limit", type=int, help="first N test questions only")
    sub.add_parser("project", help="step 3")
    args = ap.parse_args()
    if args.cmd == "prior":
        prior()
    elif args.cmd == "accept":
        accept(args.bf16, args.limit)
    else:
        project()


if __name__ == "__main__":
    main()
