import argparse, csv, json, os, random, re, statistics, time

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
DATASET = "another-symato/VMTEB-Zalo-legel-retrieval"
TOP_K = 5
ARTICLE_TOKENS = 256
QUOTE_TOKENS = 128
DEV_SIZE = 200
DATA = "data"
LORA = "build/lora"
RESULTS = "results"
END = "<|im_end|>"
CHOICES = [str(i + 1) for i in range(TOP_K)]
INSTRUCTION = ("Điều luật nào trả lời câu hỏi? Trả lời bằng số thứ tự của điều luật, "
               "sau đó trích dẫn điều luật đó.")


def name(model):
    # "Qwen/Qwen3-1.7B" -> "Qwen3-1.7B"
    return model.split("/")[-1]


def words(text):
    # text -> lowercase syllables plus syllable pairs, for BM25
    s = re.findall(r"\w+", text.lower())
    return s + [a + "_" + b for a, b in zip(s, s[1:])]


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def split_rows(split):
    # "train" / "test" -> rows written by `data`
    rows = load_jsonl(f"{DATA}/legal_{split}.jsonl")
    if not rows:
        raise SystemExit(f"missing {DATA}/legal_{split}.jsonl: run `python train.py data`")
    return rows


def prompt(tok, row):
    # row -> chat prompt text, ending where the answer starts (thinking off)
    arts = "\n\n".join(f"[{i + 1}] {a}" for i, a in enumerate(row["articles"]))
    msg = f"Câu hỏi: {row['question']}\n\nCác điều luật:\n{arts}\n\n{INSTRUCTION}"
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def has_gold(row):
    return any(c in row["gold"] for c in row["cands"])


def answer(tok, row):
    # row with gold in its candidates -> "<gold number>\n<first QUOTE_TOKENS tokens of that article>"
    i = next(i for i, c in enumerate(row["cands"]) if c in row["gold"])
    quote = tok.decode(tok(row["articles"][i], add_special_tokens=False)["input_ids"][:QUOTE_TOKENS])
    return f"{i + 1}\n{quote}"


def encode(tok, row, with_answer=False):
    # row -> (token ids of prompt [+ answer + end token], prompt length)
    p = tok(prompt(tok, row), add_special_tokens=False)["input_ids"]
    a = tok(answer(tok, row) + END, add_special_tokens=False)["input_ids"] if with_answer else []
    return p + a, len(p)


def choice_ids(tok):
    # tokenizer -> token ids of "1".."5"
    ids = [tok(c, add_special_tokens=False)["input_ids"] for c in CHOICES]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def hit_rr(ranked, gold):
    # ranked article ids, gold ids -> 1.0 if the top one is gold, reciprocal rank of the first gold
    r = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
    return float(r == 1), (1 / r if r else 0.0)


def rank_by_logits(row, logits):
    # row, logits of "1".."5" -> candidate ids, best first
    order = sorted(range(TOP_K), key=lambda i: -logits[i])
    return [row["cands"][i] for i in order]


def summary(hits):
    # list of (hit, rr) -> {"acc1", "mrr"}
    return {"acc1": round(statistics.mean(h for h, _ in hits), 4),
            "mrr": round(statistics.mean(r for _, r in hits), 4)}


def data():
    import bm25s
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[0])
    corpus = load_dataset(DATASET, "corpus", split="train")
    ids, titles, texts = corpus["id"], corpus["title"], corpus["text"]
    index = {c: i for i, c in enumerate(ids)}
    queries = {r["query_id"]: r["question"] for r in load_dataset(DATASET, "queries", split="train")}
    print(f"indexing {len(ids)} articles")
    bm25 = bm25s.BM25()
    bm25.index([words(f"{t}\n{x}") for t, x in zip(titles, texts)], show_progress=False)

    cut = {}

    def article(cid):
        # article id -> title\ntext cut to ARTICLE_TOKENS tokens
        if cid not in cut:
            i = index[cid]
            toks = tok(f"{titles[i]}\n{texts[i]}", add_special_tokens=False)["input_ids"]
            cut[cid] = tok.decode(toks[:ARTICLE_TOKENS])
        return cut[cid]

    for split in ["train", "test"]:
        gold = {}
        for r in load_dataset(DATASET, "data_ir", split=split):
            gold.setdefault(r["query_id"], []).append(r["corpus_id"])
        rows = []
        for qid, g in gold.items():
            t0 = time.perf_counter()
            top, _ = bm25.retrieve([words(queries[qid])], k=TOP_K, show_progress=False)
            ms = (time.perf_counter() - t0) * 1000
            ranked = [ids[i] for i in top[0]]
            cands = ranked[:]
            random.Random(qid).shuffle(cands)
            row = {"id": qid, "question": queries[qid], "gold": g, "bm25": ranked, "cands": cands,
                   "articles": [article(c) for c in cands], "bm25_ms": round(ms, 3)}
            row["prompt_tokens"] = encode(tok, row)[1]
            rows.append(row)
        write_jsonl(f"{DATA}/legal_{split}.jsonl", rows)
        print(f"{split}: {len(rows)} questions, BM25 recall@{TOP_K} {statistics.mean(map(has_gold, rows)):.3f}, "
              f"{statistics.mean(r['bm25_ms'] for r in rows):.1f} ms, "
              f"prompt {statistics.mean(r['prompt_tokens'] for r in rows):.0f} tokens")


def load_model(path, device="cuda"):
    # HF model id or folder -> BF16 model in eval mode, tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return model, AutoTokenizer.from_pretrained(path)


def llm_hits(model, tok, rows):
    # model, rows -> (hit, rr) per row from the logits of "1".."5" at the first answer token
    import torch
    from tqdm import tqdm
    cids = choice_ids(tok)
    out = []
    with torch.no_grad():
        for row in tqdm(rows, desc="bench"):
            x = torch.tensor([encode(tok, row)[0]], device=model.device)
            logits = model(x, logits_to_keep=1).logits[0, -1, cids].float().tolist()
            out.append(hit_rr(rank_by_logits(row, logits), row["gold"]))
    return out


def train(model_id):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    model, tok = load_model(model_id, "cpu")
    rows = [r for r in split_rows("train") if has_gold(r)]
    random.Random(0).shuffle(rows)
    dev, rows = rows[:DEV_SIZE], rows[DEV_SIZE:]

    def enc(row):
        ids, n = encode(tok, row, True)
        return {"input_ids": ids, "labels": [-100] * n + ids[n:]}

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
        ids = [b["input_ids"] for b in batch]
        return {"input_ids": torch.tensor(pad(ids, tok.pad_token_id)),
                "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
                "attention_mask": torch.tensor(pad([[1] * len(x) for x in ids], 0))}

    train_set, dev_set = [enc(r) for r in rows], [enc(r) for r in dev]
    got = tok.decode([t for t in train_set[0]["labels"] if t != -100])
    assert got == answer(tok, rows[0]) + END, f"label mask wrong: {got[:80]!r}"
    print(f"train {len(train_set)}, dev {len(dev_set)}")

    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    out = f"{LORA}/{name(model_id)}"
    args = TrainingArguments(
        output_dir=f"{out}/ckpt", num_train_epochs=2, learning_rate=2e-4,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=8,
        lr_scheduler_type="cosine", warmup_steps=20, bf16=True, logging_steps=10,
        eval_strategy="epoch", save_strategy="no", report_to="none", remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=dev_set,
                      data_collator=collate)
    before = trainer.evaluate()["eval_loss"]
    trainer.train()
    after = trainer.evaluate()["eval_loss"]
    print(f"dev loss {before:.4f} -> {after:.4f}")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}")


def bench():
    from peft import PeftModel

    test = split_rows("test")
    recall = round(statistics.mean(map(has_gold, test)), 4)
    rows = [{"cond": 0, "condition": "bm25", "model": "-",
             **summary([hit_rr(r["bm25"], r["gold"]) for r in test])}]
    for m in MODELS:
        model, tok = load_model(m)
        rows.append({"cond": 1, "condition": "rag", "model": name(m), **summary(llm_hits(model, tok, test))})
        model = PeftModel.from_pretrained(model, f"{LORA}/{name(m)}").eval()
        rows.append({"cond": 2, "condition": "rag+lora", "model": name(m), **summary(llm_hits(model, tok, test))})
        del model
    for r in rows:
        r["recall5"] = recall
        print(r)
    for m in MODELS:
        a = {r["cond"]: r["acc1"] for r in rows if r["model"] in ("-", name(m))}
        print(f"{name(m)}: RAG {a[1] - a[0]:+.4f}, LoRA {a[2] - a[1]:+.4f} acc@1 (ceiling {recall:.4f})")
    write_csv(f"{RESULTS}/train.csv", rows)


def main():
    ap = argparse.ArgumentParser(description="BM25, RAG prompt, LoRA, benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="BM25 top 5 and BM25 time per question")
    sub.add_parser("train", help="LoRA").add_argument("--model", required=True, choices=MODELS)
    sub.add_parser("bench", help="conditions 0-2")
    args = ap.parse_args()
    if args.cmd == "data":
        data()
    elif args.cmd == "train":
        train(args.model)
    else:
        bench()


if __name__ == "__main__":
    main()
