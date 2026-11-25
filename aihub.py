import argparse, math, os, re, subprocess

from train import RESULTS, name, read_csv, write_csv
from quant import CONTEXT, QUALITY, RECIPES

RUNTIME = "geniex_qairt"
SEQ_LENS = "128,1"
PP = 128
EXPORT = "build/export"
LATENCY = f"{RESULTS}/latency.csv"
# Qualcomm's published Qwen3-1.7B tok/s at context 2048 (GenieX, w4a16), README Section 6
PUBLISHED = {"Samsung Galaxy S26 (Family)": 39.9, "Samsung Galaxy S25 (Family)": 36.0,
             "Snapdragon X Elite CRD": 31.5}
# profile job name: <recipe>_..._<prompt|token>_ar<seq>_cl<ctx>_<part>_of_<parts>
JOB_NAME = re.compile(r"^(qwen3_\d+_\d+b)_.*_(prompt|token)_ar(\d+)_cl(\d+)_(\d+)_of_(\d+)$")


def slug(device):
    # "Samsung Galaxy S25 (Family)" -> "samsung_galaxy_s25_family"
    return re.sub(r"\W+", "_", device).strip("_").lower()


def export(recipe, checkpoint, device, extra):
    # W4A16 checkpoint -> compile, link and profile jobs on AI Hub, bundle in build/export/<recipe>/<device>
    out = f"{EXPORT}/{recipe}/{slug(device)}"
    cmd = ["qai-hub-models", "export", recipe, "--checkpoint", checkpoint, "--device", device,
           "--target-runtime", RUNTIME, "--context-lengths", str(CONTEXT), "--sequence-lengths", SEQ_LENS,
           "--output-dir", out, *extra]
    print(" ".join(cmd))
    # export stages GBs of weights in TMPDIR; WSL's /tmp is a small RAM disk, so use build/tmp on disk
    tmp = os.path.abspath("build/tmp")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(cmd, check=True, env={**os.environ, "TMPDIR": tmp})


def asset_mb(recipe, device):
    # exported bundle -> size of its compiled context binaries (MB), or "" if not downloaded
    files = [os.path.join(d, f) for d, _, fs in os.walk(f"{EXPORT}/{recipe}/{slug(device)}") for f in fs]
    size = sum(os.path.getsize(f) for f in files if f.endswith(".bin"))
    return round(size / 2**20, 1) if size else ""


def profile_jobs():
    # -> {(recipe, device, kind, seq_len, context, parts): {part: newest successful profile job id}}
    import qai_hub as hub
    groups, offset = {}, 0
    while True:
        jobs = hub.get_job_summaries(offset=offset, limit=50, type=hub.JobType.PROFILE,
                                     state=hub.JobStatus.State.SUCCESS)
        for j in jobs:
            m = JOB_NAME.match(j.name)
            if m:
                recipe, kind, ar, cl, part, parts = m.groups()
                key = (recipe, j.device_name, kind, int(ar), int(cl), int(parts))
                groups.setdefault(key, {}).setdefault(int(part), j.job_id)
        if len(jobs) < 50:
            return groups
        offset += 50


def collect():
    import qai_hub as hub
    from qai_hub_models._version import __version__ as qaihm
    from qai_hub_models.configs.tool_versions import ToolVersions

    bf16_mb = {r["model"]: float(r["weights_mb"]) for r in read_csv(QUALITY) if r["precision"] == "bf16"}
    model_name = {v: name(k) for k, v in RECIPES.items()}
    rows = []
    for (recipe, device, kind, ar, cl, parts), jobs in sorted(profile_jobs().items()):
        if len(jobs) != parts:
            print(f"skip {recipe} {device} {kind}_ar{ar}: {len(jobs)} of {parts} parts profiled")
            continue
        ms = peak = npu = ops = 0
        for job_id in jobs.values():
            job = hub.get_job(job_id)
            prof = job.download_profile()
            s = prof["execution_summary"]
            ms += s["estimated_inference_time"] / 1000
            peak += s["inference_memory_peak_range"][1] / 2**20
            units = [op.get("compute_unit", "").lower() for op in prof["execution_detail"]]
            npu += units.count("npu")
            ops += len(units)
        qairt = ToolVersions.from_job(job).qairt
        row = {"model": model_name.get(recipe, recipe), "device": device, "graph": f"{kind}_ar{ar}",
               "seq_len": ar, "context": cl, "parts": parts, "ms": round(ms, 3), "peak_mb": round(peak, 1),
               "npu_pct": round(100 * npu / ops, 1) if ops else "", "tok_s": "", "published_tok_s": "",
               "error_pct": "", "bf16_tok_s_est": "", "ttft_2048_s": "", "asset_mb": asset_mb(recipe, device),
               "qaihm": qaihm, "qairt": str(qairt or ""), "runtime": RUNTIME, "jobs": " ".join(jobs.values())}
        if kind == "token":
            row["tok_s"] = round(1000 / ms, 2)
            pub = PUBLISHED.get(device) if recipe == "qwen3_1_7b" else None
            if pub:
                row["published_tok_s"] = pub
                row["error_pct"] = round(100 * (row["tok_s"] - pub) / pub, 1)
            if row["asset_mb"] and row["model"] in bf16_mb:
                # bandwidth estimate: tok/s scales with 1 / weight bytes
                row["bf16_tok_s_est"] = round(row["tok_s"] * row["asset_mb"] / bf16_mb[row["model"]], 2)
        elif ar == PP:
            row["ttft_2048_s"] = round(math.ceil(CONTEXT / PP) * ms / 1000, 3)
        print({k: v for k, v in row.items() if k != "jobs"})
        rows.append(row)
    if not rows:
        raise SystemExit("no complete profile job groups found")
    write_csv(LATENCY, rows)


def main():
    ap = argparse.ArgumentParser(description="AI Hub export, profile, latencies")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="export and profile one checkpoint on one device")
    p.add_argument("recipe", choices=list(RECIPES.values()))
    p.add_argument("checkpoint")
    p.add_argument("device")
    sub.add_parser("collect", help="latencies of all profile jobs")
    args, extra = ap.parse_known_args()
    if args.cmd == "export":
        export(args.recipe, args.checkpoint, args.device, extra)
    else:
        if extra:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
        collect()


if __name__ == "__main__":
    main()
