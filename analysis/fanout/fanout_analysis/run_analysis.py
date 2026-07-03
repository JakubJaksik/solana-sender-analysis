"""Orchestrator: load once -> run all section modules in order -> build report.

Usage:
  python -m fanout_analysis.run_analysis [--run-dir ...] [--only S2,S7] [...]
"""
import argparse
import importlib
import json
from pathlib import Path

from fanout_analysis import constants, loader

# (section id, dotted module). errors.py implements the ERR section.
SECTION_MODULES = [
    ("S0", "fanout_analysis.s0_integrity"),
    ("S1", "fanout_analysis.s1_outcomes"),
    ("S2", "fanout_analysis.s2_winrate"),
    ("S3", "fanout_analysis.s3_pairwise"),
    ("S4", "fanout_analysis.s4_latency"),
    ("S5", "fanout_analysis.s5_latency_shift"),
    ("S6", "fanout_analysis.s6_intraslot"),
    ("SC", "fanout_analysis.s_slotchange"),
    ("S7", "fanout_analysis.s7_geo"),
    ("S8", "fanout_analysis.s8_per_validator"),
    ("S9", "fanout_analysis.s9_pwin_model"),
    ("S10", "fanout_analysis.s10_cost"),
    ("S11", "fanout_analysis.s11_synthesis"),
    ("ERR", "fanout_analysis.errors"),
]


def _run_file(r):
    p = Path(r)
    return p / "triggers.jsonl" if p.is_dir() else p


def build_ctx(args):
    dropped = []
    if args.runs:                      # multi-run pooled analysis (same epoch)
        run_paths = [_run_file(r) for r in args.runs]
        run_ids = [p.parent.name for p in run_paths]
        outdir = Path(args.out) if args.out else (constants.ANALYSIS_OUT / ("combined-" + "_".join(run_ids)))
        (outdir / "plots").mkdir(parents=True, exist_ok=True)
        (outdir / "summary").mkdir(parents=True, exist_ok=True)
        dropped = []
        pq = outdir / "enriched.parquet"
        if pq.exists():                       # reuse cached pooled frame -> --only is fast
            import pandas as pd
            df = pd.read_parquet(pq)
            print(f"  (loaded cached {pq.name}; delete it to force re-pool)", flush=True)
        else:
            df = loader.load_enriched_multi(run_paths, args.vepoch, args.svcsv)
            df.to_parquet(pq)
            dropped = df.attrs.get("dropped_senders", [])
            inc = df.attrs.get("dropped_incomplete_per_run", {})
            if any(inc.values()):
                print(f"  NOTE: dropped incomplete (ragged/tail) triggers per run: "
                      f"{ {k: v for k, v in inc.items() if v} }", flush=True)
            ml = df.attrs.get("dropped_multiland_per_run", {})
            if any(ml.values()):
                print(f"  NOTE: dropped multi-land (durable-nonce anomaly) triggers per run: "
                      f"{ {k: v for k, v in ml.items() if v} }", flush=True)
        run_id = "combined(" + ",".join(run_ids) + ")"
    else:                              # single run
        run_path = _run_file(args.run_dir)
        run_id = run_path.parent.name
        outdir = Path(args.out) if args.out else (constants.ANALYSIS_OUT / run_id)
        (outdir / "plots").mkdir(parents=True, exist_ok=True)
        (outdir / "summary").mkdir(parents=True, exist_ok=True)
        df = loader.load_cached(run_path, args.vepoch, args.svcsv, outdir)

    wide = df.pivot(index="trigger_id", columns="sender_name", values="land").fillna(0).astype(int)
    config = json.load(open(args.config))
    ctx = {"df": df, "wide": wide, "outdir": outdir, "config": config,
           "min_n": args.min_n, "min_indicative": args.min_indicative,
           "stake_weighted": args.stake_weighted}
    manifest = {"run_id": run_id, "epoch": args.epoch, "n_rows": len(df),
                "n_triggers": int(df["trigger_id"].nunique()),
                "n_senders": int(df["sender_name"].nunique()),
                "n_leaders": int(df["leader_identity"].nunique()),
                "n_runs_pooled": len(args.runs) if args.runs else 1,
                "dropped_senders": dropped}
    if dropped:
        print(f"  NOTE: dropped senders not common to all runs: {dropped}", flush=True)
    return ctx, manifest, outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(constants.DEFAULT_RUN.parent))
    ap.add_argument("--runs", nargs="+", default=None,
                    help="two or more run dirs/files to POOL into one analysis (same epoch + sender set)")
    ap.add_argument("--epoch", type=int, default=constants.EPOCH)
    ap.add_argument("--vepoch", default=str(constants.DEFAULT_VEPOCH))
    ap.add_argument("--svcsv", default=str(constants.DEFAULT_SVCSV))
    ap.add_argument("--config", default=str(constants.DEFAULT_CONFIG))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma list of section ids, e.g. S2,S7")
    ap.add_argument("--min-n", type=int, default=constants.GATE_INFERENTIAL)
    ap.add_argument("--min-indicative", type=int, default=constants.GATE_INDICATIVE)
    ap.add_argument("--stake-weighted", action="store_true")
    args = ap.parse_args()

    ctx, manifest, outdir = build_ctx(args)
    only = set(args.only.split(",")) if args.only else None
    todo = [(sid, mod) for sid, mod in SECTION_MODULES if (only is None or sid in only)]

    results = []
    for k, (sid, mod_name) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {sid} {mod_name} ...", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            res = mod.run(ctx)
            results.append(res)
            print(f"      done: {list((res or {}).get('key_results', {}).items())[:4]}", flush=True)
        except Exception as e:
            print(f"      ERROR in {sid}: {e}", flush=True)
            results.append({"id": sid, "title": mod_name, "key_results": {"ERROR": str(e)},
                            "tables": {}, "figures": {}, "notes": [f"section failed: {e}"]})

    if only is None:
        from fanout_analysis import report
        md, htmlp = report.build_report(results, outdir, manifest)
        print(f"\nReport: {md}\n        {htmlp}\nOutput dir: {outdir}")
    else:
        # --only is a non-destructive refresh of selected sections; do NOT rebuild
        # report.md (that would drop the other sections). Charts/CSVs are written in place.
        print(f"\n[--only] refreshed {sorted(only)}; report.md left untouched. Output dir: {outdir}")


if __name__ == "__main__":
    main()
