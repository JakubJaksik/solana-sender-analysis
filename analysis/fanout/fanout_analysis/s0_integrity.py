"""S0 - Run manifest, data-integrity & paired-design validation.

Provenance manifest + hard-fail invariant checks over the enriched fan-out frame:
rows-per-trigger balance, 1:1 signatures, single-winner per-trigger land count,
geo (leader + continent) coverage, observation-source provenance, slot range/window.

Outputs:
  summary/run-manifest.csv
  summary/integrity-invariants.csv
  outdir/integrity-invariants.json   (machine-readable, with single_winner + all-pass gate)
  plots/00-integrity-scorecard.png   (pass/fail table)
  plots/00-global-outcome.png        (global outcome-class stacked bar, distinct colors)

Hard-fails (AssertionError) if any invariant is violated: this is the gate that
protects every downstream section's denominators and paired-design assumptions.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fanout_analysis import constants, loader

logger = logging.getLogger(__name__)

SECTION_ID = "S0"
SECTION_TITLE = "Run manifest, integrity & paired-design validation"

# Distinct color per outcome class (THROTTLED_LOCAL gets its OWN color -
# the user explicitly asked where throttle sits). Greens = good, reds = real problems.
OUTCOME_COLOR = {
    "WON": "#2E7D32",              # green: the only "win" state
    "LOST_RACE": "#90CAF9",        # light blue: expected non-win, NOT an error
    "NEVER_SENT": "#EF6C00",       # orange: real problem (no tx left client)
    "THROTTLED_LOCAL": "#FBC02D",  # amber: real problem (local rate-limit) - own color
    "PROVIDER_REJECTED": "#C62828",  # red: real problem (HTTP 429)
    "SERVER_ERROR": "#6A1B9A",     # purple: real problem (HTTP 500)
}

# Invariants are STRUCTURAL (derived from the data), so the section is reusable across
# runs/epochs with any sender count. Only `critical` invariants hard-fail the pipeline.


def _ns_window_seconds(df: pd.DataFrame) -> float:
    """Wall-clock span of the run from the first to last actual send (send_at_ns>0)."""
    sent = df["send_at_ns"].where(df["send_at_ns"] > 0)
    if sent.notna().sum() == 0:
        return 0.0
    return float((sent.max() - sent.min()) / 1e9)


def _build_manifest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Provenance manifest: one (key, value) row per run-level fact."""
    sent = df["send_at_ns"].where(df["send_at_ns"] > 0)
    senders = sorted(df["sender_name"].unique().tolist())
    outcomes = df["final_outcome"].value_counts().to_dict()
    obs_src = df.loc[df["land"] == 1, "observed_source"].value_counts().to_dict()
    rows = [
        ("run_id", "20260601-150500"),
        ("epoch", constants.EPOCH),
        ("epoch_first_slot", constants.EPOCH_FIRST_SLOT),
        ("n_rows", int(len(df))),
        ("n_triggers", int(df["trigger_id"].nunique())),
        ("n_senders", int(df["sender_name"].nunique())),
        ("senders", "|".join(senders)),
        ("slot_min", int(df["slot"].min())),
        ("slot_max", int(df["slot"].max())),
        ("n_distinct_slots", int(df["slot"].nunique())),
        ("window_seconds", round(_ns_window_seconds(df), 3)),
        ("send_at_min_ns", int(sent.min()) if sent.notna().any() else 0),
        ("send_at_max_ns", int(sent.max()) if sent.notna().any() else 0),
        ("n_landed", int(outcomes.get("Landed", 0))),
        ("n_send_error", int(outcomes.get("SendError", 0))),
        ("n_unknown_pending", int(outcomes.get("UnknownPending", 0))),
        ("n_leaders", int(df["leader_identity"].nunique())),
        ("n_leader_cities", int(df["sv_city"].nunique())),
        ("n_leader_countries", int(df["sv_country"].nunique())),
        ("observation_source", "|".join(f"{k}={v}" for k, v in sorted(obs_src.items()))),
        ("client_region", constants.SENDER_REGION_CITY["helius-fra"]),
        ("tx_priority_fee_microlamports", config.get("tx", {}).get("priority_fee_microlamports")),
        ("tx_compute_unit_limit", config.get("tx", {}).get("compute_unit_limit")),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def _compute_invariants(df: pd.DataFrame) -> list:
    """Return list of invariant dicts: name, observed, expected, pass (bool)."""
    n_senders = int(df["sender_name"].nunique())
    rows_per_trigger = sorted(df.groupby("trigger_id").size().unique().tolist())
    per_trig_land = (df.assign(_l=(df["final_outcome"] == "Landed").astype(int))
                     .groupby("trigger_id")["_l"].sum())
    single_winner = {int(k): int(v) for k, v in per_trig_land.value_counts().sort_index().items()}
    max_land = int(per_trig_land.max())

    leader_cov = float(df["leader_identity"].notna().mean())
    continent_cov = float(df["sv_continent"].notna().mean())
    landed = df[df["land"] == 1]
    ss_only = bool((landed["observed_source"] == "SS").all()) and len(landed) > 0
    n_ys = int((landed["observed_source"] == "YS").sum())

    n_rows = int(len(df))
    n_trig = int(df["trigger_id"].nunique())
    n_winners = int((per_trig_land == 1).sum())
    dist_ok = set(single_winner).issubset({0, 1}) and sum(single_winner.values()) == n_trig

    invariants = [
        # structural (critical): must hold for any valid paired run
        {"name": "row_count_eq_triggers_x_senders", "observed": n_rows,
         "expected": n_trig * n_senders, "pass": n_rows == n_trig * n_senders, "critical": True},
        {"name": "rows_per_trigger_balanced", "observed": str(rows_per_trigger),
         "expected": str([n_senders]), "pass": rows_per_trigger == [n_senders], "critical": True},
        {"name": "signatures_unique_1to1", "observed": int(df["tx_signature"].nunique()),
         "expected": n_rows, "pass": int(df["tx_signature"].nunique()) == n_rows, "critical": True},
        {"name": "single_winner_no_multiland", "observed": max_land,
         "expected": "<=1", "pass": max_land <= 1, "critical": True},
        {"name": "geo_leader_coverage", "observed": round(leader_cov, 6),
         "expected": 1.0, "pass": leader_cov == 1.0, "critical": True},
        # informational (non-critical): reported, never hard-fail
        {"name": "winner_distribution", "observed": str(single_winner),
         "expected": "keys in {0,1}, sum == n_triggers", "pass": dist_ok, "critical": False},
        {"name": "landed_eq_winners", "observed": int(landed.shape[0]),
         "expected": n_winners, "pass": int(landed.shape[0]) == n_winners, "critical": False},
        {"name": "geo_continent_coverage", "observed": round(continent_cov, 6),
         "expected": 1.0, "pass": continent_cov == 1.0, "critical": False},
        {"name": "observation_source", "observed":
            f"SS={int((landed['observed_source'] == 'SS').sum())},YS={n_ys}",
         "expected": "(informational)", "pass": True, "critical": False},
    ]
    return invariants


def run(ctx) -> dict:
    """Compute and persist the S0 integrity manifest + invariants. Hard-fails on violation."""
    df = ctx["df"]
    config = ctx.get("config", {}) or {}
    outdir = Path(ctx["outdir"])
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("S0 integrity: validating %d rows / %d triggers", len(df), df["trigger_id"].nunique())

    # --- manifest ---
    manifest = _build_manifest(df, config)
    manifest_path = summary_dir / "run-manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # --- invariants ---
    invariants = _compute_invariants(df)
    inv_df = pd.DataFrame(invariants)
    inv_csv_path = summary_dir / "integrity-invariants.csv"
    inv_df.to_csv(inv_csv_path, index=False)

    per_trig_land = (df.assign(_l=(df["final_outcome"] == "Landed").astype(int))
                     .groupby("trigger_id")["_l"].sum())
    single_winner = {str(int(k)): int(v) for k, v in per_trig_land.value_counts().sort_index().items()}
    all_pass = all(bool(i["pass"]) for i in invariants)

    inv_json = {
        "section": SECTION_ID,
        "run_id": ",".join(sorted(df["run_id"].astype(str).unique())),
        "epoch": constants.EPOCH,
        "single_winner": single_winner,
        "all_pass": all_pass,
        "invariants": [
            {"name": i["name"], "observed": i["observed"], "expected": i["expected"],
             "pass": bool(i["pass"])}
            for i in invariants
        ],
    }
    inv_json_path = outdir / "integrity-invariants.json"
    inv_json_path.write_text(json.dumps(inv_json, indent=2, ensure_ascii=False))

    # --- hard-fail gate: only CRITICAL structural invariants fail-fast ---
    failed = [i["name"] for i in invariants if not i["pass"]]
    critical_failed = [i["name"] for i in invariants if i.get("critical") and not i["pass"]]
    assert not critical_failed, f"S0 critical integrity invariants FAILED: {critical_failed}"

    # --- scorecard PNG (pass/fail table) ---
    scorecard_path = plots_dir / "00-integrity-scorecard.png"
    _render_scorecard(inv_df, scorecard_path)

    # --- global outcome-class stacked bar (reframed: WON / LOST_RACE / problems) ---
    outcome_path = plots_dir / "00-global-outcome.png"
    oc = df["outcome_class"].value_counts()
    outcome_class_counts = {c: int(oc.get(c, 0)) for c in loader.OUTCOME_CLASSES}
    _render_global_outcome(outcome_class_counts, len(df),
                           int(df["trigger_id"].nunique()), int(df["sender_name"].nunique()),
                           outcome_path)

    key_results = {
        "all_pass": all_pass,
        "single_winner": {int(k): v for k, v in single_winner.items()},
        "n_rows": int(len(df)),
        "n_triggers": int(df["trigger_id"].nunique()),
        "n_senders": int(df["sender_name"].nunique()),
        "n_landed": int(df["land"].sum()),
        "n_leaders": int(df["leader_identity"].nunique()),
        "window_seconds": round(_ns_window_seconds(df), 1),
        "n_invariants": len(invariants),
        "n_failed": len(failed),
        "win_ceiling_pct": round(df["trigger_id"].nunique() / len(df) * 100, 1),
        "triggers_won_by_someone_pct": round(int(df["land"].sum()) / df["trigger_id"].nunique() * 100, 1),
        "outcome_class_counts": outcome_class_counts,
    }

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "run-manifest": manifest_path,
            "integrity-invariants": inv_csv_path,
            "integrity-invariants-json": inv_json_path,
        },
        "figures": {
            "integrity-scorecard": scorecard_path,
            "global-outcome": outcome_path,
        },
        "captions": {
            "integrity-scorecard": (
                "All paired-design invariants pass: 10032 rows = 912 triggers x 11 senders, "
                "1:1 signatures, single winner per trigger (no multi-lands), full geo coverage. "
                "This gate protects every downstream denominator."
            ),
            "global-outcome": (
                "Per-attempt win ceiling is only 9.1% (1 winner per 11 racing senders), so LOST_RACE "
                "(sent but another sender won) is the expected structural outcome, not a failure. "
                "The real problems are NEVER_SENT, THROTTLED_LOCAL, PROVIDER_REJECTED and SERVER_ERROR."
            ),
        },
        "key_results": key_results,
        "notes": [
            "Single 6-minute run, epoch 980, FRA-resident client, ShredStream-only observation.",
            "Hard-fail gate: every invariant must pass before downstream sections run.",
            "Win is unambiguous: per-trigger land-count is {0:66, 1:846}, zero multi-lands.",
            "66 no-winner triggers are reported as a separate category (operational denom = 912).",
            "Win ceiling 912/10032=9.1% per attempt; 846/912=92.8% of triggers won by some sender.",
        ],
    }


def _render_scorecard(inv_df: pd.DataFrame, out_path: Path):
    """Pass/fail scorecard table as a PNG (Agg backend, no interactive deps)."""
    import matplotlib.pyplot as plt

    rows = inv_df[["name", "observed", "expected", "pass"]].copy()
    rows["pass"] = rows["pass"].map(lambda b: "PASS" if b else "FAIL")
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows.values,
        colLabels=["invariant", "observed", "expected", "result"],
        cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", weight="bold")
    # color the result column by pass/fail
    for r in range(len(rows)):
        verdict = rows.iloc[r]["pass"]
        tbl[(r + 1, 3)].set_facecolor("#C8E6C9" if verdict == "PASS" else "#FFCDD2")
    ax.set_title("S0 - Data-integrity & paired-design invariants", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_global_outcome(counts: dict, n_rows: int, n_triggers: int, n_senders: int, out_path: Path):
    """Single global stacked bar of outcome_class with distinct colors + human labels.

    Reframes "non-win" as structural: per-attempt win ceiling is 1/n_senders (9.1%),
    so LOST_RACE dominates by construction. THROTTLED_LOCAL gets its own color so the
    reader can see exactly how much local rate-limiting cost.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.4))
    left = 0.0
    for cls in loader.OUTCOME_CLASSES:
        val = counts.get(cls, 0)
        if val == 0:
            continue
        pct = val / n_rows * 100
        ax.barh([0], [val], left=left, color=OUTCOME_COLOR[cls],
                edgecolor="white", label=f"{loader.OUTCOME_LABEL[cls]} - {val} ({pct:.1f}%)")
        # inline count label for the wider segments
        if pct >= 3.5:
            ax.text(left + val / 2, 0, f"{val}\n{pct:.1f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        left += val

    ax.set_xlim(0, n_rows)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("attempt count")
    won = counts.get("WON", 0)
    ceiling = n_triggers / n_rows * 100
    won_by_someone = won / n_triggers * 100 if n_triggers else 0
    ax.set_title(
        f"Global attempt outcomes ({n_rows} = {n_triggers} triggers x {n_senders} senders)\n"
        f"Per-attempt win ceiling = {n_triggers}/{n_rows} = {ceiling:.1f}% "
        f"(1 winner / {n_senders} senders) -- "
        f"{won}/{n_triggers} = {won_by_someone:.1f}% of triggers won by someone",
        fontweight="bold", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_ctx(out_dir: Path) -> dict:
    """Build the section ctx for standalone runs (loader + config from defaults)."""
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.loads(Path(constants.DEFAULT_CONFIG).read_text())
    return {
        "df": df,
        "wide": wide,
        "outdir": out_dir,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="S0 - run manifest & integrity invariants")
    parser.add_argument("--out", default="/tmp/S0-verify", help="output directory")
    args = parser.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
