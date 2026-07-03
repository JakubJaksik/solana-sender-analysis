"""S11 - Synthesis scorecard.

Composite per-sender row computed DIRECTLY from the enriched LONG frame (no
dependency on other section CSVs): conditional win-rate, send->obs p50 (landed),
coverage, cost-per-landing (config tips), EU-vs-non-EU land-rate gap, dominant
blocker; plus a rule-based verdict tag INVEST / IMPROVE / DEPRIORITIZE / FIX-BUG
with the single lever.

CLI:
    python -m fanout_analysis.s11_synthesis --out /tmp/S11-verify
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fanout_analysis import constants, loader  # noqa: E402

logger = logging.getLogger(__name__)

SECTION_ID = "S11"
SECTION_TITLE = "Synthesis scorecard"

# Verdict thresholds (single-run, conditional estimand). INVEST requires both a
# high conditional win-rate and near-full attempt coverage; the rate-limited
# senders are routed to IMPROVE first (their SendError blocker means we cannot
# fairly judge their race performance).
INVEST_MIN_CONDITIONAL = 0.20
INVEST_MIN_COVERAGE = 0.95

VERDICT_FIX_BUG = "FIX-BUG"
VERDICT_IMPROVE = "IMPROVE"
VERDICT_INVEST = "INVEST"
VERDICT_DEPRIORITIZE = "DEPRIORITIZE"

VERDICT_COLOR = {
    VERDICT_INVEST: "#2E7D32",        # green
    VERDICT_IMPROVE: "#F9A825",       # amber
    VERDICT_DEPRIORITIZE: "#9E9E9E",  # grey
    VERDICT_FIX_BUG: "#C62828",       # red
}


def _tips_from_config(config: dict) -> dict:
    """sender_name -> tip_lamports from the run-config snapshot (never hardcoded)."""
    return {s["name"]: int(s["tip_lamports"]) for s in config["senders"]}


def _dominant_blocker(non_landed: pd.DataFrame) -> str:
    """Largest single failure bucket among this sender's non-landed attempts."""
    se = int((non_landed["final_outcome"] == "SendError").sum())
    never = int(non_landed["never_sent"].sum())
    h500 = int(non_landed["helius_500"].sum())
    sent_lost = int(((non_landed["final_outcome"] == "UnknownPending")
                     & (~non_landed["never_sent"])
                     & (~non_landed["helius_500"])).sum())
    buckets = {
        "SendError": se,
        "never_sent": never,
        "helius_500": h500,
        "sent_but_lost": sent_lost,
    }
    return max(buckets, key=buckets.get)


def _verdict_and_lever(row: dict) -> tuple:
    """Rule-based verdict + the single lever to pull.

    Rules (in priority order):
      1. never sent at all (coverage 0) -> FIX-BUG (dead PoP).
      2. dominant blocker is rate-limiting (SendError) -> IMPROVE (cannot fairly judge).
      3. high conditional win-rate + good coverage -> INVEST.
      4. otherwise -> DEPRIORITIZE.
    """
    if row["coverage"] == 0.0:
        return VERDICT_FIX_BUG, "dead PoP: never sent any attempt (send_at_ns==0)"
    if row["dominant_blocker"] == "SendError":
        return VERDICT_IMPROVE, "rate-limited (SendError) -> lift throttle / raise rate limit"
    if (row["conditional_winrate"] >= INVEST_MIN_CONDITIONAL
            and row["coverage"] >= INVEST_MIN_COVERAGE):
        return VERDICT_INVEST, "top conditional win-rate at full coverage -> scale up"
    return VERDICT_DEPRIORITIZE, f"low conditional win-rate ({row['conditional_winrate']:.1%})"


def _scorecard(df: pd.DataFrame, tips: dict) -> pd.DataFrame:
    df = df.copy()
    df["is_eu"] = df["sv_continent"] == "Europe"
    rows = []
    for sender, sub in df.groupby("sender_name"):
        landed = int(sub["land"].sum())
        unknown = int((sub["final_outcome"] == "UnknownPending").sum())
        send_error = int((sub["final_outcome"] == "SendError").sum())
        cond_denom = landed + unknown
        conditional = landed / cond_denom if cond_denom else float("nan")

        sent = int((~sub["never_sent"]).sum())
        coverage = sent / len(sub) if len(sub) else float("nan")   # len(sub) == n_triggers

        landed_rows = sub[sub["land"] == 1]
        p50 = float(landed_rows["send_to_obs_ms"].median()) if landed else float("nan")

        tip = tips.get(sender)                                     # None if not in run-config
        # on-chain tip is paid only on a LANDED tx, so per-win cost = tip (base fee ~6k lamp
        # negligible). NOT tip*attempts - non-landed submissions cost nothing on-chain.
        cost_per_landing_sol = (tip / 1e9) if (landed and tip is not None) else float("nan")

        eu = sub[sub["is_eu"]]
        non_eu = sub[~sub["is_eu"]]
        eu_lr = float(eu["land"].mean()) if len(eu) else float("nan")
        non_eu_lr = float(non_eu["land"].mean()) if len(non_eu) else float("nan")
        geo_gap = eu_lr - non_eu_lr

        blocker = _dominant_blocker(sub[sub["land"] == 0])

        row = {
            "sender_name": sender,
            "protocol_class": constants.PROTOCOL_OF.get(sender, "UNKNOWN"),
            "landed": landed,
            "unknown_pending": unknown,
            "send_error": send_error,
            "conditional_winrate": conditional,
            "send_to_obs_p50_ms": p50,
            "coverage": coverage,
            "tip_lamports": tip,
            "cost_per_landing_sol": cost_per_landing_sol,
            "eu_land_rate": eu_lr,
            "non_eu_land_rate": non_eu_lr,
            "geo_dependence_gap": geo_gap,
            "dominant_blocker": blocker,
        }
        verdict, lever = _verdict_and_lever(row)
        row["verdict"] = verdict
        row["lever"] = lever
        rows.append(row)

    out = pd.DataFrame(rows)
    # rank by conditional win-rate (headline estimand), best first
    out = out.sort_values("conditional_winrate", ascending=False).reset_index(drop=True)
    return out


def _plot_scorecard_table(scorecard: pd.DataFrame, out_path: Path) -> Path:
    cols = ["sender_name", "protocol_class", "conditional_winrate", "send_to_obs_p50_ms",
            "coverage", "cost_per_landing_sol", "dominant_blocker", "verdict"]
    disp = scorecard[cols].copy()
    disp["conditional_winrate"] = disp["conditional_winrate"].map(lambda v: f"{v:.1%}")
    disp["send_to_obs_p50_ms"] = disp["send_to_obs_p50_ms"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    disp["coverage"] = disp["coverage"].map(lambda v: f"{v:.1%}")
    disp["cost_per_landing_sol"] = disp["cost_per_landing_sol"].map(
        lambda v: "inf" if v == float("inf") else f"{v:.4f}")
    disp.columns = ["sender", "protocol", "cond. win-rate", "send->obs p50 (ms)",
                    "coverage", "cost/land (SOL)", "blocker", "verdict"]

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(disp) + 1.5))
    ax.axis("off")
    table = ax.table(cellText=disp.values, colLabels=disp.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    verdict_col = list(disp.columns).index("verdict")
    for r in range(len(disp)):
        verdict = disp.iloc[r]["verdict"]
        cell = table[r + 1, verdict_col]
        cell.set_facecolor(VERDICT_COLOR.get(verdict, "#FFFFFF"))
        cell.set_text_props(color="white", weight="bold")
    ax.set_title(f"{SECTION_ID} sender verdict scorecard (conditional estimand)", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_quadrant(scorecard: pd.DataFrame, out_path: Path) -> Path:
    """2D quadrant: x = conditional win-rate, y = -(send->obs p50), size = coverage,
    color = verdict. Senders with no landed row (p50 NaN) are drawn at the bottom."""
    fig, ax = plt.subplots(figsize=(12, 8))
    p50 = scorecard["send_to_obs_p50_ms"]
    y_floor = p50.max(skipna=True)
    y_floor = 0.0 if pd.isna(y_floor) else float(y_floor)
    for _, row in scorecard.iterrows():
        x = row["conditional_winrate"]
        raw_p50 = row["send_to_obs_p50_ms"]
        y = -(raw_p50 if not pd.isna(raw_p50) else y_floor)
        size = 80 + row["coverage"] * 1200
        color = VERDICT_COLOR.get(row["verdict"], "#000000")
        ax.scatter(x, y, s=size, color=color, alpha=0.7, edgecolors="black", linewidths=0.6)
        ax.annotate(row["sender_name"], (x, y), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          markerfacecolor=c, markeredgecolor="black", label=v)
               for v, c in VERDICT_COLOR.items()]
    ax.legend(handles=handles, title="verdict", loc="lower right", fontsize=9)
    ax.set_xlabel("conditional win-rate  (Landed / (Landed + UnknownPending))")
    ax.set_ylabel("- send->obs p50 (ms)   (higher = faster)")
    ax.set_title(f"{SECTION_ID} sender quadrant (size = attempt coverage)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"

    tips = _tips_from_config(ctx["config"])
    scorecard = _scorecard(df, tips)
    logger.info("S11 scorecard built for %d senders", len(scorecard))

    csv_path = summary_dir / "sender-scorecard.csv"
    scorecard.to_csv(csv_path, index=False)

    table_png = plots_dir / "11-synthesis-scorecard.png"
    quadrant_png = plots_dir / "11-synthesis-quadrant.png"
    _plot_scorecard_table(scorecard, table_png)
    _plot_quadrant(scorecard, quadrant_png)

    verdicts = dict(zip(scorecard["sender_name"], scorecard["verdict"]))
    counts = scorecard["verdict"].value_counts().to_dict()
    invest = sorted(scorecard.loc[scorecard["verdict"] == VERDICT_INVEST, "sender_name"])
    improve = sorted(scorecard.loc[scorecard["verdict"] == VERDICT_IMPROVE, "sender_name"])

    notes = [
        "Single 6-min run, FRA-resident client, ShredStream-only observation; verdicts are "
        "within-run signal, NOT a persistent ranking. Multi-run replication recommended.",
        "Headline estimand is conditional-on-acceptance (excludes SendError); throttled senders "
        "(jito-multi, syncro-fra) are routed to IMPROVE because rate-limiting confounds the race.",
        "Tips read from run-config snapshot, not hardcoded; cost-per-landing is per-tx-tip "
        "archetype only (subscription archetypes priced 0).",
        "Latency (send->obs p50) is Landed-only and unpaired; used descriptively in the quadrant.",
    ]

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {"sender-scorecard": str(csv_path)},
        "figures": {"scorecard-table": str(table_png), "quadrant": str(quadrant_png)},
        "key_results": {
            "verdicts": verdicts,
            "verdict_counts": counts,
            "invest": invest,
            "improve": improve,
        },
        "notes": notes,
    }


def _build_ctx(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": out,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="S11 synthesis scorecard")
    ap.add_argument("--out", default="/tmp/S11-verify", type=Path)
    args = ap.parse_args()
    ctx = _build_ctx(args.out)
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))


if __name__ == "__main__":
    main()
