"""S2 - Head-to-head win rates (3 estimands + Wilson CIs).

Three named win-rate estimands per sender, each with Wilson score CIs (never a
bare proportion), plus a corrected-conditional sensitivity variant:

  1. operational   = Landed / 912                       (SendError counts as a loss)
  2. conditional   = Landed / (Landed + UnknownPending)  (exclude SendError; headline)
  3. share         = Landed / 846                        (contested-pool share)
  4. corrected     = (Landed & not helius_500) / (accepted & not never_sent)
                     (helius HTTP-500 reclassified as a send-failure, never-sent excluded)

`paired_bootstrap_winrate` over the 912x11 wide matrix supplies rank CIs.

Outputs (filenames per spec S2):
  summary/win-rate-estimands.csv, summary/no-winner-triggers.csv
  plots/02-winrate-forest.png        (3-panel forest)
  plots/02-winrate-slopegraph.png    (operational -> conditional, colored by protocol)
  plots/02-winrate-bootstrap-rank-heatmap.png
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fanout_analysis import constants, loader, statutils, plotutils  # noqa: E402

logger = logging.getLogger(__name__)

SECTION_ID = "S2"
SECTION_TITLE = "Head-to-head win rates (3 estimands + Wilson CIs)"
SECTION_NUM = "02"

# Distinct color per protocol class for the slopegraph.
_PROTOCOL_COLORS = {
    "HTTP_JSONRPC": "#1976D2",
    "JITO": "#D32F2F",
    "RELAY": "#388E3C",
    "QUIC_TPU": "#F57C00",
}


def _bootstrap_seed(ctx):
    return int(ctx.get("config", {}).get("_bootstrap_seed", 0)) if isinstance(ctx.get("config"), dict) else 0


def _estimands_table(df, wide, min_n, min_indicative):
    """Build one row per sender with all four estimands + Wilson CIs + bootstrap ranks."""
    senders = list(constants.SENDER_META.keys())

    # Bootstrap rank CIs over the wide (paired) matrix. Reorder cols to the canonical order.
    wide_ord = wide[[s for s in senders if s in wide.columns]]
    boot = statutils.paired_bootstrap_winrate(wide_ord, B=10000, seed=0)

    rows = []
    for s in senders:
        sub = df[df["sender_name"] == s]
        landed = int((sub["final_outcome"] == "Landed").sum())

        # 1. operational = Landed / 912
        op_n = int(len(sub))
        op_lo, op_hi = statutils.wilson_ci(landed, op_n)

        # 2. conditional = Landed / (Landed + UnknownPending) [exclude SendError]
        cond_n = int((sub["final_outcome"] != "SendError").sum())
        cond_lo, cond_hi = statutils.wilson_ci(landed, cond_n)

        # 3. share = Landed / 846 (contested pool of winning triggers)
        share_n = int((df["land"] == 1).sum())
        share_lo, share_hi = statutils.wilson_ci(landed, share_n)

        # 4. corrected conditional: helius_500 -> fail; exclude never_sent rows
        corr_landed = int(((sub["final_outcome"] == "Landed") & (~sub["helius_500"])).sum())
        corr_n = int(((sub["final_outcome"] != "SendError") & (~sub["never_sent"])).sum())
        corr_lo, corr_hi = statutils.wilson_ci(corr_landed, corr_n)

        # gating annotation keyed off the (smallest meaningful) conditional denominator
        if cond_n >= min_n:
            gate = "inferential"
        elif cond_n >= min_indicative:
            gate = "indicative"
        else:
            gate = "suppressed"

        b = boot.get(s, {})
        rows.append({
            "sender_name": s,
            "protocol_class": constants.PROTOCOL_OF[s],
            "landed": landed,
            "operational_n": op_n,
            "operational_rate": landed / op_n if op_n else float("nan"),
            "operational_lo": op_lo,
            "operational_hi": op_hi,
            "conditional_landed": landed,
            "conditional_n": cond_n,
            "conditional_rate": landed / cond_n if cond_n else float("nan"),
            "conditional_lo": cond_lo,
            "conditional_hi": cond_hi,
            "share_n": share_n,
            "share_rate": landed / share_n if share_n else float("nan"),
            "share_lo": share_lo,
            "share_hi": share_hi,
            "corrected_landed": corr_landed,
            "corrected_n": corr_n,
            "corrected_rate": corr_landed / corr_n if corr_n else float("nan"),
            "corrected_lo": corr_lo,
            "corrected_hi": corr_hi,
            "never_sent": int(sub["never_sent"].sum()),
            "helius_500": int(sub["helius_500"].sum()),
            "boot_rank_median": b.get("rank_median"),
            "boot_rank_lo": b.get("rank_lo"),
            "boot_rank_hi": b.get("rank_hi"),
            "gate": gate,
        })

    out = pd.DataFrame(rows).sort_values("operational_rate", ascending=False).reset_index(drop=True)
    return out


def _no_winner_table(df):
    """Per-trigger view of the 66 triggers where no sender landed."""
    land_per_trig = df.groupby("trigger_id")["land"].sum()
    nowin_ids = land_per_trig[land_per_trig == 0].index
    sub = df[df["trigger_id"].isin(nowin_ids)]
    agg = sub.groupby("trigger_id").agg(
        slot=("slot", "first"),
        tick=("tick", "first"),
        leader_identity=("leader_identity", "first"),
        leader_name=("leader_name", "first"),
        sv_continent=("sv_continent", "first"),
        sv_country=("sv_country", "first"),
        n_senderror=("final_outcome", lambda s: int((s == "SendError").sum())),
        n_unknownpending=("final_outcome", lambda s: int((s == "UnknownPending").sum())),
    ).reset_index().sort_values("slot").reset_index(drop=True)
    return agg


def _forest_3panel(est, out_path):
    """3-panel forest: operational | conditional | share, one row per sender."""
    panels = [
        ("operational_rate", "operational_lo", "operational_hi", "Operational (Landed/912)"),
        ("conditional_rate", "conditional_lo", "conditional_hi", "Conditional (Landed/accepted)"),
        ("share_rate", "share_lo", "share_hi", "Win share (Landed/846)"),
    ]
    # order rows by conditional rate (headline estimand) descending; top at top of plot
    order = est.sort_values("conditional_rate", ascending=True).reset_index(drop=True)
    labels = order["sender_name"].tolist()
    ys = list(range(len(order)))

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.55 * len(order))), sharey=True)
    for ax, (rate_c, lo_c, hi_c, title) in zip(axes, panels):
        for y in ys:
            lo = order.loc[y, lo_c]
            hi = order.loc[y, hi_c]
            pt = order.loc[y, rate_c]
            ax.plot([lo, hi], [y, y], color="#555")
            ax.plot(pt, y, "o", color="#1976D2")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_xlabel("rate")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels)
    fig.suptitle("S2 - Win-rate estimands with Wilson 95% CIs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _slopegraph(est, out_path):
    """operational -> conditional rate slope, colored by protocol class."""
    left = {r.sender_name: r.operational_rate for r in est.itertuples()}
    right = {r.sender_name: r.conditional_rate for r in est.itertuples()}
    color_by = {r.sender_name: _PROTOCOL_COLORS.get(r.protocol_class, "#777")
                for r in est.itertuples()}
    return plotutils.slopegraph(left, right, "operational\n(Landed/912)",
                                "conditional\n(Landed/accepted)",
                                "S2 - Operational -> conditional re-ranking (throttle penalty)",
                                out_path, color_by=color_by)


def _bootstrap_rank_heatmap(wide, out_path):
    """Heatmap of P(rank = r) per sender from the paired bootstrap over triggers."""
    senders = [s for s in constants.SENDER_META.keys() if s in wide.columns]
    wide_ord = wide[senders]
    arr = wide_ord.to_numpy()
    n, k = arr.shape
    rng = np.random.default_rng(0)
    B = 10000
    counts = np.zeros((k, k), dtype=float)  # [sender, rank-1]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = arr[idx].mean(axis=0)
        ranks = (-r).argsort().argsort()  # 0 = highest rate
        for j in range(k):
            counts[j, ranks[j]] += 1
    prob = counts / B
    mat = pd.DataFrame(prob, index=senders, columns=[f"rank{i+1}" for i in range(k)])
    # order rows by their modal (most-likely) rank then by p_hat
    phat = arr.mean(axis=0)
    order = np.lexsort((-phat, np.argmax(prob, axis=1)))
    mat = mat.iloc[order]
    return plotutils.heatmap(mat, "S2 - Bootstrap rank distribution P(rank | resample)",
                             out_path, fmt=".2f", cmap="magma", annot=True)


def _per_sender_wins(df):
    """Classic per-sender wins table (matches the bench's stdout summary).

    sent = n_triggers - blocked(SendError); win% = wins/sent; share% = wins/total_wins.
    rtt_avg(us) = mean send-RTT over acked attempts; obs_avg(us) = mean send->observed over wins.
    """
    n_trig = int(df["trigger_id"].nunique())
    total_wins = int(df["land"].sum())
    rows = []
    for name, sub in df.groupby("sender_name"):
        sid = int(sub["sender_id"].iloc[0])
        wins = int(sub["land"].sum())
        blocked = int((sub["final_outcome"] == "SendError").sum())
        sent = n_trig - blocked
        rtt = sub.loc[sub["rtt_ms"].notna(), "rtt_ms"]
        obs = sub.loc[sub["land"] == 1, "send_to_obs_ms"]
        rows.append({
            "id": sid, "name": name, "wins": wins, "sent": sent,
            "win_pct": round(wins / sent * 100, 1) if sent else 0.0,
            "share_pct": round(wins / total_wins * 100, 1) if total_wins else 0.0,
            "blocked": blocked,
            "rtt_avg_us": round(rtt.mean() * 1000) if len(rtt) else None,
            "obs_avg_us": round(obs.mean() * 1000) if wins else None,
        })
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def _format_wins_txt(t) -> str:
    head = ("--- Per-sender wins (at most 1 variant per trigger lands in nonce mode) ---\n"
            f"{'id':<3} {'name':<20}{'wins':>5} {'sent':>6}{'win%':>7}{'share%':>8}{'blocked':>9}"
            f"{'rtt_avg(us)':>13}{'obs_avg(us)':>13}")
    lines = [head]
    for r in t.itertuples():
        rtt = "-" if pd.isna(r.rtt_avg_us) else f"{int(r.rtt_avg_us)}"
        obs = "-" if pd.isna(r.obs_avg_us) else f"{int(r.obs_avg_us)}"
        lines.append(f"{r.id:<3} {r.name:<20}{r.wins:>5} {r.sent:>6}{r.win_pct:>7}{r.share_pct:>7}%"
                     f"{r.blocked:>9}{rtt:>13}{obs:>13}")
    return "\n".join(lines)


def run(ctx) -> dict:
    df = ctx["df"]
    wide = ctx["wide"]
    outdir = Path(ctx["outdir"])
    min_n = ctx.get("min_n", constants.GATE_INFERENTIAL)
    min_indicative = ctx.get("min_indicative", constants.GATE_INDICATIVE)

    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("S2: building win-rate estimands for %d senders", df["sender_name"].nunique())

    est = _estimands_table(df, wide, min_n, min_indicative)
    nowin = _no_winner_table(df)

    est_path = summary_dir / "win-rate-estimands.csv"
    nowin_path = summary_dir / "no-winner-triggers.csv"
    est.to_csv(est_path, index=False)
    nowin.to_csv(nowin_path, index=False)
    logger.info("S2: wrote %s (%d rows), %s (%d rows)",
                est_path.name, len(est), nowin_path.name, len(nowin))

    forest_path = _forest_3panel(est, plots_dir / f"{SECTION_NUM}-winrate-forest.png")
    slope_path = _slopegraph(est, plots_dir / f"{SECTION_NUM}-winrate-slopegraph.png")
    rankheat_path = _bootstrap_rank_heatmap(wide, plots_dir / f"{SECTION_NUM}-winrate-bootstrap-rank-heatmap.png")

    wins_tbl = _per_sender_wins(df)
    wins_csv = summary_dir / "per-sender-wins.csv"
    wins_tbl.to_csv(wins_csv, index=False)
    wins_txt = summary_dir / "per-sender-wins.txt"
    wins_txt.write_text(_format_wins_txt(wins_tbl))

    op_counts = {r.sender_name: r.landed for r in est.itertuples()}
    cond_leader = est.sort_values("conditional_rate", ascending=False).iloc[0]
    op_leader = est.sort_values("operational_rate", ascending=False).iloc[0]

    key_results = {
        "operational_landed_counts": op_counts,
        "operational_landed_sum": int(est["landed"].sum()),
        "n_no_winner_triggers": int(len(nowin)),
        "operational_leader": op_leader["sender_name"],
        "operational_leader_rate": float(op_leader["operational_rate"]),
        "conditional_leader": cond_leader["sender_name"],
        "conditional_leader_rate": float(cond_leader["conditional_rate"]),
        "denominators": {"operational": int(df["trigger_id"].nunique()),
                         "share": int(est["share_n"].iloc[0])},
    }

    notes = [
        "Conditional (Landed/accepted, SendError excluded) is the headline estimand; it re-ranks "
        "the throttled senders (jito/syncro/blockrazor) that the operational estimand penalizes.",
        "Corrected-conditional reclassifies the 182 helius HTTP-500 rows as send failures and "
        "excludes never-sent attempts; allenhark-quic-tk has a zero corrected denominator (912/912 "
        "never_sent) -> Wilson CI (0,0).",
        "All CIs are within-run Wilson score intervals over 912 triggers; no persistent-ranking "
        "claim from a single 6-minute run. Bootstrap ranks resample triggers (paired).",
        f"Gating: conditional n>={min_n} inferential / {min_indicative}-{min_n - 1} indicative / "
        f"<{min_indicative} suppressed.",
    ]

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "win_rate_estimands": str(est_path),
            "no_winner_triggers": str(nowin_path),
            "per_sender_wins": str(wins_csv),
        },
        "figures": {
            "forest": str(forest_path),
            "slopegraph": str(slope_path),
            "bootstrap_rank_heatmap": str(rankheat_path),
        },
        "captions": {
            "forest": (
                "Each dot = a sender's win-rate; the bar = 95% Wilson confidence interval. "
                "Three panels = three ways to count (operational = of all 912, "
                "conditional = excluding rate-limited attempts, share = of the 846 won)."
            ),
            "slopegraph": (
                "Each line links a sender's operational win-rate (left) to its conditional "
                "win-rate (right, ignoring throttled attempts). Steep upward lines = senders "
                "that were rate-limited and look much better once throttling is excluded "
                "(jito/syncro/blockrazor)."
            ),
            "bootstrap_rank_heatmap": (
                "We resample the 912 triggers 10000x; each time we re-rank senders by win-rate. "
                "Cell = probability that sender ends at rank k (rank 1 = best). "
                "Tight single-column = certain rank; spread = uncertain."
            ),
        },
        "key_results": key_results,
        "notes": notes,
    }


def _build_ctx(out_dir):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = {}
    if constants.DEFAULT_CONFIG.exists():
        config = json.loads(constants.DEFAULT_CONFIG.read_text())
    return {
        "df": df,
        "wide": wide,
        "outdir": out_dir,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run S2 win-rate section standalone.")
    parser.add_argument("--out", default="/tmp/S2-verify", help="output directory")
    args = parser.parse_args()

    ctx = _build_ctx(args.out)
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
