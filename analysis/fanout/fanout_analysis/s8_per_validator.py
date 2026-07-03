"""S8 - Per-validator (per-leader) view.

For every distinct scheduled leader (one per trigger slot) compute:
  - n_triggers       : triggers whose slot was led by this validator
  - landed_any       : triggers where >=1 sender landed (single-winner design -> land==1)
  - no_winner        : triggers where nobody landed
  - inclusion_rate   : landed_any / n_triggers (corrected per-leader inclusion)
  - no_winner_rate   : no_winner / n_triggers
  - top_sender       : sender that won the most triggers at this leader (and its count)
  - stake + geo annotation (stake_sol, city, country, continent, dc)

Plus a sparse (leader x sender) land count/rate table, and a strict-gated set of the
leaders with n_triggers>=min_n (inferential), min_ind..min_n-1 indicative, <min_ind
long-tail. All counts (leader total, gated count, etc.) are derived from the data, so
the module works unchanged on any future run.

A rate claim is only made where n_triggers>=GATE_INFERENTIAL; lower-n rows carry a
gate_label so the report can annotate the caveat. Latency is irrelevant here (this is a
win/inclusion section), so no latency stats are computed.

Outputs:
  summary/per-leader-summary.csv     : one row per leader, stake-sorted desc
  summary/per-leader-per-sender.csv  : sparse (leader, sender) land count/rate rows
  summary/per-validator-tops.txt     : PL + EN ranking sections
Plots:
  plots/08-per-validator-leader-sender-heatmap.png : leader x sender land matrix (top-N by n_triggers)
  plots/08-per-validator-winning-sender-stacked.png: winning-sender composition for the gated leaders
  plots/08-per-validator-inclusion-rate.png        : inclusion-rate bar, colored by continent
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fanout_analysis import constants, loader, plotutils  # noqa: E402

logger = logging.getLogger(__name__)

SID = "S8"
TITLE = "Per-validator (per-leader) view"
NN = "08"
# Higher cutoff for the two per-leader bar charts (keeps them readable on big pooled runs).
PLOT_MIN_TRIGGERS = 100

# how many leaders (by trigger count) to show in the leader x sender heatmap
HEATMAP_TOP_N = 20

# continent -> color for the inclusion-rate bar
_CONTINENT_COLORS = {
    "Europe": "#1976D2",
    "North America": "#E64A19",
    "Asia": "#388E3C",
    "South America": "#FBC02D",
    "Oceania": "#7B1FA2",
    "Africa": "#5D4037",
}


def _gate_label(n, min_n, min_ind):
    if n >= min_n:
        return "inferential"
    if n >= min_ind:
        return "indicative"
    return "long_tail"


def _per_leader_tables(df, min_n, min_ind):
    """Build per-leader summary (one row per distinct leader) and the sparse
    (leader x sender) land table from the enriched LONG frame."""
    # one row per trigger: all 11 sender-rows of a trigger share the same leader/slot
    per_trig = (df.groupby("trigger_id")
                .agg(leader_identity=("leader_identity", "first"),
                     leader_name=("leader_name", "first"),
                     leader_stake_sol=("leader_stake_sol", "first"),
                     leader_dc=("leader_dc", "first"),
                     sv_city=("sv_city", "first"),
                     sv_country=("sv_country", "first"),
                     sv_continent=("sv_continent", "first"),
                     land_count=("land", "sum"))
                .reset_index())

    # winning sender per landed trigger (single-winner design -> land==1 is unique within trigger)
    winners = df[df["land"] == 1][["trigger_id", "leader_identity", "sender_name"]]

    # (leader x sender) land counts -> sparse long table
    ls = (winners.groupby(["leader_identity", "sender_name"])
          .size().rename("land_count").reset_index())

    rows = []
    for leader, sub in per_trig.groupby("leader_identity"):
        n = len(sub)
        landed_any = int((sub["land_count"] >= 1).sum())
        no_winner = int((sub["land_count"] == 0).sum())
        # top winning sender at this leader
        w = winners[winners["leader_identity"] == leader]
        if len(w):
            vc = w["sender_name"].value_counts()
            top_sender = str(vc.index[0])
            top_sender_wins = int(vc.iloc[0])
        else:
            top_sender = None
            top_sender_wins = 0
        first = sub.iloc[0]
        rows.append({
            "leader_identity": leader,
            "leader_name": first["leader_name"],
            "leader_stake_sol": float(first["leader_stake_sol"]),
            "leader_dc": first["leader_dc"],
            "sv_city": first["sv_city"],
            "sv_country": first["sv_country"],
            "sv_continent": first["sv_continent"],
            "n_triggers": n,
            "landed_any": landed_any,
            "no_winner": no_winner,
            "inclusion_rate": round(landed_any / n, 4) if n else None,
            "no_winner_rate": round(no_winner / n, 4) if n else None,
            "top_sender": top_sender,
            "top_sender_wins": top_sender_wins,
            "gate_label": _gate_label(n, min_n, min_ind),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["leader_stake_sol", "n_triggers"], ascending=[False, False]).reset_index(drop=True)

    # add n_triggers + rate to the sparse per-sender table
    n_by_leader = summary.set_index("leader_identity")["n_triggers"].to_dict()
    name_by_leader = summary.set_index("leader_identity")["leader_name"].to_dict()
    ls["leader_name"] = ls["leader_identity"].map(name_by_leader)
    ls["n_triggers"] = ls["leader_identity"].map(n_by_leader)
    ls["land_rate"] = (ls["land_count"] / ls["n_triggers"]).round(4)
    ls = ls.sort_values(["n_triggers", "leader_identity", "land_count"],
                        ascending=[False, True, False]).reset_index(drop=True)
    ls = ls[["leader_identity", "leader_name", "sender_name",
             "land_count", "n_triggers", "land_rate"]]
    return summary, ls


def _tops_text(summary, min_n, min_ind):
    lines = []

    def section(title, frame, cols):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        sub = frame[cols].copy()
        sub["leader_name"] = sub["leader_name"].fillna("(unnamed)")
        lines.append(sub.to_string(index=False))

    gated = summary[summary["n_triggers"] >= min_n].copy()
    indic = summary[(summary["n_triggers"] >= min_ind) & (summary["n_triggers"] < min_n)].copy()

    cols = ["leader_name", "sv_city", "sv_continent", "n_triggers",
            "landed_any", "inclusion_rate", "top_sender", "top_sender_wins", "leader_stake_sol"]

    # ---- PL ----
    section(f"TOP LIDERZY WG LICZBY TRIGGEROW (inferencyjne, n>={min_n})",
            gated.nlargest(min(10, len(gated)), "n_triggers"), cols)
    section(f"NAJWYZSZY INCLUSION RATE (inferencyjne, n>={min_n})",
            gated.nlargest(min(10, len(gated)), "inclusion_rate"), cols)
    section(f"NAJNIZSZY INCLUSION RATE / NAJWIECEJ NO-WINNER (inferencyjne, n>={min_n})",
            gated.nsmallest(min(10, len(gated)), "inclusion_rate"), cols)
    section(f"NAJWIEKSZY STAKE (wszystkie liderzy)",
            summary.nlargest(10, "leader_stake_sol"), cols)

    # ---- EN ----
    section(f"TOP LEADERS BY TRIGGER COUNT (inferential, n>={min_n})",
            gated.nlargest(min(10, len(gated)), "n_triggers"), cols)
    section(f"HIGHEST INCLUSION RATE (inferential, n>={min_n})",
            gated.nlargest(min(10, len(gated)), "inclusion_rate"), cols)
    section(f"LOWEST INCLUSION RATE / MOST NO-WINNER (inferential, n>={min_n})",
            gated.nsmallest(min(10, len(gated)), "inclusion_rate"), cols)
    section(f"INDICATIVE LEADERS ({min_ind}<=n<{min_n}; rate caveated)",
            indic.nlargest(min(10, len(indic)), "n_triggers"), cols)

    # footer with counts
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"leaders total: {len(summary)} | "
                 f"inferential (n>={min_n}): {len(gated)} | "
                 f"indicative ({min_ind}<=n<{min_n}): {len(indic)} | "
                 f"long-tail (n<{min_ind}): {len(summary) - len(gated) - len(indic)}")
    lines.append(f"sum n_triggers: {int(summary['n_triggers'].sum())}")
    return "\n".join(lines)


def _plot_heatmap(df, summary, ls, out_path):
    """leader x sender land matrix for the top-N leaders by trigger count."""
    top_leaders = summary.nlargest(HEATMAP_TOP_N, "n_triggers")
    leader_order = list(top_leaders["leader_identity"])
    sender_order = sorted(df["sender_name"].unique())
    mat = (ls[ls["leader_identity"].isin(leader_order)]
           .pivot_table(index="leader_identity", columns="sender_name",
                        values="land_count", aggfunc="sum")
           .reindex(index=leader_order, columns=sender_order)
           .fillna(0).astype(int))
    # label rows by name (n) for readability
    label_map = {r["leader_identity"]: f"{(r['leader_name'] if (pd.notna(r['leader_name']) and str(r['leader_name']).strip()) else r['leader_identity'][:6])} (n={r['n_triggers']})"
                 for _, r in top_leaders.iterrows()}
    mat.index = [label_map[i] for i in mat.index]
    return plotutils.heatmap(
        mat, f"S8 leader x sender land count (top {HEATMAP_TOP_N} leaders by triggers)",
        out_path, fmt="d", cmap="viridis", annot=True)


def _plot_winning_sender_stacked(ls, leaders, summary, title_suffix, out_path):
    """Winning-sender composition (stacked bar) for the given leader set."""
    sub = ls[ls["leader_identity"].isin(leaders)]
    mat = (sub.pivot_table(index="leader_identity", columns="sender_name",
                           values="land_count", aggfunc="sum").fillna(0))
    # order leaders by n_triggers desc, label by name (n)
    order = (summary[summary["leader_identity"].isin(leaders)]
             .sort_values("n_triggers", ascending=False))
    mat = mat.reindex(index=list(order["leader_identity"]))
    label_map = {r["leader_identity"]: f"{(r['leader_name'] if (pd.notna(r['leader_name']) and str(r['leader_name']).strip()) else r['leader_identity'][:6])} (n={r['n_triggers']})"
                 for _, r in order.iterrows()}
    mat.index = [label_map[i] for i in mat.index]
    return plotutils.stacked_bar(
        mat, list(mat.columns),
        f"S8 winning-sender composition - {title_suffix}",
        "landed triggers", out_path, horizontal=True, pct=False)


def _plot_inclusion_rate(summary, leaders, title_suffix, out_path):
    """Inclusion-rate bar for the given leader set, colored by continent."""
    gated = (summary[summary["leader_identity"].isin(leaders)]
             .sort_values("inclusion_rate", ascending=True))
    labels = [f"{(r['leader_name'] if (pd.notna(r['leader_name']) and str(r['leader_name']).strip()) else r['leader_identity'][:6])} (n={r['n_triggers']})"
              for _, r in gated.iterrows()]
    colors = [_CONTINENT_COLORS.get(c, "#999999") for c in gated["sv_continent"]]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(gated))))
    ax.barh(labels, gated["inclusion_rate"], color=colors)
    ax.set_xlabel("inclusion rate (landed_any / n_triggers)")
    ax.set_title(f"S8 per-leader inclusion rate - {title_suffix}; color = leader continent")
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3, axis="x")
    seen = {}
    for c, name in [(_CONTINENT_COLORS.get(c, "#999999"), c) for c in gated["sv_continent"]]:
        seen[name] = c
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in seen.values()]
    ax.legend(handles, list(seen.keys()), fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    min_n = ctx.get("min_n", constants.GATE_INFERENTIAL)
    min_ind = ctx.get("min_indicative", constants.GATE_INDICATIVE)
    summ_dir = outdir / "summary"
    plot_dir = outdir / "plots"
    summ_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("S8 per-validator: building per-leader tables")
    summary, ls = _per_leader_tables(df, min_n, min_ind)

    gated_leaders = list(summary[summary["n_triggers"] >= min_n]["leader_identity"])
    logger.info("S8 per-validator: %d leaders, %d inferential (n>=%d), sum n_triggers=%d",
                len(summary), len(gated_leaders), min_n, int(summary["n_triggers"].sum()))

    # ---- CSVs / txt ----
    summary_csv = summ_dir / "per-leader-summary.csv"
    persender_csv = summ_dir / "per-leader-per-sender.csv"
    tops_txt = summ_dir / "per-validator-tops.txt"
    summary.to_csv(summary_csv, index=False)
    ls.to_csv(persender_csv, index=False)
    tops_txt.write_text(_tops_text(summary, min_n, min_ind))

    # ---- plots ----
    heatmap_png = plot_dir / f"{NN}-per-validator-leader-sender-heatmap.png"
    stacked_png = plot_dir / f"{NN}-per-validator-winning-sender-stacked.png"
    inclusion_png = plot_dir / f"{NN}-per-validator-inclusion-rate.png"
    _plot_heatmap(df, summary, ls, heatmap_png)
    # The two per-leader bar charts use a higher cutoff (PLOT_MIN_TRIGGERS) so they stay
    # readable on big pooled runs; fall back to the top leaders by n on small runs.
    hi = summary[summary["n_triggers"] >= PLOT_MIN_TRIGGERS]
    if len(hi) >= 3:
        plot_leaders = list(hi["leader_identity"])
        suffix = f"{len(plot_leaders)} leaders (>= {PLOT_MIN_TRIGGERS} triggers)"
    else:
        top = summary.nlargest(min(15, len(summary)), "n_triggers")
        plot_leaders = list(top["leader_identity"])
        suffix = f"top {len(plot_leaders)} leaders by trigger count"
    _plot_winning_sender_stacked(ls, plot_leaders, summary, suffix, stacked_png)
    _plot_inclusion_rate(summary, plot_leaders, suffix, inclusion_png)

    # gated leaders headline
    gated_rows = summary[summary["n_triggers"] >= min_n].sort_values("n_triggers", ascending=False)
    gated_view = [
        {"leader_name": r["leader_name"], "sv_continent": r["sv_continent"],
         "n_triggers": int(r["n_triggers"]), "inclusion_rate": r["inclusion_rate"],
         "top_sender": r["top_sender"], "top_sender_wins": int(r["top_sender_wins"])}
        for _, r in gated_rows.iterrows()
    ]

    key_results = {
        "n_leaders": int(len(summary)),
        "sum_n_triggers": int(summary["n_triggers"].sum()),
        "n_inferential_leaders": int(len(gated_leaders)),
        "n_indicative_leaders": int(((summary["n_triggers"] >= min_ind) &
                                     (summary["n_triggers"] < min_n)).sum()),
        "n_long_tail_leaders": int((summary["n_triggers"] < min_ind).sum()),
        "total_landed_any": int(summary["landed_any"].sum()),
        "total_no_winner": int(summary["no_winner"].sum()),
        "gated_leaders": gated_view,
    }

    notes = [
        f"Strict gating: only {len(gated_leaders)} leaders reach n_triggers>={min_n} (inferential); "
        "all other inclusion/no-winner rates are indicative or long-tail and must not anchor a claim.",
        "Single-winner design: at most one sender lands per trigger, so landed_any == sum(land) and "
        "the leader x sender matrix counts are mutually exclusive per trigger.",
        "Per-leader inclusion is descriptive of one FRA-resident run; no persistent "
        "validator-ranking claim from a single run.",
    ]

    n_inferential = len(gated_leaders)
    captions = {
        "leader_sender_heatmap": (
            f"For the top {min(HEATMAP_TOP_N, len(summary))} leaders by trigger count, how many "
            "triggers each sender won at each leader (one winner per trigger). Brighter = more wins."
        ),
        "winning_sender_stacked": (
            f"Which senders won the triggers at the {n_inferential} well-sampled leaders "
            f"(>= {min_n} triggers each). Bar length = landed triggers; segments = winning sender."
        ),
        "inclusion_rate_bar": (
            f"Share of each well-sampled leader's triggers won by some sender "
            f"(>= {min_n} triggers each); bar color = leader's continent. Low bars = more no-winner slots."
        ),
    }

    return {
        "id": SID,
        "title": TITLE,
        "tables": {
            "per_leader_summary": str(summary_csv),
            "per_leader_per_sender": str(persender_csv),
            "per_validator_tops": str(tops_txt),
        },
        "figures": {
            "leader_sender_heatmap": str(heatmap_png),
            "winning_sender_stacked": str(stacked_png),
            "inclusion_rate_bar": str(inclusion_png),
        },
        "captions": captions,
        "key_results": key_results,
        "notes": notes,
    }


def _build_ctx(out: Path) -> dict:
    out = Path(out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = {}
    if Path(constants.DEFAULT_CONFIG).exists():
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="S8 per-validator (per-leader) view")
    ap.add_argument("--out", type=Path, default=Path("/tmp/S8-verify"))
    args = ap.parse_args()
    ctx = _build_ctx(args.out)
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
