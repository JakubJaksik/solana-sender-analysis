"""S5 - By-how-much-faster (unpaired latency shift).

For every pair of senders whose Landed sample sizes are both >= GATE_INFERENTIAL,
estimate how much faster one is than the other on ``send_to_obs_ms`` (Landed-only):

  * Hodges-Lehmann shift estimate (median of pairwise differences a - b),
  * Mann-Whitney U two-sided test (BH-FDR adjusted across all gated pairs),
  * independent (unpaired) bootstrap 95% CIs on the p50 and p90 differences.

Latency is observable only for Landed rows and CANNOT be paired at the trigger
level (no trigger has two landers), so a same-trigger paired delta is NOT
estimable; every comparison here is an unpaired, distribution-free shift.

GEO CONFOUND: senders landed in different leader continents (e.g. 0slot-de1 had
far more North-American landings than helius-dual), so a raw cross-sender shift
mixes "who is faster" with "who landed where". The per-continent view restricts
each comparison to one observed-leader continent to neutralise that confound.

Outputs:
  summary/pairwise-latency-shift.csv
  summary/latency-shift-by-continent.csv
  plots/05-latency-shift-hl-heatmap.png       (HL shift matrix, gated senders)
  plots/05-latency-shift-p50-p90-dumbbell.png  (absolute p1..p99 spread per sender)
  plots/05-latency-shift-by-continent.png      (per-continent p50 + HL-vs-fastest)
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fanout_analysis import constants, loader, plotutils, statutils

SECTION_ID = "S5"
SECTION_TITLE = "By-how-much-faster (unpaired latency shift)"
METRIC = "send_to_obs_ms"

CSV_NAME = "pairwise-latency-shift.csv"
CONTINENT_CSV_NAME = "latency-shift-by-continent.csv"
HEATMAP_NAME = "05-latency-shift-hl-heatmap.png"
DUMBBELL_NAME = "05-latency-shift-p50-p90-dumbbell.png"
CONTINENT_NAME = "05-latency-shift-by-continent.png"

# Percentiles drawn on the absolute-spread dumbbell.
SPREAD_PCTS = [1, 10, 50, 90, 99]

PAIRED_NOTE = (
    "Same-trigger paired delta is NOT estimable: latency is observable only for "
    "Landed rows and no trigger has two landers, so all shifts are unpaired."
)


def _boot_quantile_diff(a, b, q, B=10000, seed=0):
    """Independent (unpaired) bootstrap 95% CI on quantile_q(a) - quantile_q(b).

    a is resampled from a, b from b, independently each iteration. Vectorized:
    draw all B resamples at once (B x n) and take the quantile along axis 1.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    ra = a[rng.integers(0, na, size=(B, na))]
    rb = b[rng.integers(0, nb, size=(B, nb))]
    diffs = np.quantile(ra, q, axis=1) - np.quantile(rb, q, axis=1)
    point = float(np.quantile(a, q) - np.quantile(b, q))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    min_n = int(ctx.get("min_n", constants.GATE_INFERENTIAL))

    all_senders = sorted(df["sender_name"].unique())
    landed = df[df["land"] == 1].copy()
    # Landed-only latency series per sender (drop any null metric defensively).
    series = {
        s: sub[METRIC].dropna().to_numpy()
        for s, sub in landed.groupby("sender_name")
    }
    # Every sender gets a Landed-n, including zero-landers (e.g. allenhark-quic-tk).
    n_land = {s: int(len(series.get(s, []))) for s in all_senders}

    # Gate: only senders with Landed n >= min_n are inference-eligible.
    gated = sorted([s for s in all_senders if n_land[s] >= min_n])
    excluded = sorted([s for s in all_senders if n_land[s] < min_n])

    notes = [
        PAIRED_NOTE,
        f"Gated to Landed n>={min_n}: {gated}.",
        f"Excluded (n<{min_n}): " + str(excluded) + ".",
        f"Metric is {METRIC} (Landed-only); HL/Mann-Whitney are distribution-free.",
        "Mann-Whitney p-values are BH-FDR adjusted across all gated pairs.",
        "Raw cross-sender shift is geo-confounded (senders landed in different "
        "leader continents); see latency-shift-by-continent for a within-continent view.",
    ]

    # Pairwise shifts. Convention: hl_shift_a_minus_b > 0 means sender_a is SLOWER
    # (larger latency) than sender_b. We order each pair so sender_a is the
    # slower-median sender for readability, but the sign carries the full info.
    rows = []
    for a, b in combinations(gated, 2):
        sa, sb = series[a], series[b]
        hl = statutils.hodges_lehmann(sa, sb)
        u, p = statutils.mann_whitney(sa, sb)
        p50_diff, p50_lo, p50_hi = _boot_quantile_diff(sa, sb, 0.50, seed=0)
        p90_diff, p90_lo, p90_hi = _boot_quantile_diff(sa, sb, 0.90, seed=0)
        rows.append({
            "sender_a": a,
            "sender_b": b,
            "n_a": n_land[a],
            "n_b": n_land[b],
            "p50_a_ms": float(np.median(sa)),
            "p50_b_ms": float(np.median(sb)),
            "hl_shift_a_minus_b_ms": hl,
            "p50_diff_a_minus_b_ms": p50_diff,
            "p50_diff_ci_lo": p50_lo,
            "p50_diff_ci_hi": p50_hi,
            "p90_diff_a_minus_b_ms": p90_diff,
            "p90_diff_ci_lo": p90_lo,
            "p90_diff_ci_hi": p90_hi,
            "mw_u": u,
            "mw_p": p,
        })

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df["mw_p_fdr"] = statutils.fdr_adjust(pair_df["mw_p"].tolist())
        pair_df["significant_fdr"] = pair_df["mw_p_fdr"] < 0.05
    else:
        pair_df["mw_p_fdr"] = []
        pair_df["significant_fdr"] = []

    csv_path = summary_dir / CSV_NAME
    pair_df.to_csv(csv_path, index=False)

    # HL shift matrix (gated senders only). Cell [i, j] = HL(series_i - series_j):
    # positive => row sender i is slower than column sender j.
    hl_mat = pd.DataFrame(np.nan, index=gated, columns=gated, dtype=float)
    for i in gated:
        for j in gated:
            if i == j:
                hl_mat.loc[i, j] = 0.0
            else:
                hl_mat.loc[i, j] = statutils.hodges_lehmann(series[i], series[j])

    heatmap_path = plots_dir / HEATMAP_NAME
    plotutils.heatmap(
        hl_mat,
        title=f"S5 Hodges-Lehmann shift on {METRIC} (ms): row - column, gated n>={min_n}",
        out_path=heatmap_path,
        fmt=".0f",
        cmap="RdBu_r",
        center=0.0,
        annot=True,
    )

    # Absolute per-sender percentile spread (p1..p99), ordered by p50. No baseline
    # subtraction: every dot is this sender's own landed-latency percentile in ms.
    spread = {
        s: {f"p{q}": float(np.percentile(series[s], q)) for q in SPREAD_PCTS}
        for s in gated
    }
    dumbbell_path = plots_dir / DUMBBELL_NAME
    _plot_spread_dumbbell(spread, n_land, min_n, dumbbell_path)

    # Per-continent within-leader-continent comparison (reduces geo confound).
    cont_df, cont_path, cont_fig_path = _per_continent(
        landed, gated, min_n, summary_dir, plots_dir)

    p50_by = {s: float(np.median(series[s])) for s in gated}
    fastest = min(p50_by, key=p50_by.get)

    key_results = {
        "metric": METRIC,
        "min_n": min_n,
        "gated_senders": gated,
        "n_gated": len(gated),
        "excluded_senders": excluded,
        "n_pairs": int(len(pair_df)),
        "fastest_p50_sender": fastest,
        "fastest_p50_ms": p50_by[fastest],
        "n_landed_per_gated": {s: n_land[s] for s in gated},
        "continents_compared": sorted(cont_df["continent"].unique().tolist())
        if not cont_df.empty else [],
    }

    figures = {
        "hl_heatmap": str(heatmap_path),
        "p50_p90_dumbbell": str(dumbbell_path),
    }
    tables = {"pairwise_latency_shift": str(csv_path)}
    if cont_fig_path is not None:
        figures["latency_shift_by_continent"] = str(cont_fig_path)
        tables["latency_shift_by_continent"] = str(cont_path)

    captions = {
        "hl_heatmap": (
            "Cell = median ms by which the row sender is slower(+)/faster(-) than the "
            "column sender, over all landing pairs (Hodges-Lehmann). WARNING: unpaired "
            "& geo-confounded - senders served different leader continents, so also see "
            "the per-continent view."
        ),
        "p50_p90_dumbbell": (
            "Absolute landed send->observed latency per sender (no cross-sender baseline): "
            "each row's dots are that sender's own p1, p10, p50, p90, p99 in ms, ordered by p50."
        ),
    }
    if cont_fig_path is not None:
        captions["latency_shift_by_continent"] = (
            "Within a single observed-leader continent (so geo is held fixed): left = each "
            "sender's p50 landed latency in ms; right = its Hodges-Lehmann shift vs the "
            "fastest sender on that continent (+ slower). Only senders with n>=20 on that "
            "continent are shown."
        )

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": tables,
        "figures": figures,
        "captions": captions,
        "key_results": key_results,
        "notes": notes,
    }


def _per_continent(landed, gated, min_n, summary_dir, plots_dir):
    """Per observed-leader-continent p50 + HL-shift vs the fastest sender there.

    Restricts comparisons to a single leader continent to neutralise the geo
    confound. A continent is reported only if >=2 gated senders have n>=min_n
    landings there. Returns (df, csv_path, fig_path); fig_path is None if no
    continent qualifies.
    """
    rows = []
    # Stable, sensible continent ordering for the figure.
    cont_order = ["Europe", "North America", "Asia", "South America"]
    cont_values = [c for c in cont_order if c in set(landed["observed_leader_continent"])]
    cont_values += [c for c in sorted(set(landed["observed_leader_continent"].dropna()))
                    if c not in cont_order]

    for cont in cont_values:
        sub = landed[landed["observed_leader_continent"] == cont]
        cseries = {
            s: sub.loc[sub["sender_name"] == s, METRIC].dropna().to_numpy()
            for s in gated
        }
        eligible = [s for s in gated if len(cseries[s]) >= min_n]
        if len(eligible) < 2:
            continue
        p50 = {s: float(np.median(cseries[s])) for s in eligible}
        fastest = min(p50, key=p50.get)
        for s in eligible:
            hl = statutils.hodges_lehmann(cseries[s], cseries[fastest])
            rows.append({
                "continent": cont,
                "sender": s,
                "n": int(len(cseries[s])),
                "p50_ms": p50[s],
                "p90_ms": float(np.percentile(cseries[s], 90)),
                "fastest_on_continent": fastest,
                "hl_shift_vs_fastest_ms": hl,
            })

    cont_df = pd.DataFrame(rows)
    csv_path = summary_dir / CONTINENT_CSV_NAME
    cont_df.to_csv(csv_path, index=False)

    if cont_df.empty:
        return cont_df, csv_path, None

    fig_path = plots_dir / CONTINENT_NAME
    _plot_continent(cont_df, min_n, fig_path)
    return cont_df, csv_path, fig_path


def _plot_continent(cont_df, min_n, out_path):
    """Two-panel per-continent view: p50 ms (left) and HL-shift vs fastest (right)."""
    import matplotlib.pyplot as plt

    conts = list(dict.fromkeys(cont_df["continent"]))
    n_cont = len(conts)
    fig, axes = plt.subplots(n_cont, 2, figsize=(13, max(3.0, 2.4 * n_cont)),
                             squeeze=False)
    palette = plt.get_cmap("tab10")
    for r, cont in enumerate(conts):
        sub = cont_df[cont_df["continent"] == cont].sort_values("p50_ms")
        senders = sub["sender"].tolist()
        ys = list(range(len(senders)))
        colors = [palette(i % 10) for i in range(len(senders))]

        ax0 = axes[r][0]
        ax0.barh(ys, sub["p50_ms"], color=colors)
        for y, n in zip(ys, sub["n"]):
            ax0.text(0, y, f" n={n}", va="center", ha="left", fontsize=8)
        ax0.set_yticks(ys)
        ax0.set_yticklabels(senders, fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlabel("p50 send_to_obs_ms")
        ax0.set_title(f"{cont}: p50 latency")
        ax0.grid(alpha=0.3, axis="x")

        ax1 = axes[r][1]
        fastest = sub["fastest_on_continent"].iloc[0]
        ax1.barh(ys, sub["hl_shift_vs_fastest_ms"], color=colors)
        ax1.axvline(0, color="#2E7D32", ls=":", alpha=0.6)
        ax1.set_yticks(ys)
        ax1.set_yticklabels(senders, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("HL shift vs fastest (ms, + = slower)")
        ax1.set_title(f"{cont}: vs {fastest}")
        ax1.grid(alpha=0.3, axis="x")

    fig.suptitle(f"S5 per-continent landed latency (Landed-only, gated n>={min_n} "
                 f"within continent); geo held fixed", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_spread_dumbbell(spread, n_land, min_n, out_path):
    """Absolute per-sender percentile spread (p1..p99) in ms, ordered by p50.

    No cross-sender baseline subtraction: every marker is the sender's own
    landed-latency percentile, on a shared absolute ms x-axis.
    """
    import matplotlib.pyplot as plt

    senders = sorted(spread, key=lambda s: spread[s]["p50"])
    ys = list(range(len(senders)))
    # Distinct marker/color per percentile so the full distribution is legible.
    pct_style = {
        1:  ("#90CAF9", "v", "p1"),
        10: ("#1976D2", "<", "p10"),
        50: ("#000000", "o", "p50"),
        90: ("#D32F2F", ">", "p90"),
        99: ("#7B1FA2", "^", "p99"),
    }
    fig, ax = plt.subplots(figsize=(11, max(4, 0.7 * len(senders))))
    for y, s in zip(ys, senders):
        lo = spread[s][f"p{SPREAD_PCTS[0]}"]
        hi = spread[s][f"p{SPREAD_PCTS[-1]}"]
        ax.plot([lo, hi], [y, y], color="#bbb", lw=2, zorder=1)
        for q in SPREAD_PCTS:
            color, marker, lab = pct_style[q]
            ax.scatter([spread[s][f"p{q}"]], [y], color=color, marker=marker,
                       s=70, zorder=2, edgecolor="white", linewidth=0.4,
                       label=lab if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{s} (n={n_land[s]})" for s in senders])
    ax.invert_yaxis()
    ax.set_xlabel(f"{METRIC} (absolute ms, Landed-only)")
    ax.set_title(f"S5 per-sender latency percentile spread p1..p99 "
                 f"(Landed-only, gated n>={min_n}); ordered by p50")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _build_ctx(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": out_dir,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    parser = argparse.ArgumentParser(description="S5 unpaired latency shift")
    parser.add_argument("--out", default="/tmp/S5-verify",
                        help="output dir (plots/ + summary/ created under it)")
    args = parser.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
