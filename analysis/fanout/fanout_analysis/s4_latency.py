"""S4 - Landed-only latency distributions.

Latency is observable only for ``Landed`` rows (selection bias) and CANNOT be
paired at the trigger level (no trigger has two landers) -> all comparisons are
distribution-free and UNPAIRED. We report median/percentile summaries and ECDFs
for three valid stages (the broken ``wall_prepared_to_send`` metric is never used):

- ``trigger_to_obs_ms``  : trigger fire -> first observation (end-to-end)
- ``send_to_obs_ms``     : our send -> first observation (the headline race metric)
- ``rtt_ms``             : send -> ack (send-side RTT), filtered to > 0

A per-sender ``fragile`` flag is raised when the Landed sample is too small to
support a stable percentile estimate (n_land < ``FRAGILE_N``).

Outputs (per the spec section S4):
- ``summary/latency-percentiles.csv``     : sender x metric percentile table
- ``summary/latency-component-gap.csv``   : per-sender stage decomposition (medians)
- ``summary/latency-by-continent.csv``    : per-sender x continent send_to_obs_ms p50 (n-gated)
- ``plots/04-latency-ecdf-send-to-obs.png``     : overlaid ECDF of send_to_obs_ms
- ``plots/04-latency-violin-send-to-obs.png``   : violin by median
- ``plots/04-latency-percentile-bars.png``      : clustered percentile bars (p1,p10,p50,p90,p99)
- ``plots/04-latency-top4-histograms.png``      : top-4 landers send_to_obs_ms hist
- ``plots/04-latency-by-continent.png``         : per-continent ECDF overlays + p50 heatmap

CONFOUND NOTE: senders served different leader continents (0slot landed heavily in North
America, helius almost never), so pooled latency mixes destinations. The per-continent view
isolates the destination so within-continent comparisons are like-for-like.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fanout_analysis import constants, loader, plotutils, statutils  # noqa: E402

SECTION_ID = "S4"
SECTION_TITLE = "Landed-latency distributions"
SECTION_NUM = "04"

# Stages analysed; rtt is filtered to > 0 (send-RTT==0 would be a recording bug, not a real ack).
METRICS = ["trigger_to_obs_ms", "send_to_obs_ms", "rtt_ms", "trigger_to_land_ticks"]
RTT_COL = "rtt_ms"
HEADLINE_METRIC = "send_to_obs_ms"

# Below this many Landed rows a per-sender percentile estimate is fragile/indicative only.
FRAGILE_N = 50
# Histogram panel = top-3 landers + this fixed reference sender (familiar baseline).
TOP_K = 4
REFERENCE_SENDER = "helius-dual"
# Continents analysed in the per-continent (confound-controlled) view; gated at n>=20.
CONTINENTS = ["Europe", "North America", "Asia"]


def _landed(df):
    """Landed-only rows (latency is selection-biased to winners)."""
    return df[df["final_outcome"] == "Landed"].copy()


def _metric_series(landed, sender, metric):
    """Per-sender, per-metric latency series; rtt is filtered to strictly positive."""
    s = landed.loc[landed["sender_name"] == sender, metric].dropna()
    if metric == RTT_COL:
        s = s[s > 0]
    return s


def _percentiles_table(landed):
    """Sender x metric percentile summary, one row per (sender, metric), fragile-flagged."""
    senders = sorted(landed["sender_name"].unique())
    n_land = landed.groupby("sender_name").size()
    rows = []
    for sender in senders:
        nl = int(n_land.get(sender, 0))
        fragile = nl < FRAGILE_N
        for metric in METRICS:
            s = _metric_series(landed, sender, metric)
            summ = statutils.percentile_summary(s, sender)
            # p1 = best-case "how fast can I realistically be with this sender";
            # not provided by statutils.percentile_summary, computed locally.
            p1 = float(s.quantile(.01)) if len(s) else float("nan")
            rows.append({
                "sender_name": sender,
                "metric": metric,
                "n_land": nl,
                "n": summ["n"],
                "p1": p1,
                "p10": summ["p10"],
                "p50": summ["p50"],
                "p90": summ["p90"],
                "p99": summ["p99"],
                "iqr": summ["iqr"],
                "max": summ["max"],
                "fragile": fragile,
            })
    import pandas as pd
    return pd.DataFrame(rows)


def _component_gap_table(landed):
    """Per-sender stage decomposition of the end-to-end latency (medians, ms).

    trigger_to_obs = (trigger -> send) + (send -> obs)
    send_to_obs    = (send -> ack/RTT) + (ack -> obs)
    """
    import pandas as pd
    rows = []
    for sender in sorted(landed["sender_name"].unique()):
        sub = landed[landed["sender_name"] == sender]
        tto = sub["trigger_to_obs_ms"].dropna()
        sto = sub["send_to_obs_ms"].dropna()
        rtt = sub[RTT_COL]
        rtt = rtt[rtt > 0].dropna()
        pre_send = (sub["trigger_to_obs_ms"] - sub["send_to_obs_ms"]).dropna()
        ack_to_obs = (sub["send_to_obs_ms"] - sub[RTT_COL]).dropna()
        rows.append({
            "sender_name": sender,
            "n_land": int(len(sub)),
            "trigger_to_send_p50_ms": float(pre_send.median()) if len(pre_send) else float("nan"),
            "send_rtt_p50_ms": float(rtt.median()) if len(rtt) else float("nan"),
            "ack_to_obs_p50_ms": float(ack_to_obs.median()) if len(ack_to_obs) else float("nan"),
            "send_to_obs_p50_ms": float(sto.median()) if len(sto) else float("nan"),
            "trigger_to_obs_p50_ms": float(tto.median()) if len(tto) else float("nan"),
            "fragile": len(sub) < FRAGILE_N,
        })
    return pd.DataFrame(rows)


def _plot_top4_histograms(landed, top_senders, out_path, metric=HEADLINE_METRIC, unit="ms", xlim=None):
    """2x2 grid of `metric` histograms for the top-K landers.

    Vertical lines mark p50 (red), p90 (orange) and p95 (purple) - the tail
    percentiles a caller cares about for worst-case sender behaviour. When `xlim`
    is set, bins are packed into the visible range (40 bins over xlim) so the
    cropped view stays well-resolved; values beyond xlim are dropped from the bars
    (percentile lines are still computed from the full data; off-range ones omitted).
    """
    n = len(top_senders)
    ncol = 2
    nrow = int(np.ceil(n / ncol)) if n else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    pct_lines = [(50, "red"), (90, "#F57C00"), (95, "#7B1FA2")]
    for ax, sender in zip(axes, top_senders):
        s = _metric_series(landed, sender, metric).to_numpy()
        ax.hist(s, bins=40, range=xlim, color="#1976D2", alpha=0.85)
        if len(s):
            for q, c in pct_lines:
                v = np.percentile(s, q)
                if xlim is None or (xlim[0] <= v <= xlim[1]):
                    ax.axvline(v, color=c, ls="--", alpha=0.8, label=f"p{q}={v:.0f}")
            ax.legend()
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(f"{sender} (n={len(s)})")
        ax.set_xlabel(f"{metric} ({unit})")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{metric} histograms - top landers + helius reference (Landed only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_percentile_bars(head, order, out_path):
    """Clustered percentile bars for the headline metric (p1,p10,p50,p90,p99).

    Local replacement for plotutils.plot_percentile_bar (which is fixed to
    p10/p50/p90/p99) so we can surface p1 = best realistic send-to-observe time.
    """
    import pandas as pd  # noqa: F401
    pcts = ["p1", "p10", "p50", "p90", "p99"]
    melt = head.melt(id_vars=["sender_name"], value_vars=pcts,
                     var_name="pct", value_name="ms")
    fig, ax = plt.subplots(figsize=(15, 6))
    import seaborn as sns
    sns.barplot(data=melt, x="sender_name", y="ms", hue="pct",
                hue_order=pcts, order=order, ax=ax)
    ax.set_ylabel("ms")
    ax.set_title(f"{SECTION_ID} {HEADLINE_METRIC} percentiles by sender "
                 "(p1=best case .. p99=tail, Landed only)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _continent_table(landed):
    """Per-sender x continent send_to_obs_ms summary (n, p50) for CONTINENTS, n-gated.

    Long format: one row per (sender, continent) cell with n>=GATE_INDICATIVE so the
    heatmap/table only shows cells with usable data. The confound this controls for:
    senders served very different leader continents, so pooled p50 mixes destinations.
    """
    import pandas as pd
    rows = []
    sub = landed[landed["sv_continent"].isin(CONTINENTS)]
    for sender in sorted(sub["sender_name"].unique()):
        for cont in CONTINENTS:
            s = sub.loc[(sub["sender_name"] == sender) & (sub["sv_continent"] == cont),
                        HEADLINE_METRIC].dropna()
            n = int(len(s))
            if n == 0:
                continue
            rows.append({
                "sender_name": sender,
                "continent": cont,
                "n": n,
                "p50_send_to_obs_ms": float(s.quantile(.50)),
                "p10_send_to_obs_ms": float(s.quantile(.10)),
                "p90_send_to_obs_ms": float(s.quantile(.90)),
                "gate": ("inferential" if n >= constants.GATE_INFERENTIAL
                         else "indicative" if n >= constants.GATE_INDICATIVE else "suppress"),
            })
    return pd.DataFrame(rows)


def _plot_by_continent(landed, cont_tbl, out_path):
    """Two panels: (left) per-continent send_to_obs ECDF overlays for senders with
    n>=GATE_INFERENTIAL in that continent; (right) per-sender x continent p50 heatmap
    (cells with n>=GATE_INFERENTIAL only). Like-for-like, destination-controlled view.
    """
    import pandas as pd
    import seaborn as sns

    present = [c for c in CONTINENTS if (landed["sv_continent"] == c).sum() >= constants.GATE_INFERENTIAL]
    ncol = len(present)
    fig = plt.figure(figsize=(6.5 * (ncol + 1), 6))
    gs = fig.add_gridspec(1, ncol + 1)

    for j, cont in enumerate(present):
        ax = fig.add_subplot(gs[0, j])
        csub = landed[landed["sv_continent"] == cont]
        series = {}
        for sender in sorted(csub["sender_name"].unique()):
            s = csub.loc[csub["sender_name"] == sender, HEADLINE_METRIC].dropna().to_numpy()
            if len(s) >= constants.GATE_INFERENTIAL:
                series[sender] = s
        for name, s in sorted(series.items(), key=lambda kv: np.median(kv[1])):
            s = np.sort(s)
            y = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, y, where="post", label=f"{name} (p50={np.median(s):.0f}, n={len(s)})")
        ax.set_xscale("log")
        ax.set_xlabel(f"{HEADLINE_METRIC} (ms)")
        ax.set_ylabel("ECDF")
        ax.set_title(f"{cont} (n>=20 senders)")
        ax.grid(alpha=0.3)
        if series:
            ax.legend(fontsize=7, loc="lower right")

    # heatmap of p50 across senders x continents (inferential cells only)
    axh = fig.add_subplot(gs[0, ncol])
    infer = cont_tbl[cont_tbl["gate"] == "inferential"]
    if len(infer):
        pivot = infer.pivot(index="sender_name", columns="continent",
                            values="p50_send_to_obs_ms")
        pivot = pivot.reindex(columns=[c for c in CONTINENTS if c in pivot.columns])
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", ax=axh,
                    cbar_kws={"shrink": .7, "label": "p50 send_to_obs_ms"})
    axh.set_title("p50 send_to_obs_ms (n>=20 cells)")
    plt.setp(axh.get_xticklabels(), rotation=25, ha="right")

    fig.suptitle(f"{SECTION_ID} per-continent {HEADLINE_METRIC} (destination-controlled, Landed only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _ticks_region_table(landed):
    """Per (sender x region) trigger_to_land_ticks p50/p90/p99 (winners, gated n>=20),
    for continents + the 8 top countries by winner count. Returns (table, top_countries)."""
    import pandas as pd
    top_countries = (landed.groupby("sv_country").size().sort_values(ascending=False)
                     .head(8).index.tolist())
    rows = []
    for region_col, keys in [("sv_continent", CONTINENTS), ("sv_country", top_countries)]:
        for sender in sorted(landed["sender_name"].unique()):
            for k in keys:
                s = landed.loc[(landed["sender_name"] == sender) & (landed[region_col] == k),
                               "trigger_to_land_ticks"].dropna()
                if len(s) >= constants.GATE_INFERENTIAL:
                    rows.append({"sender_name": sender, "region_type": region_col, "region": k,
                                 "n": int(len(s)),
                                 "p50_ticks": float(s.quantile(.50)),
                                 "p90_ticks": float(s.quantile(.90)),
                                 "p99_ticks": float(s.quantile(.99))})
    return pd.DataFrame(rows), top_countries


def _plot_ticks_region_heat(tbl, top_countries, pct_col, pct_label, out_path):
    """Two heatmaps (sender x continent | sender x top-countries) of a given
    trigger_to_land_ticks percentile (winners, cells n>=20). Lower = fewer ticks."""
    import seaborn as sns
    panels = [("sv_continent", CONTINENTS, "per kontynent"),
              ("sv_country", top_countries, "per kraj (top wg wygranych)")]
    fig = plt.figure(figsize=(7 + 0.9 * len(top_countries), 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(CONTINENTS), max(1, len(top_countries))])
    for i, (rtype, cols, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        sub = tbl[tbl["region_type"] == rtype]
        if len(sub):
            piv = sub.pivot(index="sender_name", columns="region", values=pct_col)
            piv = piv.reindex(columns=[c for c in cols if c in piv.columns])
            sns.heatmap(piv, annot=True, fmt=".0f", cmap="viridis_r", ax=ax,
                        cbar_kws={"shrink": .6, "label": f"{pct_label} trigger->land [ticki]"})
        ax.set_title(ttl)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle(f"{SECTION_ID} {pct_label} trigger->land w tickach, sender x region (winners, n>=20)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    min_n = ctx.get("min_n", constants.GATE_INFERENTIAL)

    landed = _landed(df)

    # --- tables ---
    perc = _percentiles_table(landed)
    gap = _component_gap_table(landed)
    cont_tbl = _continent_table(landed)

    perc_path = summary_dir / "latency-percentiles.csv"
    gap_path = summary_dir / "latency-component-gap.csv"
    cont_path = summary_dir / "latency-by-continent.csv"
    perc.to_csv(perc_path, index=False)
    gap.to_csv(gap_path, index=False)
    cont_tbl.to_csv(cont_path, index=False)

    # --- figures ---
    # Overlaid ECDF of the headline race metric, Landed only, one line per sender.
    ecdf_series = {s: _metric_series(landed, s, HEADLINE_METRIC).to_numpy()
                   for s in sorted(landed["sender_name"].unique())}
    ecdf_series = {k: v for k, v in ecdf_series.items() if len(v) > 0}
    ecdf_path = plots_dir / f"{SECTION_NUM}-latency-ecdf-send-to-obs.png"
    plotutils.plot_ecdf_multi(
        ecdf_series,
        title=f"{SECTION_ID} send_to_obs latency ECDF (Landed only, unpaired)",
        xlabel="send_to_obs_ms",
        out_path=ecdf_path,
        logx=True,
    )

    # Overlaid ECDF of trigger->land in PoH TICKS (chain-time, deterministic), Landed only.
    ticks_series = {s: _metric_series(landed, s, "trigger_to_land_ticks").to_numpy()
                    for s in sorted(landed["sender_name"].unique())}
    ticks_series = {k: v for k, v in ticks_series.items() if len(v) > 0}
    ticks_ecdf_path = plots_dir / f"{SECTION_NUM}-latency-ecdf-trigger-to-land-ticks.png"
    plotutils.plot_ecdf_multi(
        ticks_series,
        title=f"{SECTION_ID} trigger->land latency ECDF in PoH ticks (chain-time, Landed only; x capped 250t)",
        xlabel="trigger_to_land_ticks (64 ticks/slot)",
        out_path=ticks_ecdf_path,
        logx=False,
        xlim=(0, 250),
    )

    # Violin of send_to_obs_ms, ordered by ascending median (fastest left).
    order = (landed.groupby("sender_name")[HEADLINE_METRIC].median()
             .sort_values().index.tolist())
    violin_path = plots_dir / f"{SECTION_NUM}-latency-violin-send-to-obs.png"
    plotutils.plot_violin(
        landed, x="sender_name", y=HEADLINE_METRIC,
        title=f"{SECTION_ID} send_to_obs_ms by sender (ordered by median, Landed only; y capped 500ms)",
        out_path=violin_path, order=order, ylim=(0, 500),
    )

    # Clustered percentile bars for the headline metric, fastest-median order.
    # Includes p1 (best realistic case) .. p99 (tail) via local plotter.
    head = perc[perc["metric"] == HEADLINE_METRIC].set_index("sender_name").loc[order].reset_index()
    pbar_path = plots_dir / f"{SECTION_NUM}-latency-percentile-bars.png"
    _plot_percentile_bars(head, order, pbar_path)

    # Histogram panel: top-3 landers + a fixed reference sender (helius-dual) so there is
    # always a familiar baseline to compare against, even when helius is outside the top.
    by_count = landed.groupby("sender_name").size().sort_values(ascending=False).index.tolist()
    top3 = by_count[:3]
    if REFERENCE_SENDER in by_count and REFERENCE_SENDER not in top3:
        top_senders = top3 + [REFERENCE_SENDER]
    else:
        top_senders = by_count[:TOP_K]
    hist_path = plots_dir / f"{SECTION_NUM}-latency-top4-histograms.png"
    _plot_top4_histograms(landed, top_senders, hist_path, xlim=(0, 1500))

    # trigger->land in PoH ticks (chain-time): violin + top-K histograms, like send_to_obs.
    ticks_order = (landed.groupby("sender_name")["trigger_to_land_ticks"].median()
                   .sort_values().index.tolist())
    ticks_violin_path = plots_dir / f"{SECTION_NUM}-latency-violin-trigger-to-land-ticks.png"
    plotutils.plot_violin(
        landed, x="sender_name", y="trigger_to_land_ticks",
        title=f"{SECTION_ID} trigger->land ticks by sender (ordered by median, Landed only; y capped 100t)",
        out_path=ticks_violin_path, order=ticks_order, ylim=(0, 100),
    )
    ticks_hist_path = plots_dir / f"{SECTION_NUM}-latency-hist-trigger-to-land-ticks.png"
    _plot_top4_histograms(landed, top_senders, ticks_hist_path,
                          metric="trigger_to_land_ticks", unit="ticks", xlim=(0, 200))

    # Per-continent (destination-controlled) ECDF overlays + p50 heatmap.
    cont_fig_path = plots_dir / f"{SECTION_NUM}-latency-by-continent.png"
    _plot_by_continent(landed, cont_tbl, cont_fig_path)

    # trigger->land in TICKS per sender x region (continent + country), p50/p90/p99
    ticks_region_tbl, top_countries = _ticks_region_table(landed)
    ticks_region_csv = summary_dir / "latency-ticks-by-region.csv"
    ticks_region_tbl.to_csv(ticks_region_csv, index=False)
    ticks_region_fig = plots_dir / f"{SECTION_NUM}-latency-ticks-by-region.png"          # p50
    ticks_region_p90 = plots_dir / f"{SECTION_NUM}-latency-ticks-by-region-p90.png"
    ticks_region_p99 = plots_dir / f"{SECTION_NUM}-latency-ticks-by-region-p99.png"
    _plot_ticks_region_heat(ticks_region_tbl, top_countries, "p50_ticks", "p50", ticks_region_fig)
    _plot_ticks_region_heat(ticks_region_tbl, top_countries, "p90_ticks", "p90", ticks_region_p90)
    _plot_ticks_region_heat(ticks_region_tbl, top_countries, "p99_ticks", "p99", ticks_region_p99)

    # --- key results ---
    head_sorted = head.sort_values("p50")
    fragile_senders = sorted(perc.loc[perc["fragile"], "sender_name"].unique().tolist())
    inferential = sorted(perc.loc[(perc["metric"] == HEADLINE_METRIC) & (perc["n"] >= min_n),
                                  "sender_name"].unique().tolist())
    send_to_obs_n = {r["sender_name"]: int(r["n"])
                     for _, r in perc[perc["metric"] == HEADLINE_METRIC].iterrows()}
    p50_send_to_obs = {r["sender_name"]: (None if np.isnan(r["p50"]) else float(r["p50"]))
                       for _, r in head.iterrows()}

    fastest = head_sorted.iloc[0]["sender_name"] if len(head_sorted) else None
    fastest_p50 = float(head_sorted.iloc[0]["p50"]) if len(head_sorted) else None

    key_results = {
        "n_landed_total": int(len(landed)),
        "metrics": METRICS,
        "headline_metric": HEADLINE_METRIC,
        "fragile_threshold_n": FRAGILE_N,
        "fragile_senders": fragile_senders,
        "inferential_senders": inferential,
        "send_to_obs_n_by_sender": send_to_obs_n,
        "send_to_obs_p50_by_sender": p50_send_to_obs,
        "fastest_sender_send_to_obs": fastest,
        "fastest_send_to_obs_p50_ms": fastest_p50,
    }

    notes = [
        "Latency is Landed-only (selection bias toward winners) and UNPAIRED - no trigger has "
        "two landers, so same-trigger paired deltas are not estimable; comparisons are "
        "distribution-free (median/percentile/ECDF, Mann-Whitney/Hodges-Lehmann).",
        "wall_prepared_to_send is broken and is never used; only trigger->obs, send->obs and "
        "send-RTT are valid stages.",
        f"rtt_ms is filtered to strictly positive values for {RTT_COL}.",
        f"Senders with n_land < {FRAGILE_N} are flagged 'fragile' (percentile estimates "
        "indicative only): " + ", ".join(fragile_senders) + ".",
        f"allenhark-quic-tk has zero Landed rows (never sent) and therefore does not appear.",
        "Pooled send_to_obs latency is confounded by destination: senders landed in different "
        "leader continents (0slot heavily in North America, helius almost not at all). The "
        "per-continent figure/CSV controls for this with within-continent (n>=20) comparisons.",
    ]

    captions = {
        "ecdf_send_to_obs": (
            "ECDF = empirical cumulative distribution: for each x (ms), the curve shows the "
            "fraction of landings that were at least that fast. Leftmost/steepest curve = "
            "consistently fastest; read p50 where the curve crosses 0.5. "
            "CONFOUND: this pools all destinations - senders served different leader continents, "
            "so see the per-continent figure for a like-for-like comparison."
        ),
        "violin_send_to_obs": (
            "send_to_obs_ms distribution shape per sender (Landed only), ordered fastest median "
            "on the left. Wider bulges mean more landings at that latency."
        ),
        "percentile_bars": (
            "send_to_obs_ms percentiles per sender: p1 = best realistic case (how fast you can be), "
            "p50 = typical, p90/p99 = tail. Senders ordered by median, fastest left."
        ),
        "top4_histograms": (
            "send_to_obs_ms histograms for the four biggest landers; dashed lines mark p50, p90 "
            "and p95 so you can read the typical and tail latency at a glance."
        ),
        "latency_by_continent": (
            "Destination-controlled view: ECDF per leader continent (only senders with n>=20 there) "
            "plus a p50 heatmap (n>=20 cells, darker = faster). This removes the pooled-latency "
            "confound since each panel fixes the destination continent."
        ),
        "ecdf_trigger_to_land_ticks": (
            "Same idea as the send_to_obs ECDF but in PoH TICKS (chain-time, 64 ticks/slot) from "
            "trigger fire to landing - a deterministic clock independent of wall-clock/observation "
            "jitter. Leftmost curve = lands in fewest ticks. Winners only."
        ),
        "violin_trigger_to_land_ticks": (
            "trigger->land distribution per sender in PoH ticks (chain-time), ordered by median; "
            "y capped at 192 ticks (3 slots). Bulk near 0-64 = landed within the trigger's slot."
        ),
        "hist_trigger_to_land_ticks": (
            "trigger->land histograms (PoH ticks) for the biggest landers; dashed lines mark "
            "p50/p90/p95. Shows how many ticks pass from firing to landing on chain."
        ),
        "ticks_by_region": (
            "Median (p50) trigger->land in TICKS for each sender per leader region (continent + "
            "top countries). Cells only where n>=20 wins; lighter = lands in fewer ticks. "
            "Shows which sender is 'fast' (in chain-time) and where geographically."
        ),
        "ticks_by_region_p90": (
            "Same as ticks_by_region, but p90 (tail) trigger->land in ticks per sender x region: "
            "how bad it gets in 1 of 10 wins."
        ),
        "ticks_by_region_p99": (
            "Same again, but p99 (extreme tail) trigger->land in ticks per sender x region: "
            "the worst realistic case."
        ),
    }

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "latency_percentiles": perc_path,
            "latency_component_gap": gap_path,
            "latency_by_continent": cont_path,
            "latency_ticks_by_region": ticks_region_csv,
        },
        "figures": {
            "ecdf_send_to_obs": ecdf_path,
            "ecdf_trigger_to_land_ticks": ticks_ecdf_path,
            "violin_send_to_obs": violin_path,
            "violin_trigger_to_land_ticks": ticks_violin_path,
            "percentile_bars": pbar_path,
            "top4_histograms": hist_path,
            "hist_trigger_to_land_ticks": ticks_hist_path,
            "latency_by_continent": cont_fig_path,
            "ticks_by_region": ticks_region_fig,
            "ticks_by_region_p90": ticks_region_p90,
            "ticks_by_region_p99": ticks_region_p99,
        },
        "captions": captions,
        "key_results": key_results,
        "notes": notes,
    }


def _build_ctx(out):
    """Build the ctx dict from canonical inputs for standalone CLI use."""
    outdir = Path(out)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": outdir,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    parser = argparse.ArgumentParser(description="Run S4 - Landed-latency distributions.")
    parser.add_argument("--out", default="/tmp/S4-verify", help="output directory")
    args = parser.parse_args()
    ctx = _build_ctx(args.out)
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))
    print("tables:", {k: str(v) for k, v in result["tables"].items()})
    print("figures:", {k: str(v) for k, v in result["figures"].items()})


if __name__ == "__main__":
    main()
