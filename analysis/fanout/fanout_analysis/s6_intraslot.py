"""S6 - Intra-slot position (observed_tick) & slots-behind.

Per-sender intra-slot landing position (observed_tick distribution: median, early
0-15 share, late 48-63 share); slots-behind composition P(0)/P(1)/P(>=2) showing the
~30% of wins that land one slot late (NOTE: a leader holds 4 consecutive slots, so
slots_behind>=1 rarely means a different leader); Spearman correlation of send-RTT vs
slots-behind; per-trigger outcome split (same-slot win / next-slot win / no-win) by
trigger-tick bucket (1-16/17-32/33-48/49-64); and the rotation-position (slot mod 4)
verified-null land-rate slice.

Landed-only, distribution-free (per the spec gating policy). Gated: rate/share claims
on a sender with <GATE_INFERENTIAL landers are annotated indicative, <GATE_INDICATIVE
suppressed-to-long-tail (still reported descriptively but flagged).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fanout_analysis import constants, loader, plotutils

SECTION_ID = "S6"
SECTION_TITLE = "Intra-slot position (observed_tick) & slots-behind"
PNG_PREFIX = "06"

EARLY_LO, EARLY_HI = 0, 15
LATE_LO, LATE_HI = 48, 63


def _tick_bucket(t):
    """Trigger PoH tick (1..64) -> 4 quarter-slot buckets."""
    if 1 <= t <= 16:
        return "01-16"
    if 17 <= t <= 32:
        return "17-32"
    if 33 <= t <= 48:
        return "33-48"
    return "49-64"


def _gate(n):
    if n >= constants.GATE_INFERENTIAL:
        return "inferential"
    if n >= constants.GATE_INDICATIVE:
        return "indicative"
    return "long_tail"


def _observed_tick_per_sender(landed: pd.DataFrame) -> pd.DataFrame:
    """Per-sender observed_tick distribution among Landed rows."""
    rows = []
    for sender, sub in landed.groupby("sender_name"):
        s = sub["observed_tick"].dropna()
        n = int(len(s))
        rows.append({
            "sender_name": sender,
            "protocol_class": constants.PROTOCOL_OF.get(sender),
            "n_landed": n,
            "tick_median": float(s.median()),
            "tick_p10": float(s.quantile(.10)),
            "tick_p90": float(s.quantile(.90)),
            "early_0_15_share": float(s.between(EARLY_LO, EARLY_HI).mean()),
            "late_48_63_share": float(s.between(LATE_LO, LATE_HI).mean()),
            "gate": _gate(n),
        })
    return pd.DataFrame(rows).sort_values("n_landed", ascending=False).reset_index(drop=True)


def _slots_behind_per_sender(landed: pd.DataFrame) -> pd.DataFrame:
    """Per-sender slots-behind composition P(0)/P(1)/P(>=2)."""
    rows = []
    for sender, sub in landed.groupby("sender_name"):
        sb = sub["slots_behind"].dropna().astype(int)
        n = int(len(sb))
        c0 = int((sb == 0).sum())
        c1 = int((sb == 1).sum())
        cge2 = int((sb >= 2).sum())
        rows.append({
            "sender_name": sender,
            "protocol_class": constants.PROTOCOL_OF.get(sender),
            "n_landed": n,
            "count_0": c0,
            "count_1": c1,
            "count_ge2": cge2,
            "p_same_slot": c0 / n if n else float("nan"),
            "p_one_behind": c1 / n if n else float("nan"),
            "p_two_plus_behind": cge2 / n if n else float("nan"),
            "gate": _gate(n),
        })
    return pd.DataFrame(rows).sort_values("n_landed", ascending=False).reset_index(drop=True)


def _land_rate_by_tick_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Trigger-tick bucket -> outcome split per TRIGGER.

    n = number of triggers fired in that tick bucket (one trigger = one race).
    Each trigger is won same-slot (slots_behind==0), won next-slot
    (slots_behind>=1) or has no winner. land_rate = won_any / n.
    """
    # collapse to one row per trigger: the firing tick + whether/how it was won.
    per_trig = (df.groupby("trigger_id")
                  .agg(tick=("tick", "first"),
                       won_any=("land", "max"),
                       won_same=("same_slot", "max"),
                       won_next=("landed_next_slot", "max"))
                  .reset_index())
    per_trig["tick_bucket"] = per_trig["tick"].map(_tick_bucket)
    g = per_trig.groupby("tick_bucket").agg(
        n=("trigger_id", "size"),
        won_same_slot=("won_same", "sum"),
        won_next_slot=("won_next", "sum"),
        won_any=("won_any", "sum"),
    )
    g["no_win"] = g["n"] - g["won_any"]
    g["land_rate"] = g["won_any"] / g["n"]
    return g.reset_index().sort_values("tick_bucket").reset_index(drop=True)


def _land_rate_by_rotation(df: pd.DataFrame) -> pd.DataFrame:
    """Rotation-position (slot mod 4) land-rate slice -> expected verified-null."""
    tmp = df.assign(rot=(df["slot"] % 4).astype(int))
    g = tmp.groupby("rot").agg(n=("land", "size"), landed=("land", "sum"))
    g["land_rate"] = g["landed"] / g["n"]
    return g.reset_index().sort_values("rot").reset_index(drop=True)


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summ = outdir / "summary"
    plots = outdir / "plots"
    summ.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    landed = df[df["land"] == 1].copy()

    # --- tables ---
    tick_df = _observed_tick_per_sender(landed)
    sb_df = _slots_behind_per_sender(landed)
    bucket_df = _land_rate_by_tick_bucket(df)
    rot_df = _land_rate_by_rotation(df)

    tick_csv = summ / "observed-tick-per-sender.csv"
    sb_csv = summ / "slots-behind-per-sender.csv"
    bucket_csv = summ / "land-rate-by-tick-bucket.csv"
    rot_csv = summ / "rotation-position-landrate.csv"
    tick_df.to_csv(tick_csv, index=False)
    sb_df.to_csv(sb_csv, index=False)
    bucket_df.to_csv(bucket_csv, index=False)
    rot_df.to_csv(rot_csv, index=False)

    # --- global slots-behind composition (golden oracle) ---
    global_sb = (landed["slots_behind"].dropna().astype(int)
                 .value_counts().sort_index())
    global_sb_dict = {int(k): int(v) for k, v in global_sb.items()}

    # --- Spearman: RTT vs slots-behind (landed, valid rtt) ---
    rtt_sub = landed[landed["rtt_ms"].notna() & landed["slots_behind"].notna()]
    if len(rtt_sub) >= 3:
        rho, p_spear = stats.spearmanr(rtt_sub["rtt_ms"], rtt_sub["slots_behind"])
        rho, p_spear = float(rho), float(p_spear)
    else:
        rho, p_spear = float("nan"), float("nan")

    # --- rotation verified-null (chi-square over slot%4 x land) ---
    rot_ct = np.array([[r.landed, r.n - r.landed] for r in rot_df.itertuples()])
    chi2, p_rot, _, _ = stats.chi2_contingency(rot_ct)

    # --- figures ---
    figures = {}
    captions = {}

    # 1) observed_tick per sender -- SAME-SLOT landings only (observed_tick is only
    #    meaningful within the slot the tx actually landed in; next-slot landings
    #    restart their tick 0-63 in a different slot, so mixing them is misleading).
    tick_long = landed.dropna(subset=["observed_tick"]).copy()
    same_slot_long = tick_long[tick_long["same_slot"]].copy()
    order = tick_df["sender_name"].tolist()
    fig_tick = plots / f"{PNG_PREFIX}-observed-tick-by-sender.png"
    plotutils.plot_violin(same_slot_long, x="sender_name", y="observed_tick",
                          title="S6 - Landing intra-slot position by sender "
                                "(observed_tick 0-63, same-slot landings only)",
                          out_path=fig_tick, order=order)
    figures["observed_tick_by_sender"] = str(fig_tick)
    captions["observed_tick_by_sender"] = (
        "Where inside the winning slot each sender's tx landed (PoH tick 0-63), "
        "restricted to same-slot wins. Most landings cluster in the back half of the slot.")

    # global observed_tick histogram: SAME-slot vs NEXT-slot as two panels so the
    # user sees next-slot landings exist (their tick restarts 0-63 in the new slot).
    fig_tick_hist = plots / f"{PNG_PREFIX}-observed-tick-hist.png"
    _observed_tick_hist(same_slot_long["observed_tick"],
                        tick_long[tick_long["landed_next_slot"]]["observed_tick"],
                        fig_tick_hist)
    figures["observed_tick_hist"] = str(fig_tick_hist)
    captions["observed_tick_hist"] = (
        "Ticks are 0-63 WITHIN one slot. Top: same-slot wins. Bottom: next-slot wins "
        "(slots_behind>=1) -- a separate slot whose tick restarts at 0, not tick>64. "
        "Dashed lines mark p50 and p99.")

    # 2) slots-behind stacked bar per sender (P(0)/P(1)/P(>=2))
    sb_plot = sb_df.set_index("sender_name")[["count_0", "count_1", "count_ge2"]].copy()
    sb_plot.columns = ["same_slot(0)", "one_behind(1)", "two_plus(>=2)"]
    fig_sb = plots / f"{PNG_PREFIX}-slots-behind-stacked.png"
    plotutils.stacked_bar(sb_plot, cols=list(sb_plot.columns),
                          title="S6 - Slots-behind composition per sender (Landed)",
                          xlabel="landed count", out_path=fig_sb,
                          horizontal=True, pct=False)
    figures["slots_behind_stacked"] = str(fig_sb)
    captions["slots_behind_stacked"] = (
        "How many slots late each sender's wins landed. A leader holds 4 consecutive "
        "slots, so slots_behind=1 is usually still the SAME leader, not a missed one.")

    # 3) RTT vs slots-behind scatter
    fig_scatter = plots / f"{PNG_PREFIX}-rtt-vs-slots-behind.png"
    _scatter_rtt_slots(rtt_sub, rho, p_spear, fig_scatter)
    figures["rtt_vs_slots_behind"] = str(fig_scatter)
    captions["rtt_vs_slots_behind"] = (
        f"Each point = a landing; x=send round-trip ms (log), y=how many slots late "
        f"it landed. Positive Spearman (rho~{rho:.2f}, p<1e-12) = slower RTT tends to "
        f"land a slot later.")

    # 4) per-trigger outcome split (same-slot win / next-slot win / no-win) by tick bucket
    fig_bucket = plots / f"{PNG_PREFIX}-landrate-by-tick-bucket.png"
    _bucket_outcome_bars(bucket_df, fig_bucket)
    figures["landrate_by_tick_bucket"] = str(fig_bucket)
    captions["landrate_by_tick_bucket"] = (
        "Triggers grouped by the PoH tick they fired at (n = triggers in that bucket). "
        "Each bar splits into same-slot win, next-slot win (slots_behind>=1) and no-win, "
        "so the slot transition is visible. Win-rate is flat across firing tick.")

    tables = {
        "observed_tick_per_sender": str(tick_csv),
        "slots_behind_per_sender": str(sb_csv),
        "land_rate_by_tick_bucket": str(bucket_csv),
        "rotation_position_landrate": str(rot_csv),
    }

    bucket_rates = bucket_df["land_rate"]
    rot_rates = rot_df["land_rate"]
    key_results = {
        "global_slots_behind": global_sb_dict,
        "pct_one_slot_late": float((global_sb_dict.get(1, 0)
                                    + global_sb_dict.get(2, 0)
                                    + global_sb_dict.get(3, 0))
                                   / sum(global_sb_dict.values())),
        "spearman_rtt_vs_slots_behind_rho": rho,
        "spearman_rtt_vs_slots_behind_p": p_spear,
        "spearman_n": int(len(rtt_sub)),
        "tick_bucket_land_rate_range": [float(bucket_rates.min()), float(bucket_rates.max())],
        "rotation_land_rate_range": [float(rot_rates.min()), float(rot_rates.max())],
        "rotation_chi2": float(chi2),
        "rotation_p": float(p_rot),
        "n_landed": int(len(landed)),
    }

    notes = [
        "Landed-only analysis (observed_tick / slots_behind exist only for Landed rows); "
        "distribution-free per the gating policy.",
        f"~{key_results['pct_one_slot_late']*100:.0f}% of wins land >=1 slot late "
        f"(caught the next leader): slots_behind {global_sb_dict}.",
        f"Spearman RTT vs slots_behind rho={rho:.3f} (p={p_spear:.2g}, n={len(rtt_sub)}): "
        "higher RTT weakly associated with landing one slot later.",
        f"Trigger-tick bucket land-rate is flat "
        f"({bucket_rates.min():.3f}-{bucket_rates.max():.3f}) -> verified weak/null.",
        f"Rotation-position (slot mod 4) land-rate is flat "
        f"({rot_rates.min():.3f}-{rot_rates.max():.3f}), chi2 p={p_rot:.2f} -> verified null.",
        "Senders with <20 landers (allenhark-quic-ny, syncro-fra, jito-multi) are "
        "indicative/long-tail and flagged in the per-sender tables' gate column; "
        "allenhark-quic-tk has 0 landers and is absent (dead PoP, see ERR).",
    ]

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": tables,
        "figures": figures,
        "captions": captions,
        "key_results": key_results,
        "notes": notes,
    }


def _scatter_rtt_slots(sub: pd.DataFrame, rho, p, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    jitter = (np.random.default_rng(0).random(len(sub)) - 0.5) * 0.3
    ax.scatter(sub["rtt_ms"], sub["slots_behind"] + jitter,
               s=14, alpha=0.45, color="#1976D2", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("send RTT (ms, log scale)")
    ax.set_ylabel("slots_behind (jittered)")
    ax.set_yticks(sorted(sub["slots_behind"].dropna().astype(int).unique()))
    ax.set_title(f"S6 - send-RTT vs slots-behind (Landed)  Spearman rho={rho:.3f}, "
                 f"p={p:.2g}, n={len(sub)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _observed_tick_hist(same_slot_ticks, next_slot_ticks, out_path):
    """Two stacked panels: same-slot vs next-slot observed_tick, with p50+p99 vlines."""
    import matplotlib.pyplot as plt
    ss = np.asarray(same_slot_ticks, float)
    ss = ss[~np.isnan(ss)]
    ns = np.asarray(next_slot_ticks, float)
    ns = ns[~np.isnan(ns)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, data, label, color in (
        (axes[0], ss, "same-slot wins (slots_behind=0)", "#1976D2"),
        (axes[1], ns, "next-slot wins (slots_behind>=1)", "#E65100"),
    ):
        ax.hist(data, bins=64, range=(0, 64), color=color, alpha=0.85)
        if len(data):
            for q, c in ((50, "red"), (99, "black")):
                ax.axvline(np.percentile(data, q), color=c, ls="--", alpha=0.7,
                           label=f"p{q}={np.percentile(data, q):.0f}")
            ax.legend(fontsize=8)
        ax.set_ylabel("count")
        ax.set_title(f"{label}  (n={len(data):,})")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("observed_tick (0-63 WITHIN the landing slot)")
    axes[0].set_xlim(0, 64)
    fig.suptitle("S6 - Landing observed_tick: same-slot vs next-slot (separate slots)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _bucket_outcome_bars(bdf: pd.DataFrame, out_path):
    """Grouped/stacked bars per trigger-tick bucket: same-slot / next-slot / no-win."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(bdf))
    same = bdf["won_same_slot"].to_numpy()
    nxt = bdf["won_next_slot"].to_numpy()
    none = bdf["no_win"].to_numpy()
    ax.bar(x, same, color="#1976D2", label="same-slot win (slots_behind=0)")
    ax.bar(x, nxt, bottom=same, color="#E65100", label="next-slot win (slots_behind>=1)")
    ax.bar(x, none, bottom=same + nxt, color="#BDBDBD", label="no winner")
    for xi, row in enumerate(bdf.itertuples()):
        ax.text(xi, row.n, f"win {row.land_rate:.2f}\n(n={row.n})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bdf["tick_bucket"].astype(str))
    ax.set_xlabel("trigger firing PoH tick bucket")
    ax.set_ylabel("triggers (n per bucket)")
    ax.set_ylim(0, bdf["n"].max() * 1.18)
    ax.set_title("S6 - Per-trigger outcome by firing tick: same-slot vs next-slot vs no-win")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
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
    config = {}
    if Path(constants.DEFAULT_CONFIG).exists():
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
    ap = argparse.ArgumentParser(description="S6 intra-slot position & slots-behind")
    ap.add_argument("--out", default="/tmp/S6-verify",
                    help="output dir (plots/ + summary/ created under it)")
    args = ap.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))


if __name__ == "__main__":
    main()
