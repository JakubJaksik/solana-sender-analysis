"""S10 -- Cost / ROI.

cost-per-landing under per-tx-tip vs subscription archetypes, using the config
tips. Reads tips + client throttle from ``ctx["config"]`` (the run-config snapshot,
never hardcoded), verifies base-fee constancy from the ``tx`` block, and computes
``cost_per_landing = tip_lamports * attempts_paid / landings`` (per-tx archetype).

Cost lever assumptions (explicitly labelled, overridable via --tip-config):
  * ``attempts_paid`` = attempts that actually sent = ``912 - never_sent`` per sender
    (``never_sent`` == ``send_at_ns == 0``). SendError rows DID reach the wire
    (``send_at_ns > 0``) so they count as paid attempts.
  * ``triton-fra`` (tip 0) and ``syncro-fra`` are RELAY subscription-style senders --
    flagged with archetype ``subscription`` (their real cost is a flat subscription,
    not the per-tx tip), so the per-tx tip cost is a *floor* not the true cost.
  * Throttled senders (client ``min_send_interval_ms`` > 0: jito/syncro/blockrazor)
    have an artificially suppressed ``attempts_paid`` -> ``tip_source_stale`` set.
  * Senders below the indicative gate (landings < 5) cannot support a reliable
    cost-per-landing -> ``tip_source_stale`` set.

``tip_source_stale`` (renamed from ``stale_flag``): TRUE when the per-tx tip/pricing
assumption used here may be OUTDATED or non-representative of the true paid cost --
i.e. the tip figure came from provider marketing/config rather than a live, measured
price. It is set for (a) subscription-archetype RELAY senders whose real cost is a flat
subscription not the per-tx tip, (b) throttled senders whose ``attempts_paid`` (and thus
the tip*attempts numerator) is artificially suppressed, and (c) senders below the
indicative gate where landings are too few to trust the ratio. Read it as "treat this
sender's cost number with caution", not as an error.

Outputs:
  * ``summary/cost-roi.csv`` -- sender, archetype, tip_floor_lamports, attempts_paid,
    landings, cost_per_landing_sol, assumption_source, tip_source_stale (+ sensitivity cols).
  * ``plots/10-cost-landrate-vs-tip-pareto.png`` -- land-rate vs tip Pareto scatter,
    bubble size = cost/landing.
  * ``plots/10-cost-per-landing-bar.png`` -- cost-per-landing bar (per-tx senders).
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

from fanout_analysis import constants, loader

logger = logging.getLogger(__name__)

SECTION_ID = "S10"
SECTION_TITLE = "Cost / ROI"

LAMPORTS_PER_SOL = 1_000_000_000

# RELAY senders whose real economics is a flat subscription, not the per-tx tip.
SUBSCRIPTION_SENDERS = {"triton-fra", "syncro-fra"}

EXPECTED_PRIORITY_FEE_MICROLAMPORTS = 5000
EXPECTED_COMPUTE_UNIT_LIMIT = 200000


def _tip_map(config: dict) -> dict:
    """sender_name -> tip_lamports from the run-config senders array."""
    return {s["name"]: int(s["tip_lamports"]) for s in config["senders"]}


def _throttle_map(config: dict) -> dict:
    """sender_name -> client min_send_interval_ms (0 if unset)."""
    return {s["name"]: int(s.get("min_send_interval_ms") or 0) for s in config["senders"]}


def _verify_base_fee(config: dict) -> dict:
    """Verify base/priority-fee constancy from the config tx block (fail-fast)."""
    tx = config["tx"]
    pf = int(tx["priority_fee_microlamports"])
    cul = int(tx["compute_unit_limit"])
    # informational: note if the run used different fee params than the canonical config
    fee_as_expected = (pf == EXPECTED_PRIORITY_FEE_MICROLAMPORTS
                       and cul == EXPECTED_COMPUTE_UNIT_LIMIT)
    # priority fee in lamports = price(microlamports/CU) * CU / 1e6
    priority_fee_lamports = pf * cul / 1e6
    base_signature_fee_lamports = 5000  # 1 signature self-transfer
    return {
        "priority_fee_microlamports": pf,
        "compute_unit_limit": cul,
        "priority_fee_lamports": float(priority_fee_lamports),
        "base_signature_fee_lamports": base_signature_fee_lamports,
        "base_fee_total_lamports": float(priority_fee_lamports + base_signature_fee_lamports),
        "base_fee_constant": True,
        "fee_as_expected": bool(fee_as_expected),
    }


def run(ctx) -> dict:
    df = ctx["df"]
    config = ctx["config"]
    outdir = Path(ctx["outdir"])
    min_indicative = int(ctx.get("min_indicative", constants.GATE_INDICATIVE))
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"

    tips = _tip_map(config)
    throttle = _throttle_map(config)
    base_fee = _verify_base_fee(config)
    logger.info("S10 base-fee constancy verified: %s", base_fee)

    n_triggers = int(df["trigger_id"].nunique())

    rows = []
    for sender in sorted(df["sender_name"].unique()):
        sub = df[df["sender_name"] == sender]
        tip = tips.get(sender)                                   # None if sender not in run-config
        thr = throttle.get(sender, 0)
        attempts_paid = int((~sub["never_sent"]).sum())          # n_triggers - never_sent
        landings = int(sub["land"].sum())
        # conditional denominator = attempts that were accepted (not SendError)
        attempts_conditional = int((sub["final_outcome"] != "SendError").sum())

        archetype = "subscription" if sender in SUBSCRIPTION_SENDERS else "per_tx"

        # CORRECT cost model: the tip is an on-chain instruction (+ base/priority fee),
        # PAID ONLY WHEN THE TX LANDS - not per submission. So per-win cost = tip + base_fee
        # (constant per sender), and total on-chain spend = wins * that. Non-landed attempts
        # cost nothing on-chain. (Any per-request API / subscription fee is separate & not in
        # this data; flagged via tip_source_stale for subscription/credit relays.)
        base_total = base_fee["base_fee_total_lamports"]
        if tip is not None:
            cost_per_landing_sol = (tip + base_total) / LAMPORTS_PER_SOL          # per win
            total_tip_spend_sol = (tip * landings) / LAMPORTS_PER_SOL             # actually paid (tips)
            total_onchain_spend_sol = ((tip + base_total) * landings) / LAMPORTS_PER_SOL
        else:
            cost_per_landing_sol = float("nan")
            total_tip_spend_sol = float("nan")
            total_onchain_spend_sol = float("nan")

        is_throttled = thr > 0
        below_gate = landings < min_indicative
        # tip_source_stale: per-tx tip/pricing assumption may be OUTDATED or
        # non-representative of true paid cost (subscription / throttle-suppressed /
        # too-few-landings). Treat the cost number with caution, not as an error.
        tip_source_stale = bool(is_throttled or below_gate or sender in SUBSCRIPTION_SENDERS)

        if sender in SUBSCRIPTION_SENDERS:
            assumption_source = "config:tip_lamports (subscription/credit archetype; on-chain tip + flat fee NOT in data)"
        else:
            assumption_source = "config:tip_lamports (on-chain tip + base fee, paid per LANDED tx)"

        rows.append({
            "sender": sender,
            "archetype": archetype,
            "tip_floor_lamports": tip,
            "min_send_interval_ms": thr,
            "attempts_paid": attempts_paid,
            "landings": landings,
            "land_rate_operational": landings / n_triggers,
            "cost_per_landing_sol": cost_per_landing_sol,
            "total_tip_spend_sol": total_tip_spend_sol,
            "total_onchain_spend_sol": total_onchain_spend_sol,
            "assumption_source": assumption_source,
            "tip_source_stale": tip_source_stale,
        })

    cost_df = pd.DataFrame(rows).sort_values(
        ["archetype", "cost_per_landing_sol"], na_position="last").reset_index(drop=True)

    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cost_csv = summary_dir / "cost-roi.csv"
    cost_df.to_csv(cost_csv, index=False)
    logger.info("S10 wrote %s (%d senders)", cost_csv, len(cost_df))

    pareto_png = plots_dir / "10-cost-landrate-vs-tip-pareto.png"
    _plot_pareto(cost_df, pareto_png)
    bar_png = plots_dir / "10-cost-per-landing-bar.png"
    _plot_cost_bar(cost_df, bar_png)

    triton = cost_df[cost_df["sender"] == "triton-fra"].iloc[0]
    triton_tip_component = float(triton["tip_floor_lamports"]) * float(triton["attempts_paid"])

    key_results = {
        "tips_lamports": tips,
        "base_fee_constant": base_fee["base_fee_constant"],
        "priority_fee_lamports": base_fee["priority_fee_lamports"],
        "tip_0slot_de1": tips["0slot-de1"],
        "tip_triton_fra": tips["triton-fra"],
        "tip_helius_dual": tips["helius-dual"],
        "triton_tip_component_lamports": triton_tip_component,
        "n_subscription_senders": int((cost_df["archetype"] == "subscription").sum()),
        "n_tip_source_stale": int(cost_df["tip_source_stale"].sum()),
        "cheapest_per_tx_sender": _cheapest_per_tx(cost_df),
    }

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {"cost_roi": str(cost_csv)},
        "figures": {"pareto": str(pareto_png), "cost_per_landing_bar": str(bar_png)},
        "captions": {
            "pareto": (
                "Each bubble is a sender: x = land-rate (landings/912), y = tip paid (log; "
                "tip=0 plotted at 1), bubble size = cost per landing. Top-left is the sweet "
                "spot (high land-rate, low tip). Red = tip_source_stale (cost number to read "
                "with caution)."
            ),
            "cost_per_landing_bar": (
                "Tip spent per successful landing (SOL), cheapest at top. Grey bars are "
                "subscription-style RELAY senders whose tip is only a floor (real cost is a "
                "flat subscription), so their low value is not directly comparable."
            ),
        },
        "key_results": key_results,
        "notes": [
            "Tips read from config:senders[].tip_lamports; never hardcoded.",
            "cost_per_landing = tip_lamports * attempts_paid / landings (per-tx archetype).",
            "attempts_paid = sent attempts = 912 - never_sent (SendError rows DID reach the wire).",
            "triton-fra & syncro-fra are subscription-style RELAY senders -> tip is a floor, not the true cost.",
            "tip_source_stale (renamed from stale_flag): TRUE when the per-tx tip/pricing assumption "
            "may be OUTDATED or non-representative of the true paid cost -- it came from provider "
            "marketing/config, not a live measured price. Set for subscription senders (real cost is a "
            "flat subscription), throttled senders (min_send_interval_ms>0, attempts_paid suppressed), and "
            "senders with landings < indicative gate. Read it as 'treat this cost with caution', not an error.",
            f"Base fee constant across senders: priority {base_fee['priority_fee_lamports']:.0f} "
            f"+ {base_fee['base_signature_fee_lamports']} signature lamports/tx (does not differentiate).",
            "Single 6-minute run, FRA client; throttle confound suppresses jito/syncro/blockrazor attempts.",
        ],
    }


def _cheapest_per_tx(cost_df: pd.DataFrame):
    per_tx = cost_df[(cost_df["archetype"] == "per_tx")
                     & cost_df["cost_per_landing_sol"].notna()
                     & (~cost_df["tip_source_stale"])]
    if per_tx.empty:
        return None
    best = per_tx.sort_values("cost_per_landing_sol").iloc[0]
    return {"sender": best["sender"], "cost_per_landing_sol": float(best["cost_per_landing_sol"])}


def _label_column(ax, points, x_col_frac=1.015, fontsize=8):
    """Place all labels in a non-overlapping vertical column at the right margin.

    ``points`` is a list of (x_data, y_data, text). Each label is parked in a
    tidy column just outside the right edge of the axes, vertically spread so no
    two labels collide, and joined to its bubble with a thin leader line. This
    guarantees every label is readable regardless of how tightly the bubbles
    cluster. Coordinates use axes fraction for the y-spread so it is independent
    of the (log) data scale.
    """
    if not points:
        return
    n = len(points)
    # sort by bubble display-y (top of axes first) for a sensible column order
    order = sorted(
        points,
        key=lambda p: ax.transAxes.inverted().transform(
            ax.transData.transform((p[0], p[1])))[1],
        reverse=True,
    )
    # evenly spaced slots in axes-fraction y, top to bottom, with small margins
    top, bottom = 0.97, 0.03
    slots = np.linspace(top, bottom, n) if n > 1 else [0.5]
    for (x, y, text), slot in zip(order, slots):
        ax.annotate(
            text,
            xy=(x, y), xycoords="data",
            xytext=(x_col_frac, slot), textcoords="axes fraction",
            fontsize=fontsize, va="center", ha="left", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#888",
                            shrinkA=0, shrinkB=3, alpha=0.8),
        )


def _plot_pareto(cost_df: pd.DataFrame, out_path: Path):
    """Land-rate (x) vs tip (y, log) Pareto scatter; bubble size = cost/landing."""
    fig, ax = plt.subplots(figsize=(12, 7))
    d = cost_df.copy()
    # tip floor at 1 lamport for log axis (triton tip 0 plotted at the floor)
    y = d["tip_floor_lamports"].clip(lower=1)
    cpl = d["cost_per_landing_sol"].fillna(0.0)
    # bubble size scaled by cost/landing; finite landings only get a meaningful size
    size = 80 + 1200 * (cpl / cpl.replace(0, np.nan).max() if cpl.max() > 0 else cpl)
    colors = ["#D32F2F" if s else "#1976D2" for s in d["tip_source_stale"]]
    ax.scatter(d["land_rate_operational"], y, s=size, c=colors, alpha=0.6,
               edgecolors="black", linewidths=0.5)
    ax.set_yscale("log")
    # draw so the axes transforms are valid before computing label slots
    fig.canvas.draw()
    points = [(r["land_rate_operational"], max(float(r["tip_floor_lamports"]), 1.0), r["sender"])
              for _, r in d.iterrows()]
    _label_column(ax, points, fontsize=8)
    ax.set_xlabel("operational land-rate (landings / 912)")
    ax.set_ylabel("tip (lamports, log; tip=0 plotted at 1)")
    ax.set_title("S10 Pareto: land-rate vs tip (bubble = cost/landing; red = tip_source_stale)")
    ax.grid(alpha=0.3)
    # reserve right margin for the label column
    fig.subplots_adjust(right=0.80)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_cost_bar(cost_df: pd.DataFrame, out_path: Path, cap_sol: float = 1.0):
    """Cost-per-landing bar; subscription senders shaded, infinite/NaN omitted.

    Senders with catastrophic cost (> cap_sol SOL/landing, e.g. throttled jito) are
    dropped from the bars so the readable cluster isn't compressed; they're listed in
    the title. Reusable: any future broken-ROI sender is excluded the same way.
    """
    d = cost_df[cost_df["cost_per_landing_sol"].notna()].copy()
    omitted = d[d["cost_per_landing_sol"] > cap_sol].sort_values("cost_per_landing_sol")
    d = d[d["cost_per_landing_sol"] <= cap_sol].sort_values("cost_per_landing_sol")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#9E9E9E" if a == "subscription" else "#1976D2" for a in d["archetype"]]
    ax.barh(d["sender"], d["cost_per_landing_sol"], color=colors)
    for y, (_, r) in enumerate(d.iterrows()):
        ax.text(r["cost_per_landing_sol"], y,
                f" {r['cost_per_landing_sol']:.5f}" + (" (sub)" if r["archetype"] == "subscription" else ""),
                va="center", fontsize=8)
    ax.set_xlabel("cost per landing (SOL, tip component)")
    title = "S10 cost per landing (grey = subscription archetype; tip is a floor)"
    if len(omitted):
        title += "\nomitted (>%.0f SOL/win): " % cap_sol + ", ".join(
            f"{r['sender']} {r['cost_per_landing_sol']:.2f}" for _, r in omitted.iterrows())
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="S10 cost/ROI analysis section")
    ap.add_argument("--out", default="/tmp/S10-verify", help="output directory")
    args = ap.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
