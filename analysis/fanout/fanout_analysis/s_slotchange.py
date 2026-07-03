"""S6B - Slot-change / leader-change between send and landing.

Answers the user's explicit ask: "gdzie zestawienie sytuacji gdzie miedzy wyslaniem
a wejsciem zmienil sie slot/leader" - a focused summary of how often a winning tx
landed in a LATER slot than the one targeted, and whether that later slot was still
the SAME validator or a DIFFERENT one.

KEY FACT: a Solana leader holds 4 CONSECUTIVE slots, so slots_behind>=1 (landed one
slot late) does NOT mean a different leader. The true "leader changed" event is
landed_next_leader (observed_leader_identity != intended leader_identity).

Among the 846 wins: same_slot 586 (69.3%), next_slot 260 (30.7%) landed >=1 slot late,
but the leader actually CHANGED only 63 times (7.4%). So ~76% of "late" landings stayed
with the intended leader (next slot of the same 4-slot rotation).

Landed-only (slot-change is defined only for wins). Per-sender rates gated:
n>=GATE_INFERENTIAL inferential, GATE_INDICATIVE..19 indicative, <GATE_INDICATIVE long_tail.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fanout_analysis import constants, loader, plotutils

SECTION_ID = "SC"
SECTION_TITLE = "Slot-change / leader-change between send and landing"
PNG_PREFIX = "06b"


def _gate(n):
    if n >= constants.GATE_INFERENTIAL:
        return "inferential"
    if n >= constants.GATE_INDICATIVE:
        return "indicative"
    return "long_tail"


def _name_or_id(name, identity):
    """Prefer human leader name; fall back to truncated identity when metadata is missing."""
    if isinstance(name, str) and name:
        return name
    if isinstance(identity, str) and identity:
        return identity[:8] + "..."
    return "(unknown)"


def _slot_change_per_sender(landed: pd.DataFrame) -> pd.DataFrame:
    """Per-sender landed / same_slot / next_slot(same leader) / next_leader counts + pct."""
    rows = []
    for sender, sub in landed.groupby("sender_name"):
        n = int(len(sub))
        same = int(sub["same_slot"].sum())
        next_slot = int(sub["landed_next_slot"].sum())
        next_leader = int(sub["landed_next_leader"].sum())
        # of the next_slot landings, how many stayed with the same leader
        next_slot_same_leader = next_slot - next_leader
        rows.append({
            "sender_name": sender,
            "pop": constants.SENDER_META[sender][2],
            "protocol_class": constants.PROTOCOL_OF.get(sender),
            "n_landed": n,
            "same_slot": same,
            "next_slot": next_slot,
            "next_slot_same_leader": next_slot_same_leader,
            "next_leader": next_leader,
            "pct_same_slot": same / n * 100 if n else float("nan"),
            "pct_next_slot": next_slot / n * 100 if n else float("nan"),
            "pct_next_leader": next_leader / n * 100 if n else float("nan"),
            "gate": _gate(n),
        })
    return pd.DataFrame(rows).sort_values("n_landed", ascending=False).reset_index(drop=True)


def _leader_change_detail(landed: pd.DataFrame) -> pd.DataFrame:
    """One row per landing where the leader actually changed (landed_next_leader)."""
    nl = landed[landed["landed_next_leader"]].copy()
    rows = []
    for r in nl.itertuples():
        rows.append({
            "trigger_id": r.trigger_id,
            "intended_slot": int(r.slot),
            "intended_leader": _name_or_id(r.leader_name, r.leader_identity),
            "intended_continent": r.sv_continent,
            "observed_slot": int(r.observed_slot),
            "observed_leader": _name_or_id(r.observed_leader_name, r.observed_leader_identity),
            "observed_continent": r.observed_leader_continent,
            "sender_name": r.sender_name,
            "pop": constants.SENDER_META[r.sender_name][2],
            "slots_behind": int(r.slots_behind),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["sender_name", "intended_slot"]).reset_index(drop=True)
    return out


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summ = outdir / "summary"
    plots = outdir / "plots"
    summ.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    landed = df[df["land"] == 1].copy()

    # --- global tallies (golden oracle) ---
    g_same = int(landed["same_slot"].sum())
    g_next_slot = int(landed["landed_next_slot"].sum())
    g_next_leader = int(landed["landed_next_leader"].sum())
    g_next_slot_same_leader = g_next_slot - g_next_leader
    n_landed = int(len(landed))

    # --- tables ---
    sc_df = _slot_change_per_sender(landed)
    detail_df = _leader_change_detail(landed)

    sc_csv = summ / "slot-change-summary.csv"
    detail_csv = summ / "leader-change-detail.csv"
    sc_df.to_csv(sc_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    # --- figures ---
    figures = {}
    captions = {}

    # (c) stacked bar per sender: same-slot / next-slot(same leader) / next-leader shares
    sc_plot = sc_df.set_index("sender_name")[
        ["same_slot", "next_slot_same_leader", "next_leader"]].copy()
    sc_plot.columns = ["same slot", "next slot (same leader)", "next leader (changed)"]
    fig_per_sender = plots / f"{PNG_PREFIX}-SC-slotchange-per-sender.png"
    plotutils.stacked_bar(
        sc_plot, cols=list(sc_plot.columns),
        title="SC - Slot/leader change per sender (Landed): same slot vs next slot vs leader changed",
        xlabel="landed count", out_path=fig_per_sender, horizontal=True, pct=False)
    figures["slotchange_per_sender"] = str(fig_per_sender)
    captions["slotchange_per_sender"] = (
        "Per sender, how winning txs landed: same slot, one slot late but same validator "
        "(a leader owns 4 consecutive slots), or in a slot owned by a different leader. "
        "Most 'late' landings stay with the intended leader.")

    # (d) global small bar: 586 same / 197 next-slot-same-leader / 63 next-leader
    fig_global = plots / f"{PNG_PREFIX}-SC-global-slot-vs-leader.png"
    _global_bar(g_same, g_next_slot_same_leader, g_next_leader, n_landed, fig_global)
    figures["global_slot_vs_leader"] = str(fig_global)
    captions["global_slot_vs_leader"] = (
        f"Across all {n_landed} wins: {g_same} landed in the targeted slot, "
        f"{g_next_slot_same_leader} one slot late but with the SAME leader, and only "
        f"{g_next_leader} ({g_next_leader/n_landed*100:.1f}%) in a slot held by a different leader. "
        "Landing one slot late rarely means losing the intended validator.")

    tables = {
        "slot_change_summary": str(sc_csv),
        "leader_change_detail": str(detail_csv),
    }

    # per-sender next_slot pct keyed by pop label (matches spec oracle)
    next_slot_pct_by_pop = {
        r.pop: round(float(r.pct_next_slot), 1) for r in sc_df.itertuples()
    }

    key_results = {
        "n_landed": n_landed,
        "same_slot": g_same,
        "next_slot": g_next_slot,
        "next_slot_same_leader": g_next_slot_same_leader,
        "next_leader": g_next_leader,
        "pct_same_slot": round(g_same / n_landed * 100, 1),
        "pct_next_slot": round(g_next_slot / n_landed * 100, 1),
        "pct_next_leader": round(g_next_leader / n_landed * 100, 1),
        "next_slot_pct_by_pop": next_slot_pct_by_pop,
        "leader_change_rows": int(len(detail_df)),
    }

    notes = [
        "Landed-only: slot/leader change is defined only for the winning attempt per trigger.",
        "A leader holds 4 CONSECUTIVE slots, so slots_behind>=1 (next_slot) does NOT imply a "
        f"different leader: of {g_next_slot} next-slot landings only {g_next_leader} actually "
        f"changed leader; {g_next_slot_same_leader} stayed with the intended validator.",
        f"Global: same_slot {g_same} ({g_same/n_landed*100:.1f}%), next_slot {g_next_slot} "
        f"({g_next_slot/n_landed*100:.1f}%), leader changed {g_next_leader} "
        f"({g_next_leader/n_landed*100:.1f}%).",
        "next_slot share varies by sender PoP: 0slot-de1 44.7%, allenhark-fra 15.7%, "
        "allenhark-ny 85.7% (the NY PoP lands one slot late far more often - long network path).",
        "Per-sender rates gated; senders with <20 landers (allenhark-ny, syncro-fra, jito-multi) "
        "are indicative/long_tail and flagged in the gate column.",
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


def _global_bar(same, next_slot_same_leader, next_leader, n_landed, out_path):
    import matplotlib.pyplot as plt
    cats = ["same slot", "next slot\n(same leader)", "next leader\n(changed)"]
    vals = [same, next_slot_same_leader, next_leader]
    colors = ["#2E7D32", "#1976D2", "#C62828"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(cats, vals, color=colors, alpha=0.88)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/n_landed*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("landed count")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(f"SC - Where the {n_landed} wins landed: slot change vs leader change")
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
    ap = argparse.ArgumentParser(description="S6B slot-change / leader-change summary")
    ap.add_argument("--out", default="/tmp/SC-verify",
                    help="output dir (plots/ + summary/ created under it)")
    args = ap.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))


if __name__ == "__main__":
    main()
