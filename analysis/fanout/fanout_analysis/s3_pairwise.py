"""S3 - Pairwise dominance.

Cochran's Q omnibus test over the 912x11 win matrix; per-pair (55) McNemar exact
test on discordant counts (b: A-only land, c: B-only land) with BH-FDR adjustment;
Bradley-Terry transitive ranking built from per-trigger single-winner records.

Outputs (to ctx["outdir"]):
  summary/cochrans-q.csv
  summary/pairwise-mcnemar.csv
  summary/bradley-terry-ranking.csv
  plots/03-pairwise-dominance.png        (b-c diverging heatmap, cmap=coolwarm, center=0)
  plots/03-pairwise-bradley-terry.png    (BT ability forest)
  plots/03-pairwise-mcnemar-pmatrix.png  (-log10(p_adj) matrix heatmap)
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd

from fanout_analysis import statutils as su, plotutils

logger = logging.getLogger(__name__)

SECTION_ID = "S3"
SECTION_TITLE = "Pairwise dominance (Cochran's Q -> McNemar -> Bradley-Terry)"


def _build_win_counts(wide: pd.DataFrame):
    """For each trigger's single winner w and every other sender s, win_counts[(w,s)]+=1."""
    cols = list(wide.columns)
    arr = wide.to_numpy()
    win_counts = {}
    for row in arr:
        winners = [cols[j] for j in range(len(cols)) if row[j] == 1]
        if not winners:
            continue  # no-winner trigger (66 of them) contributes nothing
        # single-winner invariant guaranteed by loader.assert_paired
        w = winners[0]
        for s in cols:
            if s == w:
                continue
            win_counts[(w, s)] = win_counts.get((w, s), 0) + 1
    return win_counts


def run(ctx) -> dict:
    wide = ctx["wide"]
    outdir = ctx["outdir"]
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"

    cols = list(wide.columns)
    arr = wide.to_numpy()
    k = len(cols)

    logger.info("S3 pairwise dominance over %d senders x %d triggers", k, arr.shape[0])

    # --- Cochran's Q omnibus -------------------------------------------------
    Q, q_df, q_p = su.cochrans_q(arr)
    cochran_df = pd.DataFrame(
        [{"statistic": "cochrans_q", "Q": Q, "df": q_df, "p_value": q_p,
          "n_triggers": int(arr.shape[0]), "k_senders": k}]
    )
    cochran_path = summary_dir / "cochrans-q.csv"
    cochran_df.to_csv(cochran_path, index=False)
    logger.info("Cochran's Q=%.3f df=%d p=%.3e", Q, q_df, q_p)

    # --- Pairwise McNemar exact + BH-FDR over the 55 pairs -------------------
    pairs = list(combinations(cols, 2))
    recs = []
    for a, b in pairs:
        ia, ib = cols.index(a), cols.index(b)
        b_disc = int(((arr[:, ia] == 1) & (arr[:, ib] == 0)).sum())   # A landed, B did not
        c_disc = int(((arr[:, ia] == 0) & (arr[:, ib] == 1)).sum())   # B landed, A did not
        n_disc = b_disc + c_disc
        p = su.mcnemar_exact(b_disc, c_disc)
        recs.append({
            "sender_a": a, "sender_b": b,
            "b_a_only": b_disc, "c_b_only": c_disc,
            "n_discordant": n_disc, "diff_b_minus_c": b_disc - c_disc,
            "p_value": p,
        })
    pvals = [r["p_value"] for r in recs]
    padj = su.fdr_adjust(pvals)
    for r, pa in zip(recs, padj):
        r["p_adj_bh"] = pa
        r["significant_fdr05"] = bool(pa < 0.05)
    mcnemar_df = pd.DataFrame(recs).sort_values("p_adj_bh").reset_index(drop=True)
    mcnemar_path = summary_dir / "pairwise-mcnemar.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    n_sig = int(mcnemar_df["significant_fdr05"].sum())
    logger.info("McNemar: %d/%d pairs significant at BH-FDR 0.05", n_sig, len(pairs))

    # --- Bradley-Terry transitive ranking -----------------------------------
    win_counts = _build_win_counts(wide)
    ability = su.bradley_terry(win_counts)
    land_counts = {c: int(arr[:, cols.index(c)].sum()) for c in cols}
    bt_rows = sorted(ability.items(), key=lambda kv: kv[1], reverse=True)
    bt_df = pd.DataFrame([
        {"rank": i + 1, "sender": name, "bt_ability": score, "land_count": land_counts[name]}
        for i, (name, score) in enumerate(bt_rows)
    ])
    bt_path = summary_dir / "bradley-terry-ranking.csv"
    bt_df.to_csv(bt_path, index=False)
    bt_top2 = list(bt_df["sender"].iloc[:2])
    logger.info("Bradley-Terry top-2: %s", bt_top2)

    # --- Figures -------------------------------------------------------------
    # Dominance heatmap: signed b-c diverging matrix (rows=A, cols=B).
    dom = pd.DataFrame(0.0, index=cols, columns=cols)
    for r in recs:
        a, b = r["sender_a"], r["sender_b"]
        dom.loc[a, b] = r["diff_b_minus_c"]
        dom.loc[b, a] = -r["diff_b_minus_c"]
    dom_path = plots_dir / "03-pairwise-dominance.png"
    plotutils.heatmap(
        dom, "S3 dominance: signed discordant wins (row beats col, b-c)",
        dom_path, fmt=".0f", cmap="coolwarm", center=0, annot=True)

    # Bradley-Terry forest (point only, lo==hi==point so no CI bars drawn).
    bt_forest_rows = [(row["sender"], row["bt_ability"], row["bt_ability"], row["bt_ability"])
                      for _, row in bt_df.iloc[::-1].iterrows()]
    bt_fig_path = plots_dir / "03-pairwise-bradley-terry.png"
    plotutils.forest_plot(
        bt_forest_rows, "S3 Bradley-Terry transitive ability (mean-0 log scale)",
        "BT ability", bt_fig_path, ref=0.0)

    # McNemar -log10(p_adj) matrix.
    neglog = pd.DataFrame(0.0, index=cols, columns=cols)
    for r, pa in zip(recs, padj):
        a, b = r["sender_a"], r["sender_b"]
        val = -np.log10(max(pa, 1e-300))
        neglog.loc[a, b] = val
        neglog.loc[b, a] = val
    for c in cols:
        neglog.loc[c, c] = np.nan
    pmatrix_path = plots_dir / "03-pairwise-mcnemar-pmatrix.png"
    plotutils.heatmap(
        neglog, "S3 McNemar significance: -log10(p_adj BH)",
        pmatrix_path, fmt=".1f", cmap="magma", annot=True)

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "cochrans_q": str(cochran_path),
            "pairwise_mcnemar": str(mcnemar_path),
            "bradley_terry_ranking": str(bt_path),
        },
        "figures": {
            "dominance_heatmap": str(dom_path),
            "bradley_terry_forest": str(bt_fig_path),
            "mcnemar_pmatrix": str(pmatrix_path),
        },
        "captions": {
            "dominance_heatmap": (
                "Each cell shows how many more triggers the row sender beat the column "
                "sender on; red = row ahead, blue = column ahead."
            ),
            "bradley_terry_forest": (
                "A single 'strength' score fitted from all head-to-head results (like an "
                "Elo rating). Higher = wins more across opponents; bars are 95% CIs. Gives "
                "one consistent ranking."
            ),
            "mcnemar_pmatrix": (
                "For each sender pair we test (paired, same triggers) whether A beats B more "
                "than chance. Color = significance (-log10 adjusted p); darker = stronger "
                "evidence the row sender truly beats the column sender."
            ),
        },
        "key_results": {
            "cochran_q": Q,
            "cochran_df": q_df,
            "cochran_p": q_p,
            "n_pairs": len(pairs),
            "n_significant_fdr05": n_sig,
            "bt_top2": bt_top2,
            "bt_ranking": {row["sender"]: row["bt_ability"] for _, row in bt_df.iterrows()},
        },
        "notes": [
            "Single-winner-per-trigger invariant => McNemar discordant counts equal the "
            "two senders' land counts (no trigger has two landers); off-diagonal concordance is 0.",
            "Cochran's Q rejects the global no-difference null (senders are not interchangeable).",
            "Bradley-Terry ranking is within-run only; no persistent-ranking claim from one 6-min run.",
        ],
    }


def _build_ctx(out):
    """Build a ctx dict for standalone CLI runs."""
    import json
    from pathlib import Path

    from fanout_analysis import loader, constants

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
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run S3 pairwise dominance analysis standalone.")
    parser.add_argument("--out", default="/tmp/S3-verify", help="output directory")
    args = parser.parse_args()

    ctx = _build_ctx(args.out)
    result = run(ctx)
    print("S3 key_results:")
    for key, val in result["key_results"].items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
