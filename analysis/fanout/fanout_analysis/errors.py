"""ERR -- Error catalog, UnknownPending decomposition, SendError taxonomy, 2 bugs.

REFRAMED around outcome_class: losing the race is EXPECTED, not an error. With 11 senders
racing for one win per trigger, the per-attempt win ceiling is ~9%, so the bulk of the 7549
UnknownPending rows are simply LOST_RACE (sent, another sender won) -- not defects. This
section separates that structural non-win from the two things that ARE problems:

  * NEVER_SENT   -- no tx ever left the client (send_at_ns==0, rtt_ms null). The dominant case
                    is allenhark-quic-tk's dead Tokyo QUIC PoP (912/912), plus throttle-skipped
                    rows on jito/syncro/blockrazor and 1 allenhark-ny row.
  * SERVER_ERROR -- 182 helius HTTP-500 responses mislabeled UnknownPending by the recorder.

It also enumerates the SendError taxonomy. For the reason chart the two provider rate-limit
buckets (jito rpc -32097 "Limit 1/s" and syncro rpc -32005, both HTTP 429) are MERGED into one
"provider_rate_limit (HTTP-429)" category; the raw rpc_err_code/message detail is preserved in
error-catalog.csv so nothing is lost.

Two confirmed recorder bugs are flagged with evidence rows + a recommended re-classification:
  Bug A  allenhark-quic-tk: 912 never_sent rows recorded UnknownPending (dead PoP -> NEVER_SENT).
  Bug B  helius_500: 182 rows http_status==500 recorded UnknownPending (server 500 -> SERVER_ERROR).

Emits per-sender denominator corrections for the *conditional* estimand (reclassify the
helius-500 rows to SendError, drop never_sent rows from the denominator) and renders three
figures (UnknownPending decomposition stacked bar, SendError reason matrix heatmap,
recorded-vs-corrected outcome side-by-side bars).

Distribution-free / descriptive only; this section makes no inferential rate claim.
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

SECTION_ID = "ERR"
SECTION_TITLE = "Error catalog, UnknownPending decomposition & confirmed bugs"
SECTION_NUM = "ERR"  # PNG prefix; ERR has no numeric section index in the suite


# --- bug A: allenhark-quic-tk dead PoP, all 912 never_sent recorded as UnknownPending
BUG_A_SENDER = "allenhark-quic-tk"
# --- bug B: helius HTTP-500 server errors recorded as UnknownPending
BUG_B_HTTP_STATUS = 500


def _is_send_error_outcome(df: pd.DataFrame) -> pd.Series:
    return df["final_outcome"] == "SendError"


def _build_error_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Long error catalog: one row per (category, sender, reason, http_status, rpc_err_code).

    Categories: SendError (full taxonomy), UnknownPending_never_sent, UnknownPending_sent_but_lost,
    plus the dedicated bug rows (BUG_A, BUG_B) carrying explicit recommended re-classification.
    """
    rows = []

    # --- SendError taxonomy: sender x send_error x http_status x rpc_err_code ---
    se = df[_is_send_error_outcome(df)]
    grp = se.groupby(
        ["sender_name", "send_error", "http_status", "rpc_err_code", "rpc_err_message"],
        dropna=False,
    ).size()
    for (sender, send_error, http_status, rpc_code, rpc_msg), count in grp.items():
        rows.append({
            "category": "SendError",
            "sender": sender,
            "count": int(count),
            "send_error": send_error,
            "http_status": http_status,
            "rpc_err_code": rpc_code,
            "rpc_err_message": rpc_msg,
            "classification_in_run": "SendError",
            "recommended_classification": "SendError",
            "affects_denominator": False,
        })

    # --- UnknownPending decomposition (excluding rows captured by the bug rows below) ---
    up = df[df["final_outcome"] == "UnknownPending"].copy()
    up["is_bug_a"] = up["sender_name"] == BUG_A_SENDER
    up["is_bug_b"] = up["helius_500"]

    # sent-but-lost: had a real send (not never_sent) and not a bug-B server-500 row.
    sent_lost = up[(~up["never_sent"]) & (~up["is_bug_b"]) & (~up["is_bug_a"])]
    for sender, count in sent_lost.groupby("sender_name").size().items():
        rows.append({
            "category": "UnknownPending_sent_but_lost",
            "sender": sender,
            "count": int(count),
            "send_error": None,
            "http_status": None,
            "rpc_err_code": None,
            "rpc_err_message": "sent (send_at_ns>0, ack received) but never observed landing",
            "classification_in_run": "UnknownPending",
            "recommended_classification": "UnknownPending",
            "affects_denominator": False,
        })

    # never_sent (genuine no-send: send_at_ns==0 / rtt null), excluding bug-A tk rows.
    never_sent = up[up["never_sent"] & (~up["is_bug_a"])]
    for sender, count in never_sent.groupby("sender_name").size().items():
        rows.append({
            "category": "UnknownPending_never_sent",
            "sender": sender,
            "count": int(count),
            "send_error": None,
            "http_status": None,
            "rpc_err_code": None,
            "rpc_err_message": "never sent (send_at_ns==0, rtt_ms null) -- excluded from conditional denom",
            "classification_in_run": "UnknownPending",
            "recommended_classification": "exclude (never_sent)",
            "affects_denominator": True,
        })

    # --- BUG A: allenhark-quic-tk dead PoP, 912 never_sent recorded UnknownPending ---
    bug_a = up[up["is_bug_a"]]
    rows.append({
        "category": "BUG_A_dead_pop_misclassified",
        "sender": BUG_A_SENDER,
        "count": int(len(bug_a)),
        "send_error": "never_sent",
        "http_status": None,
        "rpc_err_code": None,
        "rpc_err_message": "dead PoP (84.247.153.145): send never attempted, recorded UnknownPending",
        "classification_in_run": "UnknownPending",
        "recommended_classification": "SendError (exclude from conditional denom)",
        "affects_denominator": True,
    })

    # --- BUG B: helius HTTP-500 recorded UnknownPending (182 rows) ---
    bug_b = df[df["helius_500"]]
    for sender, count in bug_b.groupby("sender_name").size().items():
        rows.append({
            "category": "BUG_B_helius_500_misclassified",
            "sender": sender,
            "count": int(count),
            "send_error": None,
            "http_status": float(BUG_B_HTTP_STATUS),
            "rpc_err_code": None,
            "rpc_err_message": "HTTP 500 server error recorded as UnknownPending",
            "classification_in_run": "UnknownPending",
            "recommended_classification": "SendError (reclassify, exclude from conditional denom)",
            "affects_denominator": True,
        })

    # --- BUG C: rtt==0 (client-throttle, send recorded with 0 RTT) vs rtt==null (no-send) ---
    n_rtt_zero = int((df["rtt_ms"] == 0).sum())
    n_rtt_null = int(df["rtt_ms"].isna().sum())
    rows.append({
        "category": "BUG_C_rtt_zero_throttle",
        "sender": "<throttled SendError rows>",
        "count": n_rtt_zero,
        "send_error": "throttled_local",
        "http_status": None,
        "rpc_err_code": None,
        "rpc_err_message": "rtt_ms==0: client-throttled send attempt recorded with zero RTT (already SendError)",
        "classification_in_run": "SendError",
        "recommended_classification": "SendError (distinguish from never_sent by rtt_ms==0)",
        "affects_denominator": False,
    })
    rows.append({
        "category": "BUG_C_rtt_null_no_send",
        "sender": "<never_sent rows>",
        "count": n_rtt_null,
        "send_error": None,
        "http_status": None,
        "rpc_err_code": None,
        "rpc_err_message": "rtt_ms==null: genuine no-send (send_at_ns==0) == never_sent indicator",
        "classification_in_run": "UnknownPending/SendError mix",
        "recommended_classification": "exclude (never_sent), keyed by rtt_ms==null",
        "affects_denominator": True,
    })

    cols = ["category", "sender", "count", "send_error", "http_status", "rpc_err_code",
            "rpc_err_message", "classification_in_run", "recommended_classification",
            "affects_denominator"]
    return pd.DataFrame(rows, columns=cols)


def _build_denominator_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sender conditional-denominator corrections (11 rows).

    raw_unknown_pending           = UnknownPending count for the sender
    reclassified_to_senderror     = helius-500 rows reclassified to SendError (bug B)
    never_sent_excluded           = UnknownPending rows that never sent (bugs A & C) dropped
    corrected_conditional_denominator = Landed + raw_unknown_pending
                                        - reclassified_to_senderror - never_sent_excluded
    """
    rows = []
    for sender, sub in df.groupby("sender_name"):
        landed = int((sub["final_outcome"] == "Landed").sum())
        raw_up = int((sub["final_outcome"] == "UnknownPending").sum())
        raw_denom = landed + raw_up  # raw conditional denominator (excludes SendError)
        reclassified = int(sub["helius_500"].sum())  # bug B -> SendError
        never_sent_up = int(((sub["final_outcome"] == "UnknownPending") & sub["never_sent"]).sum())
        corrected = raw_denom - reclassified - never_sent_up
        rows.append({
            "sender": sender,
            "landed": landed,
            "raw_unknown_pending": raw_up,
            "raw_conditional_denominator": raw_denom,
            "reclassified_to_senderror": reclassified,
            "never_sent_excluded": never_sent_up,
            "corrected_conditional_denominator": corrected,
        })
    out = pd.DataFrame(rows).sort_values("sender").reset_index(drop=True)
    return out


def _plot_unknown_decomposition(df: pd.DataFrame, out_path: Path) -> Path:
    """Stacked bar per sender: split UnknownPending via outcome_class into the EXPECTED
    non-win (LOST_RACE) vs the two real problems (NEVER_SENT, SERVER_ERROR)."""
    up = df[df["final_outcome"] == "UnknownPending"]
    kinds = ["LOST_RACE", "NEVER_SENT", "SERVER_ERROR"]
    pivot = (up.groupby(["sender_name", "outcome_class"]).size()
               .unstack(fill_value=0))
    for col in kinds:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[kinds].sort_index()
    return plotutils.stacked_bar(
        pivot, kinds,
        title="UnknownPending decomposition: LOST_RACE (expected) vs NEVER_SENT / SERVER_ERROR (problems)",
        xlabel="rows", out_path=out_path, horizontal=True, pct=False)


# merged chart category for the two provider HTTP-429 rate-limit buckets
# (jito rpc -32097 "Limit 1/s" + syncro rpc -32005). Raw rpc detail stays in the catalog CSV.
PROVIDER_RATE_LIMIT_REASON = "provider_rate_limit (HTTP-429)"


def _chart_reason(row) -> str:
    """Collapse the two provider rate-limit rpc strings into one HTTP-429 category."""
    if row["http_status"] == 429:
        return PROVIDER_RATE_LIMIT_REASON
    return row["send_error"] if pd.notna(row["send_error"]) else "(none)"


def _plot_senderror_matrix(df: pd.DataFrame, out_path: Path) -> Path:
    """Heatmap: sender x SendError reason counts. The two provider rate-limit rpc codes
    (jito -32097, syncro -32005) are merged into one 'provider_rate_limit (HTTP-429)' column."""
    se = df[_is_send_error_outcome(df)].copy()
    se["reason"] = se.apply(_chart_reason, axis=1)
    matrix = (se.groupby(["sender_name", "reason"]).size()
                .unstack(fill_value=0).sort_index())
    return plotutils.heatmap(
        matrix, title="SendError reason matrix (sender x reason; provider rate-limits merged)",
        out_path=out_path, fmt="d", cmap="Reds", annot=True)


def _plot_recorded_vs_corrected(corrections: pd.DataFrame, out_path: Path) -> Path:
    """Side-by-side (grouped, not stacked) bars: raw vs corrected conditional denominator
    per sender, so the shift from the two bug fixes is visible per sender."""
    plot_df = corrections.set_index("sender")[
        ["raw_conditional_denominator", "corrected_conditional_denominator"]].sort_index()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(plot_df))))
    plot_df.plot(kind="barh", ax=ax, color=["#90A4AE", "#1976D2"])
    ax.set_xlabel("denominator")
    ax.set_title("Conditional denominator: recorded (raw) vs corrected")
    ax.grid(alpha=0.3)
    ax.legend(["recorded (raw)", "corrected"], fontsize=8)
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

    logger.info("ERR: building error catalog over %d rows", len(df))

    catalog = _build_error_catalog(df)
    corrections = _build_denominator_corrections(df)

    catalog_csv = summary_dir / "error-catalog.csv"
    corrections_csv = summary_dir / "denominator-corrections.csv"
    catalog.to_csv(catalog_csv, index=False)
    corrections.to_csv(corrections_csv, index=False)

    fig_decomp = plots_dir / "ERR-unknownpending-decomposition.png"
    fig_matrix = plots_dir / "ERR-senderror-reason-matrix.png"
    fig_recvscorr = plots_dir / "ERR-recorded-vs-corrected.png"
    _plot_unknown_decomposition(df, fig_decomp)
    _plot_senderror_matrix(df, fig_matrix)
    _plot_recorded_vs_corrected(corrections, fig_recvscorr)

    # --- key results / golden numbers ---
    bug_a_row = catalog[(catalog["category"] == "BUG_A_dead_pop_misclassified")
                        & (catalog["sender"] == BUG_A_SENDER)]
    bug_a_count = int(bug_a_row["count"].iloc[0]) if len(bug_a_row) else 0
    bug_b_count = int(catalog.loc[catalog["category"] == "BUG_B_helius_500_misclassified",
                                  "count"].sum())
    n_send_error = int(_is_send_error_outcome(df).sum())
    n_unknown = int((df["final_outcome"] == "UnknownPending").sum())
    n_never_sent_up = int(((df["final_outcome"] == "UnknownPending") & df["never_sent"]).sum())
    n_sent_but_lost = n_unknown - n_never_sent_up

    n_lost_race = int((df["outcome_class"] == "LOST_RACE").sum())
    key_results = {
        "n_send_error": n_send_error,
        "n_unknown_pending": n_unknown,
        "unknown_lost_race_expected": n_lost_race,
        "unknown_sent_but_lost": n_sent_but_lost,
        "unknown_never_sent": n_never_sent_up,
        "unknown_server_error": bug_b_count,
        "bug_A_allenhark_tk_never_sent": bug_a_count,
        "bug_B_helius_500_count": bug_b_count,
        "bug_C_rtt_zero_throttle": int((df["rtt_ms"] == 0).sum()),
        "bug_C_rtt_null_no_send": int(df["rtt_ms"].isna().sum()),
        "denominator_corrections_rows": int(len(corrections)),
    }

    captions = {
        "unknownpending_decomposition":
            "Splits each sender's UnknownPending rows by outcome_class. LOST_RACE (sent, but "
            "another of the 11 senders won the trigger) is the EXPECTED structural outcome, not a "
            "defect. NEVER_SENT (send_at_ns==0: no tx ever left the client; allenhark-tk = dead "
            "Tokyo QUIC PoP, 912/912) and SERVER_ERROR (helius HTTP-500) are the only real problems.",
        "senderror_reason_matrix":
            "SendError reasons per sender. The two provider rate-limit rpc codes (jito -32097 "
            "\"Limit 1/s\", syncro -32005) are merged into one provider_rate_limit (HTTP-429) "
            "category; full rpc_err_code/message detail is kept in error-catalog.csv.",
        "recorded_vs_corrected":
            "Left = outcomes as the recorder labeled them; right = after fixing two known bugs "
            "(allenhark-tk never-sent, 182 helius HTTP-500 mislabeled as pending). Shows how the "
            "corrections shift each sender's denominator.",
    }

    notes = [
        "Descriptive only: this section makes no inferential rate claim.",
        "Reframe: with 11 senders racing for 1 win/trigger, LOST_RACE (sent but another sender "
        "won) is expected, not an error -- 6249 of the 7549 UnknownPending rows are LOST_RACE.",
        "Bug A: allenhark-quic-tk dead Tokyo QUIC PoP (84.247.153.145) -- all 912 rows never_sent "
        "(send_at_ns==0), recorded UnknownPending; recommend NEVER_SENT and exclude from denom.",
        "Bug B: 182 helius HTTP-500 server errors recorded UnknownPending; recommend SERVER_ERROR.",
        "Bug C: rtt_ms==0 marks client-throttled send attempts (already SendError); "
        "rtt_ms==null marks genuine never_sent rows -- the two must not be conflated.",
        "never_sent == (rtt_ms is null) == (send_at_ns==0) for all rows.",
    ]

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "error_catalog": str(catalog_csv),
            "denominator_corrections": str(corrections_csv),
        },
        "figures": {
            "unknownpending_decomposition": str(fig_decomp),
            "senderror_reason_matrix": str(fig_matrix),
            "recorded_vs_corrected": str(fig_recvscorr),
        },
        "captions": captions,
        "key_results": key_results,
        "notes": notes,
    }


def _build_ctx(out_dir: Path) -> dict:
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Run the ERR error-catalog section.")
    parser.add_argument("--out", default="/tmp/ERR-verify", help="output directory")
    args = parser.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2, default=str))


if __name__ == "__main__":
    main()
