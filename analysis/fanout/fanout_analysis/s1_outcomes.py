"""S1 - Sender outcome profile & SendError taxonomy.

Per-sender outcome mix shown over the 6 outcome_class buckets (WON / LOST_RACE /
NEVER_SENT / THROTTLED_LOCAL / PROVIDER_REJECTED / SERVER_ERROR) with Wilson CIs
on the operational and conditional estimands; decomposition of SendError into
throttled_local / HTTP-429 / server-500; attempt-coverage (sent vs never-sent);
and the UnknownPending sub-split (never-sent vs sent-but-lost).

Framing: with 11 senders racing for 1 win/trigger, ~91% of attempts cannot win by
construction, so LOST_RACE and NEVER_SENT are structural, not failures.

Outputs (per spec S1):
  summary/per-sender-outcomes.csv
  summary/per-sender-outcome-class.csv  (6-class count matrix)
  summary/senderror-taxonomy.csv
  plots/01-outcomes-stacked.png        (100%-stacked 6-class outcome bar)
  plots/01-outcomes-senderror-reasons.png  (SendError-reason stacked bar)
  plots/01-outcomes-coverage.png       (attempt-coverage bar)
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fanout_analysis import constants, loader, plotutils, statutils

logger = logging.getLogger(__name__)

SECTION_ID = "S1"
SECTION_TITLE = "Sender outcome profile & SendError taxonomy"
SECTION_NUM = "01"

# SendError reason buckets (decomposition). server_500 is included for completeness
# even though in this run all helius HTTP-500 are recorded as UnknownPending (bug B).
SENDERROR_REASONS = ["throttled_local", "http_429", "server_500"]


def _per_sender_outcome_class(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sender count matrix over the 6 outcome_class buckets (loader order)."""
    ct = (
        pd.crosstab(df["sender_name"], df["outcome_class"])
        .reindex(columns=loader.OUTCOME_CLASSES, fill_value=0)
    )
    # order rows by wins desc so the chart reads top-down by performance
    ct = ct.sort_values("WON", ascending=False)
    return ct


def _senderror_reason(row) -> str:
    """Classify a SendError row into a reason bucket.

    throttled_local: client-side throttle (send_error text 'throttled_local', which
                     coincides with rtt_ms==0 -- the tx never reached the provider).
    http_429:        provider HTTP 429 rate-limit (http_status==429).
    server_500:      provider HTTP 500 server error.
    """
    if row["http_status"] == 429:
        return "http_429"
    if row["http_status"] == 500:
        return "server_500"
    if row["send_error"] == "throttled_local" or row["rtt_ms"] == 0:
        return "throttled_local"
    return "other"


def _per_sender_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """3-state counts + Wilson CIs + coverage + UnknownPending split per sender."""
    rows = []
    for sender, sub in df.groupby("sender_name"):
        n_attempts = len(sub)
        landed = int((sub["final_outcome"] == "Landed").sum())
        send_error = int((sub["final_outcome"] == "SendError").sum())
        pending = int((sub["final_outcome"] == "UnknownPending").sum())

        # attempt-coverage: rows actually transmitted (send_at_ns > 0)
        never_sent = int(sub["never_sent"].sum())
        sent = n_attempts - never_sent

        # UnknownPending sub-split
        up_never_sent = int(sub.loc[sub["final_outcome"] == "UnknownPending", "never_sent"].sum())
        up_sent_but_lost = pending - up_never_sent

        # operational estimand: landed / attempts
        op_lo, op_hi = statutils.wilson_ci(landed, n_attempts)
        # conditional estimand: landed / (landed + pending), excluding SendError
        cond_denom = landed + pending
        cond_lo, cond_hi = statutils.wilson_ci(landed, cond_denom)

        rows.append({
            "sender_name": sender,
            "protocol_class": constants.PROTOCOL_OF.get(sender),
            "n_attempts": n_attempts,
            "landed": landed,
            "send_error": send_error,
            "unknown_pending": pending,
            "n_sent": sent,
            "never_sent": never_sent,
            "coverage": sent / n_attempts if n_attempts else float("nan"),
            "up_never_sent": up_never_sent,
            "up_sent_but_lost": up_sent_but_lost,
            "operational_rate": landed / n_attempts if n_attempts else float("nan"),
            "operational_lo": op_lo,
            "operational_hi": op_hi,
            "conditional_denom": cond_denom,
            "conditional_rate": landed / cond_denom if cond_denom else float("nan"),
            "conditional_lo": cond_lo,
            "conditional_hi": cond_hi,
        })
    out = pd.DataFrame(rows).sort_values("landed", ascending=False).reset_index(drop=True)
    return out


def _senderror_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sender x reason SendError decomposition with rpc_err_code annotation."""
    se = df[df["final_outcome"] == "SendError"].copy()
    rows = []
    if len(se) == 0:
        return pd.DataFrame(
            columns=["sender_name", "protocol_class", "send_error_total",
                     *SENDERROR_REASONS, "rpc_err_codes"])
    se["reason"] = se.apply(_senderror_reason, axis=1)
    for sender, sub in se.groupby("sender_name"):
        counts = sub["reason"].value_counts().to_dict()
        codes = sorted(int(c) for c in sub["rpc_err_code"].dropna().unique())
        rows.append({
            "sender_name": sender,
            "protocol_class": constants.PROTOCOL_OF.get(sender),
            "send_error_total": len(sub),
            **{r: int(counts.get(r, 0)) for r in SENDERROR_REASONS},
            "rpc_err_codes": ";".join(str(c) for c in codes),
        })
    out = pd.DataFrame(rows).sort_values("send_error_total", ascending=False).reset_index(drop=True)
    return out


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summary_dir = outdir / "summary"
    plots_dir = outdir / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("S1: computing per-sender outcomes over %d rows", len(df))
    outcomes = _per_sender_outcomes(df)
    outcome_class = _per_sender_outcome_class(df)
    taxonomy = _senderror_taxonomy(df)

    outcomes_csv = summary_dir / "per-sender-outcomes.csv"
    outcome_class_csv = summary_dir / "per-sender-outcome-class.csv"
    taxonomy_csv = summary_dir / "senderror-taxonomy.csv"
    outcomes.to_csv(outcomes_csv, index=False)
    outcome_class.to_csv(outcome_class_csv)
    taxonomy.to_csv(taxonomy_csv, index=False)

    # --- figures ---
    idx = outcomes.set_index("sender_name")

    # 100%-stacked outcome bar over the 6 outcome_class buckets. Labelled columns
    # so allenhark-quic-tk visibly shows NEVER_SENT (not UnknownPending).
    stacked = outcome_class.rename(columns=loader.OUTCOME_LABEL)
    stacked_cols = [loader.OUTCOME_LABEL[c] for c in loader.OUTCOME_CLASSES]
    fig_stacked = plots_dir / f"{SECTION_NUM}-outcomes-stacked.png"
    plotutils.stacked_bar(
        stacked, cols=stacked_cols,
        title="S1 - Per-sender outcome mix (100%-stacked, 6-class)",
        xlabel="share of attempts", out_path=fig_stacked, horizontal=True, pct=True)

    # SendError-reason stacked bar (only senders with any SendError)
    fig_reasons = plots_dir / f"{SECTION_NUM}-outcomes-senderror-reasons.png"
    if len(taxonomy) > 0:
        reasons_idx = taxonomy.set_index("sender_name")[SENDERROR_REASONS]
        plotutils.stacked_bar(
            reasons_idx, cols=SENDERROR_REASONS,
            title="S1 - SendError reason decomposition (counts)",
            xlabel="SendError count", out_path=fig_reasons, horizontal=True, pct=False)
    else:
        fig_reasons = None

    # attempt-coverage bar
    coverage_df = idx[["coverage"]].copy()
    fig_coverage = plots_dir / f"{SECTION_NUM}-outcomes-coverage.png"
    plotutils.stacked_bar(
        coverage_df, cols=["coverage"],
        title="S1 - Attempt coverage (fraction of attempts actually sent)",
        xlabel="coverage", out_path=fig_coverage, horizontal=True, pct=False)

    # --- key results ---
    senderror_totals = {r["sender_name"]: int(r["send_error"])
                        for _, r in outcomes.iterrows()}
    never_sent_counts = {r["sender_name"]: int(r["never_sent"])
                         for _, r in outcomes.iterrows()}
    up_split = {
        "never_sent": int(outcomes["up_never_sent"].sum()),
        "sent_but_lost": int(outcomes["up_sent_but_lost"].sum()),
    }

    # never-sent cause split: structural dead-PoP (allenhark-quic-tk) vs partial
    # client-throttle skips on the rate-limited senders.
    never_sent_cause = {
        "allenhark_quic_tk_dead_pop": int(never_sent_counts.get("allenhark-quic-tk", 0)),
        "throttle_skip_jito": int(never_sent_counts.get("jito-multi", 0)),
        "throttle_skip_syncro": int(never_sent_counts.get("syncro-fra", 0)),
        "throttle_skip_blockrazor": int(never_sent_counts.get("blockrazor", 0)),
        "throttle_skip_allenhark_ny": int(never_sent_counts.get("allenhark-quic-ny", 0)),
        "note": ("allenhark-quic-tk 912/912 send_at_ns==0 (QUIC PoP in Tokyo "
                 "unreachable from the FRA client) is a total dead-PoP, distinct "
                 "from jito/syncro/blockrazor whose never-sent rows are partial "
                 "client-throttle skips, not connectivity loss."),
    }

    key_results = {
        "senderror_totals": senderror_totals,
        "never_sent_counts": never_sent_counts,
        "never_sent_cause": never_sent_cause,
        "unknown_pending_split": up_split,
        "total_landed": int(outcomes["landed"].sum()),
        "total_send_error": int(outcomes["send_error"].sum()),
        "total_unknown_pending": int(outcomes["unknown_pending"].sum()),
    }

    notes = [
        "throttled_local SendError == rtt_ms==0 (client-side min_send_interval_ms "
        "throttle on jito/syncro/blockrazor; tx never reached the provider).",
        "All 182 helius HTTP-500 rows are recorded as UnknownPending, not SendError "
        "(bug B); server_500 SendError bucket is therefore 0 in this run.",
        "allenhark-quic-tk has 912/912 never_sent (dead PoP, bug A) -> 0 coverage.",
        "Operational denominator = all attempts; conditional denominator excludes "
        "SendError (fair race for the rate-limited senders).",
    ]

    captions = {
        "outcomes_stacked": (
            "Per-sender outcome mix over 6 classes. WON and LOST_RACE are structural "
            "(11 senders race, only 1 can win per trigger); allenhark-quic-tk is "
            "entirely NEVER_SENT because its Tokyo QUIC PoP never transmitted."),
        "coverage": (
            "Fraction of attempts actually transmitted. Coverage<1 means attempts "
            "were not sent: client-side throttle skips for jito/syncro/blockrazor, "
            "and a total dead PoP (0 coverage) for allenhark-quic-tk."),
    }
    if fig_reasons:
        captions["senderror_reasons"] = (
            "SendError breakdown for the rate-limited senders: throttled_local is the "
            "client min-send-interval (tx never reached the provider), http_429 is a "
            "provider rate-limit rejection.")

    return {
        "id": SECTION_ID,
        "title": SECTION_TITLE,
        "tables": {
            "per_sender_outcomes": str(outcomes_csv),
            "per_sender_outcome_class": str(outcome_class_csv),
            "senderror_taxonomy": str(taxonomy_csv),
        },
        "figures": {
            "outcomes_stacked": str(fig_stacked),
            **({"senderror_reasons": str(fig_reasons)} if fig_reasons else {}),
            "coverage": str(fig_coverage),
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
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": out,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="S1 - Sender outcome profile & SendError taxonomy")
    ap.add_argument("--out", default="/tmp/S1-verify", help="output directory")
    args = ap.parse_args(argv)
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2))
    print("tables:", json.dumps(result["tables"], indent=2))
    print("figures:", json.dumps(result["figures"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
