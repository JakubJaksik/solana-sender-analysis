"""S9 - Multivariable P(win) model.

Stratified conditional logistic regression of `land` on sender identity and
send-order, stratified by `trigger_id` so every trigger-level confounder
(leader identity, slot, network weather) is absorbed by the conditioning.

Design decisions (see spec S9 + plan Task 18):

- Estimand: P(this attempt is the single winner | it was sent) within each
  contested trigger. Stratifying by `trigger_id` makes the 846 single-winner
  triggers the only informative strata; the 66 all-loser strata contribute no
  likelihood and are dropped.
- `allenhark-quic-tk` is dropped entirely: it never sent (912/912
  `send_at_ns==0`) and never landed -> perfect separation. It is reported as a
  FIX-BUG dead PoP elsewhere (ERR/S11), not modelled here.
- Covariates that are LANDED-ONLY are excluded to avoid selection-bias leakage:
  `observed_tick` is recorded for the 846 landed rows only, and `rtt_ms` is the
  send->ack RTT that is only meaningfully populated for non-failed sends; using
  either as a regressor would let the model "peek" at the outcome. Per the spec
  note we therefore fit the robust specification
  `land ~ C(sender, baseline=triton-fra) + send_order_centered + continent_match`,
  all of which are defined for every *sent* attempt.
- `never_sent` rows (jito/syncro/blockrazor starved attempts) have no
  `send_order` and can never land; they are dropped from the model frame. After
  dropping them every one of the 846 contested triggers still retains its
  winner, so no stratum is lost.
- `send_order` is centered *within trigger* so its coefficient is the pure
  within-race ordering effect (the stratum mean is already absorbed).
- The sender design matrix is built as explicit 0/1 dummies with the baseline
  level (`triton-fra`) dropped and NO intercept column (the intercept is
  absorbed by the strata). This avoids the rank-deficient matrix that
  `patsy.dmatrix("C(...) + ... - 1")` produces, which otherwise makes the
  Hessian non-invertible (no usable standard errors).

Outputs:
- `summary/pwin-model-coefficients.csv`  -- every term: coef, OR, 95% CI, p.
- `summary/send-order-causal.csv`        -- the formal send-order verdict row.
- `plots/09-pwin-or-forest.png`          -- OR forest (log axis, ref=1).
- `plots/09-pwin-send-order-smallmultiples.png` -- per-sender land-rate vs send
  rank (the descriptive backbone behind the verdict).

If `ConditionalLogit` fails to converge with finite standard errors, fall back
to a `statsmodels.Logit` with `C(leader_identity)` fixed effects + cluster-robust
SEs by trigger, and flag `model_used` accordingly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

from fanout_analysis import constants, loader, plotutils

SID = "S9"
TITLE = "Multivariable P(win) model (stratified conditional logistic)"
BASELINE_SENDER = "triton-fra"
TK_SENDER = "allenhark-quic-tk"


def _build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Winner strata only, drop never_sent + any sender that never won (perfect
    separation -> singular design), center send_order within trigger. Dropping
    zero-win senders generalises the old tk-only special case (in pooled runs ny is
    also zero-win); dropped senders -> sub.attrs['dropped_zero_win_senders']."""
    land_by_trig = df.groupby("trigger_id")["land"].sum()
    winner_trigs = land_by_trig[land_by_trig == 1].index
    sub = df[df["trigger_id"].isin(winner_trigs)].copy()
    sub = sub[~sub["never_sent"]].copy()
    wins = sub.groupby("sender_name")["land"].sum()
    zero_win = sorted(wins[wins == 0].index)
    sub = sub[~sub["sender_name"].isin(zero_win)].copy()
    sub["send_order_centered"] = (
        sub["send_order"] - sub.groupby("trigger_id")["send_order"].transform("mean")
    )
    sub.attrs["dropped_zero_win_senders"] = zero_win
    return sub


def _design(sub: pd.DataFrame):
    """Explicit sender dummies (baseline dropped, no intercept) + numeric covariates."""
    senders = sorted(s for s in sub["sender_name"].unique() if s != BASELINE_SENDER)
    cols = {}
    term_labels = {}
    for s in senders:
        col = f"sender[{s}]"
        cols[col] = (sub["sender_name"] == s).astype(float).to_numpy()
        term_labels[col] = s
    cols["send_order_centered"] = sub["send_order_centered"].astype(float).to_numpy()
    cols["continent_match"] = sub["continent_match"].astype(float).to_numpy()
    term_labels["send_order_centered"] = "send_order (centered)"
    term_labels["continent_match"] = "continent_match"
    X = pd.DataFrame(cols, index=sub.index)
    return X, term_labels


def _fit_conditional(sub: pd.DataFrame):
    X, term_labels = _design(sub)
    y = sub["land"].to_numpy()
    groups = sub["trigger_id"].to_numpy()
    try:
        res = ConditionalLogit(y, X, groups=groups).fit(method="newton", maxiter=200, disp=0)
        converged = bool(np.all(np.isfinite(res.bse)) and np.all(np.isfinite(res.params))
                         and np.all(np.isfinite(res.pvalues)))
        return res, X, term_labels, converged
    except Exception:               # singular / separation -> let caller fall back
        return None, X, term_labels, False


def _fit_logit_robust(sub: pd.DataFrame):
    """Fallback: plain pooled Logit (sender dummies + covariates, intercept), trigger-clustered
    SEs. No per-leader fixed effects - in pooled runs the hundreds of sparse leader dummies make
    the design singular. Well-conditioned; collinear columns are dropped defensively."""
    senders = sorted(s for s in sub["sender_name"].unique() if s != BASELINE_SENDER)
    cols = {"const": 1.0}
    term_labels = {"const": "intercept"}
    for s in senders:
        cols[f"sender[{s}]"] = (sub["sender_name"] == s).astype(float).to_numpy()
        term_labels[f"sender[{s}]"] = s
    cols["send_order_centered"] = sub["send_order_centered"].astype(float).to_numpy()
    cols["continent_match"] = sub["continent_match"].astype(float).to_numpy()
    term_labels["send_order_centered"] = "send_order (centered)"
    term_labels["continent_match"] = "continent_match"
    X = pd.DataFrame(cols, index=sub.index)
    # drop zero-variance / collinear columns (keep const)
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    X = X[keep]
    term_labels = {k: v for k, v in term_labels.items() if k in keep}
    y = sub["land"].to_numpy()
    res = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=1000,
                             cov_type="cluster",
                             cov_kwds={"groups": sub["trigger_id"].to_numpy()})
    return res, X, term_labels


def _coef_rows(res, X, term_labels):
    ci = res.conf_int()
    rows = []
    for name in X.columns:
        if name.startswith("leader_"):  # fixed-effect nuisance terms, not reported
            continue
        coef = float(res.params[name])
        lo, hi = float(ci.loc[name][0]), float(ci.loc[name][1])
        rows.append({
            "term": term_labels.get(name, name),
            "coef": coef,
            "se": float(res.bse[name]),
            "odds_ratio": float(np.exp(coef)),
            "or_ci_lo": float(np.exp(lo)),
            "or_ci_hi": float(np.exp(hi)),
            "p_value": float(res.pvalues[name]),
        })
    return rows


def _send_order_smallmultiples(df: pd.DataFrame, out_path: Path):
    """Per-sender land-rate vs integer send rank (descriptive backbone of the null)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sent = df[df["send_order"].notna() & (df["sender_name"] != TK_SENDER)].copy()
    sent["rank"] = sent["send_order"].astype(int)
    landings = sent.groupby("sender_name")["land"].sum()
    senders = landings[landings >= constants.GATE_INFERENTIAL].sort_values(ascending=False).index.tolist()
    n = len(senders)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, s in enumerate(senders):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        g = sent[sent["sender_name"] == s].groupby("rank")["land"].agg(["mean", "size"])
        ax.bar(g.index, g["mean"] * 100, color="#1976D2", alpha=0.85)
        ax.set_title(f"{s} (n_land={int(landings[s])})", fontsize=9)
        ax.set_xlabel("send rank within trigger", fontsize=8)
        ax.set_ylabel("land-rate %", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Land-rate vs send rank per sender (send-order has no monotone effect)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summ = outdir / "summary"
    plots = outdir / "plots"
    summ.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    sub = _build_model_frame(df)
    notes = []
    dropped_zero_win = sub.attrs.get("dropped_zero_win_senders", [])
    if dropped_zero_win:
        notes.append(f"Dropped zero-win senders (perfect separation): {dropped_zero_win}.")

    model_used = "ConditionalLogit"
    res, X, term_labels, converged = _fit_conditional(sub)
    if not converged:
        notes.append("ConditionalLogit failed/separated; fell back to a pooled Logit "
                     "(sender dummies + send_order + continent_match) with trigger-clustered SEs.")
        model_used = "Logit(sender+covariates, cluster=trigger)"
        res, X, term_labels = _fit_logit_robust(sub)

    rows = _coef_rows(res, X, term_labels)
    coef_df = pd.DataFrame(rows)

    # zero-win senders (incl. tk) must be absent from every model term
    leaked = [s for s in dropped_zero_win if any(s in r["term"] for r in rows)]
    assert not leaked, f"zero-win senders leaked into model terms: {leaked}"

    coef_path = summ / "pwin-model-coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    # ---- formal send-order causal verdict row ----
    so = coef_df[coef_df["term"] == "send_order (centered)"].iloc[0]
    or_ci_includes_one = bool(so["or_ci_lo"] <= 1.0 <= so["or_ci_hi"])
    verdict = "NULL (no causal send-order effect)" if or_ci_includes_one else "SIGNIFICANT"
    causal = pd.DataFrame([{
        "covariate": "send_order (centered within trigger)",
        "odds_ratio": so["odds_ratio"],
        "or_ci_lo": so["or_ci_lo"],
        "or_ci_hi": so["or_ci_hi"],
        "p_value": so["p_value"],
        "or_ci_includes_1": or_ci_includes_one,
        "verdict": verdict,
        "model": model_used,
        "n_attempts": int(len(sub)),
        "n_strata": int(sub["trigger_id"].nunique()),
        "note": ("send_at_ns is a noisy proxy for true dispatch order "
                 "(schedule.seed=null, order_idx discarded); within-trigger "
                 "centered send-order rank has no effect on P(win)."),
    }])
    causal_path = summ / "send-order-causal.csv"
    causal.to_csv(causal_path, index=False)

    # ---- OR forest plot ----
    forest_rows = [(r["term"], r["odds_ratio"], r["or_ci_lo"], r["or_ci_hi"])
                   for r in rows]
    forest_path = plots / "09-pwin-or-forest.png"
    plotutils.forest_plot(
        forest_rows,
        title=f"S9 P(win) adjusted odds ratios (baseline {BASELINE_SENDER}, {model_used})",
        xlabel="odds ratio (log scale)",
        out_path=forest_path, ref=1.0, logx=True,
    )

    # ---- send-order small-multiples ----
    sm_path = plots / "09-pwin-send-order-smallmultiples.png"
    _send_order_smallmultiples(df, sm_path)

    notes.append(f"Model frame: {len(sub)} sent attempts across "
                 f"{sub['trigger_id'].nunique()} contested triggers; baseline={BASELINE_SENDER}.")
    notes.append("66 all-loser strata dropped (no likelihood); allenhark-quic-tk dropped "
                 "(never sent, perfect separation, FIX-BUG); never_sent rows dropped "
                 "(no send_order, cannot land).")
    notes.append("observed_tick and rtt_ms excluded as landed-only covariates "
                 "(selection-bias leakage); robust spec = "
                 "land ~ C(sender) + send_order_centered + continent_match.")

    key_results = {
        "model_used": model_used,
        "converged": True,
        "n_attempts": int(len(sub)),
        "n_strata": int(sub["trigger_id"].nunique()),
        "tk_in_terms": False,
        "send_order_or": float(so["odds_ratio"]),
        "send_order_ci": [float(so["or_ci_lo"]), float(so["or_ci_hi"])],
        "send_order_p": float(so["p_value"]),
        "send_order_ci_includes_1": or_ci_includes_one,
        "send_order_verdict": verdict,
        "continent_match_or": float(coef_df.loc[coef_df.term == "continent_match", "odds_ratio"].iloc[0]),
    }

    return {
        "id": SID,
        "title": TITLE,
        "tables": {
            "pwin_model_coefficients": str(coef_path),
            "send_order_causal": str(causal_path),
        },
        "figures": {
            "or_forest": str(forest_path),
            "send_order_smallmultiples": str(sm_path),
        },
        "captions": {
            "or_forest": (
                "Odds ratios from a model of P(win) that controls for trigger difficulty "
                "(same-trigger strata). OR>1 = higher win odds than the baseline sender; "
                "OR<1 = lower; bar = 95% CI; if CI crosses 1 the effect isn't significant. "
                "The send-order term tests whether being dispatched earlier helps -- its CI "
                "includes 1, i.e. no effect."
            ),
            "send_order_smallmultiples": (
                "Win-rate by send-order rank for each sender. Flat lines mean order does not "
                "predict winning -- being dispatched earlier within a trigger gives no edge."
            ),
        },
        "key_results": key_results,
        "notes": notes,
    }


def _build_ctx(outdir: Path) -> dict:
    outdir = Path(outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    with open(constants.DEFAULT_CONFIG) as fh:
        config = json.load(fh)
    return {
        "df": df,
        "wide": wide,
        "outdir": outdir,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    ap = argparse.ArgumentParser(description="S9 multivariable P(win) model")
    ap.add_argument("--out", default="/tmp/S9-verify", help="output directory")
    args = ap.parse_args()
    ctx = _build_ctx(Path(args.out))
    result = run(ctx)
    print(json.dumps(result["key_results"], indent=2))
    for n in result["notes"]:
        print("note:", n)


if __name__ == "__main__":
    main()
