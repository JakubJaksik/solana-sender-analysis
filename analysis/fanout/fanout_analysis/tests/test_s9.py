"""Golden tests for S9 multivariable P(win) model (plan Task 18)."""
import json

import pandas as pd

from fanout_analysis import constants, loader, s9_pwin_model


def _ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "df": loader.load_enriched(),
        "wide": loader.load_wide(),
        "outdir": tmp_path,
        "config": json.load(open(constants.DEFAULT_CONFIG)),
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s9_send_order_is_verified_null(tmp_path):
    res = s9_pwin_model.run(_ctx(tmp_path))
    kr = res["key_results"]
    # send_order OR 95% CI includes 1.0 -> verified null
    lo, hi = kr["send_order_ci"]
    assert lo <= 1.0 <= hi
    assert kr["send_order_ci_includes_1"] is True
    assert "NULL" in kr["send_order_verdict"]


def test_s9_model_converges_and_tk_absent(tmp_path):
    res = s9_pwin_model.run(_ctx(tmp_path))
    kr = res["key_results"]
    assert kr["converged"] is True
    assert kr["tk_in_terms"] is False
    # allenhark-quic-tk must not appear in any coefficient term
    coef = pd.read_csv(res["tables"]["pwin_model_coefficients"])
    assert not coef["term"].astype(str).str.contains("allenhark-quic-tk").any()
    # baseline (triton-fra) must not be its own term either
    assert not coef["term"].astype(str).str.fullmatch("triton-fra").any()


def test_s9_send_order_causal_csv_row(tmp_path):
    res = s9_pwin_model.run(_ctx(tmp_path))
    causal = pd.read_csv(res["tables"]["send_order_causal"])
    assert len(causal) == 1
    row = causal.iloc[0]
    assert bool(row["or_ci_includes_1"]) is True
    assert row["or_ci_lo"] <= row["odds_ratio"] <= row["or_ci_hi"]
    assert row["or_ci_lo"] <= 1.0 <= row["or_ci_hi"]


def test_s9_files_created(tmp_path):
    res = s9_pwin_model.run(_ctx(tmp_path))
    coef = tmp_path / "summary" / "pwin-model-coefficients.csv"
    causal = tmp_path / "summary" / "send-order-causal.csv"
    forest = tmp_path / "plots" / "09-pwin-or-forest.png"
    smul = tmp_path / "plots" / "09-pwin-send-order-smallmultiples.png"
    assert coef.exists() and causal.exists()
    assert forest.exists() and smul.exists()
    # coefficient table must hold sender terms + the two numeric covariates
    df = pd.read_csv(coef)
    assert "send_order (centered)" in set(df["term"])
    assert "continent_match" in set(df["term"])
