import json

from fanout_analysis import loader, constants, s0_integrity


def _build_ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    config = json.loads(constants.DEFAULT_CONFIG.read_text())
    return {
        "df": loader.load_enriched(),
        "wide": loader.load_wide(),
        "outdir": tmp_path,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s0_golden_single_winner_and_all_pass(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s0_integrity.run(ctx)

    # golden: single_winner == {"0":66, "1":846}, all invariants pass
    inv_json = json.loads((tmp_path / "integrity-invariants.json").read_text())
    assert inv_json["single_winner"] == {"0": 66, "1": 846}
    assert inv_json["all_pass"] is True
    assert all(i["pass"] is True for i in inv_json["invariants"])

    assert result["id"] == "S0"
    assert result["key_results"]["all_pass"] is True
    assert result["key_results"]["single_winner"] == {0: 66, 1: 846}
    assert result["key_results"]["n_rows"] == 10032
    assert result["key_results"]["n_triggers"] == 912
    assert result["key_results"]["n_senders"] == 11
    assert result["key_results"]["n_landed"] == 846


def test_s0_writes_named_files(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s0_integrity.run(ctx)

    assert (tmp_path / "summary" / "run-manifest.csv").exists()
    assert (tmp_path / "summary" / "integrity-invariants.csv").exists()
    assert (tmp_path / "integrity-invariants.json").exists()
    assert (tmp_path / "plots" / "00-integrity-scorecard.png").exists()
    assert (tmp_path / "plots" / "00-global-outcome.png").exists()

    # returned paths point at the real files
    assert result["tables"]["run-manifest"].exists()
    assert result["figures"]["integrity-scorecard"].exists()
    assert result["figures"]["global-outcome"].exists()


def test_s0_hard_fails_on_violated_invariant(tmp_path):
    import pytest

    ctx = _build_ctx(tmp_path)
    # drop one row -> row_count + rows_per_trigger balance must fail the gate
    ctx["df"] = ctx["df"].iloc[1:].copy()
    with pytest.raises(AssertionError):
        s0_integrity.run(ctx)
