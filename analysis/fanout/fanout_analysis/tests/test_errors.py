import json

import pandas as pd

from fanout_analysis import constants, errors, loader


def _ctx(tmp_path):
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "df": df,
        "wide": wide,
        "outdir": tmp_path,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_run_returns_section_contract(tmp_path):
    result = errors.run(_ctx(tmp_path))
    assert result["id"] == "ERR"
    assert result["title"]
    assert set(result["tables"]) == {"error_catalog", "denominator_corrections"}
    assert set(result["figures"]) == {
        "unknownpending_decomposition", "senderror_reason_matrix", "recorded_vs_corrected"}


def test_error_catalog_has_bug_a_and_bug_b(tmp_path):
    result = errors.run(_ctx(tmp_path))
    catalog_path = result["tables"]["error_catalog"]
    catalog = pd.read_csv(catalog_path)

    # Bug A: allenhark-quic-tk never_sent count 912 (dead PoP misclassified)
    bug_a = catalog[(catalog["category"] == "BUG_A_dead_pop_misclassified")
                    & (catalog["sender"] == "allenhark-quic-tk")]
    assert len(bug_a) == 1
    assert int(bug_a["count"].iloc[0]) == 912
    assert bug_a["recommended_classification"].iloc[0].startswith("SendError")
    assert bool(bug_a["affects_denominator"].iloc[0]) is True

    # Bug B: helius_500 count 182 (sum across helius-dual + helius-fra)
    bug_b = catalog[catalog["category"] == "BUG_B_helius_500_misclassified"]
    assert int(bug_b["count"].sum()) == 182
    assert (bug_b["http_status"] == 500).all()


def test_denominator_corrections_has_11_sender_rows(tmp_path):
    result = errors.run(_ctx(tmp_path))
    corrections = pd.read_csv(result["tables"]["denominator_corrections"])
    assert len(corrections) == 11
    assert corrections["sender"].nunique() == 11
    # corrected = raw - reclassified - never_sent_excluded
    recomputed = (corrections["raw_conditional_denominator"]
                  - corrections["reclassified_to_senderror"]
                  - corrections["never_sent_excluded"])
    assert (recomputed == corrections["corrected_conditional_denominator"]).all()
    # tk fully excluded -> corrected denominator 0
    tk = corrections[corrections["sender"] == "allenhark-quic-tk"].iloc[0]
    assert int(tk["corrected_conditional_denominator"]) == 0
    # helius senders each reclassify 91 server-500 rows
    for s in ("helius-dual", "helius-fra"):
        row = corrections[corrections["sender"] == s].iloc[0]
        assert int(row["reclassified_to_senderror"]) == 91


def test_key_results_golden_numbers(tmp_path):
    result = errors.run(_ctx(tmp_path))
    kr = result["key_results"]
    assert kr["bug_A_allenhark_tk_never_sent"] == 912
    assert kr["bug_B_helius_500_count"] == 182
    assert kr["n_send_error"] == 1637
    assert kr["n_unknown_pending"] == 7549
    assert kr["unknown_never_sent"] == 1118
    assert kr["unknown_sent_but_lost"] == 6431
    assert kr["bug_C_rtt_zero_throttle"] == 1409
    assert kr["bug_C_rtt_null_no_send"] == 1118
    assert kr["denominator_corrections_rows"] == 11


def test_run_returns_caption_for_every_figure(tmp_path):
    result = errors.run(_ctx(tmp_path))
    captions = result["captions"]
    assert set(captions) == set(result["figures"])
    for cap in captions.values():
        assert isinstance(cap, str) and cap.strip()
    # the never_sent definition and the LOST_RACE-is-expected framing must be present
    decomp = captions["unknownpending_decomposition"]
    assert "send_at_ns==0" in decomp
    assert "EXPECTED" in decomp
    assert "912/912" in decomp


def test_senderror_chart_merges_provider_rate_limits(tmp_path):
    df = loader.load_enriched()
    se = df[df["final_outcome"] == "SendError"].copy()
    se["reason"] = se.apply(errors._chart_reason, axis=1)
    matrix = se.groupby(["sender_name", "reason"]).size().unstack(fill_value=0)
    # the two rpc rate-limit codes collapse into one HTTP-429 chart column
    assert errors.PROVIDER_RATE_LIMIT_REASON in matrix.columns
    assert int(matrix[errors.PROVIDER_RATE_LIMIT_REASON].sum()) == 228  # jito 177 + syncro 51
    assert set(matrix.columns) == {errors.PROVIDER_RATE_LIMIT_REASON, "throttled_local"}


def test_unknown_decomposition_keeps_lost_race_separate(tmp_path):
    result = errors.run(_ctx(tmp_path))
    kr = result["key_results"]
    # LOST_RACE (expected) is reported separately from the real problems
    assert kr["unknown_lost_race_expected"] == 6249
    assert kr["unknown_server_error"] == 182
    assert kr["unknown_never_sent"] == 1118
    # the three add back to the full UnknownPending count
    assert (kr["unknown_lost_race_expected"] + kr["unknown_server_error"]
            + kr["unknown_never_sent"]) == kr["n_unknown_pending"]


def test_csv_and_png_files_created(tmp_path):
    result = errors.run(_ctx(tmp_path))
    assert (tmp_path / "summary" / "error-catalog.csv").exists()
    assert (tmp_path / "summary" / "denominator-corrections.csv").exists()
    assert (tmp_path / "plots" / "ERR-unknownpending-decomposition.png").exists()
    assert (tmp_path / "plots" / "ERR-senderror-reason-matrix.png").exists()
    assert (tmp_path / "plots" / "ERR-recorded-vs-corrected.png").exists()
