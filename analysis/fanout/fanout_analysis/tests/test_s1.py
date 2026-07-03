"""Golden tests for S1 - sender outcome profile & SendError taxonomy."""
import json

import pandas as pd

from fanout_analysis import constants, loader, s1_outcomes


def _build_ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": tmp_path,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s1_senderror_totals_and_never_sent_golden(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s1_outcomes.run(ctx)

    se = result["key_results"]["senderror_totals"]
    # Golden: SendError totals jito-multi 746, syncro-fra 626, blockrazor 265, all others 0.
    assert se["jito-multi"] == 746
    assert se["syncro-fra"] == 626
    assert se["blockrazor"] == 265
    for sender, n in se.items():
        if sender not in ("jito-multi", "syncro-fra", "blockrazor"):
            assert n == 0, f"{sender} should have 0 SendError, got {n}"

    # Golden: allenhark-quic-tk never_sent count == 912.
    assert result["key_results"]["never_sent_counts"]["allenhark-quic-tk"] == 912


def test_s1_per_sender_outcomes_csv_written_and_consistent(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s1_outcomes.run(ctx)

    outcomes_csv = tmp_path / "summary" / "per-sender-outcomes.csv"
    taxonomy_csv = tmp_path / "summary" / "senderror-taxonomy.csv"
    assert outcomes_csv.exists()
    assert taxonomy_csv.exists()

    out = pd.read_csv(outcomes_csv)
    assert len(out) == 11
    assert out["landed"].sum() == 846
    assert out["send_error"].sum() == 1637
    assert out["unknown_pending"].sum() == 7549
    # every cell is one of the three outcomes
    assert ((out["landed"] + out["send_error"] + out["unknown_pending"]) == out["n_attempts"]).all()
    assert (out["n_attempts"] == 912).all()

    tk = out[out["sender_name"] == "allenhark-quic-tk"].iloc[0]
    assert tk["never_sent"] == 912
    assert tk["n_sent"] == 0
    assert tk["coverage"] == 0.0


def test_s1_senderror_taxonomy_decomposition(tmp_path):
    ctx = _build_ctx(tmp_path)
    s1_outcomes.run(ctx)

    tax = pd.read_csv(tmp_path / "summary" / "senderror-taxonomy.csv")
    # only the 3 throttled senders appear
    assert set(tax["sender_name"]) == {"jito-multi", "syncro-fra", "blockrazor"}
    # reason buckets sum back to the SendError total per sender
    assert (
        (tax["throttled_local"] + tax["http_429"] + tax["server_500"])
        == tax["send_error_total"]
    ).all()
    # global decomposition: 1409 throttled_local, 228 http_429, 0 server_500
    assert tax["throttled_local"].sum() == 1409
    assert tax["http_429"].sum() == 228
    assert tax["server_500"].sum() == 0
    assert tax["send_error_total"].sum() == 1637


def test_s1_unknown_pending_split(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s1_outcomes.run(ctx)
    split = result["key_results"]["unknown_pending_split"]
    # 1118 never-sent (incl. tk 912) + 6431 sent-but-lost == 7549 UnknownPending
    assert split["never_sent"] == 1118
    assert split["sent_but_lost"] == 6431
    assert split["never_sent"] + split["sent_but_lost"] == 7549


def test_s1_figures_written(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s1_outcomes.run(ctx)
    for key in ("outcomes_stacked", "senderror_reasons", "coverage"):
        path = result["figures"][key]
        assert path.endswith(".png")
        from pathlib import Path
        assert Path(path).exists()
    assert (tmp_path / "plots" / "01-outcomes-stacked.png").exists()
    assert (tmp_path / "plots" / "01-outcomes-senderror-reasons.png").exists()
    assert (tmp_path / "plots" / "01-outcomes-coverage.png").exists()
