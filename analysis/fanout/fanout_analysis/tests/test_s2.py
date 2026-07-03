"""Golden-number tests for S2 (head-to-head win rates).

Golden oracle (validated run 20260601-150500): operational landed counts
0slot-de1 253, triton-fra 221, blockrazor 108, helius-dual 98, allenhark-quic-fra 51,
allenhark-quic-ams 47, helius-fra 46, allenhark-quic-ny 14, syncro-fra 7, jito-multi 1,
allenhark-quic-tk 0 (sum 846); no-winner-triggers.csv has 66 rows.
"""
import pandas as pd

from fanout_analysis import loader, constants, s2_winrate


GOLDEN_OPERATIONAL = {
    "0slot-de1": 253,
    "triton-fra": 221,
    "blockrazor": 108,
    "helius-dual": 98,
    "allenhark-quic-fra": 51,
    "allenhark-quic-ams": 47,
    "helius-fra": 46,
    "allenhark-quic-ny": 14,
    "syncro-fra": 7,
    "jito-multi": 1,
    "allenhark-quic-tk": 0,
}


def _ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "df": loader.load_enriched(),
        "wide": loader.load_wide(),
        "outdir": tmp_path,
        "config": {},
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s2_operational_counts_match_golden_leaderboard(tmp_path):
    result = s2_winrate.run(_ctx(tmp_path))

    op = result["key_results"]["operational_landed_counts"]
    assert op == GOLDEN_OPERATIONAL
    assert sum(op.values()) == 846
    assert result["key_results"]["operational_landed_sum"] == 846
    # all 11 senders present (including the 0-land allenhark-quic-tk)
    assert len(op) == 11


def test_s2_no_winner_triggers_has_66_rows(tmp_path):
    result = s2_winrate.run(_ctx(tmp_path))
    assert result["key_results"]["n_no_winner_triggers"] == 66

    nowin = pd.read_csv(tmp_path / "summary" / "no-winner-triggers.csv")
    assert len(nowin) == 66
    # each no-winner trigger has 11 attempts split between SendError and UnknownPending
    assert (nowin["n_senderror"] + nowin["n_unknownpending"] == 11).all()


def test_s2_estimands_csv_shape_and_values(tmp_path):
    s2_winrate.run(_ctx(tmp_path))

    est = pd.read_csv(tmp_path / "summary" / "win-rate-estimands.csv")
    assert len(est) == 11
    by_sender = est.set_index("sender_name")

    # operational counts match golden
    for sender, count in GOLDEN_OPERATIONAL.items():
        assert int(by_sender.loc[sender, "landed"]) == count

    # operational denominator is 912 for every sender
    assert (est["operational_n"] == 912).all()
    # share denominator is the contested pool of 846 winning triggers
    assert (est["share_n"] == 846).all()

    # conditional excludes SendError: throttled senders have a smaller denominator
    assert int(by_sender.loc["jito-multi", "conditional_n"]) == 166
    assert int(by_sender.loc["syncro-fra", "conditional_n"]) == 286
    assert int(by_sender.loc["blockrazor", "conditional_n"]) == 647
    # un-throttled senders keep the full 912
    assert int(by_sender.loc["0slot-de1", "conditional_n"]) == 912

    # corrected variant: tk has a zero denominator (all never_sent) -> Wilson CI (0,0)
    assert int(by_sender.loc["allenhark-quic-tk", "corrected_n"]) == 0
    assert by_sender.loc["allenhark-quic-tk", "corrected_lo"] == 0.0
    assert by_sender.loc["allenhark-quic-tk", "corrected_hi"] == 0.0

    # Wilson CI sanity for the operational leader
    row = by_sender.loc["0slot-de1"]
    assert row["operational_lo"] < row["operational_rate"] < row["operational_hi"]


def test_s2_files_produced(tmp_path):
    result = s2_winrate.run(_ctx(tmp_path))

    assert (tmp_path / "summary" / "win-rate-estimands.csv").exists()
    assert (tmp_path / "summary" / "no-winner-triggers.csv").exists()
    assert (tmp_path / "plots" / "02-winrate-forest.png").exists()
    assert (tmp_path / "plots" / "02-winrate-slopegraph.png").exists()
    assert (tmp_path / "plots" / "02-winrate-bootstrap-rank-heatmap.png").exists()

    # returned dict points at the produced files
    for path in list(result["tables"].values()) + list(result["figures"].values()):
        from pathlib import Path
        assert Path(path).exists()
