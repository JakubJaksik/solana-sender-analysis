"""Golden tests for S4 - Landed-latency distributions.

Golden oracle (validated run 20260601-150500): per-sender n in latency-percentiles
equals that sender's Landed count (e.g. send_to_obs_ms n for 0slot-de1 == 253);
jito-multi is flagged fragile with n == 1.
"""
import pandas as pd

from fanout_analysis import loader, constants, s4_latency


def _ctx(tmp_path):
    df = loader.load_enriched()
    wide = loader.load_wide()
    return {
        "df": df,
        "wide": wide,
        "outdir": tmp_path,
        "config": {},
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_run_returns_expected_shape(tmp_path):
    res = s4_latency.run(_ctx(tmp_path))
    assert res["id"] == "S4"
    assert set(res["tables"]) == {
        "latency_percentiles", "latency_component_gap", "latency_by_continent"
    }
    assert set(res["figures"]) == {
        "ecdf_send_to_obs", "ecdf_trigger_to_land_ticks",
        "violin_send_to_obs", "violin_trigger_to_land_ticks",
        "percentile_bars", "top4_histograms", "hist_trigger_to_land_ticks",
        "latency_by_continent", "ticks_by_region",
        "ticks_by_region_p90", "ticks_by_region_p99",
    }
    # every figure must carry a concise caption
    assert set(res["captions"]) == set(res["figures"])


def test_send_to_obs_n_matches_landed_counts(tmp_path):
    s4_latency.run(_ctx(tmp_path))
    perc = pd.read_csv(tmp_path / "summary" / "latency-percentiles.csv")
    sto = perc[perc["metric"] == "send_to_obs_ms"].set_index("sender_name")
    # GOLDEN: send_to_obs n equals the per-sender Landed count.
    assert int(sto.loc["0slot-de1", "n"]) == 253
    assert int(sto.loc["triton-fra", "n"]) == 221
    assert int(sto.loc["blockrazor", "n"]) == 108
    assert int(sto.loc["helius-dual", "n"]) == 98
    assert int(sto.loc["jito-multi", "n"]) == 1
    # n equals n_land for every sender / every metric (rtt is all > 0 in this run).
    assert (perc["n"] == perc["n_land"]).all()


def test_jito_flagged_fragile_with_n_one(tmp_path):
    s4_latency.run(_ctx(tmp_path))
    perc = pd.read_csv(tmp_path / "summary" / "latency-percentiles.csv")
    jito = perc[perc["sender_name"] == "jito-multi"]
    assert jito["fragile"].all()
    assert (jito["n"] == 1).all()
    # 0slot-de1 (n=253) is NOT fragile.
    assert not perc[perc["sender_name"] == "0slot-de1"]["fragile"].any()


def test_allenhark_tk_absent(tmp_path):
    s4_latency.run(_ctx(tmp_path))
    perc = pd.read_csv(tmp_path / "summary" / "latency-percentiles.csv")
    assert "allenhark-quic-tk" not in set(perc["sender_name"])  # zero Landed rows


def test_files_created(tmp_path):
    res = s4_latency.run(_ctx(tmp_path))
    assert (tmp_path / "summary" / "latency-percentiles.csv").exists()
    assert (tmp_path / "summary" / "latency-component-gap.csv").exists()
    for fig in res["figures"].values():
        assert fig.exists()
    # PNG prefix is the section number.
    for fig in res["figures"].values():
        assert fig.name.startswith("04-")


def test_percentiles_has_p1_column(tmp_path):
    s4_latency.run(_ctx(tmp_path))
    perc = pd.read_csv(tmp_path / "summary" / "latency-percentiles.csv")
    # p1 (best-case) added; ordered <= p10 for every row that has data.
    assert "p1" in perc.columns
    sto = perc[(perc["metric"] == "send_to_obs_ms") & (perc["n"] > 0)]
    assert (sto["p1"] <= sto["p10"] + 1e-9).all()


def test_latency_by_continent_gating(tmp_path):
    s4_latency.run(_ctx(tmp_path))
    cont = pd.read_csv(tmp_path / "summary" / "latency-by-continent.csv")
    # only the three analysed continents appear
    assert set(cont["continent"]) <= {"Europe", "North America", "Asia"}
    # GOLDEN cell counts (validated run 20260601-150500): destination = leader continent.
    cell = cont.set_index(["sender_name", "continent"])
    assert int(cell.loc[("0slot-de1", "North America"), "n"]) == 95
    assert int(cell.loc[("0slot-de1", "Europe"), "n"]) == 129
    assert int(cell.loc[("triton-fra", "Asia"), "n"]) == 25
    # inferential gate applied at n>=20
    assert (cont.loc[cont["n"] >= 20, "gate"] == "inferential").all()


def test_key_results_golden(tmp_path):
    res = s4_latency.run(_ctx(tmp_path))
    kr = res["key_results"]
    assert kr["n_landed_total"] == 846
    assert kr["send_to_obs_n_by_sender"]["0slot-de1"] == 253
    assert kr["send_to_obs_n_by_sender"]["jito-multi"] == 1
    assert "jito-multi" in kr["fragile_senders"]
    # inferential set (n>=20) excludes the small landers.
    for small in ["jito-multi", "syncro-fra", "allenhark-quic-ny"]:
        assert small not in kr["inferential_senders"]
