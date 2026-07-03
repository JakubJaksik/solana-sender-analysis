"""Golden tests for S5 - unpaired latency shift (Task 14)."""
import pandas as pd

from fanout_analysis import constants, loader, s5_latency_shift

# Senders with Landed n>=20 (validated): these MUST be the only ones in the
# pairwise output. jito-multi (n=1), syncro-fra (n=7), allenhark-quic-ny (n=14)
# are excluded; allenhark-quic-tk has zero landings.
GATED = {
    "0slot-de1", "triton-fra", "blockrazor", "helius-dual",
    "allenhark-quic-fra", "allenhark-quic-ams", "helius-fra",
}
EXCLUDED = {"jito-multi", "syncro-fra", "allenhark-quic-ny", "allenhark-quic-tk"}


def _ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "df": loader.load_enriched(),
        "wide": loader.load_wide(),
        "outdir": tmp_path,
        "config": {},
        "min_n": 20,
        "min_indicative": 5,
    }


def test_only_gated_senders_appear(tmp_path):
    res = s5_latency_shift.run(_ctx(tmp_path))

    assert res["id"] == "S5"
    assert set(res["key_results"]["gated_senders"]) == GATED
    assert set(res["key_results"]["excluded_senders"]) == EXCLUDED

    csv_path = tmp_path / "summary" / "pairwise-latency-shift.csv"
    assert csv_path.exists()
    pair_df = pd.read_csv(csv_path)

    senders_in_pairs = set(pair_df["sender_a"]) | set(pair_df["sender_b"])
    assert senders_in_pairs == GATED
    # none of the excluded senders may leak into any pair
    assert senders_in_pairs.isdisjoint(EXCLUDED)
    # 7 gated senders -> C(7,2) = 21 unordered pairs
    assert len(pair_df) == 21


def test_pairwise_columns_and_fdr(tmp_path):
    res = s5_latency_shift.run(_ctx(tmp_path))
    pair_df = pd.read_csv(tmp_path / "summary" / "pairwise-latency-shift.csv")

    for col in ["hl_shift_a_minus_b_ms", "mw_u", "mw_p", "mw_p_fdr",
                "p50_diff_a_minus_b_ms", "p50_diff_ci_lo", "p50_diff_ci_hi",
                "p90_diff_a_minus_b_ms", "p90_diff_ci_lo", "p90_diff_ci_hi"]:
        assert col in pair_df.columns

    # FDR-adjusted p >= raw p, and within [0, 1]
    assert (pair_df["mw_p_fdr"] >= pair_df["mw_p"] - 1e-9).all()
    assert pair_df["mw_p_fdr"].between(0.0, 1.0).all()
    # CI brackets the point estimate
    assert (pair_df["p50_diff_ci_lo"] <= pair_df["p50_diff_a_minus_b_ms"] + 1e-6).all()
    assert (pair_df["p50_diff_ci_hi"] >= pair_df["p50_diff_a_minus_b_ms"] - 1e-6).all()


def test_fastest_baseline_and_files(tmp_path):
    res = s5_latency_shift.run(_ctx(tmp_path))

    # helius-dual has the smallest p50 send_to_obs_ms among gated senders.
    assert res["key_results"]["fastest_p50_sender"] == "helius-dual"

    assert (tmp_path / "plots" / "05-latency-shift-hl-heatmap.png").exists()
    assert (tmp_path / "plots" / "05-latency-shift-p50-p90-dumbbell.png").exists()

    # paired-not-estimable caveat must be stated
    assert any("paired" in n.lower() and "not estimable" in n.lower() for n in res["notes"])


def test_captions_cover_every_figure(tmp_path):
    res = s5_latency_shift.run(_ctx(tmp_path))
    captions = res["captions"]
    # every figure key must have a non-empty caption
    for name in res["figures"]:
        assert name in captions and captions[name].strip()


def test_per_continent_view(tmp_path):
    res = s5_latency_shift.run(_ctx(tmp_path))

    # Europe (all 7 gated have n>=20) and North America (0slot-de1, triton-fra)
    # both qualify with >=2 senders.
    assert set(res["key_results"]["continents_compared"]) == {"Europe", "North America"}

    cont_csv = tmp_path / "summary" / "latency-shift-by-continent.csv"
    assert cont_csv.exists()
    cont_df = pd.read_csv(cont_csv)

    for col in ["continent", "sender", "n", "p50_ms", "p90_ms",
                "fastest_on_continent", "hl_shift_vs_fastest_ms"]:
        assert col in cont_df.columns

    # within each continent at least 2 senders are gated, all with n>=20
    assert (cont_df["n"] >= 20).all()
    eu = cont_df[cont_df["continent"] == "Europe"]
    assert len(eu) == 7
    # helius-dual is the fastest in Europe -> its own HL shift vs itself is 0
    assert eu.loc[eu["sender"] == "helius-dual", "hl_shift_vs_fastest_ms"].iloc[0] == 0.0

    assert (tmp_path / "plots" / "05-latency-shift-by-continent.png").exists()
