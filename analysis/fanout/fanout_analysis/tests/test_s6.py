"""Golden tests for S6 (intra-slot position & slots-behind)."""
import json
from pathlib import Path

from fanout_analysis import constants, loader, s6_intraslot


def _ctx(tmp_path):
    (tmp_path / "plots").mkdir(parents=True, exist_ok=True)
    (tmp_path / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = {}
    if Path(constants.DEFAULT_CONFIG).exists():
        config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df,
        "wide": wide,
        "outdir": tmp_path,
        "config": config,
        "min_n": constants.GATE_INFERENTIAL,
        "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s6_global_slots_behind_golden(tmp_path):
    res = s6_intraslot.run(_ctx(tmp_path))
    assert res["id"] == "S6"
    # GOLDEN: global slots_behind value_counts among Landed
    assert res["key_results"]["global_slots_behind"] == {0: 586, 1: 255, 2: 4, 3: 1}
    assert res["key_results"]["n_landed"] == 846


def test_s6_csv_and_png_written(tmp_path):
    res = s6_intraslot.run(_ctx(tmp_path))
    summ = tmp_path / "summary"
    plots = tmp_path / "plots"
    # named CSVs from the spec section
    assert (summ / "observed-tick-per-sender.csv").exists()
    assert (summ / "slots-behind-per-sender.csv").exists()
    # the four PNGs
    for fig_path in res["figures"].values():
        assert Path(fig_path).exists()
    assert (plots / "06-observed-tick-by-sender.png").exists()
    assert (plots / "06-slots-behind-stacked.png").exists()
    assert (plots / "06-rtt-vs-slots-behind.png").exists()
    assert (plots / "06-landrate-by-tick-bucket.png").exists()
    # every figure has a concise caption (caption mechanism is required)
    caps = res.get("captions") or {}
    assert set(caps) == set(res["figures"])
    assert all(c.strip() for c in caps.values())


def test_s6_slots_behind_per_sender_sums_match_global(tmp_path):
    import pandas as pd
    s6_intraslot.run(_ctx(tmp_path))
    sb = pd.read_csv(tmp_path / "summary" / "slots-behind-per-sender.csv")
    # 10 senders land at least once (allenhark-quic-tk lands 0 -> absent)
    assert len(sb) == 10
    assert int(sb["n_landed"].sum()) == 846
    # P(same)+P(one)+P(>=2) == 1 for every sender
    tot = sb["p_same_slot"] + sb["p_one_behind"] + sb["p_two_plus_behind"]
    assert ((tot - 1.0).abs() < 1e-9).all()


def test_s6_rotation_and_tick_bucket_are_flat_null(tmp_path):
    res = s6_intraslot.run(_ctx(tmp_path))
    kr = res["key_results"]
    # rotation slot%4 land-rate flat -> verified null (p well above 0.05)
    rlo, rhi = kr["rotation_land_rate_range"]
    assert (rhi - rlo) < 0.01
    assert kr["rotation_p"] > 0.05
    # per-trigger win-rate (won_any / n) is flat across firing tick (~91-94%);
    # this is per-TRIGGER now (846/912 framing), not per-attempt, so the spread
    # is wider than the old per-attempt rate but still flat/null.
    blo, bhi = kr["tick_bucket_land_rate_range"]
    assert (bhi - blo) < 0.05
    assert blo > 0.90 and bhi < 0.95


def test_s6_tick_bucket_outcome_split_shows_slot_transition(tmp_path):
    import pandas as pd
    s6_intraslot.run(_ctx(tmp_path))
    b = pd.read_csv(tmp_path / "summary" / "land-rate-by-tick-bucket.csv")
    # n = triggers per bucket -> totals are 912 triggers / 846 wins
    assert int(b["n"].sum()) == 912
    assert int(b["won_any"].sum()) == 846
    # same-slot + next-slot + no-win == n per bucket
    assert (b["won_same_slot"] + b["won_next_slot"] + b["no_win"] == b["n"]).all()
    # slot transition: early-tick triggers win same-slot, late-tick triggers win next-slot
    early = b.set_index("tick_bucket").loc["01-16"]
    late = b.set_index("tick_bucket").loc["49-64"]
    assert early["won_same_slot"] > early["won_next_slot"]
    assert late["won_next_slot"] > late["won_same_slot"]


def test_s6_spearman_rtt_slots_behind_positive(tmp_path):
    res = s6_intraslot.run(_ctx(tmp_path))
    kr = res["key_results"]
    assert kr["spearman_n"] == 846
    assert kr["spearman_rtt_vs_slots_behind_rho"] > 0
    assert kr["spearman_rtt_vs_slots_behind_p"] < 0.05
