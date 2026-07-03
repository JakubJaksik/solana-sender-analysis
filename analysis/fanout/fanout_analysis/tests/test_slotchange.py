"""Golden tests for S6B (slot-change / leader-change summary)."""
import json
from pathlib import Path

from fanout_analysis import constants, loader, s_slotchange


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


def test_slotchange_global_golden(tmp_path):
    res = s_slotchange.run(_ctx(tmp_path))
    assert res["id"] == "SC"
    kr = res["key_results"]
    # GOLDEN: among 846 wins
    assert kr["n_landed"] == 846
    assert kr["same_slot"] == 586
    assert kr["next_slot"] == 260
    assert kr["next_leader"] == 63
    # same + next_slot == total landed (every win is either same-slot or late)
    assert kr["same_slot"] + kr["next_slot"] == kr["n_landed"]
    # of the 260 next-slot landings, 63 changed leader, 197 stayed with same leader
    assert kr["next_slot_same_leader"] == 197


def test_slotchange_per_sender_next_slot_pct_golden(tmp_path):
    res = s_slotchange.run(_ctx(tmp_path))
    by_pop = res["key_results"]["next_slot_pct_by_pop"]
    assert by_pop["0slot-de1"] == 44.7
    assert by_pop["allenhark-fra"] == 15.7
    assert by_pop["allenhark-ny"] == 85.7


def test_slotchange_csv_and_png_written(tmp_path):
    res = s_slotchange.run(_ctx(tmp_path))
    summ = tmp_path / "summary"
    plots = tmp_path / "plots"
    assert (summ / "slot-change-summary.csv").exists()
    assert (summ / "leader-change-detail.csv").exists()
    assert (plots / "06b-SC-slotchange-per-sender.png").exists()
    assert (plots / "06b-SC-global-slot-vs-leader.png").exists()
    for fig_path in res["figures"].values():
        assert Path(fig_path).exists()
    # every figure has a caption
    assert set(res["captions"].keys()) == set(res["figures"].keys())


def test_slotchange_leader_change_detail_rows(tmp_path):
    import pandas as pd
    s_slotchange.run(_ctx(tmp_path))
    detail = pd.read_csv(tmp_path / "summary" / "leader-change-detail.csv")
    # one row per landing where the leader actually changed
    assert len(detail) == 63
    # detail observed_slot is always strictly later than the intended slot
    assert (detail["observed_slot"] > detail["intended_slot"]).all()
