"""Golden tests for S10 (cost / ROI)."""
import json

import pandas as pd

from fanout_analysis import constants, loader, s10_cost


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


def test_s10_tips_match_config_and_triton_tip_zero(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s10_cost.run(ctx)
    kr = res["key_results"]

    # GOLDEN: tips loaded match config exactly
    assert kr["tips_lamports"]["0slot-de1"] == 1000000
    assert kr["tips_lamports"]["triton-fra"] == 0
    assert kr["tips_lamports"]["helius-dual"] == 200000
    assert kr["tip_0slot_de1"] == 1000000
    assert kr["tip_triton_fra"] == 0
    assert kr["tip_helius_dual"] == 200000

    # GOLDEN: triton-fra tip component cost == 0 (tip is 0)
    assert kr["triton_tip_component_lamports"] == 0


def test_s10_cost_csv_and_archetypes(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s10_cost.run(ctx)

    cost_csv = tmp_path / "summary" / "cost-roi.csv"
    assert cost_csv.exists()
    cost = pd.read_csv(cost_csv)

    # required columns from spec (stale_flag renamed -> tip_source_stale)
    for col in ["sender", "archetype", "tip_floor_lamports", "attempts_paid",
                "landings", "cost_per_landing_sol", "assumption_source", "tip_source_stale"]:
        assert col in cost.columns, f"missing column {col}"

    assert len(cost) == 11
    assert set(cost["archetype"].unique()) <= {"per_tx", "subscription"}

    # triton-fra: subscription archetype, tip 0 -> cost_per_landing_sol == 0
    triton = cost[cost["sender"] == "triton-fra"].iloc[0]
    assert triton["archetype"] == "subscription"
    assert triton["tip_floor_lamports"] == 0
    assert triton["cost_per_landing_sol"] == 0.0

    # 0slot-de1: per-tx, tip 1_000_000, attempts_paid 912, landings 253
    z = cost[cost["sender"] == "0slot-de1"].iloc[0]
    assert z["archetype"] == "per_tx"
    assert z["tip_floor_lamports"] == 1000000
    assert z["attempts_paid"] == 912
    assert z["landings"] == 253
    expected_cost = 1000000 * 912 / 253 / 1e9
    assert abs(z["cost_per_landing_sol"] - expected_cost) < 1e-12

    # throttled senders are tip_source_stale flagged
    for thr in ["jito-multi", "syncro-fra", "blockrazor"]:
        assert bool(cost[cost["sender"] == thr].iloc[0]["tip_source_stale"]) is True


def test_s10_figures_created(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s10_cost.run(ctx)
    pareto = tmp_path / "plots" / "10-cost-landrate-vs-tip-pareto.png"
    bar = tmp_path / "plots" / "10-cost-per-landing-bar.png"
    assert pareto.exists() and pareto.stat().st_size > 0
    assert bar.exists() and bar.stat().st_size > 0
    # returned figure paths point at the created files
    assert res["figures"]["pareto"] == str(pareto)
    assert res["figures"]["cost_per_landing_bar"] == str(bar)
