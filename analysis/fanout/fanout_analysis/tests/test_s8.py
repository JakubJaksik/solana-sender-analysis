"""Golden tests for S8 per-validator (per-leader) view (plan Task 17).

Golden oracle (this fixed run):
  - per-leader-summary rows == 133
  - sum of n_triggers == 912
  - the inferential count (leaders with n_triggers >= min_n) is DERIVED from the data
    and cross-checked against key_results; for THIS run it equals 6.
Plus: named CSV/TXT/PNG outputs are created and internally consistent.
"""
import json

import pandas as pd
import pytest

from fanout_analysis import constants, loader, s8_per_validator


@pytest.fixture(scope="module")
def enriched():
    return loader.load_enriched()


@pytest.fixture(scope="module")
def wide():
    return loader.load_wide()


def _make_ctx(df, wide, outdir):
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    return {
        "df": df,
        "wide": wide,
        "outdir": outdir,
        "config": {},
        "min_n": 20,
        "min_indicative": 5,
    }


def test_s8_golden_numbers_and_outputs(tmp_path, enriched, wide):
    ctx = _make_ctx(enriched, wide, tmp_path)
    result = s8_per_validator.run(ctx)

    assert result["id"] == "S8"

    summary_csv = tmp_path / "summary" / "per-leader-summary.csv"
    persender_csv = tmp_path / "summary" / "per-leader-per-sender.csv"
    tops_txt = tmp_path / "summary" / "per-validator-tops.txt"
    heatmap_png = tmp_path / "plots" / "08-per-validator-leader-sender-heatmap.png"
    stacked_png = tmp_path / "plots" / "08-per-validator-winning-sender-stacked.png"
    inclusion_png = tmp_path / "plots" / "08-per-validator-inclusion-rate.png"

    # files exist
    for p in (summary_csv, persender_csv, tops_txt, heatmap_png, stacked_png, inclusion_png):
        assert p.exists(), f"missing output {p}"
        assert p.stat().st_size > 0, f"empty output {p}"

    summary = pd.read_csv(summary_csv)
    min_n = ctx["min_n"]

    # --- GOLDEN ---
    assert len(summary) == 133
    assert int(summary["n_triggers"].sum()) == 912

    # inferential count is DERIVED from the data (works on any run), then cross-checked
    n_inferential = int((summary["n_triggers"] >= min_n).sum())
    # for THIS fixed run the derived value is 6
    assert n_inferential == 6

    # key_results mirror the golden numbers
    kr = result["key_results"]
    assert kr["n_leaders"] == len(summary)
    assert kr["sum_n_triggers"] == int(summary["n_triggers"].sum())
    assert kr["n_inferential_leaders"] == n_inferential

    # bucket counts sum to all leaders
    assert (kr["n_inferential_leaders"] + kr["n_indicative_leaders"]
            + kr["n_long_tail_leaders"]) == len(summary)

    # single-winner design accounting (846 landed, 66 no-winner)
    assert kr["total_landed_any"] == 846
    assert kr["total_no_winner"] == 66
    assert int(summary["landed_any"].sum()) == 846
    assert int(summary["no_winner"].sum()) == 66

    # landed_any + no_winner == n_triggers per leader
    assert ((summary["landed_any"] + summary["no_winner"]) == summary["n_triggers"]).all()

    # summary is stake-sorted (descending) by default
    assert summary["leader_stake_sol"].is_monotonic_decreasing

    # per-sender sparse table land counts sum to all landed triggers
    ls = pd.read_csv(persender_csv)
    assert int(ls["land_count"].sum()) == 846
    # every leader/sender land_count <= that leader's n_triggers (mutually exclusive winners)
    assert (ls["land_count"] <= ls["n_triggers"]).all()

    # gated leaders surfaced in key_results match the derived inferential count
    assert len(kr["gated_leaders"]) == n_inferential

    # every figure has a concise caption (required by the report renderer)
    captions = result["captions"]
    assert set(captions) == set(result["figures"])
    assert all(isinstance(c, str) and c.strip() for c in captions.values())


def test_s8_main_builds_and_runs(tmp_path):
    """The standalone _build_ctx + run path produces the golden summary."""
    ctx = s8_per_validator._build_ctx(tmp_path)
    result = s8_per_validator.run(ctx)
    kr = result["key_results"]
    assert kr["n_leaders"] == 133
    assert kr["sum_n_triggers"] == 912

    # n_inferential is derived from the summary (gating with ctx min_n), then cross-checked
    summary = pd.read_csv(tmp_path / "summary" / "per-leader-summary.csv")
    n_inferential = int((summary["n_triggers"] >= ctx["min_n"]).sum())
    assert kr["n_inferential_leaders"] == n_inferential
    # for THIS fixed run the derived value is 6
    assert n_inferential == 6
