"""Golden tests for S7 (geography). Oracle: continent trigger totals
Europe 605, NA 219, Asia 72, SA 16; world-map html created; CSVs written."""
from fanout_analysis import loader, constants, s7_geo


def _build_ctx(tmp_path):
    df = loader.load_enriched()
    wide = loader.load_wide()
    import json
    config = json.load(open(constants.DEFAULT_CONFIG))
    outdir = tmp_path
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    return {
        "df": df, "wide": wide, "outdir": outdir, "config": config,
        "min_n": constants.GATE_INFERENTIAL, "min_indicative": constants.GATE_INDICATIVE,
    }


def test_s7_continent_trigger_totals_golden(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s7_geo.run(ctx)
    totals = res["key_results"]["continent_trigger_totals"]
    assert totals["Europe"] == 605
    assert totals["North America"] == 219
    assert totals["Asia"] == 72
    assert totals["South America"] == 16
    assert sum(totals.values()) == 912


def test_s7_writes_named_csvs(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s7_geo.run(ctx)
    summary = tmp_path / "summary"
    for name in ["sender-by-continent-landrate.csv", "sender-by-country-landrate.csv",
                 "sender-by-datacenter-landrate.csv", "geo-proximity-test.csv"]:
        assert (summary / name).exists(), f"missing {name}"
        assert name in str(res["tables"].values()) or any(name in str(p) for p in res["tables"].values())


def test_s7_world_map_html_created(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s7_geo.run(ctx)
    world_html = res["figures"]["world_map_html"]
    from pathlib import Path
    assert Path(world_html).exists(), "world map html not created"
    assert str(world_html).endswith(".html")


def test_s7_allenhark_proximity_ordering(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s7_geo.run(ctx)
    # operational land counts: fra 51, ams 47, ny 14, tk 0 (validated)
    overall = res["key_results"]["allenhark_overall_landed"]
    assert overall["allenhark-quic-fra"] == 51
    assert overall["allenhark-quic-ams"] == 47
    assert overall["allenhark-quic-ny"] == 14
    assert overall["allenhark-quic-tk"] == 0


def test_s7_continent_heatmap_and_plots_created(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s7_geo.run(ctx)
    from pathlib import Path
    for key in ["continent_heatmap", "allenhark_bar", "eu_noneu_slope", "country_heatmap"]:
        assert Path(res["figures"][key]).exists(), f"missing figure {key}"
