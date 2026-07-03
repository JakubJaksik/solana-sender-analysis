import json

from fanout_analysis import constants, loader
from fanout_analysis import s11_synthesis as s11


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


def test_s11_golden_verdicts(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s11.run(ctx)

    verdicts = result["key_results"]["verdicts"]
    # golden: dead PoP -> FIX-BUG
    assert verdicts["allenhark-quic-tk"] == "FIX-BUG"
    # golden: rate-limited -> IMPROVE
    assert verdicts["jito-multi"] == "IMPROVE"
    assert verdicts["syncro-fra"] == "IMPROVE"
    # golden: high conditional win-rate + good coverage -> INVEST
    assert verdicts["0slot-de1"] == "INVEST"
    assert verdicts["triton-fra"] == "INVEST"

    # exactly the two best senders are INVEST
    assert set(result["key_results"]["invest"]) == {"0slot-de1", "triton-fra"}
    assert set(result["key_results"]["improve"]) == {"jito-multi", "syncro-fra"}


def test_s11_scorecard_csv_contents(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s11.run(ctx)

    csv_path = tmp_path / "summary" / "sender-scorecard.csv"
    assert csv_path.exists()
    sc = __import__("pandas").read_csv(csv_path)
    assert sc.shape[0] == 11
    assert set(sc["verdict"].unique()) <= {"INVEST", "IMPROVE", "DEPRIORITIZE", "FIX-BUG"}

    row = sc.set_index("sender_name")
    # tip wired from config, not hardcoded
    assert int(row.loc["0slot-de1", "tip_lamports"]) == 1000000
    assert int(row.loc["triton-fra", "tip_lamports"]) == 0
    # conditional win-rate matches Landed/(Landed+UnknownPending)
    assert abs(row.loc["0slot-de1", "conditional_winrate"] - 253 / (253 + 659)) < 1e-9
    assert abs(row.loc["triton-fra", "conditional_winrate"] - 221 / (221 + 691)) < 1e-9
    # triton tips are zero -> zero cost-per-landing
    assert row.loc["triton-fra", "cost_per_landing_sol"] == 0.0
    # tk never sent -> zero coverage
    assert row.loc["allenhark-quic-tk", "coverage"] == 0.0


def test_s11_files_created(tmp_path):
    ctx = _build_ctx(tmp_path)
    result = s11.run(ctx)

    assert (tmp_path / "summary" / "sender-scorecard.csv").exists()
    assert (tmp_path / "plots" / "11-synthesis-scorecard.png").exists()
    assert (tmp_path / "plots" / "11-synthesis-quadrant.png").exists()
    # returned paths point at the real files
    assert (tmp_path / result["tables"]["sender-scorecard"]).exists() or \
        __import__("pathlib").Path(result["tables"]["sender-scorecard"]).exists()
    for fig_path in result["figures"].values():
        assert __import__("pathlib").Path(fig_path).exists()
