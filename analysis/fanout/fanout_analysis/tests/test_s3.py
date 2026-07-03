from fanout_analysis import loader, constants, s3_pairwise


def _build_ctx(tmp_path):
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


def test_s3_golden_cochran_q_and_bradley_terry(tmp_path):
    ctx = _build_ctx(tmp_path)
    res = s3_pairwise.run(ctx)

    # Golden: Cochran's Q rejects global no-difference null at p < 1e-6
    assert res["key_results"]["cochran_p"] < 1e-6
    assert res["key_results"]["cochran_df"] == 10

    # Golden: 55 unordered pairs over 11 senders
    assert res["key_results"]["n_pairs"] == 55

    # Golden: Bradley-Terry top-2 ranks are 0slot-de1 and triton-fra
    assert set(res["key_results"]["bt_top2"]) == {"0slot-de1", "triton-fra"}

    # named CSVs created
    assert (tmp_path / "summary" / "cochrans-q.csv").exists()
    assert (tmp_path / "summary" / "pairwise-mcnemar.csv").exists()
    assert (tmp_path / "summary" / "bradley-terry-ranking.csv").exists()

    # named PNGs created
    assert (tmp_path / "plots" / "03-pairwise-dominance.png").exists()
    assert (tmp_path / "plots" / "03-pairwise-bradley-terry.png").exists()
    assert (tmp_path / "plots" / "03-pairwise-mcnemar-pmatrix.png").exists()


def test_s3_pairwise_mcnemar_has_55_rows(tmp_path):
    ctx = _build_ctx(tmp_path)
    s3_pairwise.run(ctx)
    import pandas as pd
    mc = pd.read_csv(tmp_path / "summary" / "pairwise-mcnemar.csv")
    assert len(mc) == 55
    # discordant counts equal the two senders' land counts (mutually exclusive landers)
    assert (mc["n_discordant"] == mc["b_a_only"] + mc["c_b_only"]).all()
