import json

from fanout_analysis import loader, constants

RUN985 = constants.FANOUT_DIR / "runs" / "20260610-161813" / "triggers.jsonl"
VEP = constants.ANALYSIS_DIR / "validators-epoch-985.json"
SV = constants.FANOUT_DIR / "solanaview-crosscheck-985.csv"


def _split_run(tmp_path):
    rows = [json.loads(l) for l in open(RUN985)]
    tids = sorted({r["trigger_id"] for r in rows})
    half = set(tids[: len(tids) // 2])
    a = tmp_path / "A"; b = tmp_path / "B"; a.mkdir(); b.mkdir()
    with open(a / "triggers.jsonl", "w") as fa, open(b / "triggers.jsonl", "w") as fb:
        for r in rows:
            r2 = dict(r)
            if r["trigger_id"] in half:
                r2["run_id"] = "A"; fa.write(json.dumps(r2) + "\n")
            else:
                r2["run_id"] = "B"; fb.write(json.dumps(r2) + "\n")
    return a / "triggers.jsonl", b / "triggers.jsonl"


def test_multi_pool_preserves_blocks(tmp_path):
    a, b = _split_run(tmp_path)
    total = sum(1 for _ in open(RUN985))
    df = loader.load_enriched_multi([a, b], VEP, SV)
    assert len(df) == total
    assert df.groupby("trigger_id").size().unique().tolist() == [17]
    assert df.attrs["dropped_senders"] == []
    assert set(df.attrs["run_ids"]) == {"A", "B"}
    # composite ids keep runs separate
    assert df["trigger_id"].str.startswith("A#").any()
    assert df["trigger_id"].str.startswith("B#").any()


def test_multi_restricts_to_common_senders(tmp_path):
    a, b = _split_run(tmp_path)
    rows = [json.loads(l) for l in open(b)]
    keep = [r for r in rows if r["sender_name"] != "nozomi-fra"]
    with open(b, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in keep)
    df = loader.load_enriched_multi([a, b], VEP, SV)
    assert "nozomi-fra" in df.attrs["dropped_senders"]
    assert "nozomi-fra" not in df["sender_name"].unique()
    assert df.groupby("trigger_id").size().unique().tolist() == [16]
