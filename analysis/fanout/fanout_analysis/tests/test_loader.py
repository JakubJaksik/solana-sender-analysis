import pytest

from fanout_analysis import loader, constants, geo_offline

RUN = constants.DEFAULT_RUN
VEPOCH = constants.DEFAULT_VEPOCH
SVCSV = constants.DEFAULT_SVCSV


def test_load_long_shape_and_invariants():
    df = loader.load_long(RUN)
    assert len(df) == 10032
    assert df["trigger_id"].nunique() == 912
    assert df["sender_name"].nunique() == 11
    assert df.groupby("trigger_id").size().unique().tolist() == [11]
    assert df["tx_signature"].nunique() == 10032
    per_trig_land = (df.assign(l=(df.final_outcome == "Landed").astype(int))
                     .groupby("trigger_id").l.sum())
    assert per_trig_land.value_counts().to_dict() == {1: 846, 0: 66}
    assert int((df.final_outcome == "Landed").sum()) == 846


def test_assert_paired_hard_fails_on_bad_input(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trigger_id":1,"sender_name":"a","final_outcome":"Landed","tx_signature":"x"}\n'
                   '{"trigger_id":1,"sender_name":"a","final_outcome":"Landed","tx_signature":"x"}\n')
    with pytest.raises(AssertionError):
        loader.assert_paired(loader._read_raw(bad))


def test_enrich_geo_and_derived():
    df = loader.load_enriched(RUN, VEPOCH, SVCSV)
    assert df["leader_identity"].notna().all()
    assert df["sv_continent"].notna().mean() == 1.0
    assert "send_to_obs_ms" in df and "rtt_ms" in df
    assert df["land"].sum() == 846
    assert df["mask_operational"].all()
    assert (df["mask_conditional"] == (df.final_outcome != "SendError")).all()
    assert df.loc[df.sender_name == "allenhark-quic-tk", "never_sent"].all()
    assert int(df["helius_500"].sum()) == 182
    assert df.loc[df.sender_name == "allenhark-quic-tk", "send_order"].isna().all()


def test_wide_pivot():
    wide = loader.load_wide(RUN, VEPOCH, SVCSV)
    assert wide.shape == (912, 11)
    assert wide.isna().sum().sum() == 0
    assert int(wide.sum().sum()) == 846


def test_outcome_class_taxonomy():
    df = loader.load_enriched(RUN, VEPOCH, SVCSV)
    vc = df["outcome_class"].value_counts().to_dict()
    assert vc["WON"] == 846
    assert vc["THROTTLED_LOCAL"] == 1409
    assert vc["PROVIDER_REJECTED"] == 228
    assert vc["NEVER_SENT"] == 1118
    assert vc["SERVER_ERROR"] == 182
    assert vc["LOST_RACE"] == 6249
    assert sum(vc.values()) == 10032


def test_slot_and_leader_change_columns():
    df = loader.load_enriched(RUN, VEPOCH, SVCSV)
    assert int(df["same_slot"].sum()) == 586
    assert int(df["landed_next_slot"].sum()) == 260
    # leader holds 4 consecutive slots, so next-slot != next-leader
    assert int(df["landed_next_leader"].sum()) == 63
    assert int(df["landed_next_leader"].sum()) < int(df["landed_next_slot"].sum())


def test_all_leader_cities_have_coords():
    df = loader.load_enriched(RUN, VEPOCH, SVCSV)
    cities = set(df["sv_city"].dropna().unique())
    missing = cities - set(geo_offline.CITY_COORDS)
    assert not missing, f"cities without coords: {missing}"
