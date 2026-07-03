"""Single source of truth: read benchmark JSONL -> enriched LONG + WIDE frames.

Hard-fails on paired-design violations. Joins leader identity + geo, derives ms
columns, denominator masks, reclassification flags, and within-trigger send order.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fanout_analysis import constants

# Human-meaningful per-attempt outcome classes. LOST_RACE and NEVER_SENT are
# NOT errors: with 11 senders racing for 1 win/trigger, ~91% of attempts cannot
# win by construction. Only NEVER_SENT/THROTTLED/PROVIDER/SERVER are problems.
OUTCOME_CLASSES = [
    "WON", "LOST_RACE", "NEVER_SENT", "THROTTLED_LOCAL", "PROVIDER_REJECTED", "SERVER_ERROR",
]
OUTCOME_LABEL = {
    "WON": "Won (landed)",
    "LOST_RACE": "Lost race (sent, another sender won)",
    "NEVER_SENT": "Never sent (no tx left client)",
    "THROTTLED_LOCAL": "Throttled locally (client rate-limit)",
    "PROVIDER_REJECTED": "Provider rejected (HTTP 429)",
    "SERVER_ERROR": "Server error (HTTP 500)",
}


def _read_raw(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def assert_paired(df: pd.DataFrame) -> None:
    """Hard-fail if the matched-block precondition is violated."""
    n_senders = df["sender_name"].nunique()
    sizes = sorted(df.groupby("trigger_id").size().unique().tolist())
    assert sizes == [n_senders], f"unbalanced block: rows-per-trigger {sizes} != [{n_senders}]"
    assert df["tx_signature"].nunique() == len(df), "signatures not 1:1 with rows"
    land = df.assign(_l=(df["final_outcome"] == "Landed").astype(int)).groupby("trigger_id")["_l"].sum()
    assert land.max() <= 1, f"multi-land trigger survived cleaning: max={land.max()}"


def _drop_incomplete_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only triggers carrying the full (modal) sender set.

    Real runs have ragged triggers - almost always the shutdown tail (recorder still
    collecting when the run is killed) or a sender that missed a single dispatch. Those
    break the paired design. We take the modal sender set (what most triggers have) and
    drop triggers that don't match it. Dropped count -> df.attrs['dropped_incomplete_triggers'].
    """
    nsen = df.groupby("trigger_id")["sender_name"].nunique()
    modal = int(nsen.mode().iloc[0])
    ref_ids = nsen[nsen == modal].index
    full = set(df[df["trigger_id"].isin(ref_ids)]["sender_name"].unique())
    sets = df.groupby("trigger_id")["sender_name"].agg(lambda s: set(s))
    keep = sets[sets == full].index
    dropped = int(df["trigger_id"].nunique() - len(keep))
    out = df[df["trigger_id"].isin(keep)].copy()
    out.attrs["dropped_incomplete_triggers"] = dropped
    return out


def _drop_multiland_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop triggers where 2+ senders' txs both landed.

    Under durable-nonce a trigger's nonce is consumed once, so a double-land is an
    anomaly (separate nonce accounts / nonce-advance race), not a normal race result.
    We drop these as artifacts. Count -> df.attrs['dropped_multiland_triggers'].
    """
    landed = df[df["final_outcome"] == "Landed"]
    counts = landed.groupby("trigger_id").size()
    bad = counts[counts > 1].index
    out = df[~df["trigger_id"].isin(bad)].copy()
    out.attrs["dropped_multiland_triggers"] = int(len(bad))
    return out


def load_long(path, drop_incomplete=True, drop_multiland=True) -> pd.DataFrame:
    df = _read_raw(Path(path))
    inc = ml = 0
    if drop_incomplete:
        df = _drop_incomplete_triggers(df)
        inc = df.attrs.get("dropped_incomplete_triggers", 0)
    if drop_multiland:
        df = _drop_multiland_triggers(df)
        ml = df.attrs.get("dropped_multiland_triggers", 0)
    assert_paired(df)
    df.attrs["dropped_incomplete_triggers"] = inc
    df.attrs["dropped_multiland_triggers"] = ml
    return df


def _leader_maps(vepoch_path: Path):
    vj = json.load(open(vepoch_path))
    first = vj["epoch"]["absolute_slot"]
    slot_to_id = {}
    for ident, idxs in vj["schedule"].items():
        for si in idxs:
            slot_to_id[first + si] = ident
    meta = {v["identity"]: v for v in vj["validators"]}
    id_to_vote = {v["identity"]: v.get("vote_account") for v in vj["validators"]}
    return slot_to_id, meta, id_to_vote


def _geo_map(svcsv_path: Path):
    geo = {}
    for r in csv.DictReader(open(svcsv_path)):
        geo[r["vote_account"]] = {
            "sv_city": r.get("sv_city") or None,
            "sv_country": r.get("sv_country") or None,
            "sv_continent": r.get("sv_continent") or None,
            "sv_asn": r.get("sv_asn") or None,
            "sv_asn_org": r.get("sv_asn_org") or None,
            "country_mismatch": r.get("country_mismatch") == "True",
        }
    return geo


def load_enriched(run_path=constants.DEFAULT_RUN,
                  vepoch_path=constants.DEFAULT_VEPOCH,
                  svcsv_path=constants.DEFAULT_SVCSV) -> pd.DataFrame:
    df = load_long(run_path)
    slot_to_id, meta, id_to_vote = _leader_maps(Path(vepoch_path))
    geo = _geo_map(Path(svcsv_path))

    df["leader_identity"] = df["slot"].map(slot_to_id)
    df["leader_vote"] = df["leader_identity"].map(id_to_vote)
    df["leader_name"] = df["leader_identity"].map(lambda i: (meta.get(i) or {}).get("name"))
    df["leader_stake_sol"] = df["leader_identity"].map(
        lambda i: ((meta.get(i) or {}).get("active_stake_lamports") or 0) / 1e9)
    df["leader_dc"] = df["leader_identity"].map(lambda i: (meta.get(i) or {}).get("data_center_key"))
    for k in ["sv_city", "sv_country", "sv_continent", "sv_asn", "sv_asn_org", "country_mismatch"]:
        df[k] = df["leader_vote"].map(lambda v, k=k: (geo.get(v) or {}).get(k))

    # derived ms (None where source null)
    for src, dst in [("wall_send_rtt_ns", "rtt_ms"),
                     ("wall_send_to_observed_ns", "send_to_obs_ms"),
                     ("wall_trigger_to_observed_ns", "trigger_to_obs_ms")]:
        df[dst] = df[src] / 1e6

    df["land"] = (df["final_outcome"] == "Landed").astype(int)
    df["slots_behind"] = df["observed_slot"] - df["slot"]
    # chain-time latency in PoH ticks: trigger fire (slot,tick) -> landed (observed_slot,observed_tick).
    # Deterministic clock (64 ticks/slot), winners only. Analogue of send_to_obs_ms but in ticks.
    df["trigger_to_land_ticks"] = np.where(
        df["land"] == 1,
        (df["observed_slot"] - df["slot"]) * constants.TICKS_PER_SLOT
        + (df["observed_tick"] - df["tick"]),
        np.nan)

    # which leader actually produced the landing block (observed_slot -> leader).
    # NOTE: a leader holds 4 consecutive slots, so slots_behind>=1 does NOT imply a
    # different leader. landed_next_leader is the true "leader changed" event.
    df["observed_leader_identity"] = df["observed_slot"].map(slot_to_id)
    df["observed_leader_vote"] = df["observed_leader_identity"].map(id_to_vote)
    df["observed_leader_name"] = df["observed_leader_identity"].map(
        lambda i: (meta.get(i) or {}).get("name"))
    df["observed_leader_continent"] = df["observed_leader_vote"].map(
        lambda v: (geo.get(v) or {}).get("sv_continent"))
    df["same_slot"] = ((df["land"] == 1) & (df["slots_behind"] == 0))
    df["landed_next_slot"] = ((df["land"] == 1) & (df["slots_behind"] >= 1))
    df["landed_next_leader"] = ((df["land"] == 1)
                                & (df["observed_leader_identity"] != df["leader_identity"]))

    # denominator masks
    df["mask_operational"] = True
    df["mask_conditional"] = df["final_outcome"] != "SendError"
    df["mask_share"] = df["land"] == 1

    # reclassification flags (error catalog / corrected denominator)
    df["never_sent"] = df["send_at_ns"].fillna(0) == 0
    df["helius_500"] = (df["http_status"] == 500) & (df["final_outcome"] == "UnknownPending")

    # human-meaningful per-attempt outcome class (see OUTCOME_CLASSES)
    se = df["final_outcome"] == "SendError"
    up = df["final_outcome"] == "UnknownPending"
    throttled = se & (df["send_error"].astype(str).str.contains("throttl", case=False, na=False)
                      | ((df["rtt_ms"].fillna(-1) == 0) & (df["http_status"] != 429)))
    df["outcome_class"] = np.select(
        [df["land"] == 1,
         se & (df["http_status"] == 429),
         throttled,
         se,
         up & df["never_sent"],
         up & df["helius_500"]],
        ["WON", "PROVIDER_REJECTED", "THROTTLED_LOCAL", "PROVIDER_REJECTED",
         "NEVER_SENT", "SERVER_ERROR"],
        default="LOST_RACE")

    # protocol + sender region
    df["protocol_class"] = df["sender_name"].map(constants.PROTOCOL_OF)
    df["sender_region_continent"] = df["sender_name"].map(constants.SENDER_REGION_CONTINENT)
    df["continent_match"] = (df["sender_region_continent"] == df["sv_continent"]).astype(int)

    # send order within trigger from send_at_ns (>0 only; never-sent -> NaN)
    s = df["send_at_ns"].where(df["send_at_ns"] > 0)
    df["send_order"] = s.groupby(df["trigger_id"]).rank(method="first")
    return df


def load_wide(run_path=constants.DEFAULT_RUN,
              vepoch_path=constants.DEFAULT_VEPOCH,
              svcsv_path=constants.DEFAULT_SVCSV) -> pd.DataFrame:
    df = load_enriched(run_path, vepoch_path, svcsv_path)
    return df.pivot(index="trigger_id", columns="sender_name", values="land").fillna(0).astype(int)


def load_enriched_multi(run_paths, vepoch_path=constants.DEFAULT_VEPOCH,
                        svcsv_path=constants.DEFAULT_SVCSV, restrict_common=True) -> pd.DataFrame:
    """Pool several runs of the SAME epoch into one paired dataset.

    Each run is enriched independently (so per-run send_order/invariants hold), then
    trigger_id is namespaced as ``run_id#trigger_id`` so blocks never collide. If runs
    differ in sender set, we keep only the senders common to ALL runs (paired design
    needs identical columns); dropped senders are recorded in df.attrs['dropped_senders'].
    All runs must belong to the epoch of `vepoch_path` (else slots won't map to leaders).
    """
    if isinstance(run_paths, (str, Path)):
        run_paths = [run_paths]
    frames, sender_sets = [], []
    for p in run_paths:
        d = load_enriched(p, vepoch_path, svcsv_path)
        if d["leader_identity"].isna().any():
            raise ValueError(
                f"{p}: some slots do not map to a leader in {vepoch_path} - "
                f"runs must be from the same epoch as the validators file.")
        frames.append(d)
        sender_sets.append(set(d["sender_name"].unique()))
    common = set.intersection(*sender_sets) if sender_sets else set()
    dropped = sorted(set().union(*sender_sets) - common) if sender_sets else []
    norm = []
    for d in frames:
        if restrict_common and dropped:
            d = d[d["sender_name"].isin(common)].copy()
        norm.append(d)
    incomplete_per_run = {
        str(f["run_id"].iloc[0]): int(f.attrs.get("dropped_incomplete_triggers", 0))
        for f in frames}
    multiland_per_run = {
        str(f["run_id"].iloc[0]): int(f.attrs.get("dropped_multiland_triggers", 0))
        for f in frames}
    df = pd.concat(norm, ignore_index=True)
    df["trigger_id"] = df["run_id"].astype(str) + "#" + df["trigger_id"].astype(str)
    assert_paired(df)
    df.attrs["dropped_senders"] = dropped
    df.attrs["dropped_incomplete_per_run"] = incomplete_per_run
    df.attrs["dropped_multiland_per_run"] = multiland_per_run
    df.attrs["run_ids"] = sorted(str(x) for x in df["run_id"].unique())
    return df


def load_cached(run_path=constants.DEFAULT_RUN,
                vepoch_path=constants.DEFAULT_VEPOCH,
                svcsv_path=constants.DEFAULT_SVCSV,
                out_dir=None) -> pd.DataFrame:
    """Enrich once, cache to out_dir/enriched.parquet for fast re-runs."""
    if out_dir is None:
        return load_enriched(run_path, vepoch_path, svcsv_path)
    pq = Path(out_dir) / "enriched.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = load_enriched(run_path, vepoch_path, svcsv_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)
    return df
