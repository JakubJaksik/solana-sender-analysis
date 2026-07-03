"""
Per-leader breakdown for a bench run.

For each validator that appeared as either schedule_leader OR observed_leader,
produce a row with:

  Counts:
    scheduled        : # (slot, tick) pairs in our schedule where this validator
                       was the slot's leader (per leader_cache at run start)
    observed_here    : # tx OBSERVED with observed_leader = this validator
    landed_here      : # tx where schedule_leader == observed_leader == this
                       (we wanted them there AND they landed there)
    slipped_away     : # tx where schedule_leader == this but observed_leader != this
                       (their slot, but tx slipped to next leader)
    received_slipped : # tx where schedule_leader != this but observed_leader == this
                       (someone else's scheduled slot, but tx caught up here)
    pending_unknown  : # tx where schedule_leader == this but status = UNKNOWN_PENDING
    missed_prune     : # scheduled (slot, tick) pairs where this leader was scheduled
                       but observer never fired (tick never detected → no parquet row)

  Latency (only for tx where observed_leader = this):
    n, p50/p90/p99 send→observed (ms), median tick_delta

  Derived:
    inclusion_rate     : landed_here / scheduled    (% of "their" scheduled tx that
                                                    landed at them - best possible)
    slip_rate          : slipped_away / scheduled   (% slipped to next slot's leader)
    pending_rate       : pending_unknown / scheduled
    missed_rate        : missed_prune / scheduled
    capture_efficiency : (landed_here + slipped_away + pending + missed) / scheduled
                       (sanity check: should equal 1.0)

Plus geo (country, continent, dc, stake) from validators.app.

Outputs:
  - per-leader.csv       : full table, one row per validator
  - per-leader-tops.txt  : top-10 rankings by various metrics
"""
import argparse
import json
import sys
from pathlib import Path

import base58
import duckdb
import pandas as pd

CONTINENT = {
    "US": "North America", "CA": "North America", "MX": "North America",
    "DE": "Europe", "NL": "Europe", "FR": "Europe", "GB": "Europe", "UK": "Europe",
    "CH": "Europe", "IE": "Europe", "FI": "Europe", "SE": "Europe", "NO": "Europe",
    "DK": "Europe", "ES": "Europe", "IT": "Europe", "PT": "Europe", "AT": "Europe",
    "PL": "Europe", "CZ": "Europe", "RO": "Europe", "BG": "Europe", "BE": "Europe",
    "LT": "Europe", "LV": "Europe", "EE": "Europe", "HU": "Europe", "GR": "Europe",
    "UA": "Europe", "RU": "Europe", "IS": "Europe", "LU": "Europe", "MT": "Europe",
    "JP": "Asia", "SG": "Asia", "KR": "Asia", "IN": "Asia", "CN": "Asia",
    "HK": "Asia", "TW": "Asia", "TH": "Asia", "VN": "Asia", "ID": "Asia",
    "MY": "Asia", "PH": "Asia", "AE": "Asia", "IL": "Asia", "TR": "Asia",
    "AU": "Oceania", "NZ": "Oceania",
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "PE": "South America", "UY": "South America",
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "KE": "Africa", "MA": "Africa",
}


def parse_country(dck):
    if not dck or not isinstance(dck, str):
        return None
    parts = dck.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return None


def load_validators(path: Path) -> dict:
    vjson = json.load(open(path))
    out = {}
    for v in vjson["validators"]:
        cc = parse_country(v.get("data_center_key"))
        out[v["identity"]] = {
            "name": (v.get("name") or "").strip() or None,
            "country": cc,
            "continent": CONTINENT.get(cc, "Unknown") if cc else "Unknown",
            "dc": v.get("data_center_key"),
            "stake_lamports": v.get("active_stake_lamports") or 0,
        }
    return out


def load_leader_schedule(path: Path) -> dict:
    """slot (int) -> identity (base58).

    Actual on-disk shape: {"slots": {"<slot_str>": "<identity>", ...}}
    """
    raw = json.load(open(path))
    slots = raw.get("slots", raw)   # tolerate both shapes
    return {int(k): v for k, v in slots.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--validators", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="output path prefix (no extension)")
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading validators metadata")
    vmap = load_validators(args.validators)

    print(f"[2/5] loading leader schedule")
    leader_sched = load_leader_schedule(args.run_dir / "leader-schedule.json")

    print(f"[3/5] loading bench schedule (all scheduled (slot, tick) pairs)")
    bench_sched = json.load(open(args.run_dir / "schedule.json"))
    sched_entries = bench_sched["entries"]   # list of {slot, tick}
    print(f"      {len(sched_entries)} scheduled (slot, tick) entries")

    # Build dataframe of scheduled entries with leader assignment
    sched_df = pd.DataFrame(sched_entries)
    sched_df["schedule_leader"] = sched_df["slot"].map(leader_sched)
    sched_df = sched_df.rename(columns={"slot": "schedule_slot", "tick": "schedule_tick"})

    print(f"[4/5] loading parquet")
    parquet = args.run_dir / "tx-events.parquet"
    con = duckdb.connect()
    pq = con.execute(f"""
        SELECT
            schedule_slot,
            schedule_tick,
            observed_slot,
            observed_leader,
            status,
            (CAST(observed_at_ns AS BIGINT) - CAST(send_at_ns AS BIGINT))/1e6 AS wall_ms,
            tick_delta
        FROM '{parquet}'
    """).df()
    def _b58(b):
        if b is None or (isinstance(b, float) and pd.isna(b)):
            return None
        try:
            return base58.b58encode(bytes(b)).decode()
        except (TypeError, ValueError):
            return None
    pq["observed_leader_b58"] = pq["observed_leader"].apply(_b58)
    print(f"      {len(pq)} parquet rows")

    # Join schedule with parquet by (slot, tick)
    print(f"[5/5] joining + computing per-leader stats")
    merged = sched_df.merge(
        pq[["schedule_slot", "schedule_tick", "observed_slot",
            "observed_leader_b58", "status", "wall_ms", "tick_delta"]],
        on=["schedule_slot", "schedule_tick"],
        how="left",
    )
    merged["fate"] = merged.apply(
        lambda r: ("missed_prune" if pd.isna(r["status"])
                   else "pending_unknown" if r["status"] == "UNKNOWN_PENDING"
                   else "landed_here" if r["observed_leader_b58"] == r["schedule_leader"]
                   else "slipped_away"),
        axis=1
    )

    # All leaders appearing anywhere
    leaders = set(merged["schedule_leader"].dropna()) | set(merged["observed_leader_b58"].dropna())

    rows = []
    for pk in leaders:
        v = vmap.get(pk, {})
        # scheduled
        own_sched = merged[merged["schedule_leader"] == pk]
        scheduled = len(own_sched)
        landed_here = (own_sched["fate"] == "landed_here").sum()
        slipped_away = (own_sched["fate"] == "slipped_away").sum()
        pending = (own_sched["fate"] == "pending_unknown").sum()
        missed = (own_sched["fate"] == "missed_prune").sum()

        # observed here (regardless of who scheduled)
        here_obs = merged[(merged["status"] == "OBSERVED") &
                          (merged["observed_leader_b58"] == pk)]
        received_slipped = (here_obs["schedule_leader"] != pk).sum()

        latency = here_obs["wall_ms"].dropna()
        ticks   = here_obs["tick_delta"].dropna()
        rows.append({
            "identity": pk,
            "name": v.get("name"),
            "country": v.get("country"),
            "continent": v.get("continent"),
            "dc": v.get("dc"),
            "stake_sol": v.get("stake_lamports", 0) // 1_000_000_000,
            # scheduled-side
            "scheduled": scheduled,
            "landed_here": int(landed_here),
            "slipped_away": int(slipped_away),
            "pending_unknown": int(pending),
            "missed_prune": int(missed),
            "inclusion_rate":  round(landed_here / scheduled, 4) if scheduled else None,
            "slip_rate":       round(slipped_away / scheduled, 4) if scheduled else None,
            "missed_rate":     round(missed / scheduled, 4) if scheduled else None,
            "pending_rate":    round(pending / scheduled, 4) if scheduled else None,
            # observed-side (latency)
            "observed_here":  len(here_obs),
            "received_slipped": int(received_slipped),
            "p50_ms": round(latency.median(), 2) if len(latency) else None,
            "p90_ms": round(latency.quantile(0.90), 2) if len(latency) else None,
            "p99_ms": round(latency.quantile(0.99), 2) if len(latency) else None,
            "min_ms": round(latency.min(), 2) if len(latency) else None,
            "max_ms": round(latency.max(), 2) if len(latency) else None,
            "p50_tick_delta": ticks.median() if len(ticks) else None,
            "p99_tick_delta": ticks.quantile(0.99) if len(ticks) else None,
        })

    df = pd.DataFrame(rows)
    # Sort by stake desc as default
    df = df.sort_values("stake_sol", ascending=False)

    csv_path = args.out_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"      wrote {csv_path} ({len(df)} leaders)")

    # ---------- text summary: top rankings ----------
    lines = []

    def section(title, sub):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        sub = sub.copy()
        sub["name"] = sub["name"].fillna("(unnamed)")
        lines.append(sub.to_string(index=False))

    # only validators with enough sample for meaningful latency stats
    REL = df[df["observed_here"] >= 20].copy()

    section(
        "TOP 10 FASTEST (median wall_ms; min 20 tx)",
        REL.nsmallest(10, "p50_ms")[
            ["name", "country", "dc", "observed_here", "p50_ms", "p99_ms",
             "p50_tick_delta", "stake_sol"]]
    )

    section(
        "TOP 10 SLOWEST (median wall_ms; min 20 tx)",
        REL.nlargest(10, "p50_ms")[
            ["name", "country", "dc", "observed_here", "p50_ms", "p99_ms",
             "p50_tick_delta", "stake_sol"]]
    )

    section(
        "TOP 10 WORST TAIL (p99 wall_ms; min 20 tx)",
        REL.nlargest(10, "p99_ms")[
            ["name", "country", "dc", "observed_here", "p50_ms", "p99_ms",
             "stake_sol"]]
    )

    # had slot but tx didn't reach (highest missed_rate among leaders we scheduled to)
    SCHED = df[df["scheduled"] >= 20].copy()
    section(
        "TOP 10 MOST PRUNED (scheduled but tick never observed; min 20 scheduled)",
        SCHED.nlargest(10, "missed_rate")[
            ["name", "country", "dc", "scheduled", "missed_prune", "missed_rate",
             "landed_here", "stake_sol"]]
    )

    section(
        "TOP 10 MOST SLIP (tx fell through to the next leader; min 20 scheduled)",
        SCHED.nlargest(10, "slip_rate")[
            ["name", "country", "dc", "scheduled", "slipped_away", "slip_rate",
             "landed_here", "stake_sol"]]
    )

    section(
        "TOP 10 MOST PENDING (tx sent, never observed; min 20 scheduled)",
        SCHED.nlargest(10, "pending_rate")[
            ["name", "country", "dc", "scheduled", "pending_unknown", "pending_rate",
             "stake_sol"]]
    )

    section(
        "TOP 10 HIGHEST INCLUSION RATE (landed_here/scheduled; min 50 scheduled)",
        df[df["scheduled"] >= 50].nlargest(10, "inclusion_rate")[
            ["name", "country", "dc", "scheduled", "landed_here", "inclusion_rate",
             "slip_rate", "p50_ms"]]
    )

    section(
        "TOP 10 RECEIVED-SLIPPED (leader caught tx scheduled for the PREVIOUS leader; benefits from others' slip)",
        df[df["received_slipped"] >= 5].nlargest(10, "received_slipped")[
            ["name", "country", "dc", "observed_here", "received_slipped",
             "scheduled", "landed_here", "p50_ms"]]
    )

    txt_path = args.out_prefix.with_name(args.out_prefix.name + "-tops.txt")
    txt_path.write_text("\n".join(lines))
    print(f"      wrote {txt_path}")

    # Global sanity
    total_sched = df["scheduled"].sum()
    total_landed = df["landed_here"].sum()
    total_slipped = df["slipped_away"].sum()
    total_pending = df["pending_unknown"].sum()
    total_missed = df["missed_prune"].sum()
    print()
    print("=== Sanity ===")
    print(f"  scheduled total:   {total_sched}")
    print(f"    landed_here:     {total_landed} ({total_landed/total_sched*100:.2f}%)")
    print(f"    slipped_away:    {total_slipped} ({total_slipped/total_sched*100:.2f}%)")
    print(f"    pending_unknown: {total_pending} ({total_pending/total_sched*100:.2f}%)")
    print(f"    missed_prune:    {total_missed} ({total_missed/total_sched*100:.2f}%)")
    print(f"  sum check:         {total_landed+total_slipped+total_pending+total_missed} (should = scheduled)")


if __name__ == "__main__":
    main()
