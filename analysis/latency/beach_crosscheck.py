"""
Cross-check validator location: Solana Beach vs validators.app vs latency.

For each validator OBSERVED in a bench run:
  1. Pull location from Solana Beach API (best label source, per our findings).
  2. Compare with current validators-epoch-NNN.json (validators.app).
  3. Add the latency-based sanity check: median wall_ms vs expected RTT
     for the claimed country.

Outputs:
  - <out>.csv      : full table with all sources side-by-side
  - <out>-patches.json : suggested patches to validators-epoch-NNN.json
                         (only entries where Solana Beach disagrees with vapp)

Solana Beach API:
  Free tier requires registration at https://app.solanabeach.io/dashboard.
  Export key as env var SOLANA_BEACH_API_KEY before running.

Endpoint:
  GET https://api.solanabeach.io/v1/validator/{vote_account}
  Header: Authorization: Bearer <SOLANA_BEACH_API_KEY>

Rate limit:
  Free tier ~50 req/min. Script auto-sleeps if 429.

Usage:
  export SOLANA_BEACH_API_KEY=...
  python beach_crosscheck.py \
      --run-dir path/to/run \
      --validators path/to/validators-epoch-NNN.json \
      --out summary/beach-crosscheck
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import base58
import duckdb
import pandas as pd
import requests

BEACH_API = "https://api.solanabeach.io/v1"


# Expected wall_ms (Fra → validator + shred back) by country.
# Floor estimate using ~2× speed-of-light + propagation overhead.
# If measured median is < floor by 2x, the geo label is suspect.
EXPECTED_MIN_WALL_MS = {
    # Western/Central EU (close to Fra)
    "DE": 5, "NL": 10, "FR": 15, "BE": 10, "LU": 5, "CH": 10, "AT": 15,
    # Northern / Western EU
    "GB": 25, "IE": 30, "DK": 20, "SE": 30, "NO": 30, "FI": 35,
    # Southern EU
    "ES": 30, "IT": 25, "PT": 40, "GR": 40,
    # Eastern EU
    "PL": 20, "CZ": 15, "RO": 35, "HU": 25, "BG": 40,
    "LT": 30, "LV": 35, "EE": 40, "UA": 40, "RU": 40,
    "IS": 50, "MT": 40,
    # Near East / W Asia
    "TR": 70, "IL": 70, "AE": 130,
    # E Asia
    "IN": 130, "JP": 230, "SG": 180, "KR": 240, "HK": 200,
    "TW": 230, "TH": 200, "VN": 200, "ID": 220, "MY": 200, "PH": 240, "CN": 200,
    # NA
    "US": 90, "CA": 100, "MX": 130,
    # SA
    "BR": 200, "AR": 220, "CL": 230, "CO": 200, "PE": 230, "UY": 220,
    # Oceania / Africa
    "AU": 280, "NZ": 290,
    "ZA": 160, "EG": 80, "NG": 130, "KE": 170, "MA": 60,
}


def parse_country(data_center_key):
    if not data_center_key or not isinstance(data_center_key, str):
        return None
    parts = data_center_key.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return None


def load_vapp(path: Path):
    vjson = json.load(open(path))
    by_identity = {}
    by_vote = {}
    for v in vjson["validators"]:
        rec = {
            "identity": v["identity"],
            "vote_account": v.get("vote_account"),
            "name": (v.get("name") or "").strip() or None,
            "vapp_country": parse_country(v.get("data_center_key")),
            "vapp_dc": v.get("data_center_key"),
            "vapp_asn": v.get("asn"),
            "vapp_asn_org": v.get("asn_organization"),
            "vapp_ip": v.get("ip"),
        }
        by_identity[v["identity"]] = rec
        if v.get("vote_account"):
            by_vote[v["vote_account"]] = rec
    return by_identity, by_vote


def load_observed(run_dir: Path):
    """Pubkeys + median measured latency per validator."""
    parquet = run_dir / "tx-events.parquet"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            observed_leader,
            (CAST(observed_at_ns AS BIGINT) - CAST(send_at_ns AS BIGINT))/1e6 AS wall_ms
        FROM '{parquet}'
        WHERE status='OBSERVED' AND observed_leader IS NOT NULL
    """).df()
    df["identity"] = df["observed_leader"].apply(
        lambda b: base58.b58encode(bytes(b)).decode() if b is not None else None
    )
    stats = df.groupby("identity").agg(
        n=("wall_ms", "count"),
        min_wall=("wall_ms", "min"),
        median_wall=("wall_ms", "median"),
        p99_wall=("wall_ms", lambda s: s.quantile(0.99)),
    ).reset_index()
    return stats


def fetch_beach(vote_account: str, api_key: str):
    """Fetch single validator from Solana Beach. Returns dict or None.

    Response shape (typical):
      { 'votePubkey': ..., 'nodePubkey': ..., 'name': ...,
        'dcInfo': { 'countryCode': 'IE', 'city': 'Dublin',
                    'asn': 1072, 'asnInfo': {'name': 'Payward'} },
        'ip': '193.118.169.103', ... }
    Endpoint and shape based on observed responses; adjust if Beach changes API.
    """
    url = f"{BEACH_API}/validator/{vote_account}"
    headers = {"Authorization": f"Bearer {api_key}"}
    while True:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 429:
            print("  429 rate-limited; sleeping 60s")
            time.sleep(60)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


def extract_beach_geo(beach_data: dict) -> dict:
    """Normalize possible Beach response shapes."""
    if not beach_data:
        return {"beach_country": None, "beach_city": None,
                "beach_asn": None, "beach_asn_org": None, "beach_ip": None}
    # try multiple possible field names
    dc = beach_data.get("dcInfo") or beach_data.get("dc") or {}
    country = (dc.get("countryCode") or beach_data.get("countryCode")
               or dc.get("country") or beach_data.get("country"))
    city = dc.get("city") or beach_data.get("city") or dc.get("region")
    asn = dc.get("asn") or beach_data.get("asn")
    asn_org = (dc.get("asnInfo", {}).get("name") if isinstance(dc.get("asnInfo"), dict) else None) \
              or beach_data.get("asnOrg") or beach_data.get("asn_organization")
    ip = beach_data.get("ip") or beach_data.get("gossipIP")
    return {
        "beach_country": country.upper() if country else None,
        "beach_city": city,
        "beach_asn": asn,
        "beach_asn_org": asn_org,
        "beach_ip": ip,
    }


def latency_consistent(country: str | None, median_wall: float) -> bool | None:
    """True if measured median is plausible for the country, False if not,
    None if we have no expectation."""
    if not country or pd.isna(median_wall):
        return None
    floor = EXPECTED_MIN_WALL_MS.get(country)
    if floor is None:
        return None
    return median_wall >= 0.5 * floor   # allow 2× slack downward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--validators", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output prefix (no extension)")
    args = ap.parse_args()

    api_key = os.environ.get("SOLANA_BEACH_API_KEY")
    if not api_key:
        print("ERROR: set SOLANA_BEACH_API_KEY env var")
        print("Sign up: https://app.solanabeach.io/dashboard")
        sys.exit(2)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading validators.app from {args.validators}")
    by_id, by_vote = load_vapp(args.validators)
    print(f"      {len(by_id)} entries")

    print(f"[2/4] loading observed pubkeys + latency stats from {args.run_dir}")
    obs = load_observed(args.run_dir)
    print(f"      {len(obs)} unique observed leaders")

    print(f"[3/4] querying Solana Beach for {len(obs)} validators...")
    rows = []
    for i, row in enumerate(obs.itertuples(index=False)):
        pk = row.identity
        vapp = by_id.get(pk, {})
        vote = vapp.get("vote_account")
        beach_geo = {"beach_country": None, "beach_city": None,
                     "beach_asn": None, "beach_asn_org": None, "beach_ip": None}
        if vote:
            try:
                beach_data = fetch_beach(vote, api_key)
                beach_geo = extract_beach_geo(beach_data)
            except Exception as e:
                print(f"  fail {pk[:8]}: {e}")
        rows.append({
            "identity": pk,
            "vote_account": vote,
            "name": vapp.get("name"),
            "n_tx": row.n,
            "median_wall_ms": round(row.median_wall, 2),
            "min_wall_ms":    round(row.min_wall, 2),
            "vapp_country":   vapp.get("vapp_country"),
            "vapp_dc":        vapp.get("vapp_dc"),
            "vapp_asn_org":   vapp.get("vapp_asn_org"),
            **beach_geo,
            "latency_ok_vapp":  latency_consistent(vapp.get("vapp_country"),    row.median_wall),
            "latency_ok_beach": latency_consistent(beach_geo["beach_country"],  row.median_wall),
            "country_mismatch": (
                vapp.get("vapp_country") is not None
                and beach_geo["beach_country"] is not None
                and vapp.get("vapp_country") != beach_geo["beach_country"]
            ),
        })
        # gentle rate limit: 50/min = 1.2s
        time.sleep(1.3)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(obs)}")

    df = pd.DataFrame(rows).sort_values(
        ["country_mismatch", "latency_ok_vapp", "median_wall_ms"],
        ascending=[False, True, True],
    )
    csv_path = args.out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[4/4] saved {csv_path}")

    # Suggested patches: where vapp ≠ beach AND beach is latency-consistent
    patches = {}
    for r in df.itertuples(index=False):
        if r.country_mismatch and (r.latency_ok_beach is True or r.latency_ok_beach is None):
            patches[r.identity] = {
                "data_center_key": f"{by_id.get(r.identity, {}).get('vapp_asn') or 0}"
                                   f"-{r.beach_country}-{r.beach_city or 'Unknown'}",
                "asn_organization": r.beach_asn_org or by_id.get(r.identity, {}).get('vapp_asn_org'),
                "_old_dc": r.vapp_dc,
                "_median_wall_ms": r.median_wall_ms,
                "_n_tx": r.n_tx,
            }
    patches_path = args.out.with_name(args.out.name + "-patches.json")
    json.dump(patches, open(patches_path, "w"), indent=2)
    print(f"      {len(patches)} suggested patches → {patches_path}")

    # Print summary
    print()
    print("=== Summary ===")
    print(f"  Total observed leaders:           {len(df)}")
    print(f"  Beach data fetched successfully:  {df['beach_country'].notna().sum()}")
    print(f"  Country mismatch (vapp vs beach): {df['country_mismatch'].sum()}")
    print(f"  Latency inconsistent w/ vapp:     {(df['latency_ok_vapp']==False).sum()}")
    print(f"  Latency inconsistent w/ beach:    {(df['latency_ok_beach']==False).sum()}")
    print(f"  Suggested patches:                {len(patches)}")
    print()
    print("Top 10 mismatches by tx count:")
    sub = df[df["country_mismatch"]].head(10)
    print(sub[[
        "name", "n_tx", "median_wall_ms",
        "vapp_country", "vapp_dc",
        "beach_country", "beach_city",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
