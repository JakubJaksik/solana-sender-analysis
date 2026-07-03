"""
Cross-check validator geolocation between three sources:
  1. validators.app (`data_center_key` in validators-epoch-NNN.json) - labeled
  2. Solana RPC `getClusterNodes` → ip-api.com GeoIP - live gossip + BGP-derived
  3. (optional) Solana Beach API - third-party labeled

Outputs a CSV with one row per validator OBSERVED in a bench run, listing
country/region/ASN from each source and flagging discrepancies.

Free GeoIP via ip-api.com:
  - 45 req/min, batched up to 100 IPs at once → 4 batches = ~5s for 400 IPs
  - no API key needed for ≤45 req/min
  - response includes: country, regionName, city, isp, org, as

Usage:
  python geo_crosscheck.py \
      --run-dir path/to/run \
      --validators path/to/validators-epoch-NNN.json \
      --rpc-url $SOLANA_RPC_URL \
      --out summary/geo-crosscheck.csv
"""
import argparse
import json
import time
from pathlib import Path

import base58
import duckdb
import pandas as pd
import requests


def parse_country(data_center_key):
    if not data_center_key or not isinstance(data_center_key, str):
        return None
    parts = data_center_key.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return None


def load_validators_app(path: Path) -> dict:
    """identity -> {country, dc, asn, asn_org, ip}"""
    vjson = json.load(open(path))
    out = {}
    for v in vjson["validators"]:
        out[v["identity"]] = {
            "vapp_country": parse_country(v.get("data_center_key")),
            "vapp_dc": v.get("data_center_key"),
            "vapp_asn": v.get("asn"),
            "vapp_asn_org": v.get("asn_organization"),
            "vapp_ip": v.get("ip"),
            "name": (v.get("name") or "").strip() or None,
        }
    return out


def fetch_cluster_nodes(rpc_url: str) -> dict:
    """pubkey -> ip address from live gossip."""
    print(f"[1/4] fetching cluster nodes from {rpc_url}...")
    resp = requests.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": 1, "method": "getClusterNodes"},
        timeout=15,
    )
    resp.raise_for_status()
    nodes = resp.json()["result"]
    pubkey_to_ip = {}
    for n in nodes:
        # `gossip` is "ip:port"; take ip part. Some nodes have null gossip.
        g = n.get("gossip")
        if not g:
            continue
        ip = g.split(":")[0]
        pubkey_to_ip[n["pubkey"]] = ip
    print(f"      {len(pubkey_to_ip)} pubkeys with IPs")
    return pubkey_to_ip


def batch_geoip(ips: list[str]) -> dict:
    """IP -> {country, region, city, asn, asn_org} via ip-api.com batch.
    Up to 100 IPs per batch, ≤45 req/min.
    """
    print(f"[3/4] geoip lookup for {len(ips)} unique IPs (batched)...")
    out = {}
    for i in range(0, len(ips), 100):
        batch = ips[i:i+100]
        # Free endpoint accepts POST with list of objects with `query` field
        # or raw IP strings. Use simple array.
        body = [{"query": ip,
                 "fields": "status,query,country,countryCode,regionName,city,isp,org,as,asname"}
                for ip in batch]
        resp = requests.post(
            "http://ip-api.com/batch",
            json=body,
            timeout=30,
        )
        if resp.status_code == 429:
            print("      rate-limited; sleeping 60s")
            time.sleep(60)
            resp = requests.post("http://ip-api.com/batch", json=body, timeout=30)
        resp.raise_for_status()
        for entry in resp.json():
            ip = entry.get("query")
            if entry.get("status") != "success":
                out[ip] = {"gip_country": None, "gip_city": None,
                           "gip_asn": None, "gip_asn_org": None}
                continue
            # Parse "AS20326 KDDI CORPORATION" → asn=20326, asn_org="KDDI CORPORATION"
            asn_str = entry.get("as", "") or ""
            asn_num = None
            asn_org = None
            if asn_str.startswith("AS"):
                parts = asn_str.split(" ", 1)
                try:
                    asn_num = int(parts[0][2:])
                except ValueError:
                    pass
                asn_org = parts[1] if len(parts) > 1 else None
            out[ip] = {
                "gip_country": entry.get("countryCode"),
                "gip_country_full": entry.get("country"),
                "gip_region": entry.get("regionName"),
                "gip_city": entry.get("city"),
                "gip_asn": asn_num,
                "gip_asn_org": asn_org or entry.get("asname") or entry.get("isp"),
            }
        # respect rate limit: 45 req/min ≈ wait 1.4s between batches
        time.sleep(2)
    print(f"      {sum(1 for v in out.values() if v.get('gip_country'))} successful lookups")
    return out


def load_observed_pubkeys(run_dir: Path) -> set:
    parquet = run_dir / "tx-events.parquet"
    print(f"[2/4] loading observed pubkeys from {parquet}...")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT DISTINCT observed_leader, schedule_leader
        FROM '{parquet}'
    """).df()
    seen = set()
    for col in ("observed_leader", "schedule_leader"):
        for b in df[col]:
            if b is None:
                continue
            seen.add(base58.b58encode(bytes(b)).decode())
    print(f"      {len(seen)} unique pubkeys (observed + scheduled)")
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--validators", type=Path, required=True)
    ap.add_argument("--rpc-url", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. validators.app labels
    vapp = load_validators_app(args.validators)
    print(f"[1/4] validators.app: {len(vapp)} entries")

    # 2. observed pubkeys (only check these - much smaller than full cluster)
    observed = load_observed_pubkeys(args.run_dir)

    # 3. live IPs from cluster nodes
    pk_to_ip = fetch_cluster_nodes(args.rpc_url)

    # 4. geoip
    ips_for_observed = sorted({ip for pk, ip in pk_to_ip.items() if pk in observed})
    geo = batch_geoip(ips_for_observed)

    # 5. join
    print(f"[4/4] building cross-check table...")
    rows = []
    for pk in observed:
        v = vapp.get(pk, {})
        ip = pk_to_ip.get(pk)
        g = geo.get(ip, {}) if ip else {}
        rows.append({
            "identity": pk,
            "name": v.get("name"),
            "gossip_ip": ip,
            # validators.app
            "vapp_country": v.get("vapp_country"),
            "vapp_dc": v.get("vapp_dc"),
            "vapp_asn": v.get("vapp_asn"),
            "vapp_asn_org": v.get("vapp_asn_org"),
            # geoip
            "gip_country": g.get("gip_country"),
            "gip_city": g.get("gip_city"),
            "gip_asn": g.get("gip_asn"),
            "gip_asn_org": g.get("gip_asn_org"),
            # mismatch flags
            "country_mismatch": (
                v.get("vapp_country") is not None
                and g.get("gip_country") is not None
                and v.get("vapp_country") != g.get("gip_country")
            ),
            "asn_mismatch": (
                v.get("vapp_asn") is not None
                and g.get("gip_asn") is not None
                and v.get("vapp_asn") != g.get("gip_asn")
            ),
        })
    df = pd.DataFrame(rows).sort_values(["country_mismatch", "asn_mismatch", "name"],
                                          ascending=[False, False, True])
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print()
    print("=== DISCREPANCIES (validators.app vs live GeoIP) ===")
    bad = df[df["country_mismatch"] | df["asn_mismatch"]]
    if bad.empty:
        print("  none - all observed validators agree across sources")
    else:
        print(bad[["identity", "name", "gossip_ip",
                   "vapp_country", "vapp_dc", "vapp_asn_org",
                   "gip_country", "gip_city", "gip_asn_org"]].to_string(index=False))


if __name__ == "__main__":
    main()
