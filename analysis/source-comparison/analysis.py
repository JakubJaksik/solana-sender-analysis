"""
Source-comparison analysis: Jito ShredStream (SS) vs Yellowstone gRPC (YS).

The `entry-comparator` crate writes one Parquet row per observed shred-entry,
tagged with which source(s) saw it and the reception timestamp from each. This
script loads that Parquet, keeps the rows both sources matched, and measures how
much earlier ShredStream delivered each entry:

    ss_advantage_ns = ys_observed_ns - ss_fec_complete_ns   # positive => SS first

It then joins every entry to the producing validator's geolocation and reports
the advantage globally and split by continent / country, plus a set of figures.

Usage:
    python analysis.py \
        --run-dir path/to/run \
        --validators path/to/validators-epoch-NNN.json \
        --out plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import base58
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# country -> continent (ISO-3166 alpha-2)
# --------------------------------------------------------------------------- #
CONTINENT = {
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    # Europe
    "DE": "Europe", "NL": "Europe", "FR": "Europe", "GB": "Europe", "UK": "Europe",
    "CH": "Europe", "IE": "Europe", "FI": "Europe", "SE": "Europe", "NO": "Europe",
    "DK": "Europe", "ES": "Europe", "IT": "Europe", "PT": "Europe", "AT": "Europe",
    "PL": "Europe", "CZ": "Europe", "RO": "Europe", "BG": "Europe", "BE": "Europe",
    "LT": "Europe", "LV": "Europe", "EE": "Europe", "HU": "Europe", "GR": "Europe",
    "UA": "Europe", "RU": "Europe", "IS": "Europe", "LU": "Europe", "MT": "Europe",
    # Asia
    "JP": "Asia", "SG": "Asia", "KR": "Asia", "IN": "Asia", "CN": "Asia",
    "HK": "Asia", "TW": "Asia", "TH": "Asia", "VN": "Asia", "ID": "Asia",
    "MY": "Asia", "PH": "Asia", "AE": "Asia", "IL": "Asia", "TR": "Asia",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania",
    # South America
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "PE": "South America", "UY": "South America",
    # Africa
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "KE": "Africa", "MA": "Africa",
}

PERCENTILES = [("p1", 0.01), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90), ("p99", 0.99)]
PERCENTILE_ORDER = [p[0] for p in PERCENTILES]
PERCENTILE_PALETTE = ["#1976D2", "#42A5F5", "#4CAF50", "#FFC107", "#F44336"]


def parse_country(data_center_key: str | None) -> str | None:
    """Extract the ISO country code from an '<ASN>-<CC>-<city>' data-center key."""
    if not data_center_key or not isinstance(data_center_key, str):
        return None
    parts = data_center_key.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return None


def load_validators(path: Path) -> pd.DataFrame:
    """Build a validator -> (country, continent, stake) table from a snapshot."""
    with open(path) as f:
        vjson = json.load(f)
    rows = []
    for v in vjson["validators"]:
        cc = parse_country(v.get("data_center_key"))
        rows.append({
            "identity": v["identity"],
            "name": (v.get("name") or "").strip() or "(unnamed)",
            "country": cc or "??",
            "continent": CONTINENT.get(cc, "Unknown"),
            "data_center_key": v.get("data_center_key"),
            "stake_lamports": v.get("active_stake_lamports") or 0,
        })
    return pd.DataFrame(rows)


def load_matched(run_dir: Path, validators: pd.DataFrame) -> pd.DataFrame:
    """Load matched (BOTH-source) entries and join validator geo. adv_ms > 0 => SS first."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW d AS SELECT * FROM '{run_dir / 'diff.parquet'}'")
    df = con.execute("""
        SELECT
            slot,
            leader_pubkey,
            ys_observed_ns,
            ss_fec_complete_ns,
            CAST(ys_observed_ns AS BIGINT) - CAST(ss_fec_complete_ns AS BIGINT) AS ss_advantage_ns
        FROM d
        WHERE source = 'BOTH'
          AND ys_observed_ns IS NOT NULL
          AND ss_fec_complete_ns IS NOT NULL
    """).df()

    df["leader"] = df["leader_pubkey"].apply(
        lambda b: base58.b58encode(b).decode() if b is not None else None
    )
    df = df.merge(validators, left_on="leader", right_on="identity", how="left")
    df["country"] = df["country"].fillna("UNKNOWN")
    df["continent"] = df["continent"].fillna("Unknown")
    df["adv_ms"] = df["ss_advantage_ns"] / 1e6
    return df


def print_and_save_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    print("\nGlobal stats:")
    print(f"  n = {len(df):,}")
    print(f"  SS faster: {(df['adv_ms'] > 0).mean() * 100:.2f}%")
    print(f"  median advantage: {df['adv_ms'].median():.2f} ms")
    print(f"  p95: {df['adv_ms'].quantile(0.95):.2f} ms")
    print(f"  p99: {df['adv_ms'].quantile(0.99):.2f} ms")

    def summarize(by: str, head: int | None = None) -> pd.DataFrame:
        g = df.groupby(by)["adv_ms"].agg(
            n="count",
            p50="median",
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
            pct_ss_faster=lambda x: (x > 0).mean() * 100,
        ).round(2).sort_values("n", ascending=False)
        return g.head(head) if head else g

    cont = summarize("continent")
    ctry = summarize("country", head=15)
    print("\n=== Per-continent summary ===")
    print(cont.to_string())
    print("\n=== Per-country summary (top 15) ===")
    print(ctry.to_string())
    cont.to_csv(out_dir / "summary-continent.csv")
    ctry.to_csv(out_dir / "summary-country.csv")


def plot_global_cdf(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 6))
    sorted_adv = df["adv_ms"].sort_values().to_numpy()
    cdf = (pd.Series(range(len(sorted_adv))) + 1) / len(sorted_adv)
    plt.plot(sorted_adv, cdf, linewidth=1.5)
    plt.axvline(0, color="red", linestyle="--", alpha=0.6, label="YS faster <- | -> SS faster")
    plt.xlabel("SS advantage (ms)")
    plt.ylabel("CDF")
    plt.title(f"SS-vs-YS latency advantage CDF (n={len(df):,}, matched only)")
    plt.grid(alpha=0.3)
    plt.xlim(-50, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01-global-cdf.png", dpi=120)
    plt.close()


def plot_continent_violin(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 6))
    order = df.groupby("continent")["adv_ms"].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x="continent", y="adv_ms", order=order, cut=0, density_norm="width")
    plt.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.ylabel("SS advantage (ms)")
    plt.title("SS advantage distribution per continent")
    plt.ylim(-30, 60)
    plt.tight_layout()
    plt.savefig(out_dir / "02-per-continent-violin.png", dpi=120)
    plt.close()


def plot_percentiles(df: pd.DataFrame, out_dir: Path) -> None:
    """p1/p10/p50/p90/p99 of SS advantage, per continent and per top-10 country."""
    def percentile_frame(group_col: str, groups: list[str]) -> pd.DataFrame:
        rows = []
        for key, grp in df[df[group_col].isin(groups)].groupby(group_col):
            for name, q in PERCENTILES:
                rows.append({group_col: key, "percentile": name,
                             "advantage_ms": grp["adv_ms"].quantile(q)})
        return pd.DataFrame(rows)

    cont_order = (df.groupby("continent")["adv_ms"].median()
                    .sort_values(ascending=False).index.tolist())
    cont_df = percentile_frame("continent", cont_order)
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=cont_df, x="continent", y="advantage_ms", hue="percentile",
                     hue_order=PERCENTILE_ORDER, order=cont_order, palette=PERCENTILE_PALETTE)
    plt.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.ylabel("SS advantage (ms), positive = SS faster")
    plt.title("SS advantage at p1 / p10 / p50 / p90 / p99 per continent")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "08-percentile-continent.png", dpi=120)
    plt.close()

    top10 = df["country"].value_counts().head(10).index.tolist()
    ctry_df = percentile_frame("country", top10)
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(data=ctry_df, x="country", y="advantage_ms", hue="percentile",
                     hue_order=PERCENTILE_ORDER, order=top10, palette=PERCENTILE_PALETTE)
    plt.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.ylabel("SS advantage (ms), positive = SS faster")
    plt.title("SS advantage per percentile per country (top 10 by sample count)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=3, fontsize=7)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "09-percentile-country.png", dpi=120)
    plt.close()


def plot_source_breakdown(run_dir: Path, out_dir: Path) -> None:
    """How many entries each source saw: BOTH (matched) vs *_ONLY (unmatched)."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW d AS SELECT * FROM '{run_dir / 'diff.parquet'}'")
    miss = con.execute("SELECT source, COUNT(*) AS n FROM d GROUP BY source").df()
    total = miss["n"].sum()
    miss["pct"] = miss["n"] / total * 100
    print("\n=== Entries by source ===")
    print(miss.to_string(index=False))
    colors = {"BOTH": "#4CAF50", "YS_ONLY": "#FF9800", "SS_ONLY": "#2196F3"}
    plt.figure(figsize=(8, 5))
    plt.bar(miss["source"], miss["n"], color=[colors.get(s, "#999") for s in miss["source"]])
    for i, (s, n, p) in enumerate(zip(miss["source"], miss["n"], miss["pct"])):
        plt.text(i, n, f"{n:,}\n({p:.2f}%)", ha="center", va="bottom")
    plt.ylabel("Entry count")
    plt.title("Entries by source (BOTH = matched, *_ONLY = unmatched)")
    plt.tight_layout()
    plt.savefig(out_dir / "05-missing-breakdown.png", dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="ShredStream vs Yellowstone source comparison.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run directory containing diff.parquet (entry-comparator output).")
    ap.add_argument("--validators", type=Path, required=True,
                    help="Validator snapshot JSON (solana-leader-map output).")
    ap.add_argument("--out", type=Path, default=Path("plots"), help="Output directory for figures.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading validators...")
    validators = load_validators(args.validators)
    print(f"  {len(validators)} validators, {validators['country'].nunique()} countries")

    print("Loading matched entries...")
    df = load_matched(args.run_dir, validators)
    print(f"  {len(df):,} matched rows")

    print_and_save_summaries(df, args.out)
    plot_global_cdf(df, args.out)
    plot_continent_violin(df, args.out)
    plot_percentiles(df, args.out)
    plot_source_breakdown(args.run_dir, args.out)
    print(f"\nDone. Figures written to {args.out}/")


if __name__ == "__main__":
    main()
