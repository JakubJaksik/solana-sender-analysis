"""
Tick-trigger-bench latency analysis.

Reads tx-events.parquet from a bench run, joins with validator metadata
(country / continent / data center) and produces:
  • Global stats (counts, p50/p90/p95/p99) for each latency metric
  • Per-continent and per-country breakdowns
  • Histograms / CDFs / boxplots for each metric

Four primary metrics, each computed end-to-end:
  M1. trigger → send      (ms)          - our internal pipeline overhead
  M2. helius RTT          (ms)          - POST → 200 OK from Helius Sender
  M3. send → observed     (ms)          - wall-clock until tx visible in shred-stream
  M4. trigger → include   (ticks, hash) - chain progress (PoH) between trigger and inclusion
  M4'.send → include      (ticks, hash) - same but from send_at (subtracts pipeline overhead)

Plus secondary breakdowns: slot_delta distribution, leader_changed impact,
schedule-tick bucket vs latency, shred propagation delta.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base58

# ---------- config ----------
# Raw run data is large and lives outside the repo. Override with
# ANALYSIS_DATA_DIR, or pass paths explicitly on the CLI.
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANALYSIS_DATA_DIR", HERE / "data"))
DEFAULT_RUN = DATA_DIR / "tick-trigger" / "20260512-120035"
DEFAULT_VALIDATORS = DATA_DIR / "validators-epoch-970.json"
DEFAULT_OUT = HERE / "plots"
SUMMARY_DIR = HERE / "summary"

# Solana mainnet constants
HASHES_PER_TICK = 62_500
TICKS_PER_SECOND = 160               # 10M hashes/s ÷ 62 500 hashes/tick
NS_PER_TICK = 1_000_000_000 // TICKS_PER_SECOND   # 6_250_000 ns

# ---------- country → continent map (shared with the source-comparison analysis) ----------
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

PERCENTILES = [("p1", 0.01), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90),
               ("p95", 0.95), ("p99", 0.99)]
PERCENTILE_ORDER = [p[0] for p in PERCENTILES]
PERCENTILE_PALETTE = ["#1976D2", "#42A5F5", "#4CAF50", "#FFC107", "#FF7043", "#F44336"]


def parse_country(data_center_key):
    if not data_center_key or not isinstance(data_center_key, str):
        return None
    parts = data_center_key.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return None


def load_validators(path: Path) -> pd.DataFrame:
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


def load_bench_parquet(run_dir: Path) -> pd.DataFrame:
    """Load tx-events.parquet, derive latency columns, return DataFrame."""
    parquet = run_dir / "tx-events.parquet"
    if not parquet.exists():
        sys.exit(f"missing parquet: {parquet}")

    con = duckdb.connect()
    # `observed_cumulative_hashes_in_slot` and `hash_delta` are new in the
    # post-2026-05-12 schema. Probe for them so old parquet files still load.
    cols = {row[0] for row in con.execute(
        f"DESCRIBE SELECT * FROM '{parquet}'"
    ).fetchall()}
    optional = []
    if "observed_cumulative_hashes_in_slot" in cols:
        optional.append("observed_cumulative_hashes_in_slot")
    else:
        optional.append("NULL::UBIGINT AS observed_cumulative_hashes_in_slot")
    if "hash_delta" in cols:
        optional.append("hash_delta")
    else:
        optional.append("NULL::UBIGINT AS hash_delta")
    df = con.execute(f"""
        SELECT
            schedule_slot,
            schedule_tick,
            schedule_leader,
            tx_signature,
            trigger_observed_at_ns,
            send_at_ns,
            response_at_ns,
            send_error,
            observed_at_ns,
            observed_slot,
            observed_entry_index,
            observed_tick_in_slot,
            {optional[0]},
            observed_leader,
            tick_delta,
            slot_delta,
            time_delta_ns,
            {optional[1]},
            leader_changed,
            status
        FROM '{parquet}'
        WHERE status = 'OBSERVED'
    """).df()
    con.close()
    return df


def derive_latencies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # M1: trigger → send (ms)
    df["m1_trigger_to_send_ms"] = (df["send_at_ns"] - df["trigger_observed_at_ns"]) / 1e6
    # M2: helius RTT (ms)
    df["m2_helius_rtt_ms"] = (df["response_at_ns"] - df["send_at_ns"]) / 1e6
    # M3: send → observed (ms, wall-clock)
    df["m3_send_to_observed_ms"] = (df["observed_at_ns"] - df["send_at_ns"]) / 1e6

    # M4: trigger → include in TICKS (chain time)
    df["m4_trigger_to_include_ticks"] = df["tick_delta"].astype(float)
    # Hashes: prefer precise hash_delta column from parquet (post-2026-05-12
    # schema). Fall back to tick × HASHES_PER_TICK approximation when missing.
    if "hash_delta" in df.columns and df["hash_delta"].notna().any():
        df["m4_trigger_to_include_hashes"] = df["hash_delta"].astype(float)
        df["hash_resolution"] = "precise (sub-tick)"
    else:
        df["m4_trigger_to_include_hashes"] = df["m4_trigger_to_include_ticks"] * HASHES_PER_TICK
        df["hash_resolution"] = "tick-rounded (approx)"
    # also in ms (= ticks × 6.25); precise version uses hashes / 10M/s
    df["m4_trigger_to_include_ms_chain"] = df["m4_trigger_to_include_hashes"] / 10_000.0  # 10M hash/ms

    # M4': send → include = trigger→include MINUS (trigger→send) converted
    # to ticks/hashes via PoH rate (10M hash/s = 1 hash/100 ns).
    pipeline_ns = (df["send_at_ns"] - df["trigger_observed_at_ns"]).astype(float)
    pipeline_ticks  = pipeline_ns / NS_PER_TICK
    pipeline_hashes = pipeline_ns / 100.0  # 10M hash / 1e9 ns = 1/100
    df["m4p_send_to_include_ticks"]   = df["m4_trigger_to_include_ticks"]  - pipeline_ticks
    df["m4p_send_to_include_hashes"]  = df["m4_trigger_to_include_hashes"] - pipeline_hashes
    df["m4p_send_to_include_ms_chain"] = df["m4p_send_to_include_hashes"] / 10_000.0

    # leader bytes → base58
    df["observed_leader_b58"] = df["observed_leader"].apply(
        lambda b: base58.b58encode(bytes(b)).decode() if b is not None else None
    )
    df["schedule_leader_b58"] = df["schedule_leader"].apply(
        lambda b: base58.b58encode(bytes(b)).decode() if b is not None else None
    )
    return df


def attach_geo(df: pd.DataFrame, vdf: pd.DataFrame, leader_col="observed_leader_b58") -> pd.DataFrame:
    out = df.merge(vdf, left_on=leader_col, right_on="identity", how="left",
                   suffixes=("", "_v"))
    out["country"] = out["country"].fillna("UNKNOWN")
    out["continent"] = out["continent"].fillna("Unknown")
    return out


# ---------- summary helpers ----------
def percentile_summary(series, name):
    out = {"n": len(series)}
    for q_name, q in PERCENTILES:
        out[q_name] = round(series.quantile(q), 3)
    out["avg"] = round(series.mean(), 3)
    out["min"] = round(series.min(), 3)
    out["max"] = round(series.max(), 3)
    return pd.Series(out, name=name)


def global_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "M1: trigger→send (ms)":             df["m1_trigger_to_send_ms"],
        "M2: helius RTT (ms)":               df["m2_helius_rtt_ms"],
        "M3: send→observed (ms, wall)":      df["m3_send_to_observed_ms"],
        "M4: trigger→include (ticks)":       df["m4_trigger_to_include_ticks"],
        "M4: trigger→include (hashes)":      df["m4_trigger_to_include_hashes"],
        "M4: trigger→include (ms, chain)":   df["m4_trigger_to_include_ms_chain"],
        "M4': send→include (ticks)":         df["m4p_send_to_include_ticks"],
        "M4': send→include (hashes)":        df["m4p_send_to_include_hashes"],
        "M4': send→include (ms, chain)":     df["m4p_send_to_include_ms_chain"],
    }
    rows = [percentile_summary(s, name) for name, s in metrics.items()]
    return pd.DataFrame(rows)


def validator_summary(df: pd.DataFrame) -> dict:
    """Counts of unique leaders involved in the bench."""
    observed_leaders = df["observed_leader_b58"].dropna().unique()
    schedule_leaders = df["schedule_leader_b58"].dropna().unique()
    # tx counts per observed leader
    per_leader = df.groupby("observed_leader_b58").size().sort_values(ascending=False)
    # geo coverage
    countries = df["country"].dropna().unique()
    continents = df["continent"].dropna().unique()
    return {
        "unique_observed_leaders": len(observed_leaders),
        "unique_schedule_leaders": len(schedule_leaders),
        "tx_per_observed_leader_p50": int(per_leader.median()),
        "tx_per_observed_leader_max": int(per_leader.max()),
        "tx_per_observed_leader_min": int(per_leader.min()),
        "unique_countries": len(countries),
        "unique_continents": len(continents),
        "per_leader_top10": per_leader.head(10).to_dict(),
    }


def grouped_summary(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[metric_col]
    rows = []
    for key, series in g:
        row = percentile_summary(series, key)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.index.name = group_col
    return out


# ---------- plotting helpers ----------
def plot_cdf(series, title, xlabel, out_path, xlim=None, ref_zero=True):
    s = series.dropna().sort_values().to_numpy()
    if len(s) == 0:
        return
    cdf = (pd.Series(range(len(s))) + 1) / len(s)
    plt.figure(figsize=(11, 6))
    plt.plot(s, cdf, linewidth=1.5)
    if ref_zero:
        plt.axvline(0, color="red", linestyle="--", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(f"{title} (n={len(s):,})")
    plt.grid(alpha=0.3)
    if xlim:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_hist(series, title, xlabel, out_path, bins=80, xlim=None,
              secondary_label=None, secondary_scale=None):
    """secondary_label/scale: e.g. ("hashes", 62500) to add a top axis showing
    the same data multiplied by `secondary_scale`. Used to display ticks AND
    hashes on the same plot."""
    s = series.dropna()
    if len(s) == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(s, bins=bins, range=xlim, color="#1976D2", alpha=0.85, edgecolor="white")
    p50, p90, p99 = s.quantile(0.50), s.quantile(0.90), s.quantile(0.99)
    if secondary_scale:
        ax.axvline(p50, color="#4CAF50", linestyle="-",
                   label=f"p50 = {p50:.2f}  ({p50*secondary_scale:,.0f} {secondary_label})")
        ax.axvline(p90, color="#FFC107", linestyle="--",
                   label=f"p90 = {p90:.2f}  ({p90*secondary_scale:,.0f} {secondary_label})")
        ax.axvline(p99, color="#F44336", linestyle="--",
                   label=f"p99 = {p99:.2f}  ({p99*secondary_scale:,.0f} {secondary_label})")
    else:
        ax.axvline(p50, color="#4CAF50", linestyle="-",  label=f"p50 = {p50:.2f}")
        ax.axvline(p90, color="#FFC107", linestyle="--", label=f"p90 = {p90:.2f}")
        ax.axvline(p99, color="#F44336", linestyle="--", label=f"p99 = {p99:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(f"{title} (n={len(s):,})")
    if xlim:
        ax.set_xlim(*xlim)
    if secondary_scale:
        # Top axis: same domain × scale
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] * secondary_scale, ax.get_xlim()[1] * secondary_scale)
        ax2.set_xlabel(secondary_label)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_per_continent_violin(df, metric_col, title, ylabel, out_path, ylim=None):
    sub = df[df["continent"] != "Unknown"]
    if sub.empty:
        return
    order = sub.groupby("continent")[metric_col].median().sort_values().index
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=sub, x="continent", y=metric_col, order=order,
                   cut=0, density_norm="width")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_percentile_bar(df, group_col, metric_col, title, ylabel, out_path, top_n=10):
    counts = df[group_col].value_counts()
    keep = counts.head(top_n).index.tolist() if len(counts) > top_n else counts.index.tolist()
    sub = df[df[group_col].isin(keep)]
    rows = []
    for key, grp in sub.groupby(group_col):
        for q_name, q in PERCENTILES:
            rows.append({group_col: key, "percentile": q_name,
                         "value": grp[metric_col].quantile(q)})
    pdf = pd.DataFrame(rows)
    order = keep
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=pdf, x=group_col, y="value",
                     hue="percentile", hue_order=PERCENTILE_ORDER, order=order,
                     palette=PERCENTILE_PALETTE)
    plt.ylabel(ylabel)
    plt.title(title)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=2, fontsize=6)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# Tick buckets chosen to make slot boundaries visible (64 ticks = 1 slot).
# Same slot is split into quarters so we can see if late-tick triggers
# (close to slot end) suffer more boundary-crossing than early-tick ones.
TICK_BUCKETS = [
    (0,   15,  "0-15  (same slot, Q1)"),
    (16,  31,  "16-31 (same slot, Q2)"),
    (32,  47,  "32-47 (same slot, Q3)"),
    (48,  63,  "48-63 (same slot, Q4)"),
    (64,  127, "64-127 (slot+1)"),
    (128, 191, "128-191 (slot+2)"),
    (192, None, "192+ (slot+3 or more)"),
]
TICK_BUCKET_ORDER = [b[2] for b in TICK_BUCKETS]


def assign_tick_bucket(td):
    if pd.isna(td):
        return None
    td = int(td)
    for lo, hi, label in TICK_BUCKETS:
        if hi is None and td >= lo:
            return label
        if hi is not None and lo <= td <= hi:
            return label
    return None


def plot_tick_delta_breakdown(df, out_path):
    """Pure histogram by tick_delta bucket - total counts."""
    df = df.copy()
    df["tick_bucket"] = df["tick_delta"].apply(assign_tick_bucket)
    counts = df["tick_bucket"].value_counts().reindex(TICK_BUCKET_ORDER, fill_value=0)
    plt.figure(figsize=(13, 6))
    bars = plt.bar(range(len(counts)), counts.values, color="#1976D2")
    plt.xticks(range(len(counts)), counts.index, rotation=20, ha="right")
    for i, n in enumerate(counts.values):
        pct = n / counts.sum() * 100 if counts.sum() else 0
        plt.text(i, n, f"{n}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.ylabel("count")
    plt.title(f"tick_delta distribution by bucket (n={int(counts.sum())})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_tick_delta_breakdown_per_continent(df, out_path):
    """Same buckets, stacked / grouped per continent."""
    df = df.copy()
    df["tick_bucket"] = df["tick_delta"].apply(assign_tick_bucket)
    pivot = (
        df[df["continent"] != "Unknown"]
        .groupby(["tick_bucket", "continent"])
        .size()
        .unstack(fill_value=0)
        .reindex(TICK_BUCKET_ORDER, fill_value=0)
    )
    # absolute counts: grouped bars
    plt.figure(figsize=(15, 6))
    ax = pivot.plot(kind="bar", ax=plt.gca(), width=0.8, edgecolor="white")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("count")
    plt.title("tick_delta distribution per continent (observed_leader)")
    plt.legend(title="continent")
    for c in ax.containers:
        labels = [f"{int(v)}" if v > 0 else "" for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    # normalized per continent: stacked 100% bars (where does each continent's tx go?)
    pivot_norm = pivot.div(pivot.sum(axis=0), axis=1) * 100
    plt.figure(figsize=(13, 6))
    pivot_norm.T.plot(kind="bar", stacked=True, ax=plt.gca(),
                       colormap="viridis", edgecolor="white")
    plt.ylabel("% of tx within continent")
    plt.title("tick_delta bucket share per continent (100% stacked)")
    plt.legend(title="tick_delta bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "-stacked100.png"), dpi=120)
    plt.close()


def plot_slot_delta_breakdown(df, out_path):
    g = df.groupby(["slot_delta", "leader_changed"]).size().reset_index(name="n")
    g["bucket"] = g.apply(
        lambda r: f"Δ={r.slot_delta}{' (leader change)' if r.leader_changed else ''}",
        axis=1
    )
    plt.figure(figsize=(11, 6))
    colors = ["#4CAF50" if not lc else "#F44336" for lc in g["leader_changed"]]
    plt.bar(g["bucket"], g["n"], color=colors)
    for i, n in enumerate(g["n"]):
        plt.text(i, n, str(n), ha="center", va="bottom", fontsize=9)
    plt.ylabel("count")
    plt.title("Distribution: slot_delta × leader_changed")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_tick_bucket_latency(df, out_path):
    df = df.copy()
    df["tick_bucket"] = pd.cut(df["schedule_tick"],
                                bins=[0, 16, 32, 48, 64],
                                labels=["1-16", "17-32", "33-48", "49-64"])
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="tick_bucket", y="m3_send_to_observed_ms",
                showfliers=False, palette="viridis")
    plt.ylabel("send→observed (ms)")
    plt.xlabel("scheduled tick bucket (within slot)")
    plt.title("Latency vs. scheduled tick position within slot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                    help="bench run dir containing tx-events.parquet")
    ap.add_argument("--validators", type=Path, default=DEFAULT_VALIDATORS,
                    help="validators-epoch-NNN.json (from solana-leader-map)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="output plot dir")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"run dir:    {args.run_dir}")
    print(f"validators: {args.validators}")
    print(f"out:        {args.out_dir}")

    # 1) load + derive
    print("\n[1/5] loading validators...")
    vdf = load_validators(args.validators)
    print(f"      {len(vdf)} validators, "
          f"{vdf['country'].nunique()} countries, "
          f"continents={vdf['continent'].value_counts().to_dict()}")

    print("\n[2/5] loading bench parquet...")
    raw = load_bench_parquet(args.run_dir)
    print(f"      {len(raw)} OBSERVED rows")
    if len(raw) == 0:
        sys.exit("no OBSERVED rows - nothing to analyse")

    print("\n[3/5] deriving latencies + attaching geo...")
    df = derive_latencies(raw)
    df = attach_geo(df, vdf, leader_col="observed_leader_b58")
    print(f"      after merge: {len(df)} rows, "
          f"matched leaders: {(df['identity'].notna()).sum()}, "
          f"unknown: {(df['identity'].isna()).sum()}")

    # 2) global summary
    print("\n[4/5] computing summaries...")
    gs = global_summary(df)
    print("\n=== Global latency summary ===")
    print(gs.to_string())
    gs.to_csv(SUMMARY_DIR / "global-summary.csv")

    # 2b) validator coverage
    vstats = validator_summary(df)
    print("\n=== Validator coverage ===")
    for k, v in vstats.items():
        if k == "per_leader_top10":
            print(f"  {k}:")
            for ldr, n in v.items():
                print(f"      {ldr[:8]}…  n={n}")
        else:
            print(f"  {k}: {v}")
    with open(SUMMARY_DIR / "validator-coverage.json", "w") as f:
        json.dump(vstats, f, indent=2)

    # 3) per-continent / per-country summaries for each primary metric
    metrics_for_breakdown = [
        ("m1_trigger_to_send_ms",            "M1-trigger-to-send-ms"),
        ("m2_helius_rtt_ms",                 "M2-helius-rtt-ms"),
        ("m3_send_to_observed_ms",           "M3-send-to-observed-ms"),
        ("m4_trigger_to_include_ticks",      "M4-trigger-to-include-ticks"),
        ("m4_trigger_to_include_hashes",     "M4-trigger-to-include-hashes"),
        ("m4p_send_to_include_ticks",        "M4p-send-to-include-ticks"),
        ("m4p_send_to_include_hashes",       "M4p-send-to-include-hashes"),
    ]
    for col, label in metrics_for_breakdown:
        cont_sum = grouped_summary(df, "continent", col)
        ctry_sum = grouped_summary(df, "country",   col)
        cont_sum.to_csv(SUMMARY_DIR / f"per-continent-{label}.csv")
        ctry_sum.to_csv(SUMMARY_DIR / f"per-country-{label}.csv")
        print(f"\n=== per-continent: {label} ===")
        print(cont_sum.to_string())

    # 4) plots
    print(f"\n[5/5] writing plots to {args.out_dir}/...")

    # Histograms (zoomed in) + CDFs (full range)
    # Tuple: (column, title, unit, filename, xlim, secondary_label, secondary_scale)
    # secondary_scale is used for tick→hash double-axis (top axis shows hashes).
    plots = [
        ("m1_trigger_to_send_ms",        "M1: trigger → send (ms)",
         "ms",     "01a-m1-hist.png",     (0, 5),         None, None),
        ("m2_helius_rtt_ms",             "M2: Helius RTT (ms)",
         "ms",     "02a-m2-hist.png",     (0, 10),        None, None),
        ("m3_send_to_observed_ms",       "M3: send → observed (wall, ms)",
         "ms",     "03a-m3-hist.png",     (0, 2000),       None, None),
        ("m4_trigger_to_include_ticks",  "M4: trigger → include (PoH ticks, top axis = hashes)",
         "ticks",  "04a-m4-ticks-hist.png", (0, 300),     "hashes", HASHES_PER_TICK),
        ("m4_trigger_to_include_hashes", "M4: trigger → include (PoH hashes, precise)",
         "hashes", "04b-m4-hashes-hist.png", (0, 6_250_000), None, None),
        ("m4p_send_to_include_ticks",    "M4': send → include (PoH ticks, top axis = hashes)",
         "ticks",  "05a-m4p-ticks-hist.png", (0, 100),    "hashes", HASHES_PER_TICK),
        ("m4p_send_to_include_hashes",   "M4': send → include (PoH hashes, precise)",
         "hashes", "05b-m4p-hashes-hist.png", (0, 6_250_000), None, None),
    ]
    for col, title, unit, fname, xlim, sec_lbl, sec_scale in plots:
        plot_hist(df[col], title, unit, args.out_dir / fname, xlim=xlim,
                  secondary_label=sec_lbl, secondary_scale=sec_scale)
    # CDFs (full range, no clipping)
    for col, title, unit, fname, _, _, _ in plots:
        plot_cdf(df[col], title, unit, args.out_dir / fname.replace("-hist", "-cdf"),
                 ref_zero=False)

    # Per-continent violin for each metric
    violins = [
        ("m3_send_to_observed_ms",        "ms",      "send→observed",          (0, 800)),
        ("m4_trigger_to_include_ticks",   "ticks",   "trigger→include",        (0, 100)),
        ("m4_trigger_to_include_hashes",  "hashes",  "trigger→include",        (0, 6_250_000)),
        ("m4p_send_to_include_ticks",     "ticks",   "send→include",           (0, 100)),
        ("m4p_send_to_include_hashes",    "hashes",  "send→include",           (0, 6_250_000)),
    ]
    for col, unit, label, ylim in violins:
        plot_per_continent_violin(
            df, col,
            f"{label} per continent of observed_leader",
            f"{label} ({unit})",
            args.out_dir / f"10-violin-{col}.png",
            ylim=ylim
        )

    # Percentile bars per continent
    for col, unit, label, _ in violins:
        plot_percentile_bar(
            df, "continent", col,
            f"{label} percentiles per continent (observed_leader)",
            f"{label} ({unit})",
            args.out_dir / f"20-pct-continent-{col}.png"
        )
    # Percentile bars per country (top 10)
    for col, unit, label, _ in violins:
        plot_percentile_bar(
            df, "country", col,
            f"{label} percentiles per country, top 10 (observed_leader)",
            f"{label} ({unit})",
            args.out_dir / f"30-pct-country-{col}.png"
        )

    # Slot-delta breakdown
    plot_slot_delta_breakdown(df, args.out_dir / "40-slot-delta-breakdown.png")

    # Latency vs. scheduled tick position (where in the slot we triggered)
    plot_tick_bucket_latency(df, args.out_dir / "41-latency-vs-scheduled-tick.png")

    # Tick-delta breakdown (global + per-continent)
    plot_tick_delta_breakdown(df, args.out_dir / "42-tick-delta-buckets.png")
    plot_tick_delta_breakdown_per_continent(
        df, args.out_dir / "43-tick-delta-buckets-per-continent.png"
    )

    # Per-continent percentile bars for tick_delta directly (already done via
    # m4_trigger_to_include_ticks, but explicit one for clarity)
    plot_percentile_bar(
        df, "continent", "tick_delta",
        "tick_delta percentiles per continent",
        "ticks (PoH)",
        args.out_dir / "44-tick-delta-percentiles-continent.png"
    )

    # Quick text summary
    summary_lines = [
        f"run: {args.run_dir.name}",
        f"observed rows: {len(df)}",
        f"unique observed leaders:  {vstats['unique_observed_leaders']}",
        f"unique schedule leaders:  {vstats['unique_schedule_leaders']}",
        f"unique countries: {vstats['unique_countries']}, "
        f"continents: {vstats['unique_continents']}",
        f"tx per observed leader  min/p50/max: "
        f"{vstats['tx_per_observed_leader_min']} / "
        f"{vstats['tx_per_observed_leader_p50']} / "
        f"{vstats['tx_per_observed_leader_max']}",
        f"continents histogram: {df['continent'].value_counts().to_dict()}",
        "",
        "M1 trigger→send         p50/p99 (ms):    "
        f"{df['m1_trigger_to_send_ms'].median():.3f} / "
        f"{df['m1_trigger_to_send_ms'].quantile(0.99):.3f}",
        "M2 Helius RTT           p50/p99 (ms):    "
        f"{df['m2_helius_rtt_ms'].median():.3f} / "
        f"{df['m2_helius_rtt_ms'].quantile(0.99):.3f}",
        "M3 send→observed        p50/p99 (ms):    "
        f"{df['m3_send_to_observed_ms'].median():.3f} / "
        f"{df['m3_send_to_observed_ms'].quantile(0.99):.3f}",
        "M4 trigger→include      p50/p99 (ticks): "
        f"{df['m4_trigger_to_include_ticks'].median():.2f} / "
        f"{df['m4_trigger_to_include_ticks'].quantile(0.99):.2f}",
        "M4 trigger→include      p50/p99 (hash):  "
        f"{df['m4_trigger_to_include_hashes'].median():.0f} / "
        f"{df['m4_trigger_to_include_hashes'].quantile(0.99):.0f}",
        "M4' send→include        p50/p99 (ticks): "
        f"{df['m4p_send_to_include_ticks'].median():.2f} / "
        f"{df['m4p_send_to_include_ticks'].quantile(0.99):.2f}",
        "M4' send→include        p50/p99 (hash):  "
        f"{df['m4p_send_to_include_hashes'].median():.0f} / "
        f"{df['m4p_send_to_include_hashes'].quantile(0.99):.0f}",
    ]
    print("\n" + "\n".join(summary_lines))
    (SUMMARY_DIR / "quick-summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"\nDone. Plots in {args.out_dir}/, summaries in {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
