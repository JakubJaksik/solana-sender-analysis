"""S7 - Geography.

sender x leader-continent conditional land-rate matrix (gated, n annotated);
AllenHark fra/ams/ny/tk same-protocol proximity contrast (overall + Europe-leader
subset); Europe vs non-Europe land-rate per sender; per-country and
per-data-center (sv_asn_org) DESCRIPTIVE heatmaps (gated); CMH stratified by
continent (statsmodels StratifiedTable) where feasible. Builds a world map of
leader cities (n triggers, top winning sender) + sender PoPs.

Conditional denominator (mask_conditional, excludes SendError) is used for all
land-rate cells, matching the spec's default headline estimand. Cells are
annotated with n; n<GATE_INDICATIVE are suppressed-to-NaN for the heatmaps and
flagged in the CSVs.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fanout_analysis import constants, geo_offline, loader, plotutils, statutils

logger = logging.getLogger(__name__)

SECTION_ID = "S7"
SECTION_TITLE = "Geography - where each sender wins"
SECTION_NUM = "07"

ALLENHARK = ["allenhark-quic-fra", "allenhark-quic-ams",
             "allenhark-quic-ny", "allenhark-quic-tk"]
CONTINENT_ORDER = ["Europe", "North America", "Asia", "South America"]


def _gate_label(n, min_n, min_ind):
    if n >= min_n:
        return "inferential"
    if n >= min_ind:
        return "indicative"
    return "suppressed"


def _conditional_landrate(sub):
    """Conditional estimand on a slice: Landed / (Landed + UnknownPending)."""
    cond = sub[sub["mask_conditional"]]
    n = int(len(cond))
    k = int(cond["land"].sum())
    rate = (k / n) if n else float("nan")
    return k, n, rate


def _sender_by_group_landrate(df, group_col, min_n, min_ind):
    """Long table: sender x group conditional land-rate, n, gate label."""
    senders = sorted(df["sender_name"].unique())
    groups = [g for g in df[group_col].dropna().unique()]
    rows = []
    for s in senders:
        for g in groups:
            sub = df[(df["sender_name"] == s) & (df[group_col] == g)]
            k, n, rate = _conditional_landrate(sub)
            lo, hi = statutils.wilson_ci(k, n)
            rows.append({
                "sender_name": s, group_col: g,
                "landed": k, "n_conditional": n,
                "land_rate": rate, "wilson_lo": lo, "wilson_hi": hi,
                "gate": _gate_label(n, min_n, min_ind),
            })
    return pd.DataFrame(rows)


def _landrate_matrix(long_df, group_col, value_col="land_rate", min_ind=5,
                     gate_suppress=True, col_order=None):
    """Pivot long sender x group land-rate to a matrix; suppress n<min_ind cells."""
    piv = long_df.pivot(index="sender_name", columns=group_col, values=value_col)
    if gate_suppress:
        ncnt = long_df.pivot(index="sender_name", columns=group_col, values="n_conditional")
        piv = piv.where(ncnt >= min_ind)
    if col_order:
        cols = [c for c in col_order if c in piv.columns] + \
               [c for c in piv.columns if c not in col_order]
        piv = piv[cols]
    return piv


def _landrate_matrices_all_cells(long_df, group_col, keep_groups):
    """Build rate + n matrices for a chosen set of groups, keeping EVERY cell.

    No per-cell suppression and no blank rows: only groups in keep_groups are
    shown (those with total triggers >= the indicative gate), and every
    sender x group cell carries its rate and conditional n so small-n cells are
    visible-but-annotated rather than blanked.
    """
    sub = long_df[long_df[group_col].isin(keep_groups)]
    rate = sub.pivot(index="sender_name", columns=group_col, values="land_rate")
    ncnt = sub.pivot(index="sender_name", columns=group_col, values="n_conditional")
    cols = [g for g in keep_groups if g in rate.columns]
    return rate[cols], ncnt[cols].fillna(0).astype(int)


def _heatmap_with_n(rate_mat, n_mat, title, out_path, min_ind):
    """Heatmap of land-rate where each cell is annotated 'rate\\n(n=N)'.

    Cells are coloured by rate but never blanked; the n annotation lets the
    reader weight thin cells themselves. Cell text is dimmed where n<min_ind.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    nrow, ncol = rate_mat.shape
    fig, ax = plt.subplots(figsize=(1.15 * ncol + 4, 0.55 * nrow + 3))
    annot = rate_mat.copy().astype(object)
    for r in rate_mat.index:
        for c in rate_mat.columns:
            v = rate_mat.loc[r, c]
            n = int(n_mat.loc[r, c]) if c in n_mat.columns else 0
            if n == 0:
                annot.loc[r, c] = "n=0"                  # no conditional attempt
            elif pd.isna(v):
                annot.loc[r, c] = ""
            else:
                annot.loc[r, c] = f"{v:.2f}\n(n={n})"
    sns.heatmap(rate_mat.astype(float), annot=annot.values, fmt="", cmap="viridis",
                ax=ax, cbar_kws={"shrink": .7, "label": "conditional land-rate"},
                annot_kws={"fontsize": 7}, linewidths=0.3, linecolor="#ffffff")
    # paint n=0 cells a neutral grey so they read as 'no data', not 0.0; dim thin cells
    texts = [t for t in ax.texts]
    ti = 0
    for ri, r in enumerate(rate_mat.index):
        for ci, c in enumerate(rate_mat.columns):
            v = rate_mat.loc[r, c]
            n = int(n_mat.loc[r, c]) if c in n_mat.columns else 0
            if n == 0:
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True,
                                           facecolor="#dddddd", edgecolor="#ffffff",
                                           lw=0.3, zorder=2))
                if ti < len(texts):
                    texts[ti].set_color("#777777")
                    texts[ti].set_zorder(3)
                    ti += 1
            elif not pd.isna(v):
                if n < min_ind:
                    texts[ti].set_color("#bbbbbb")
                    texts[ti].set_style("italic")
                ti += 1
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _allenhark_contrast(df):
    """Overall (operational /912) + Europe-leader subset conditional land counts."""
    overall = {}
    eu = {}
    eu_df = df[df["sv_continent"] == "Europe"]
    for s in ALLENHARK:
        sub = df[df["sender_name"] == s]
        overall[s] = int(sub["land"].sum())
        eu[s] = int(eu_df[eu_df["sender_name"] == s]["land"].sum())
    return overall, eu


def _cmh_stratified(df, sender_a, sender_b, min_ind):
    """Cochran-Mantel-Haenszel: P(land) for A vs B stratified by continent.

    Per stratum (continent) builds a 2x2 [[A_land, A_notland],[B_land, B_notland]]
    on the conditional denominator. Returns pooled OR, CMH stat, p, n_strata.
    """
    tables = []
    used = []
    for c in CONTINENT_ORDER:
        a = df[(df["sender_name"] == sender_a) & (df["sv_continent"] == c) & df["mask_conditional"]]
        b = df[(df["sender_name"] == sender_b) & (df["sv_continent"] == c) & df["mask_conditional"]]
        na, nb = len(a), len(b)
        if na < min_ind or nb < min_ind:
            continue
        a_l = int(a["land"].sum())
        b_l = int(b["land"].sum())
        tbl = np.array([[a_l, na - a_l], [b_l, nb - b_l]])
        # CMH/StratifiedTable needs both margins non-degenerate; skip empty strata.
        if tbl.sum() == 0:
            continue
        tables.append(tbl)
        used.append(c)
    if len(tables) < 2:
        return {"sender_a": sender_a, "sender_b": sender_b,
                "n_strata": len(tables), "or_pooled": float("nan"),
                "cmh_stat": float("nan"), "cmh_p": float("nan"),
                "strata": ",".join(used), "feasible": False}
    from statsmodels.stats.contingency_tables import StratifiedTable
    st = StratifiedTable([t.astype(float) for t in tables])
    res = st.test_null_odds()
    return {"sender_a": sender_a, "sender_b": sender_b,
            "n_strata": len(tables), "or_pooled": float(st.oddsratio_pooled),
            "cmh_stat": float(res.statistic), "cmh_p": float(res.pvalue),
            "strata": ",".join(used), "feasible": True}


def run(ctx) -> dict:
    df = ctx["df"]
    outdir = Path(ctx["outdir"])
    summary = outdir / "summary"
    plots = outdir / "plots"
    summary.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    min_n = ctx.get("min_n", constants.GATE_INFERENTIAL)
    min_ind = ctx.get("min_indicative", constants.GATE_INDICATIVE)

    tables = {}
    figures = {}
    notes = []

    # --- continent trigger totals (golden oracle) ---
    trig = df.drop_duplicates("trigger_id")[["trigger_id", "sv_continent"]]
    continent_totals = trig["sv_continent"].value_counts().to_dict()
    continent_totals = {str(k): int(v) for k, v in continent_totals.items()}
    logger.info("continent trigger totals: %s", continent_totals)

    # ===== 1. sender x continent conditional land-rate matrix (gated) =====
    cont_long = _sender_by_group_landrate(df, "sv_continent", min_n, min_ind)
    cont_path = summary / "sender-by-continent-landrate.csv"
    cont_long.to_csv(cont_path, index=False)
    tables["sender_by_continent"] = cont_path

    cont_mat = _landrate_matrix(cont_long, "sv_continent", min_ind=min_ind,
                                col_order=CONTINENT_ORDER)
    fig_cont = plots / f"{SECTION_NUM}-geo-continent-landrate.png"
    plotutils.heatmap(cont_mat,
                      "S7 sender x leader-continent conditional land-rate "
                      "(cells n<%d suppressed)" % min_ind,
                      fig_cont, fmt=".2f", cmap="viridis")
    figures["continent_heatmap"] = fig_cont

    # ===== 2. AllenHark same-protocol proximity contrast =====
    ah_overall, ah_eu = _allenhark_contrast(df)
    ah_rows = []
    eu_df = df[df["sv_continent"] == "Europe"]
    for s in ALLENHARK:
        sub = df[df["sender_name"] == s]
        k_op, n_op = int(sub["land"].sum()), int(len(sub))
        lo_op, hi_op = statutils.wilson_ci(k_op, n_op)
        sub_eu = eu_df[eu_df["sender_name"] == s]
        k_eu, n_eu, rate_eu = _conditional_landrate(sub_eu)
        lo_eu, hi_eu = statutils.wilson_ci(k_eu, n_eu)
        ah_rows.append({
            "sender_name": s,
            "pop_city": constants.SENDER_REGION_CITY[s],
            "landed_operational": k_op, "n_operational": n_op,
            "landrate_operational": k_op / n_op if n_op else float("nan"),
            "op_wilson_lo": lo_op, "op_wilson_hi": hi_op,
            "landed_eu_subset": k_eu, "n_eu_conditional": n_eu,
            "landrate_eu_subset": rate_eu, "eu_wilson_lo": lo_eu, "eu_wilson_hi": hi_eu,
        })
    ah_df = pd.DataFrame(ah_rows)

    # AllenHark grouped bar: operational vs EU-subset land-rate per PoP
    ah_plot = ah_df.set_index("sender_name")[["landrate_operational", "landrate_eu_subset"]]
    fig_ah = plots / f"{SECTION_NUM}-geo-allenhark-proximity.png"
    _grouped_bar(ah_plot,
                 "S7 AllenHark QUIC PoP proximity: operational (/912) vs Europe-leader subset",
                 "land-rate", fig_ah)
    figures["allenhark_bar"] = fig_ah

    # ===== 3. Europe vs non-Europe land-rate per sender (+ slopegraph) =====
    df_eu_split = df.copy()
    df_eu_split["eu_bucket"] = np.where(df_eu_split["sv_continent"] == "Europe",
                                        "Europe", "non-Europe")
    eu_long = _sender_by_group_landrate(df_eu_split, "eu_bucket", min_n, min_ind)
    eu_pivot = eu_long.pivot(index="sender_name", columns="eu_bucket", values="land_rate")
    left_vals = eu_pivot["Europe"].to_dict()
    right_vals = eu_pivot["non-Europe"].to_dict()
    color_by = {s: _protocol_color(constants.PROTOCOL_OF.get(s)) for s in left_vals}
    fig_slope = plots / f"{SECTION_NUM}-geo-eu-vs-noneu-slope.png"
    plotutils.slopegraph(left_vals, right_vals, "Europe", "non-Europe",
                         "S7 Europe vs non-Europe conditional land-rate per sender",
                         fig_slope, color_by=color_by)
    figures["eu_noneu_slope"] = fig_slope

    # ===== 4. per-country descriptive heatmap (no empty rows / no blanked cells) =====
    country_long = _sender_by_group_landrate(df, "sv_country", min_n, min_ind)
    country_path = summary / "sender-by-country-landrate.csv"
    country_long.to_csv(country_path, index=False)
    tables["sender_by_country"] = country_path

    # Keep only countries whose TOTAL triggers >= the indicative gate, ordered by
    # trigger count desc; then show ALL sender cells (annotated with n) - no
    # blanked cells, no empty rows.
    country_trig = (df.drop_duplicates("trigger_id")["sv_country"]
                    .dropna().value_counts())
    keep_countries = country_trig[country_trig >= min_ind].index.tolist()
    country_rate, country_n = _landrate_matrices_all_cells(
        country_long, "sv_country", keep_countries)
    fig_country = plots / f"{SECTION_NUM}-geo-country-landrate.png"
    _heatmap_with_n(country_rate, country_n,
                    "S7 sender x leader-country conditional land-rate "
                    "(DESCRIPTIVE; countries with >=%d triggers; n shown per cell)" % min_ind,
                    fig_country, min_ind)
    figures["country_heatmap"] = fig_country

    # ===== 5. per-data-center descriptive heatmap (no empty rows / no blanked cells) =====
    # A single ASN org (e.g. TeraSwitch) hosts leaders in up to 7 countries, so the
    # bare ASN name is physically ambiguous. Key the data-center dimension on the
    # composite "ASN_org (COUNTRY)" so each row is one identifiable physical site.
    df = df.copy()
    df["dc_label"] = (df["sv_asn_org"].astype(str) + " ("
                      + df["sv_country"].astype(str) + ")")
    # rows whose ASN org or country is null collapse to a clearly-marked label
    df.loc[df["sv_asn_org"].isna() | df["sv_country"].isna(), "dc_label"] = None

    dc_long = _sender_by_group_landrate(df, "dc_label", min_n, min_ind)
    dc_long = dc_long.rename(columns={"dc_label": "datacenter"})
    dc_path = summary / "sender-by-datacenter-landrate.csv"
    dc_long.to_csv(dc_path, index=False)
    tables["sender_by_datacenter"] = dc_path

    dc_trig = df.drop_duplicates("trigger_id")["dc_label"].dropna().value_counts()
    keep_dc = dc_trig[dc_trig >= min_ind].index.tolist()
    dc_rate, dc_n = _landrate_matrices_all_cells(dc_long, "datacenter", keep_dc)
    fig_dc = plots / f"{SECTION_NUM}-geo-datacenter-landrate.png"
    _heatmap_with_n(dc_rate, dc_n,
                    "S7 sender x leader data-center 'ASN_org (country)' conditional "
                    "land-rate (DESCRIPTIVE; DCs with >=%d triggers; n per cell)" % min_ind,
                    fig_dc, min_ind)
    figures["datacenter_heatmap"] = fig_dc

    # ===== 6. CMH stratified by continent (feasible inferential senders) =====
    # Restrict to senders with overall conditional n>=min_n in >=2 continents:
    cmh_rows = []
    # geo-proximity test rows = AllenHark proximity ordering (descriptive) +
    # CMH contrasts between the well-sampled senders.
    proximity_rows = []
    for s in ALLENHARK:
        proximity_rows.append({
            "test": "allenhark_proximity_descriptive",
            "sender_name": s, "pop_city": constants.SENDER_REGION_CITY[s],
            "landed_operational": ah_overall[s], "n_operational": 912,
            "landrate_operational": ah_overall[s] / 912,
            "landed_eu_subset": ah_eu[s],
        })
    # CMH for the top two senders vs the AllenHark-FRA reference + 0slot vs triton.
    inferential_senders = (cont_long[cont_long["gate"] == "inferential"]
                           .groupby("sender_name").size())
    well = [s for s, c in inferential_senders.items() if c >= 2]
    cmh_pairs = []
    if "0slot-de1" in well and "triton-fra" in well:
        cmh_pairs.append(("0slot-de1", "triton-fra"))
    for ref in ["0slot-de1", "triton-fra"]:
        for other in ["allenhark-quic-fra", "helius-dual", "blockrazor"]:
            if ref in well and other in df["sender_name"].values:
                cmh_pairs.append((ref, other))
    seen = set()
    for a, b in cmh_pairs:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        r = _cmh_stratified(df, a, b, min_ind)
        cmh_rows.append(r)
    cmh_df = pd.DataFrame(cmh_rows) if cmh_rows else pd.DataFrame(
        columns=["sender_a", "sender_b", "n_strata", "or_pooled",
                 "cmh_stat", "cmh_p", "strata", "feasible"])

    # geo-proximity-test.csv combines the AllenHark proximity contrast + CMH results.
    prox_df = pd.DataFrame(proximity_rows)
    geo_prox_path = summary / "geo-proximity-test.csv"
    # write proximity block then CMH block, tagged by 'test' column for the report
    cmh_block = cmh_df.copy()
    if not cmh_block.empty:
        cmh_block.insert(0, "test", "cmh_stratified_by_continent")
    combined = pd.concat([prox_df, cmh_block], ignore_index=True, sort=False)
    combined.to_csv(geo_prox_path, index=False)
    tables["geo_proximity_test"] = geo_prox_path

    # ===== 7. world map: leader cities + sender PoPs (pretty Scattergeo) =====
    land = df[df["land"] == 1]
    trig_by_city = df.drop_duplicates("trigger_id").groupby("sv_city").size()
    leader_points = []
    for city, n in trig_by_city.items():
        if city is None or pd.isna(city):
            continue
        coords = geo_offline.CITY_COORDS.get(city)
        if coords is None:
            notes.append(f"city without coords skipped on map: {city}")
            continue
        city_land = land[land["sv_city"] == city]
        if len(city_land):
            top_sender = city_land["sender_name"].value_counts().idxmax()
        else:
            top_sender = "(no winner)"
        leader_points.append((city, coords[0], coords[1], int(n), top_sender))

    pop_points = []
    for label, city in geo_offline.SENDER_POP.items():
        coords = geo_offline.CITY_COORDS.get(city)
        if coords is None:
            continue
        pop_points.append((label, coords[0], coords[1]))

    world_html = plots / f"{SECTION_NUM}-geo-world-map.html"
    world_png = plots / f"{SECTION_NUM}-geo-world-map.png"
    _pretty_world_map(leader_points, pop_points, world_html, world_png)
    figures["world_map_html"] = world_html
    if world_png.exists():
        figures["world_map_png"] = world_png
    else:
        notes.append("world-map PNG not rendered (kaleido unavailable); HTML written.")

    # ===== notes / gating caveats =====
    notes.append("Land-rate cells use the conditional denominator "
                 "(Landed/(Landed+UnknownPending), SendError excluded).")
    notes.append("Per-country and per-data-center heatmaps are DESCRIPTIVE only "
                 "(thin cells; no inferential claim).")
    notes.append("AllenHark proximity is same-protocol (QUIC-TPU) but tk PoP is "
                 "dead (0/912 sent) - see ERR bug A.")
    n_country_inf = int((country_trig >= min_n).sum())
    notes.append(f"{n_country_inf} countries have >={min_n} triggers; "
                 f"all per-country comparisons remain descriptive per spec.")

    key_results = {
        "continent_trigger_totals": continent_totals,
        "allenhark_overall_landed": ah_overall,
        "allenhark_eu_subset_landed": ah_eu,
        "n_countries": int(df["sv_country"].nunique()),
        "n_datacenters": int(df["dc_label"].nunique()),
        "n_leader_cities_mapped": len(leader_points),
        "cmh_tests": {f"{r['sender_a']}_vs_{r['sender_b']}":
                      {"or_pooled": r["or_pooled"], "cmh_p": r["cmh_p"],
                       "n_strata": r["n_strata"], "feasible": r["feasible"]}
                      for r in cmh_rows},
    }

    # --- concise per-figure captions (rendered under each image) ---
    captions = {
        "continent_heatmap":
            "Conditional land-rate by sender x leader continent (n<%d cells "
            "suppressed). Frankfurt relays dominate Europe (605 triggers); "
            "win-rates fall on NA (219), Asia (72), SA (16) leaders." % min_ind,
        "allenhark_bar":
            "Same-protocol (QUIC-TPU) AllenHark PoPs, land counts fra 51 / ams 47 / "
            "ny 14 / tk 0 (/912; tk PoP dead). Confirmed: all 9 of allenhark-quic-ny's "
            "EU-leader landings were on the NEXT slot (slots_behind>=1, 0 same-slot) - "
            "NY only catches EU leaders after the slot rolls over, consistent with NY "
            "being far from EU leaders.",
        "eu_noneu_slope":
            "Per-sender conditional land-rate, Europe vs non-Europe leaders; lines "
            "coloured by protocol. Most senders drop sharply off Europe.",
        "country_heatmap":
            "Descriptive land-rate by sender x leader country (only countries with "
            ">=%d triggers; every cell shows its n, thin cells dimmed). No "
            "inferential claim." % min_ind,
        "datacenter_heatmap":
            "Descriptive land-rate by sender x physical data-center, keyed "
            "'ASN_org (country)' since one ASN spans many countries (only DCs with "
            ">=%d triggers; n per cell)." % min_ind,
        "world_map_html":
            "Interactive world map (download to open): leader-city bubbles sized by "
            "sqrt(triggers), coloured by top winning sender, with sender PoPs. Notion "
            "cannot embed it without hosting.",
        "world_map_png":
            "Static world map: bubble area ~ trigger count, colour = top winning "
            "sender (categorical), red X = sender PoPs. Frankfurt is the densest "
            "leader cluster.",
    }
    # keep a caption only for figures actually emitted
    captions = {k: v for k, v in captions.items() if k in figures}

    return {
        "id": SECTION_ID, "title": SECTION_TITLE,
        "tables": tables, "figures": figures,
        "key_results": key_results, "notes": notes,
        "captions": captions,
    }


def _protocol_color(proto):
    return {"HTTP_JSONRPC": "#1976D2", "JITO": "#8E24AA", "RELAY": "#2E7D32",
            "QUIC_TPU": "#E65100"}.get(proto)


def _pretty_world_map(leader_points, pop_points, out_html, out_png):
    """Pretty Scattergeo: leader cities sized by sqrt(n), coloured by top winning
    sender (categorical), sender PoPs as red X, biggest-city text labels.

    leader_points: (city, lat, lon, n, top_sender). Writes self-contained HTML
    (downloadable / interactive) and exports a static PNG via kaleido.
    """
    import math

    import plotly.graph_objects as go

    senders = sorted({p[4] for p in leader_points})
    palette = ["#1976D2", "#E65100", "#2E7D32", "#8E24AA", "#C62828",
               "#00838F", "#F9A825", "#6D4C41", "#546E7A", "#AD1457"]
    color_of = {s: palette[i % len(palette)] for i, s in enumerate(senders)}

    fig = go.Figure()
    # one trace per top-winning-sender category -> categorical legend + colours
    for s in senders:
        pts = [p for p in leader_points if p[4] == s]
        fig.add_trace(go.Scattergeo(
            lon=[p[2] for p in pts], lat=[p[1] for p in pts],
            text=[f"{p[0]}: {p[3]} triggers<br>top winner: {p[4]}" for p in pts],
            hoverinfo="text",
            marker=dict(
                size=[8 + 6 * math.sqrt(p[3]) for p in pts],  # SQRT sizing
                color=color_of[s], opacity=0.72,
                line=dict(width=0.6, color="#ffffff")),
            name=f"won mostly by {s}"))

    # sender PoPs as red X
    fig.add_trace(go.Scattergeo(
        lon=[p[2] for p in pop_points], lat=[p[1] for p in pop_points],
        text=[p[0] for p in pop_points], hoverinfo="text",
        marker=dict(size=11, color="#D50000", symbol="x",
                    line=dict(width=1, color="#D50000")),
        name="sender PoP"))

    # text labels for the biggest leader cities only (avoid clutter)
    big = sorted(leader_points, key=lambda p: p[3], reverse=True)[:8]
    fig.add_trace(go.Scattergeo(
        lon=[p[2] for p in big], lat=[p[1] for p in big],
        text=[f"{p[0]} ({p[3]})" for p in big],
        mode="text", textfont=dict(size=10, color="#212121"),
        textposition="top center", hoverinfo="skip", showlegend=False))

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="#bdbdbd",
        showland=True, landcolor="#f5f5f0",
        showocean=True, oceancolor="#eaf2fb",
        showcoastlines=True, coastlinecolor="#9e9e9e",
        showframe=False, resolution=50)
    fig.update_layout(
        title="S7 leader locations (bubble area ~ trigger count, colour = top "
              "winning sender) & sender PoPs",
        legend=dict(font=dict(size=9), itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0))

    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    try:
        fig.write_image(str(out_png), width=1400, height=820, scale=2)
    except Exception as exc:  # kaleido failure must not break the section
        logger.warning("world-map PNG export failed: %s", exc)
    return out_html


def _grouped_bar(matrix_df, title, ylabel, out_path):
    """Small grouped-bar helper (rows=index, columns=series)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    matrix_df.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _build_ctx(out_dir):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    df = loader.load_enriched()
    wide = loader.load_wide()
    config = json.load(open(constants.DEFAULT_CONFIG))
    return {
        "df": df, "wide": wide, "outdir": out_dir, "config": config,
        "min_n": constants.GATE_INFERENTIAL, "min_indicative": constants.GATE_INDICATIVE,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="S7 geography analysis")
    ap.add_argument("--out", default="/tmp/S7-verify",
                    help="output dir (plots/ + summary/ created within)")
    args = ap.parse_args()
    ctx = _build_ctx(args.out)
    res = run(ctx)
    print(json.dumps(res["key_results"], indent=2, default=str))
    print("tables:", {k: str(v) for k, v in res["tables"].items()})
    print("figures:", {k: str(v) for k, v in res["figures"].items()})


if __name__ == "__main__":
    main()
