"""Offline city->lat/lon table for leader cities + sender PoPs, and a plotly world map.

No network: coordinates are embedded. Covers the 39 leader cities present in
solanaview-crosscheck-980 plus sender PoP locations from run-config.
"""
from pathlib import Path

import plotly.graph_objects as go

# lat, lon
CITY_COORDS = {
    "Frankfurt": (50.11, 8.68), "Amsterdam": (52.37, 4.90), "Tokyo": (35.68, 139.69),
    "London": (51.51, -0.13), "Šiauliai": (55.93, 23.31), "Singapore": (1.35, 103.82),
    "New York": (40.71, -74.01), "São Paulo": (-23.55, -46.63), "Dublin": (53.35, -6.26),
    "Chicago": (41.88, -87.63), "Ashburn": (39.04, -77.49), "Stockholm": (59.33, 18.07),
    "West Haven": (41.27, -72.95), "Corral City": (33.21, -97.13), "Newark": (40.74, -74.17),
    "Vilnius": (54.69, 25.28), "Montreal": (45.50, -73.57), "Los Angeles": (34.05, -118.24),
    "Aubervilliers": (48.92, 2.38), "Warsaw": (52.23, 21.01), "Munich": (48.14, 11.58),
    "Lansing": (42.73, -84.56), "George Town": (19.30, -81.38), "Surrey": (49.10, -122.82),
    "Boardman": (45.84, -119.70), "Wilmington": (39.74, -75.55), "Dallas": (32.78, -96.80),
    "Piscataway": (40.55, -74.46), "Strasbourg": (48.58, 7.75), "Sterling": (39.00, -77.40),
    "Bratislava": (48.15, 17.11), "Mexico City": (19.43, -99.13), "Moscow": (55.76, 37.62),
    "Atlanta": (33.75, -84.39), "Royal Oak": (42.49, -83.14), "Limburg an der Lahn": (50.39, 8.06),
    "Rüsselsheim am Main": (49.99, 8.41), "Frankfurt Main Flughafen": (50.05, 8.57),
    # sv_country was null/odd for these but city resolves:
    "Amsterdam (US)": (52.37, 4.90),
}

# sender PoP -> city (coords reused from CITY_COORDS); from run-config endpoints/regions.
SENDER_POP = {
    "allenhark-fra": "Frankfurt", "allenhark-ams": "Amsterdam",
    "allenhark-ny": "New York", "allenhark-tk": "Tokyo",
    "helius-fra": "Frankfurt", "0slot-de1": "Frankfurt", "triton-fra": "Frankfurt",
    "syncro-fra": "Frankfurt", "blockrazor-fra": "Frankfurt",
    "astralane-fra": "Frankfurt", "astralane-quic-fra": "Frankfurt",
    "bloxroute-fra": "Frankfurt", "bloxroute-quic-fra": "Frankfurt",
    "nextblock-quic-fra": "Frankfurt", "nozomi-fra": "Frankfurt",
}


def world_map(leader_points, pop_points, out_html, out_png=None):
    """leader_points: list of (city, lat, lon, n, top_sender).
    pop_points: list of (label, lat, lon). Writes interactive HTML (+ optional PNG)."""
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[p[2] for p in leader_points], lat=[p[1] for p in leader_points],
        text=[f"{p[0]}: {p[3]} triggers, top={p[4]}" for p in leader_points],
        marker=dict(size=[max(6, 4 + p[3]) for p in leader_points], color="#1976D2",
                    opacity=0.6, line=dict(width=0)),
        name="leaders"))
    fig.add_trace(go.Scattergeo(
        lon=[p[2] for p in pop_points], lat=[p[1] for p in pop_points],
        text=[p[0] for p in pop_points],
        marker=dict(size=12, color="red", symbol="x"), name="sender PoPs"))
    fig.update_layout(title="Leader locations (size=triggers) & sender PoPs",
                      geo=dict(showland=True, landcolor="#eeeeee"))
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    if out_png:
        try:
            fig.write_image(str(out_png))   # needs kaleido; optional
        except Exception:
            pass
    return out_html
