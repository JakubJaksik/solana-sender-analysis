"""Shared constants: paths, PoH/epoch constants, sender metadata, gating thresholds."""
import os
from pathlib import Path

# Raw run files (triggers.jsonl, validator snapshots) live outside the repo
# because they are large. Point ANALYSIS_DATA_DIR at your local data root, or
# drop the run files under ./data next to this package.
ANALYSIS_DIR = Path(
    os.environ.get("ANALYSIS_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)
FANOUT_DIR = ANALYSIS_DIR / "tick-trigger-fan-out"

EPOCH = 980
EPOCH_FIRST_SLOT = 423360000          # 980 * 432000
SLOTS_IN_EPOCH = 432000
TICKS_PER_SLOT = 64                   # PoH ticks; observed_tick in 0..63

DEFAULT_RUN = FANOUT_DIR / "runs" / "20260601-150500" / "triggers.jsonl"
DEFAULT_VEPOCH = ANALYSIS_DIR / "validators-epoch-980.json"
DEFAULT_SVCSV = FANOUT_DIR / "solanaview-crosscheck-980.csv"
DEFAULT_CONFIG = FANOUT_DIR / "run-config-20260601-150500.json"
ANALYSIS_OUT = FANOUT_DIR / "analysis-out"

# sender -> (protocol_class, region_city, pop_label). region_city must exist in geo_offline.CITY_COORDS.
SENDER_META = {
    "helius-dual":        ("HTTP_JSONRPC", "Frankfurt", "helius-fra"),
    "helius-fra":         ("HTTP_JSONRPC", "Frankfurt", "helius-fra"),
    "jito-multi":         ("JITO",         "Frankfurt", "jito-multiregion"),
    "triton-fra":         ("RELAY",        "Frankfurt", "triton-fra"),
    "syncro-fra":         ("RELAY",        "Frankfurt", "syncro-fra"),
    "0slot-de1":          ("RELAY",        "Frankfurt", "0slot-de1"),
    "blockrazor":         ("RELAY",        "Frankfurt", "blockrazor-fra"),
    "allenhark-quic-fra": ("QUIC_TPU",     "Frankfurt", "allenhark-fra"),
    "allenhark-quic-ams": ("QUIC_TPU",     "Amsterdam", "allenhark-ams"),
    "allenhark-quic-ny":  ("QUIC_TPU",     "New York",  "allenhark-ny"),
    "allenhark-quic-tk":  ("QUIC_TPU",     "Tokyo",     "allenhark-tk"),
    # added epoch 985 (all Frankfurt-resident)
    "astralane-fra":       ("RELAY",    "Frankfurt", "astralane-fra"),
    "astralane-quic-fra":  ("QUIC_TPU", "Frankfurt", "astralane-quic-fra"),
    "bloxroute-fra":       ("RELAY",    "Frankfurt", "bloxroute-fra"),
    "bloxroute-quic-fra":  ("QUIC_TPU", "Frankfurt", "bloxroute-quic-fra"),
    "nextblock-quic-fra":  ("QUIC_TPU", "Frankfurt", "nextblock-quic-fra"),
    "nozomi-fra":          ("RELAY",    "Frankfurt", "nozomi-fra"),
}
PROTOCOL_OF = {k: v[0] for k, v in SENDER_META.items()}
SENDER_REGION_CITY = {k: v[1] for k, v in SENDER_META.items()}

# continent of each sender's region city (for continent_match covariate)
CITY_CONTINENT = {
    "Frankfurt": "Europe", "Amsterdam": "Europe", "New York": "North America", "Tokyo": "Asia",
}
SENDER_REGION_CONTINENT = {k: CITY_CONTINENT[v[1]] for k, v in SENDER_META.items()}

GATE_INFERENTIAL = 20
GATE_INDICATIVE = 5
