# Analysis (Python)

The statistics and figures behind the write-up. Three areas, one per experiment. The Rust benches produce the
raw data (Parquet / JSONL); these scripts turn it into summaries and charts.

```
source-comparison/   ShredStream vs Yellowstone   (from entry-comparator output)
latency/             single-sender PoH latency    (from tick-trigger-bench output)
fanout/              the 17-sender race statistics (from tick-trigger-fan-out-bench output)
```

## Setup

```bash
cd analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the exact pinned environment. The heavier stats (Wilson intervals, McNemar, Bradley-Terry,
conditional logit) use `scipy` and `statsmodels`; the world map uses `plotly`.

## Data

Raw run data is large and not committed. Point the scripts at your own run outputs, or set
`ANALYSIS_DATA_DIR` to a local data root. Small summary CSVs and the integrity report from a real run are
included under `fanout/sample-output/` so you can see the shape of the results without the raw files.

## source-comparison

One script over the `entry-comparator` Parquet. It keeps the entries both sources matched, measures
`ys_observed_ns - ss_fec_complete_ns` (positive means ShredStream first), joins each entry to the producing
validator's geolocation, and writes the CDF, per-continent, and per-percentile figures.

```bash
python source-comparison/analysis.py \
  --run-dir path/to/run \
  --validators path/to/validators-epoch-NNN.json \
  --out plots
```

## latency

Single-sender latency from the `tick-trigger-bench` Parquet, plus geolocation cross-checks. `analysis.py`
produces the global and per-region breakdowns in PoH ticks and hashes; `geo_crosscheck.py` and
`beach_crosscheck.py` reconcile validator locations across validators.app, live gossip GeoIP, and Solana Beach.

```bash
python latency/analysis.py --run path/to/run --validators path/to/validators-epoch-NNN.json
python latency/per_leader_stats.py --help
```

## fanout

The statistics suite for the sender race. `fanout_analysis` is a package of numbered sections (S0 integrity
through S11 synthesis) plus a shared loader, a stats module, and a report builder. Each section writes CSVs and
PNGs and returns a dict for the HTML report.

```bash
# full run (all sections + report.html)
python -m fanout_analysis.run_analysis --run-dir path/to/runs --epoch 985 --out analysis-out

# pool several runs
python -m fanout_analysis.run_analysis --runs run1 run2 run3 --out analysis-out

# refresh only some sections (fast, reuses the cached frame)
python -m fanout_analysis.run_analysis --only S2,S7
```

The load-bearing correctness contracts have tests. Run them with:

```bash
cd fanout && python -m pytest fanout_analysis/tests -q
```

Highlights of the suite:

- **`loader.py`** is the single source of truth: it reads the JSONL, asserts the paired-design invariants
  (row count, unique signatures, single winner per trigger), joins leader identity, stake, and geolocation, and
  builds the three win-rate denominators.
- **`statutils.py`** holds the estimators: Wilson intervals, paired bootstrap, exact McNemar, Cochran's Q,
  Bradley-Terry, Hodges-Lehmann, Mann-Whitney, and Benjamini-Hochberg correction, each tested against textbook
  fixtures.
- **`s9_pwin_model.py`** fits a conditional logit stratified by trigger, which absorbs trigger-level
  confounders and isolates the effect of the sender itself (and confirms send-order does not matter).
