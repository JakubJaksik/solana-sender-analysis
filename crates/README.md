# Measurement tools (Rust workspace)

A Cargo workspace of the tools used to measure Solana transaction sending. Shared dependencies
(`tokio`, `reqwest`, `serde`, `yellowstone-grpc`, `solana-sdk`, `arrow`/`parquet`, and so on) are pinned once
in the workspace `Cargo.toml` so every crate builds against the same versions.

| Crate | What it does |
|---|---|
| [`solana-leader-map`](./solana-leader-map/) | Per-epoch validator to location map, cross-referenced with the leader schedule and stake. Country / region / data-center aggregates, JSON snapshot per epoch. |
| [`entry-sources`](./entry-sources/) | Shared ingestion layer. Reassembles Jito ShredStream shreds back into entries and subscribes to Yellowstone gRPC, behind one interface. Used by the crates below. |
| [`entry-comparator`](./entry-comparator/) | Matches the same entry across ShredStream and Yellowstone and records which arrived first. Output feeds the source-comparison analysis. |
| [`tick-trigger-bench`](./tick-trigger-bench/) | Single-sender latency. Schedules PoH ticks, pre-signs self-transfers, fires on the tick, and measures inclusion distance in ticks and hashes. |
| [`tick-trigger-fan-out-bench`](./tick-trigger-fan-out-bench/) | The 17-sender race. Durable-nonce fan-out, first-seen-wins observation across both streams, one JSONL row per attempt. |

## Requirements

- Rust 2024 stable (1.85+), see `rust-toolchain.toml`.
- Per-crate extras (API tokens, RPC endpoints) are documented in each crate's README.

## Build

```bash
# whole workspace
cargo build --release

# a single crate
cargo build --release -p solana-leader-map
cargo build --release -p tick-trigger-fan-out-bench
```

## Test

```bash
cargo test                          # everything
cargo test -p tick-trigger-fan-out-bench
```

## Configuration and secrets

Each tool reads a `config.json` for endpoints and keys. The committed `config.example.json` files use
placeholders for anything private. `config.json` and `config.local.json` are git-ignored, so no tokens, private
endpoints, or wallet keys are in this repository (public vendor endpoints appear as defaults). Copy the example,
fill in your own, and keep it local:

```bash
cp config.example.json config.json && chmod 600 config.json
```

## A note on the code

This is research code. It was written to answer questions quickly and correctly, not to ship as a library, so
the benches favour explicit pipelines and heavy instrumentation over polish. The parts that carry correctness
(the entry merger, the durable-nonce manager, the recorder) have tests; the glue does not always.
