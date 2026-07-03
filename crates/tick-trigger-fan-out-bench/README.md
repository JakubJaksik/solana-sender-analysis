# tick-trigger-fan-out-bench

The fan-out benchmark: it fires the same logical transaction through many senders at once, on a scheduled
Proof-of-History tick, and records which one lands first. Built in independent phases, each of which is:

1. **Runnable end to end on its own**, via a `phaseN_*` binary.
2. **Instrumented**, with counters exported to a JSON snapshot plus periodic log-line summaries.
3. **Tested**, with unit and integration tests.
4. **Composable**, where one phase's output is the next phase's input, over `crossbeam_channel`.

## Phase 1: verify the data sources

```text
  ShredStream gRPC --+
                     +--> entry-merger --> ordering-tracker --> metrics
  Yellowstone gRPC --+    (one entry = one emission,
                          first-seen wins, latency
                          measured internally)
```

Before building anything on top, confirm the entry stream is trustworthy. Each unique `(slot, entry_hash)`
passes through the output channel **exactly once**, so downstream sees a single unified stream. Underneath, the
merger measures who saw each entry first and by how much.

### What it measures

Merger:
- `ss_received`, `ys_received`: raw receive count per source
- `ss_first`, `ys_first`: which source saw a unique entry first
- `confirmed_by_both`: unique entries the second source also confirmed
- `confirm_latency_{sum,min,max}_ns`: inter-source latency distribution
- `duplicates`: same-source replay or a third arrival (should be near zero)

Ordering tracker (per sealed slot):
- `entries_seen`, `max_index`: slot size
- `out_of_order_count`: entries that arrived after a higher-index entry
- `max_backward_gap`: the largest backward step observed
- `missing_indices`: indices below `max_index` never seen
- `last_entry_was_tick`: whether the slot ends on a tick (it should)

Aggregates:
- `slots_fully_ordered` / `slots_sealed`: fraction of clean slots
- `total_out_of_order`: sum of backward arrivals
- `tick_ending_rate`: fraction of slots that end on a tick

### How to run

```bash
cargo build --release -p tick-trigger-fan-out-bench

./target/release/phase1_observe \
  --ss-url http://127.0.0.1:9999 \
  --ys-url https://YOUR-YELLOWSTONE-ENDPOINT \
  --ys-token PASTE-YOUR-TOKEN \
  --duration 60s \
  --output runs/phase1-$(date +%Y%m%d-%H%M%S).json
```

It prints a one-line summary every 5 seconds and a full report (with per-slot detail) as JSON at the end.

### How to read it

Healthy mainnet:
- `tick_ending_rate` near 100%, every slot ends on tick 64
- `slots_fully_ordered` above 95%, most slots arrive without interleaving
- `both_confirm_rate` above 95%, the two sources see the same entries
- `total_missing_indices` near 0, no holes inside a slot
- `avg_entries_per_slot` around 100 to 300, typical mainnet activity

Warning signs:
- `tick_ending_rate` well below 100%: a source is losing a slot's final ticks (dropped FEC shreds). This blocks
  computing the durable-nonce blockhash locally (see later phases).
- `avg_out_of_order_per_slot` well above 1: entries really do arrive reordered, so a naive last-write-wins hash
  cache is wrong; an index-aware store is required.
- `slots_with_gaps` above 0: some entries are never seen, so either the source is dropping them or the dedup
  window is too short.
- `duplicates` well above 0: a bug in the source or the merger.

## Later phases

- **Phase 2:** PoH tick tracking + schedule firing (moves the observer in).
- **Phase 3:** the sender layer (multi-vendor fan-out, fresh-blockhash mode).
- **Phase 4:** Parquet writer + finality tracker.
- **Phase 5 (optional):** durable-nonce mode with a chain-hash fallback.

Each phase gets its own binary and its own JSON output, compatible with the layer before it.
