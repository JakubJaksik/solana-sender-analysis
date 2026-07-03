# solana-leader-map

Per-epoch map of **Solana validator to physical location**. It cross-references `getLeaderSchedule` with the
validators.app dataset, writes one JSON snapshot per epoch, and exposes country, region, and data-center
aggregates.

## Why it exists

If you care about transaction inclusion, it helps to know **where the next leader physically is**, because that
drives:

- whether your sender is close to the leader (intra-metro round trips versus 100 to 230 ms cross-region);
- whether a given slot window is worth an aggressive attempt or a skip;
- what the data-center distribution of the whole network looks like this epoch, as a baseline sanity check.

The dataset is static per epoch: the leader schedule is fixed for the whole epoch (about 432,000 slots, roughly
two days). Fetch once, use for two days.

## Requirements

- **Rust 2024** stable (1.85+).
- A free **validators.app** account (https://www.validators.app/users/sign_up), then Settings, then API Tokens.
- Access to a **Solana HTTP RPC** endpoint (your own node or the public `https://api.mainnet-beta.solana.com`).

## Setup

```bash
# from the workspace root:
cargo build --release -p solana-leader-map
# binary: target/release/solana-leader-map

cd solana-leader-map/
cp config.example.json config.json
chmod 600 config.json
# edit: paste your api_token and RPC URL
```

`config.json` is git-ignored at the workspace level.

## Usage

```bash
# 1. Fetch the current epoch (validators.app + getLeaderSchedule)
target/release/solana-leader-map fetch
#    (cached in runs/leader-map-epoch-{N}.json; refetch every ~2 days when the
#     epoch changes, or pass --force to refetch over an existing cache.)

# 2. Table of country x % of slots x % of stake x validator count
target/release/solana-leader-map summary

# 3. Who leads this specific slot
target/release/solana-leader-map at 251234567

# 4. A slot range (inclusive)
target/release/solana-leader-map slots 251234500..251234520

# 5. The full snapshot as raw JSON (to feed other tools)
target/release/solana-leader-map export > snapshot.json
```

Every command except `fetch` takes an optional `--epoch <N>` to read an older cached epoch.

Useful flags: `-c <path>` / `--config <path>` for an alternate config, and `RUST_LOG=debug` for verbose tracing.

## Cache format

`runs/leader-map-epoch-{N}.json`:

```json
{
  "fetched_at": "2026-04-27T...",
  "epoch": { "epoch": 695, "absolute_slot": 0, "slot_index": 0, "slots_in_epoch": 432000 },
  "validators": [
    { "identity": "...", "name": "Helius", "country_code": "DE",
      "data_center_key": "Hetzner-DE-FRA", "active_stake_lamports": 14000000000000000 }
  ],
  "schedule": { "<identity>": [0, 4, 8], "...": [] }
}
```

Everything needed for later reanalysis is here; rebuild the slot map with `aggregate::build_slot_map`.

## Things to watch out for

1. **IP geolocation is imperfect for cloud and VPS.** A validator on AWS `eu-central-1` may be tagged "US"
   rather than "DE" if the ASN is US-registered. validators.app handles this better than most (MaxMind plus
   manual overrides), but verify outliers.
2. **Stake weighted is not slot weighted in a single window.** Averaged over an epoch the slot distribution
   tracks stake, but in one 4-slot window it can be the same validator four times in a row. `summary` gives the
   epoch average; `slots` and `at` give per-slot detail.
3. **`country_code: "??"`** is the "unknown to validators.app" bucket (in the schedule but missing from the geo
   dataset). It should be under 2 to 3% of slots; if it is more, check whether validators.app has fresh data.
4. **Refresh per epoch.** The schedule changes about every two days. This tool does not auto-refetch; add a cron
   or systemd timer if you want live data.

## Layout

```
src/
  main.rs           entry point, tracing setup
  cli.rs            clap CLI + dispatcher
  config.rs         load config.json
  validators_app.rs validators.app REST client
  solana_rpc.rs     JSON-RPC client (getEpochInfo, getLeaderSchedule)
  domain.rs         types: ValidatorInfo, EpochInfo, EpochSnapshot, SlotEntry, EpochSummary
  aggregate.rs      slot-map + summary logic + unit tests
  cache.rs          read/write runs/leader-map-epoch-{N}.json
  output.rs         pretty CLI tables (comfy-table)
  lib.rs            module re-exports
```
