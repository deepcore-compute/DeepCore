# src/cli - the deepcore-miner program

## Status

**Implemented and validated against a real `zanod` testnet daemon: both
`--dry-run` (connect + first job) and a live mining run (continuous
mining, real share acceptance, real blocks added to the daemon's chain,
clean shutdown on SIGTERM/SIGINT).**

`deepcore_miner_main.cpp` is what actually makes this project a runnable
miner: it assembles `network::zano_stratum::ZanoStratumClient` and
`mining::MiningLoop` (with `mining::CpuHashSearchBackend` - the reference
backend, not yet a production GPU backend, see `src/mining/README.md`)
behind a small CLI, and is the `deepcore-miner` executable target.

## Build

`deepcore-miner` always builds. Its sources depend on which optional
CMake layers are enabled:

- `DEEPCORE_BUILD_NETWORK_PROTOCOL AND DEEPCORE_BUILD_REFERENCE_TOOLS`
  (both `ON` by default): builds the real CLI described here, linked
  against `zano-stratum-client` and `deepcore-progpowz-work`.
- Either one `OFF`: falls back to the Phase 2 placeholder (`src/main.cpp`)
  with no extra dependencies, so the project always configures and builds
  regardless of which optional layers are enabled.

## Usage

```
deepcore-miner --url <host:port> --user <address> [options]

Required:
  --url <host:port>       Pool/daemon stratum endpoint (plain TCP only for now)
  --user <address>        Public payout address (never a private key/seed)

Options:
  --worker <name>         Worker name reported to the pool (default: none)
  --password <value>      Stratum password (default: "x")
  --threads <n>           CPU search worker threads (default: hardware_concurrency())
  --status-interval <s>   Seconds between status lines, 0 to disable (default: 10)
  --dry-run               Connect, wait for the first job, print result, then exit
  --quiet                 Suppress per-event log lines (status lines still print)
  --help, -h              Show this help text
```

`--dry-run` is meant for pool/config verification (CI, first-run sanity
checks) without actually mining: it exits 0 once connected and a job has
been received (or exits 1 after a 10s timeout).

Flags for features that don't exist yet (device/GPU selection, API port,
power/temperature limits, auto-tune, keepalive, stale-share submission,
job timeout) are deliberately not exposed - see `src/network/README.md`
and `src/mining/README.md` for what's a real, tracked gap versus silently
unsupported.

## Real-daemon validation (2026-08-20)

Built a fresh single-node `zanod` testnet daemon (same binary/setup as
the validations in `src/network/README.md` and `src/mining/README.md`)
and ran the actual `deepcore-miner` binary against it, unmodified from
what a real user would invoke:

1. `deepcore-miner --dry-run --url 127.0.0.1:<port> --user <addr> --worker
   cli_dry_run` - connected, reached `Ready`, received a real job, printed
   `PASS`, exited 0.
2. `deepcore-miner --url 127.0.0.1:<port> --user <addr> --worker
   cli_live_run --status-interval 3 --threads 2` - ran continuously:
   - 5 job cycles handled correctly.
   - Real shares `Accepted` - the daemon's own log independently confirms
     blocks were added to its chain at heights 1 through 4 (beyond
     genesis at height 0).
   - As in the earlier `MiningLoop` real-daemon test, difficulty then
     rose (Zano's stratum vardiff, `--stratum-vdiff-retarget-shares=12`)
     faster than the CPU reference backend could keep up, so most later
     submissions came back `Rejected reason=not enough work was done` -
     expected, not a bug, and further confirmation the CLI surfaces
     rejection reasons correctly.
   - `[status]` lines printed on schedule with correct counters (from
     both `MiningLoop::counters()` and `ZanoStratumClient::counters()`).
   - The test's `timeout` command sent SIGTERM, which the CLI's signal
     handler caught: printed `shutting down...`, stopped the loop,
     disconnected cleanly (`[state] Disconnected`), rather than being
     killed mid-operation.

This is the same "prove it against a real daemon, not just self-tests"
discipline used throughout this project, now covering the actual
end-user binary rather than an internal component.

## What's still missing

- **Mining throughput.** `CpuHashSearchBackend` is correct but far too
  slow for competitive hashrate - see `src/mining/README.md` and
  `src/cuda/README.md`. Swapping in a GPU-backed `IHashSearchBackend`
  needs no change to this CLI.
- **No device/GPU selection, no API/monitoring port, no power/thermal
  limits, no auto-tune.** None of these exist anywhere in the project yet
  (`src/hardware/gpu_manager.hpp` is still just an interface).
- Everything tracked as a gap in `src/network/README.md` (TLS, multi-
  endpoint failover, reconnect/backoff, keepalive, Windows verification,
  real public-testnet validation) applies here too.
- **No HiveOS process wiring yet** (`hiveos/h-manifest.conf` exists but
  the `h-config.sh`/`h-run.sh`/`h-stats.sh` scripts that would invoke this
  CLI under HiveOS do not).
