# src/cli - the deepcore-miner program

## Status

**Implemented and validated against a real `zanod` testnet daemon: both
`--dry-run` (connect + first job) and a live mining run (continuous
mining, real share acceptance, real blocks added to the daemon's chain,
clean shutdown on SIGTERM/SIGINT).**

`deepcore_miner_main.cpp` is what actually makes this project a runnable
miner: it assembles `network::zano_stratum::ZanoStratumClient` and
`mining::MiningLoop` behind a small CLI, and is the `deepcore-miner`
executable target. Uses `mining::CpuHashSearchBackend` by default; when
this build was compiled with CUDA, `--gpu` selects
`mining::GpuHashSearchBackend` instead (see `src/cuda/README.md` -
**written, not yet validated with this CLI on real hardware**, though the
backend itself has been, standalone).

**`--gpu` status: NOT yet validated end-to-end through this CLI.**
`GpuHashSearchBackend` itself passed all 11 checks in
`deepcore-gpu-backend-selftest` on a real Quadro GV100 (see
`src/cuda/README.md`), but that only exercises the backend directly, not
through `MiningLoop`+`ZanoStratumClient`+this CLI together, and not
through a mixed CXX+CUDA `deepcore-miner` build (a first for this
target - see "Build" below). Do not trust `--gpu` from this CLI until a
real run against a real daemon is recorded here, the same way the CPU
path already is.

## Build

`deepcore-miner` always builds. Its sources depend on which optional
CMake layers are enabled:

- `DEEPCORE_BUILD_NETWORK_PROTOCOL AND DEEPCORE_BUILD_REFERENCE_TOOLS`
  (both `ON` by default): builds the real CLI described here, linked
  against `zano-stratum-client` and `deepcore-progpowz-work`.
- Either one `OFF`: falls back to the Phase 2 placeholder (`src/main.cpp`)
  with no extra dependencies, so the project always configures and builds
  regardless of which optional layers are enabled.
- Additionally `DEEPCORE_WITH_CUDA` with a CUDA compiler actually found:
  `src/cuda/progpowz_gpu_backend.cu` is added to the target and
  `DEEPCORE_HAVE_GPU_BACKEND` is defined, which is what makes `--gpu`
  appear at all (see `parse_args`/`print_usage` - both are `#ifdef`-gated
  on that macro, so a CPU-only build never claims a flag it can't honor).
  This is the first `deepcore-miner` build to mix a CXX source
  (`deepcore_miner_main.cpp`) and a CUDA source in one target; the
  target's warning/optimization `target_compile_options` calls were
  scoped with `$<COMPILE_LANGUAGE:CXX>` so GCC/Clang/MSVC-specific flags
  are never handed to `nvcc`. Not yet confirmed to actually configure and
  link cleanly on real hardware - see "Status" above.

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
                          ignored if --gpu is set (one worker drives one GPU)
  --status-interval <s>   Seconds between status lines, 0 to disable (default: 10)
  --dry-run               Connect, wait for the first job, print result, then exit
  --quiet                 Suppress per-event log lines (status lines still print)
  --gpu                   Mine with the GPU backend instead of the CPU one
                          (only present in a build compiled with CUDA)
  --gpu-device <n>        CUDA device index to use with --gpu (default: 0)
  --help, -h              Show this help text
```

`--dry-run` is meant for pool/config verification (CI, first-run sanity
checks) without actually mining: it exits 0 once connected and a job has
been received (or exits 1 after a 10s timeout).

Flags for features that don't exist yet (multi-GPU selection beyond a
single `--gpu-device` index, API port, power/temperature limits,
auto-tune, keepalive, stale-share submission, job timeout) are
deliberately not exposed - see `src/network/README.md` and
`src/mining/README.md` for what's a real, tracked gap versus silently
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

- **`--gpu` not yet validated through this CLI** - see "Status" above.
- **Mining throughput even with `--gpu`.** `GpuHashSearchBackend` still
  uses the light-cache kernel and re-uploads its epoch caches on every
  batch - correct, but not yet competitive; see `src/cuda/README.md`.
- **No multi-GPU support, no API/monitoring port, no power/thermal
  limits, no auto-tune.** None of these exist anywhere in the project yet
  (`src/hardware/gpu_manager.hpp` is still just an interface).
- Everything tracked as a gap in `src/network/README.md` (TLS, multi-
  endpoint failover, reconnect/backoff, keepalive, Windows verification,
  real public-testnet validation) applies here too.
- **No HiveOS process wiring yet** (`hiveos/h-manifest.conf` exists but
  the `h-config.sh`/`h-run.sh`/`h-stats.sh` scripts that would invoke this
  CLI under HiveOS do not).
