# src/mining - the persistent mining loop

## Status

**`MiningLoop`: implemented, validated architecturally (12 self-test
checks with a fake client) AND against a real zanod daemon running
continuously for 20 seconds across 3 real job cycles.**

`mining_loop.{hpp,cpp}` is the component that actually ties the whole
project together: it receives jobs from an `IStratumClient`, searches for
a passing nonce via a pluggable `IHashSearchBackend`, and submits shares
found back through the same client - continuously, not one submission at
a time like the earlier ad-hoc validation scripts.

The search strategy is pluggable on purpose. `CpuHashSearchBackend` (using
`progpowz_hash_light`) is a real, working implementation - not a stub -
used here to validate the loop's *architecture* (job handling,
cancellation, submission, counters, clean shutdown) independent of GPU
hardware access. It is not intended for real mining throughput; a
GPU-backed `IHashSearchBackend` (wrapping `progpowz_kernel.cu`, and later
a full-DAG kernel) is the intended production backend and can be swapped
in without changing `MiningLoop` itself.

## Real-daemon continuous validation (2026-08-20)

Ran a real `zanod` testnet daemon (same setup as `src/network/README.md`'s
earlier validation) and pointed a real `ZanoStratumClient` + `MiningLoop`
(2 CPU worker threads) at it continuously for 20 seconds, requesting an
artificially low starting difficulty (1) via the login username's
`-1` suffix so shares would be found quickly.

**What happened, and why it's a good result even though the raw numbers
look mixed:**

- 3 distinct jobs received; the loop correctly transitioned between all of
  them, cancelling in-flight search each time.
- The first ~14+ shares submitted were `Accepted` - the daemon's own log
  independently confirms **2 real blocks were found and added to its
  chain** (heights 1 and 2), the same kind of confirmation as the earlier
  one-shot validation.
- After that, hundreds of subsequent submissions came back `Rejected`.
  This is **not a bug** - the daemon's log gives the exact reason:
  `block pow hash ... doesn't meet worker difficulty: 100000000`. Zano's
  stratum server has a built-in vardiff (variable difficulty) system
  (`--stratum-vdiff-retarget-shares`, default 12) that raises a worker's
  required difficulty once it submits too many shares too quickly - which
  our test deliberately triggered by starting at difficulty 1 and then
  submitting far more than 12 shares within seconds. The server updates
  its internal difficulty bar for that worker faster than it pushes a new
  job reflecting it (a new job/target is only broadcast alongside an
  actual new block template - see the `adjust_worker_difficulty_if_needed`
  comment in `stratum_server.cpp`), so our loop kept submitting against
  the old, now-too-easy target and correctly got told "no."

**Net result: this test exercised BOTH the `Accepted` and `Rejected` code
paths for real**, against genuine daemon logic - more validation coverage
than the earlier one-shot test, which only ever saw `Accepted`. The
`ShareResultStatus::Rejected` path is now proven to be wired correctly
too, not just assumed.

This also surfaces a real, documented (not hidden) inefficiency worth
knowing about for later: `MiningLoop` currently has no local awareness
that "I already found a winning share for this job" - it keeps searching
and submitting against the same job until a new one arrives. Under normal
network conditions (real difficulty, not an artificial 1) this essentially
never matters, since finding a share at all is rare. It's only visible
here because the test deliberately used a degenerate low difficulty to
make validation fast. Worth a small future improvement (pause/reduce
search intensity for a job after this worker's own share for it is
accepted), not a correctness problem.

## What's still missing

- **GPU-backed search backend.** `CpuHashSearchBackend` is correct but far
  too slow for real mining throughput - see `src/cuda/README.md`. A
  backend wrapping the CUDA kernel (and eventually a full-DAG kernel) is
  the real next step for anything resembling competitive hashrate.
- **No CLI / process wiring.** Nothing yet assembles `ZanoStratumClient` +
  `MiningLoop` + real config (pool URL, wallet address, device selection)
  into an actual runnable `deepcore-miner` program - `src/main.cpp` is
  still the Phase 2 placeholder.
- **No local "already solved this job" tracking**, per the note above.
- Everything else tracked as a gap in `src/network/README.md` (TLS,
  failover, reconnect/backoff, keepalive, Windows verification, real
  public-testnet validation) applies here too, since this loop sits on
  top of that transport.
