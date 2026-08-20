# src/network - Zano pool connectivity

## Status

**Wire protocol: implemented and validated (26 self-test checks).**
**Transport: implemented and validated end-to-end against a mock server (14 self-test checks) AND against a real zanod testnet daemon.**
**Mining glue (`src/mining/progpowz_work.*`): implemented, 17 self-test checks, and exercised live in the same real-daemon run.**

Four layers, built and proven in order (same discipline as `src/cuda/`:
prove each layer before trusting the next one):

1. `zano_stratum_protocol.{hpp,cpp}` - pure encode/decode for zanod's
   `eth_getWork`/`eth_submitWork`/`eth_submitLogin`/`eth_submitHashrate`
   JSON-RPC dialect. No sockets. Validated by `deepcore-protocol-selftest`
   against hand-verified wire strings independent of our own encoder.
2. `zano_stratum_client.{hpp,cpp}` (`ZanoStratumClient`) - a concrete
   `IStratumClient` (see `stratum_client.hpp`) built on plain TCP sockets
   and layer 1. Validated by `deepcore-network-selftest` against a mock
   server, then for real (see below).
3. `../mining/progpowz_work.{hpp,cpp}` - parses a job, checks a computed
   hash against its target, formats a share submission. Validated by
   `deepcore-mining-selftest`, then for real (see below).
4. Not started: a real end-to-end mining loop (persistent GPU-driven nonce
   search wired to the pool client) and a CLI. The real-daemon run below
   used a small ad-hoc CPU search, not the production loop.

### Real-daemon validation (2026-08-15)

Built an actual `zanod` testnet binary from source (`hyle-team/zano`,
commit `ee3de1e`, via `cmake -D TESTNET=TRUE`) and ran it locally with
`--stratum --stratum-always-online --stratum-miner-address=<testnet addr>`
(`--stratum-always-online` was needed because this environment's network
policy blocks the P2P port, so the node can't sync to real peers - the
flag makes the stratum server treat the core as synchronized regardless,
which still exercises the real share-verification code path, just against
a single-node genesis chain rather than the live public testnet).

Pointed a small integration program (not a committed self-test - it
depends on an external live process) using our real `ZanoStratumClient`
and `progpowz_work` at `127.0.0.1:18899`. Result, confirmed from **both**
sides:

- Client-side: connected, logged in, received a real job, computed a
  passing hash with `progpowz_hash_light`, submitted it, received
  `ShareResultStatus::Accepted` back.
- Daemon-side log, independently: `WORKER realtest: block found ... at
  height 1 ... found block ... was successfully added to the blockchain`
  - the daemon didn't just accept the share, it verified and added the
  resulting block to its own chain. This happened twice (heights 1 and 2)
  across two runs.

This is the strongest validation available without live public-testnet
access: real daemon binary, real consensus/verification code, real
protocol exchange - not a mock, not an assumption from reading source.

## What this milestone does NOT cover (real, tracked gaps)

- **No TLS.** `connect()` returns `not_supported` if the first endpoint
  requests it. Needed before ever pointing this at a public-internet pool.
- **No multi-endpoint failover.** Only `PoolConfig::endpoints.front()` is
  tried. The ordered-failover behavior documented in `stratum_client.hpp`
  is not implemented.
- **No reconnect/backoff loop.** On disconnect, the client reports
  `ConnectionState::Disconnected` and its worker thread exits - it does
  not retry. `PoolConfig::initial_reconnect_backoff`/`max_reconnect_backoff`
  are not read yet.
- **No keepalive, no job-timeout-triggered failover.**
- **Only the current job's header hash is retained**, so a share submitted
  against any job other than the most recent one is treated as stale
  locally (counted, no network round-trip) even if
  `PoolConfig::send_stale_shares` is set - there's no small history of
  recent jobs to actually resubmit an old one against yet.
- **Windows/Winsock code path is written but has never been compiled or
  run.** This development environment is Linux-only. The POSIX path is
  real-tested (`deepcore-network-selftest` passes reliably, run 5x with no
  flakiness); the `#ifdef _WIN32` branches mirror the same logic but are
  unverified - same honesty policy as the CUDA kernel's Ampere gap.
- **Real-daemon validation so far is a single-node, unsynced-to-the-public-
  network genesis chain** (`--stratum-always-online`, no real peers - see
  above), not the live public testnet with its real difficulty and other
  miners. It proves the protocol/verification path is genuinely correct,
  not that this client behaves correctly under real network conditions
  (real difficulty retargeting, competing miners, stale work from a
  fast-moving chain). Testing against the actual public testnet (or a
  node that's synced to it) is still a real, separate gap.

## Design notes worth knowing

- Zano has no separate job-id field on the wire; `ZanoStratumClient` uses
  the job's `pow_hash` (hex-encoded) as `JobNotification::job_id`, since
  it's unique per job and is also literally the `header_hash` value
  `eth_submitWork` needs.
- `JobNotification::raw_payload` is the raw JSON text of the work message,
  passed through opaquely - not yet parsed into a `WorkPayload` at this
  layer. The (not-yet-written) ProgPowZ mining-loop code is expected to
  call `parse_message()` again on it to get `seed_hash`/`target_boundary`/
  `height`.
- zanod's own `result:true` response is used for both a genuinely accepted
  share and a silently-dropped stale one (found while reading
  `stratum_server.cpp` - see `zano_stratum_protocol.hpp`'s header comment).
  This client only ever emits `ShareResultStatus::Accepted` for a
  `result:true` it receives *after* first confirming the job still
  matched at submit time; it cannot detect server-side staleness beyond
  that.

## Next steps, in order

1. **Done.** Real `zanod` daemon, real share acceptance, real blocks
   added to its chain - see "Real-daemon validation" above.
2. **Done.** `progpowz_work` mining glue - built, self-tested, and
   exercised live in the same real-daemon run.
3. **Done.** Production mining loop (`src/mining/mining_loop.{hpp,cpp}`) -
   see `src/mining/README.md`.
4. **Done.** CLI wiring (`src/cli/deepcore_miner_main.cpp`, target
   `deepcore-miner`) - assembles `ZanoStratumClient` + `MiningLoop` into an
   actual runnable program, validated with both `--dry-run` and a live
   mining run against a real `zanod` (see `src/cli/README.md`).
5. Remaining, not yet started: reconnect/backoff, multi-endpoint failover,
   TLS, keepalive, and validation against the real public testnet (not
   just a single-node genesis chain). A GPU-backed `IHashSearchBackend` is
   also still outstanding - see `src/mining/README.md`.
