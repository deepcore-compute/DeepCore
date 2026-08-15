# src/network - Zano pool connectivity

## Status

**Wire protocol: implemented and validated (26 self-test checks).**
**Transport: implemented and validated end-to-end against a mock server (14 self-test checks). NOT yet tested against a real zanod daemon.**

Three layers, built and proven in order (same discipline as `src/cuda/`:
prove each layer before trusting the next one):

1. `zano_stratum_protocol.{hpp,cpp}` - pure encode/decode for zanod's
   `eth_getWork`/`eth_submitWork`/`eth_submitLogin`/`eth_submitHashrate`
   JSON-RPC dialect. No sockets. Validated by `deepcore-protocol-selftest`
   against hand-verified wire strings independent of our own encoder.
2. `zano_stratum_client.{hpp,cpp}` (`ZanoStratumClient`) - a concrete
   `IStratumClient` (see `stratum_client.hpp`) built on plain TCP sockets
   and layer 1. Validated by `deepcore-network-selftest`, which runs a
   minimal mock server (also built with layer 1's helpers) and drives the
   real client through connect -> login -> receive work -> submit a share
   -> receive a result -> shutdown.
3. Not started: wiring this into an actual mining loop that takes real
   GPU-computed proofs and drives `ZanoStratumClient`, and a CLI.

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
- **Never tested against a real zanod.** The mock server in
  `tools/network_selftest` proves the client speaks the protocol
  correctly, not that a real daemon accepts it. See the root-level
  discussion of setting up `zanod --stratum` on a testnet build (e.g. via
  `canardleteer/zano-docker` with `-D TESTNET=TRUE`) for the next real
  validation step.

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

1. Get a real `zanod --stratum` testnet node running (see root project
   discussion) and point `deepcore-network-selftest`-style checks at it
   instead of the mock server - the actual "does a real daemon accept
   this" gate.
2. Build the ProgPowZ mining-loop glue: parse `JobNotification::raw_payload`
   into a `WorkPayload`, drive the GPU kernel with it, construct
   `ShareSubmission`s from results that clear the target boundary.
3. Only after (1)+(2) work end-to-end on testnet: reconnect/backoff,
   multi-endpoint failover, TLS, keepalive, CLI wiring.

Do not skip ahead to (3) before (1) and (2) are proven - a transport that
only knows how to fail over between endpoints it has never successfully
talked to a real daemon on is not meaningfully more "production-ready,"
just more complex.
