# ethash-progpow-ref

This directory is a **vendored, unmodified copy** of the ProgPoW/Ethash
reference implementation used by Zano's own daemon, taken directly from
[`hyle-team/zano`](https://github.com/hyle-team/zano)
at commit `ee3de1e5a077b60106ba88301e236474680b1028`
(path: `contrib/ethereum/libethash`), which is itself vendored from
[`chfast/ethash`](https://github.com/chfast/ethash) (Pawel Bylica,
Apache License 2.0 - see `LICENSE` in this directory).

## Why this is here

Per DeepCore's correctness-before-optimization policy, no GPU kernel is
implemented from a written spec alone - it is implemented against a known-
correct CPU reference and validated with deterministic test vectors before
any GPU work begins. This is that reference: it is the *exact* code Zano's
daemon runs for proof-of-work, so its output is authoritative for what a
valid ZANO share must look like.

## What it is NOT

- **Not** used by the `deepcore-miner` binary at runtime - there is no
  production code path that calls into this directory. It exists purely
  as a build-time/test-time correctness oracle (see
  `tools/reference/gen_vectors.cpp`).
- **Not** a CPU miner. It is never wired into pool submission or any
  mining loop.
- **Not** modified from upstream - if a fix or change is ever needed here,
  it must be pulled from upstream Zano/ethash, not hand-edited, so this
  directory stays a faithful, auditable copy.

## Confirmed facts about Zano's PoW (from reading this code)

- Algorithm: ProgPoW revision `0.9.2` (`progpow::revision` in
  `ethash/progpow.hpp`), unmodified from the public ProgPoW spec.
- Parameters: `period_length=50`, `num_regs=32`, `num_lanes=16`,
  `num_cache_accesses=12`, `num_math_operations=20`, `l1_cache_size=16KiB`.
- DAG/epoch: standard Ethash dataset generation, `epoch_length=30000`
  blocks (`ETHASH_EPOCH_LENGTH` in `ethash/ethash.h`).
- Zano-specific integration point (not part of this vendored directory,
  see `zano/src/currency_core/basic_pow_helpers.cpp` in the upstream repo):
  the `header_hash` fed into `progpow::hash()` is Zano's own
  `cn_fast_hash` (Keccak-based) of the serialized block with the nonce
  field zeroed.

## Provenance / update policy

If Zano upgrades its ProgPoW/Ethash dependency, this directory should be
re-synced wholesale from the new `contrib/ethereum/libethash` at the
commit Zano's mainnet daemon is running, with this README's commit hash
updated to match - not patched piecemeal.
