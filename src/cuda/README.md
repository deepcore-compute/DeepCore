# src/cuda - ProgPowZ GPU kernel work

## Status (current milestone)

**Algorithm correctness: validated on CPU. GPU execution: not yet attempted.**

`progpowz_portable.hpp` is a from-scratch port of ProgPowZ's per-nonce hash
algorithm (Keccak-f[1600], Keccak-f[800], dataset-item generation, the
ProgPoW mix/round loop) into `__host__ __device__`-callable free functions,
derived from the vendored reference in `third_party/ethash-progpow-ref/`
(Apache-2.0 - see that directory for provenance and license).

It is written to be single-threaded (all 16 ProgPoW "lanes" are processed
in a sequential loop, not by 16 cooperating GPU threads) specifically so it
can be:

1. Compiled and run **today, with no CUDA toolchain**, as ordinary host
   C++ (`tools/cuda_selftest/cuda_selftest.cpp`, target
   `deepcore-cuda-selftest`), and checked byte-for-byte against the real
   reference implementation's own output.
2. Later called, unmodified, from inside a CUDA `__global__` kernel with
   one thread computing one full nonce's hash - the simplest, lowest-risk
   GPU mapping, and the right first GPU milestone per this project's
   correctness-before-optimization policy.

`deepcore-cuda-selftest` currently passes on all 4 recorded test vectors
(byte-identical final_hash and mix_hash vs. the reference, for both the
end-to-end hash and two isolated sub-checks: the keccak_progpow_64/256
pipeline, and calculate_dataset_item_2048 alone).

### Bugs this self-test already caught (kept here as a record of why it exists)

Three real bugs were found and fixed by this self-test before it passed,
none of which would have been caught by "it compiles":

1. `keccakf800`: an initial draft used a generic, from-memory
   reconstruction of the Keccak permutation instead of transcribing the
   actual (fully-unrolled) reference file - wrong output.
2. `round()`'s RNG state: the reference passes its `mix_rng_state` **by
   value** into `round()`, meaning every one of the 64 rounds restarts
   from the same initial RNG state rather than carrying state forward
   between rounds. An initial port let the RNG advance continuously
   across rounds instead.
3. `random_merge`'s rotate-right case had its two shift amounts swapped
   (computed `rotl32(a, 32-x)` instead of `rotr32(a, x)`).

This is the concrete argument for why `tools/cuda_selftest` exists and
must keep passing: this algorithm has enough small, easy-to-get-wrong
details that "looks right" is not a substitute for "matches the reference
byte-for-byte."

## What does NOT exist yet

- **No actual `.cu` file / CUDA kernel.** `progpowz_portable.hpp` has never
  been compiled with `nvcc` or run on a GPU - this development environment
  has neither the CUDA Toolkit nor an NVIDIA GPU available. It is written
  to be device-compatible (`__host__ __device__`, CUDA intrinsics gated
  behind `__CUDA_ARCH__`), but that has only been checked by inspection,
  not by an actual device compile.
- **No full-DAG (mining-speed) kernel.** This only implements the
  "light"/cache-based path (recompute each dataset item on demand), which
  is what the reference vectors were generated with and is fine for a
  correctness check, but is far too slow per-hash for real mining. A real
  miner needs the full dataset resident in VRAM with O(1) lookups - that
  is a distinct, later performance milestone, not started here.
- **No lane-cooperative (16 threads/warp via `__shfl_sync`) kernel.** That
  is the real-world performance mapping for ProgPoW on GPU; the
  single-thread-per-hash approach here is deliberately the simpler,
  lower-risk first step.
- **No nonce-range search loop, no host-side launch/orchestration code,
  no integration with `gpu_manager.hpp`/`stratum_client.hpp`.**

## Next steps, in order

1. Get this code onto a machine with the CUDA Toolkit and a Volta or
   Ampere GPU (matching the project's actual hardware: Tesla V100-SXM2 /
   Tesla A100). Write a thin `.cu` wrapper: one `__global__` kernel launch
   parameter block, device-side buffers for `light_cache` / `l1_cache`
   (uploaded once per epoch via `cudaMemcpy`), one thread per nonce calling
   `progpowz_hash_light()` unmodified, results copied back and compared
   against the same reference vectors `deepcore-cuda-selftest` already
   uses. This is the first real "does this run correctly on a GPU" gate.
2. Only after (1) passes: build the full-dataset (precomputed DAG in
   VRAM) path for real mining throughput, sized dynamically from queried
   free VRAM (see `gpu_manager.hpp`'s telemetry interface) - never a
   hard-coded capacity assumption.
3. Only after (2) is measured working: lane-cooperative warp-shuffle
   optimization, launch-parameter autotuning per architecture, CUDA
   Graphs / stream overlap, etc.

Do not skip ahead to (2) or (3) before (1) has actually run and passed on
real hardware - "compiles" and "the CPU-side algorithm is correct" are
necessary but not sufficient; only real GPU execution can validate the
kernel launch/memory-transfer code, which is entirely new/untested surface
that this milestone deliberately did not touch.
