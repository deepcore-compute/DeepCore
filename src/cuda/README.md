# src/cuda - ProgPowZ GPU kernel work

## Status (current milestone)

**Algorithm correctness: validated on CPU. GPU execution: validated on real hardware.** `deepcore-gpu-selftest` has been built and run on a Quadro GV100 (Volta, `sm_70`) and all 4 recorded test vectors passed byte-for-byte (`final_hash` and `mix_hash`) against the CPU reference.

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

## What exists now: `progpowz_kernel.cu` + `gpu_selftest.cu`

`progpowz_kernel.cu` adds a `__global__` kernel (`progpowz_light_kernel`)
that is one thread = one nonce, calling `progpowz_hash_light()` from
`progpowz_portable.hpp` unmodified, plus a host-side launch wrapper
(`run_progpowz_light_gpu`: `cudaMalloc`/`cudaMemcpy`/launch/copy-back,
with `cudaGetErrorString`-based error checking).

`tools/cuda_selftest/gpu_selftest.cu` is the actual validation gate: it
builds a real epoch context via the reference library (same as
`cuda_selftest.cpp`), runs the same 4 fixed vectors through the GPU kernel,
and compares against the CPU reference byte-for-byte. Wired into CMake as
the `deepcore-gpu-selftest` target, built only when `-DDEEPCORE_WITH_CUDA=ON`
*and* a CUDA compiler is actually found (`check_language(CUDA)` - see
`CMakeLists.txt`; this degrades to a warning and skips the target rather
than failing configuration on machines without the CUDA Toolkit, which is
what this development environment is).

**`deepcore-gpu-selftest` has been built and run on real hardware.** Built
with `-DDEEPCORE_WITH_CUDA=ON` (CUDA Toolkit 12.4, `nvcc`) and run on a
machine with a Quadro GV100 GPU (Volta, `sm_70`, compute capability 7.0).
All 4 recorded test vectors passed byte-for-byte (both `final_hash` and
`mix_hash`) against the CPU reference. This confirms `progpowz_kernel.cu`
and `gpu_selftest.cu` compile with `nvcc`, the kernel launches, and it
produces correct output on real Volta hardware. Ampere (`sm_80`) and other
architectures remain unverified on real hardware.

## `GpuHashSearchBackend` (GPU milestone 2: mining_loop integration)

**Validated on real hardware (Quadro GV100): all 11 self-test checks pass.**

`progpowz_gpu_backend.{hpp,cu}` wraps `run_progpowz_light_gpu` (above) in a
`mining::IHashSearchBackend` (see `src/mining/mining_loop.hpp`) - the
adapter that lets `MiningLoop` actually drive this kernel, in place of the
CPU reference backend. `tools/gpu_backend_selftest/gpu_backend_selftest.cu`
is the validation gate for this layer, same role `gpu_selftest.cu` played
for milestone 1: it drives the backend through known-answer searches
(a target constructed from a nonce's own reference-computed hash, so the
expected winner is known ahead of time), a deterministic window-
correctness check (an always-satisfied target at two different start
nonces must return exactly that nonce, not the other), a not-found case,
cancellation, cross-run determinism, and a direct cross-check against
`CpuHashSearchBackend` on the same job/range.

Built with `-DDEEPCORE_WITH_CUDA=ON` and run on a Quadro GV100 (Volta,
`sm_70`). The first real run found 9/10 checks passing, with one failure
traced to a flawed test (not a backend bug): it used a target equal to one
specific nonce's exact hash as a stand-in for "hard to meet", but that
hash's magnitude (leading byte `0x32`) made it satisfiable by roughly 1 in
5 random nonces - over a 65536-nonce batch a coincidental extra match was
near-certain. Verified against the real reference implementation, then
replaced with a deterministic check that doesn't depend on any hash's
magnitude (an all-`0xFF` target must return exactly the requested start
nonce, proving the search window is honored rather than hoping for no
coincidence). Re-run on the same hardware: all 11 checks pass, GPU results
matching the CPU reference backend exactly.

Scope of this milestone, deliberately: still the "light" (recompute-on-
demand) kernel, and no persistent device memory - the epoch's caches are
re-uploaded to the GPU on every `search()` call, same as the kernel wrapper
it calls. A GPU-driven `MiningLoop` worker thread using this backend is
correctly wired but not yet fast; see "What does NOT exist yet" below for
what real throughput needs.

`IHashSearchBackend` also gained a `preferred_batch_size()` hook (default
0 = "no preference") so a backend like this one, whose per-call overhead
(kernel launch + cache upload) is large relative to per-nonce cost, can
ask `MiningLoop` for a much bigger batch per `search()` call than the
CPU backend's tuned default of 64 - `GpuHashSearchBackend` currently
requests 65536. This does mean coarser cancellation granularity than the
CPU backend (a CUDA kernel can't be interrupted mid-flight) - see that
class's header comment for why this is an accepted, documented tradeoff
rather than a new correctness gap.

## What does NOT exist yet

- **No full-DAG (mining-speed) kernel.** This only implements the
  "light"/cache-based path (recompute each dataset item on demand), which
  is what the reference vectors were generated with and is fine for a
  correctness check, but is far too slow per-hash for real mining. A real
  miner needs the full dataset resident in VRAM with O(1) lookups - that
  is a distinct, later performance milestone, not started here.
- **No persistent device memory in `GpuHashSearchBackend`.** The epoch's
  light_cache/l1_cache are re-uploaded on every `search()` call rather
  than kept resident in VRAM across calls for the same epoch - a real
  performance cost once this backend is otherwise validated.
- **No lane-cooperative (16 threads/warp via `__shfl_sync`) kernel.** That
  is the real-world performance mapping for ProgPoW on GPU; the
  single-thread-per-hash approach here is deliberately the simpler,
  lower-risk first step.
- **No CLI wiring.** `deepcore-miner` still always uses
  `CpuHashSearchBackend` - see `src/cli/README.md`. Selecting
  `GpuHashSearchBackend` from the CLI is a separate, later step, only
  after this backend is validated on real hardware.
- **No multi-GPU support, no integration with `gpu_manager.hpp`** (device
  enumeration/selection/telemetry) - `GpuHashSearchBackend` drives exactly
  one CUDA device index passed at construction.

## Next steps, in order

1. **Done on Volta.** Built and run on a machine with the CUDA Toolkit and
   a Quadro GV100 (Volta, `sm_70`) - configured with `-DDEEPCORE_WITH_CUDA=ON`,
   built the `deepcore-gpu-selftest` target, and ran it. All 4 test vectors
   passed byte-for-byte against the CPU reference: `progpowz_kernel.cu`
   compiles with `nvcc`, the kernel launches, and it produces correct output
   on real Volta hardware. Ampere (`sm_80`, target Tesla A100 hardware) and
   other `CMAKE_CUDA_ARCHITECTURES` this project targets remain unverified
   on real hardware.
2. **Done.** `GpuHashSearchBackend` built with `-DDEEPCORE_WITH_CUDA=ON`
   and validated on a real Quadro GV100 - all 11 self-test checks pass,
   see above.
3. Wire `GpuHashSearchBackend` into `deepcore-miner`
   (a `--gpu` flag or similar) and validate a live GPU mining run against a
   real `zanod`, same discipline as the CPU-backend CLI validation in
   `src/cli/README.md`.
4. Only after (3) works end-to-end: persistent device memory (upload once
   per epoch, not once per batch), the full-dataset (precomputed DAG in
   VRAM) path for real mining throughput sized dynamically from queried
   free VRAM (see `gpu_manager.hpp`'s telemetry interface, never a
   hard-coded capacity assumption), lane-cooperative warp-shuffle
   optimization, launch-parameter autotuning per architecture, CUDA
   Graphs / stream overlap, etc.

Do not skip ahead - "compiles" and "the CPU-side algorithm is correct" are
necessary but not sufficient at each step; only real GPU execution can
validate what that step actually added.
