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

## Persistent VRAM caching (GPU throughput milestone 1)

**Validated on real hardware (Quadro GV100): all 14 self-test checks
pass, including the new I1/I2 cache-identity checks.**

`GpuHashSearchBackend` previously re-uploaded the epoch's `light_cache`/
`l1_cache` (tens of MB) to the GPU on every `search()` batch. Real-world
testing against a live Zano mainnet pool and a cross-check against Rigel
(a known-working third-party miner, ~38 MH/s on the same Quadro GV100)
put a concrete number on the gap this project is actually working from:
this backend runs at ~13 KH/s - about 3000x below a competitively-
optimized kernel on identical hardware (see `src/network/README.md` for
the full account). Re-uploading the light_cache every ~5 seconds was one
real, avoidable cost on top of that gap.

`DeviceEpochCache` and `PersistentGpuSearcher` (added to
`progpowz_kernel.cu`, additive - the correctness-only
`run_progpowz_light_gpu` used by `gpu_selftest.cu` is untouched) keep the
light_cache/l1_cache resident in VRAM across calls, re-uploading only
when the epoch context's host pointer actually changes (which
`MiningLoop`'s own epoch-context cache guarantees happens only on a real
epoch change, not on every job). `GpuHashSearchBackend` now uses these
via a pImpl (so its public header stays plain C++, includable from
non-CUDA translation units like the CLI). `gpu_backend_selftest` gained
two new checks (I1/I2) specifically targeting the new identity-check
logic: switching between two independent epoch-0 contexts and back, to
catch any bug that would let a stale cached light_cache silently leak
into results for a different context.

**Important scope note: this alone does not close anywhere near the
3000x gap.** It removes one real but secondary cost (re-upload
overhead); the dominant cost is that light-cache mode recomputes each
dataset item from scratch (many rounds of Keccak-based mixing) on every
lookup, which is what the full-DAG kernel below actually replaces.

## Full-DAG kernel (GPU throughput milestone 2)

**Validated on real hardware (Quadro GV100): all 15 self-test checks
pass (including J, confirming full-DAG mode actually ran, not a silent
fallback), and real measured throughput against a live Zano mainnet pool:
~524 KH/s - about 40x faster than light-cache mode's ~13 KH/s, closing
the gap to a competitive miner (Rigel, ~38 MH/s on the same GV100) from
~3000x down to ~72x.**

`progpowz_hash_light` (in `progpowz_portable.hpp`) gained an optional
`full_dataset` parameter: when null (every existing caller, unaffected),
behavior is completely unchanged (recompute each dataset item on demand);
when provided, it reads `full_dataset[item_index]` instead - the same
value `calculate_dataset_item_2048` would have computed, just precomputed
once instead of recomputed on every one of the 64 mix-round lookups per
hash. This one substitution is where the real throughput gain comes from.

This mechanism was validated on CPU first, with a small synthetic dataset
built against the real epoch-0 light_cache (`tools/cuda_selftest/
cuda_selftest.cpp`): light mode and full-DAG-array mode produce
byte-identical output for several nonces. A real dataset (millions of
items, multiple GB) is only realistically buildable on a GPU, so
real-scale validation is GPU-only, same situation as the original kernel.

`progpowz_kernel.cu` adds: `progpowz_dag_generate_kernel` (one thread per
dataset item, calling the already-proven `calculate_dataset_item_2048`)
and `progpowz_full_kernel` (identical mapping to `progpowz_light_kernel`,
but sourcing dataset items from a precomputed array); `DeviceFullDataset`
builds and keeps the full dataset VRAM-resident per epoch (same host-
pointer-identity caching pattern as `DeviceEpochCache`), checking real
free VRAM via `cudaMemGetInfo` first and failing clearly rather than
assuming capacity; `PersistentFullDagSearcher` launches it with a reused
output buffer. `GpuHashSearchBackend` now prefers full-DAG mode whenever
the epoch's dataset fits, falling back to light-cache mode (with a
clearly logged reason, once per epoch) when it doesn't -
`last_search_used_full_dag()` lets `gpu_backend_selftest` assert which
path actually ran, since light mode is also correct and would otherwise
make every check pass silently either way.

## Lane-cooperative warp-shuffle kernel (GPU throughput milestone 3)

**Correctness validated on real hardware (10/10 checks in
`deepcore-warp-kernel-selftest`, first attempt). First real hashrate
measurement through the CLI was a genuine regression (~340-354 KH/s,
*slower* than the plain full-DAG kernel's ~524 KH/s) - diagnosed,
fixed, re-validation pending.**

### The regression, and why it happened

The first version had every one of the 16 cooperating lanes
independently read the FULL 256-byte dataset item from global memory
(each lane needed only its own 16-byte slice, but fetched the whole
thing for simplicity - see the design rationale originally written
here). That is 16x more DAG memory traffic than the plain kernel needs
for the same hash: 64 rounds x 16 lanes x 256 bytes = 256 KiB/hash vs.
64 rounds x 256 bytes = 16 KiB/hash. On a workload this memory-
bandwidth-bound (effectively random access across a multi-GB dataset,
which defeats caching), that redundancy measurably outweighed the
compute-parallelism gain from cooperating in the first place - a real,
reproducible result, not a guess, and an important lesson that "more
parallel threads" isn't automatically faster when the bottleneck is
memory bandwidth, not compute.

**Fix:** each lane now reads only its own fixed 16-byte identity slice
of the item (lane `l` reads bytes `[l*16, l*16+16)` - together the 16
lanes cover the full 256 bytes with zero redundant reads), then obtains
the specific slice it actually needs THIS round via `__shfl_sync` from
whichever lane's identity slice matches `(lane ^ round) % num_lanes` -
exchanging already-fetched register data instead of re-reading memory.
Total DAG traffic per hash is now identical to the plain kernel (16
KiB), while keeping the 16x compute-parallelism benefit. This is the
standard, correct way to map ProgPoW's lane concept onto a GPU warp -
what the original design should have done from the start; the
redundant-read version was an explicitly-flagged simplification that
turned out to cost real performance once measured.

Written, not yet re-validated: this changes the actual per-round data
flow, so both `deepcore-warp-kernel-selftest` (correctness) and a fresh
live hashrate measurement are needed before trusting either the
correctness or the performance of this fix.

`progpowz_light_kernel`/`progpowz_full_kernel` map ONE GPU thread to ALL
16 ProgPoW lanes, looping over them sequentially - correct, but not how
ProgPoW's lane concept was actually designed to run on a GPU (16 threads
cooperating via warp-shuffle). Tracing through the sequential algorithm
shows the real cross-lane dependency is narrow: mix initialization,
cache-access, math-operations, and the item-application step are all
already fully lane-parallel (no lane needs another lane's register
state) once every lane independently fetches the same broadcast-index
dataset item. Only two points need real cross-lane communication: the
per-round `item_index` broadcast, and the once-per-hash final reduction
of all 16 lanes' `lane_hash` into one `mix_hash` - both `__shfl_sync`-
based (full documentation in `progpowz_kernel.cu`'s header comment for
this section).

`progpowz_hash_warp_full_dag` (one thread = one lane, 16 cooperating
threads = one hash) and `progpowz_warp_kernel`/`PersistentWarpSearcher`
(the launch wrapper - every launched thread runs the full algorithm even
past `count`, since `__shfl_sync` requires the whole warp in lockstep;
only the final write is guarded) implement this. As a side effect, this
also reduces per-thread register/stack usage substantially (~384 bytes
for one lane's state vs. ~2048 bytes for all 16 lanes in the sequential
kernel), which should help occupancy independent of the parallelism
itself - not yet measured.

Since `__shfl_sync` has no CPU equivalent, this is the first code in the
whole project with no way to pre-validate the actual cross-lane mechanism
before real hardware. `tools/warp_kernel_selftest/warp_kernel_selftest.cu`
is the dedicated gate: compares this kernel's output against the
already-validated full-DAG kernel (`PersistentFullDagSearcher`) AND the
real reference implementation directly, across a single nonce, a
non-block-aligned batch (17 - exercises the padding/tail logic), and a
large multi-block batch (256). Deliberately NOT wired into
`GpuHashSearchBackend` yet - it stays isolated/opt-in until its own gate
passes for real, so the currently-working production path (plain
full-DAG, ~524 KH/s) is never put at risk by unvalidated code.

## What does NOT exist yet

- **Warp-shuffle kernel's memory-traffic fix not yet re-validated** -
  see above.
- **No launch-parameter autotuning, no CUDA Graphs / stream overlap.**
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
3. **Done.** `--gpu` wired into `deepcore-miner`, validated against a real
   local `zanod` and a real public Zano mainnet pool (LuckyPool) - found
   and fixed two real bugs along the way (byte-order, response-shape) -
   see `src/network/README.md`. Real hashrate confirmed: ~13 KH/s, vs.
   ~38 MH/s for a competitive kernel (Rigel) on the same GV100.
4. **Done.** Persistent VRAM caching (upload the epoch's caches once, not
   once per batch) - validated on a real Quadro GV100, all 14 self-test
   checks pass, see above.
5. **Done.** The full-dataset (precomputed DAG in VRAM, O(1) lookups
   instead of recompute-on-demand) path - validated on a real Quadro
   GV100 (all 15 self-test checks pass), real measured throughput ~524
   KH/s against a live Zano mainnet pool - about 40x faster than
   light-cache mode, closing the gap to a competitive miner (Rigel, ~38
   MH/s) from ~3000x to ~72x. Sized dynamically from queried free VRAM,
   never a hard-coded capacity assumption.
6. **In progress.** `l1_cache` (16KiB, identical for every thread) moved
   into on-chip `__shared__` memory in `progpowz_full_kernel`, loaded
   once per thread block instead of re-read from global memory on every
   mix-round lookup - pure memory-locality change, no algorithm
   difference, so existing correctness self-tests remain valid regression
   coverage. Written, not yet run on real hardware.
7. **In progress.** Lane-cooperative warp-shuffle kernel - correctness
   validated (10/10 checks), wired into `GpuHashSearchBackend`, but the
   first live hashrate measurement was a real regression (~340-354 KH/s,
   slower than the plain full-DAG kernel). Root-caused to 16x redundant
   per-lane DAG reads; fixed to a zero-redundancy per-lane slice fetch +
   shuffle-based redistribution (see above) - not yet re-validated for
   correctness or performance.
8. Still open, after (7) is re-validated: launch-parameter autotuning
   per architecture, CUDA Graphs / stream overlap.

Do not skip ahead - "compiles" and "the CPU-side algorithm is correct" are
necessary but not sufficient at each step; only real GPU execution can
validate what that step actually added.
