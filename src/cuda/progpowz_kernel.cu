// ProgPowZ CUDA kernel - GPU milestone 1: correctness only.
//
// This is a THIN wrapper around the already CPU-validated
// progpowz_portable.hpp (see that file and src/cuda/README.md for what has
// and hasn't been proven correct so far). Per this project's
// correctness-before-optimization policy, the mapping here is deliberately
// the simplest possible one: ONE CUDA THREAD COMPUTES ONE FULL NONCE'S HASH,
// looping over all 16 ProgPoW lanes internally exactly like the CPU code -
// no warp-shuffle lane cooperation, no precomputed full-DAG-in-VRAM, no
// nonce-range search/target-comparison logic. Those are later, separate
// performance milestones (see README.md) that must not be attempted before
// this correctness baseline is confirmed running correctly on real
// hardware.
//
// STATUS: written but NEVER COMPILED OR RUN. This development environment
// has no CUDA Toolkit and no NVIDIA GPU. Do not treat this file as working
// until tools/cuda_selftest/gpu_selftest.cu has actually been built and run
// on real hardware (Tesla V100 / Tesla A100) and reported PASS.

#include "progpowz_portable.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace deepcore::progpowz {

struct GpuHashResult
{
    hash256 final_hash;
    hash256 mix_hash;
    uint64_t nonce;
};

__global__ void progpowz_light_kernel(
    const hash512* light_cache, int64_t light_cache_num_items,
    const uint32_t* l1_cache_words, uint32_t full_dataset_num_items,
    int block_number, hash256 header_hash, uint64_t start_nonce,
    uint32_t count, GpuHashResult* out)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    uint64_t nonce = start_nonce + idx;
    progpowz_result r = progpowz_hash_light(
        light_cache, light_cache_num_items, l1_cache_words, full_dataset_num_items,
        block_number, header_hash, nonce);

    out[idx].final_hash = r.final_hash;
    out[idx].mix_hash = r.mix_hash;
    out[idx].nonce = nonce;
}

// ----------------------------------------------------------------------
// Host-side launch wrapper.
// ----------------------------------------------------------------------

#define DEEPCORE_CUDA_CHECK(expr)                                                          \
    do                                                                                      \
    {                                                                                       \
        cudaError_t deepcore_cuda_err__ = (expr);                                           \
        if (deepcore_cuda_err__ != cudaSuccess)                                             \
        {                                                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                cudaGetErrorString(deepcore_cuda_err__));                                   \
            std::exit(1);                                                                   \
        }                                                                                    \
    } while (0)

// Runs progpowz_light_kernel for `count` consecutive nonces starting at
// `start_nonce`, uploading `light_cache`/`l1_cache_words` fresh each call.
// This is intentionally simple/unoptimized (no persistent device
// allocation, no streams) - it exists to answer one question only: "does
// the ported algorithm produce the same output on a real GPU as it does on
// the CPU host?" Reusing device allocations across calls, overlapping
// transfer/compute, and batching many more nonces per launch are
// performance-milestone concerns, not part of this correctness check.
inline std::vector<GpuHashResult> run_progpowz_light_gpu(
    const hash512* host_light_cache, int64_t light_cache_num_items,
    const uint32_t* host_l1_cache_words, uint32_t full_dataset_num_items,
    int block_number, const hash256& header_hash, uint64_t start_nonce, uint32_t count)
{
    hash512* d_light_cache = nullptr;
    uint32_t* d_l1_cache = nullptr;
    GpuHashResult* d_out = nullptr;

    const size_t light_cache_bytes = static_cast<size_t>(light_cache_num_items) * sizeof(hash512);
    const size_t l1_cache_bytes = static_cast<size_t>(l1_cache_num_items) * sizeof(uint32_t);
    const size_t out_bytes = static_cast<size_t>(count) * sizeof(GpuHashResult);

    DEEPCORE_CUDA_CHECK(cudaMalloc(&d_light_cache, light_cache_bytes));
    DEEPCORE_CUDA_CHECK(cudaMalloc(&d_l1_cache, l1_cache_bytes));
    DEEPCORE_CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    DEEPCORE_CUDA_CHECK(cudaMemcpy(d_light_cache, host_light_cache, light_cache_bytes, cudaMemcpyHostToDevice));
    DEEPCORE_CUDA_CHECK(cudaMemcpy(d_l1_cache, host_l1_cache_words, l1_cache_bytes, cudaMemcpyHostToDevice));

    const uint32_t threads_per_block = 256;
    const uint32_t blocks = (count + threads_per_block - 1) / threads_per_block;

    progpowz_light_kernel<<<blocks, threads_per_block>>>(
        d_light_cache, light_cache_num_items, d_l1_cache, full_dataset_num_items,
        block_number, header_hash, start_nonce, count, d_out);

    DEEPCORE_CUDA_CHECK(cudaGetLastError());
    DEEPCORE_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GpuHashResult> results(count);
    DEEPCORE_CUDA_CHECK(cudaMemcpy(results.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_light_cache);
    cudaFree(d_l1_cache);
    cudaFree(d_out);

    return results;
}

// ----------------------------------------------------------------------
// GPU throughput milestone 1: persistent device memory.
//
// run_progpowz_light_gpu above is deliberately the simplest possible
// wrapper (GPU milestone 1: correctness only - re-upload/re-free every
// call) and is left completely unmodified; it remains the correctness
// baseline gpu_selftest.cu validates against. The classes below are
// additive: they call the SAME progpowz_light_kernel with the SAME
// per-hash algorithm (already validated correct on real hardware), and
// only change when device memory is allocated/uploaded/freed - avoiding
// re-uploading the epoch's light_cache/l1_cache (tens of MB) on every
// search batch, which the "always re-upload" wrapper above does every
// single call. GpuHashSearchBackend uses these; the correctness self-test
// keeps using the simple wrapper above.
// ----------------------------------------------------------------------

// Keeps one epoch's light_cache/l1_cache resident in VRAM across many
// calls. "Same epoch already uploaded" is tracked by host pointer
// identity of light_cache: MiningLoop's epoch_context_for() caches and
// reuses the same ethash::epoch_context (and therefore the same
// light_cache pointer) for repeated jobs in the same epoch - see
// mining_loop.cpp - so pointer identity is a correct, cheap check here,
// not a heuristic.
class DeviceEpochCache {
public:
    DeviceEpochCache() = default;
    ~DeviceEpochCache() { release(); }
    DeviceEpochCache(const DeviceEpochCache&) = delete;
    DeviceEpochCache& operator=(const DeviceEpochCache&) = delete;

    void ensure_uploaded(const hash512* host_light_cache, int64_t light_cache_num_items,
        const uint32_t* host_l1_cache_words, size_t l1_cache_num_words)
    {
        if (uploaded_identity_ == host_light_cache)
            return;

        release();

        const size_t light_bytes = static_cast<size_t>(light_cache_num_items) * sizeof(hash512);
        const size_t l1_bytes = l1_cache_num_words * sizeof(uint32_t);

        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_light_cache_, light_bytes));
        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_l1_cache_, l1_bytes));
        DEEPCORE_CUDA_CHECK(cudaMemcpy(d_light_cache_, host_light_cache, light_bytes, cudaMemcpyHostToDevice));
        DEEPCORE_CUDA_CHECK(cudaMemcpy(d_l1_cache_, host_l1_cache_words, l1_bytes, cudaMemcpyHostToDevice));

        uploaded_identity_ = host_light_cache;
    }

    [[nodiscard]] const hash512* device_light_cache() const { return d_light_cache_; }
    [[nodiscard]] const uint32_t* device_l1_cache() const { return d_l1_cache_; }

private:
    void release()
    {
        if (d_light_cache_) { cudaFree(d_light_cache_); d_light_cache_ = nullptr; }
        if (d_l1_cache_) { cudaFree(d_l1_cache_); d_l1_cache_ = nullptr; }
        uploaded_identity_ = nullptr;
    }

    const hash512* uploaded_identity_ = nullptr;
    hash512* d_light_cache_ = nullptr;
    uint32_t* d_l1_cache_ = nullptr;
};

// Launches progpowz_light_kernel against an already-uploaded
// DeviceEpochCache, reusing a persistent (grow-only) output buffer across
// calls instead of allocating/freeing it every time.
class PersistentGpuSearcher {
public:
    PersistentGpuSearcher() = default;
    ~PersistentGpuSearcher() { if (d_out_) cudaFree(d_out_); }
    PersistentGpuSearcher(const PersistentGpuSearcher&) = delete;
    PersistentGpuSearcher& operator=(const PersistentGpuSearcher&) = delete;

    std::vector<GpuHashResult> search(DeviceEpochCache& cache, int64_t light_cache_num_items,
        uint32_t full_dataset_num_items, int block_number, const hash256& header_hash,
        uint64_t start_nonce, uint32_t count)
    {
        ensure_out_capacity(count);

        const uint32_t threads_per_block = 256;
        const uint32_t blocks = (count + threads_per_block - 1) / threads_per_block;

        progpowz_light_kernel<<<blocks, threads_per_block>>>(
            cache.device_light_cache(), light_cache_num_items, cache.device_l1_cache(),
            full_dataset_num_items, block_number, header_hash, start_nonce, count, d_out_);

        DEEPCORE_CUDA_CHECK(cudaGetLastError());
        DEEPCORE_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<GpuHashResult> results(count);
        DEEPCORE_CUDA_CHECK(cudaMemcpy(results.data(), d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult),
            cudaMemcpyDeviceToHost));
        return results;
    }

private:
    void ensure_out_capacity(uint32_t count)
    {
        if (count <= d_out_capacity_)
            return;
        if (d_out_)
            cudaFree(d_out_);
        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult)));
        d_out_capacity_ = count;
    }

    GpuHashResult* d_out_ = nullptr;
    uint32_t d_out_capacity_ = 0;
};

// ----------------------------------------------------------------------
// GPU throughput milestone 2: full-DAG kernel.
//
// Light-cache mode (above) recomputes each dataset item from scratch via
// calculate_dataset_item_2048 on every lookup during mixing (256 parent
// rounds x 4 sub-items, once per one of the 64 mix rounds per hash) - this
// is the dominant real cost behind this project's confirmed ~3000x
// throughput gap versus a competitive miner on the same hardware (see
// src/cuda/README.md, src/network/README.md's Rigel cross-check). Full-DAG
// mode instead precomputes every dataset item ONCE per epoch into a large
// VRAM-resident array, so each mix-round lookup becomes an O(1) read
// instead of a full recompute.
//
// Correctness rests entirely on progpowz_hash_light's `full_dataset`
// parameter (see progpowz_portable.hpp): when provided, it reads
// full_dataset[item_index] instead of calling calculate_dataset_item_2048
// - and the array is defined to hold exactly what that function would
// have returned for each index, so populating it correctly is the whole
// correctness question. That mechanism itself was validated on CPU with a
// small synthetic dataset (tools/cuda_selftest/cuda_selftest.cpp); this
// kernel is what populates a REAL, full-scale array (which - at millions
// of items and multiple GB - is only realistically buildable on a GPU,
// not something this project's CPU-only development environment can
// exercise). Real-scale end-to-end validation is the job of
// gpu_backend_selftest, comparing full-DAG output against the
// already-validated light-mode kernel for real nonces against a real
// epoch's full dataset - see that file.
// ----------------------------------------------------------------------

// One thread computes ONE full dataset item (hash2048, i.e. 4 combined
// hash512 sub-items) - the same simple "one thread, one unit of work"
// mapping this project used for progpowz_light_kernel above, for the same
// reason: it is the lowest-risk way to parallelize an already-correct
// scalar function, deferring cooperative/warp-level optimization to a
// later, separate milestone once this is proven correct.
__global__ void progpowz_dag_generate_kernel(
    const hash512* light_cache, int64_t light_cache_num_items, uint32_t num_items, hash2048* out)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items)
        return;
    out[idx] = calculate_dataset_item_2048(light_cache, light_cache_num_items, idx);
}

// Full-DAG counterpart to progpowz_light_kernel: identical mapping (one
// thread, one nonce), but passes a precomputed dataset array through to
// progpowz_hash_light instead of light_cache/light_cache_num_items (which
// are unused - and therefore safe to leave null/zero - whenever
// full_dataset is non-null; see that function's header comment).
//
// Performance-only addition (no algorithm change - the values read are
// identical either way, only where they live differs): l1_cache_words is
// the same 16KiB (l1_cache_num_items words) for every thread in the
// kernel, so instead of every thread re-reading it from slow global
// memory on every one of the mix loop's per-round lookups, the whole
// block cooperatively loads it into on-chip __shared__ memory ONCE, and
// every thread reads from that shared copy afterward. This is the kernel
// GpuHashSearchBackend actually uses for real mining now (light mode is
// an emergency VRAM-insufficient fallback only), so it's the one worth
// optimizing here.
//
// CUDA correctness note: the cooperative load and __syncthreads() MUST
// run before any thread can return early (idx >= count) - __syncthreads()
// requires every thread in the block to reach it, or the wait is
// undefined behavior. That's why the bounds check is after the sync, not
// before, unlike progpowz_light_kernel above.
__global__ void progpowz_full_kernel(
    const uint32_t* l1_cache_words, uint32_t full_dataset_num_items, const hash2048* full_dataset,
    int block_number, hash256 header_hash, uint64_t start_nonce, uint32_t count, GpuHashResult* out)
{
    __shared__ uint32_t s_l1_cache[l1_cache_num_items];
    for (uint32_t i = threadIdx.x; i < l1_cache_num_items; i += blockDim.x)
        s_l1_cache[i] = l1_cache_words[i];
    __syncthreads();

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    uint64_t nonce = start_nonce + idx;
    progpowz_result r = progpowz_hash_light(
        /*light_cache=*/nullptr, /*light_cache_num_items=*/0, s_l1_cache, full_dataset_num_items,
        block_number, header_hash, nonce, full_dataset);

    out[idx].final_hash = r.final_hash;
    out[idx].mix_hash = r.mix_hash;
    out[idx].nonce = nonce;
}

// Builds and keeps VRAM-resident the full dataset for one epoch, rebuilding
// only when the epoch's host light_cache pointer changes (same identity-
// check pattern as DeviceEpochCache above, for the same reason: MiningLoop
// reuses the same epoch_context, and therefore the same light_cache
// pointer, across repeated jobs in the same epoch).
//
// Queries actual free VRAM before allocating (cudaMemGetInfo) and fails
// clearly via `error_out` rather than attempting an allocation that would
// exceed it - this project's stated policy is to never assume GPU
// capacity. Callers must check the return value; a false return leaves
// the dataset unbuilt/unusable (device_dataset() returns nullptr).
class DeviceFullDataset {
public:
    DeviceFullDataset() = default;
    ~DeviceFullDataset() { release(); }
    DeviceFullDataset(const DeviceFullDataset&) = delete;
    DeviceFullDataset& operator=(const DeviceFullDataset&) = delete;

    [[nodiscard]] bool ensure_built(const hash512* host_light_cache, int64_t light_cache_num_items,
        uint32_t full_dataset_num_items, std::string& error_out)
    {
        if (uploaded_identity_ == host_light_cache)
            return true;

        release();

        const uint32_t num_items = full_dataset_num_items / 2;
        const size_t dataset_bytes = static_cast<size_t>(num_items) * sizeof(hash2048);
        const size_t light_cache_bytes = static_cast<size_t>(light_cache_num_items) * sizeof(hash512);

        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        {
            error_out = "cudaMemGetInfo failed - cannot verify free VRAM before building the full DAG";
            return false;
        }

        // Headroom for the light_cache upload (freed right after
        // generation, but resident during it), output buffers, and driver/
        // other-process overhead - deliberately conservative rather than
        // trying to consume every last free byte.
        constexpr size_t kHeadroomBytes = 512ull * 1024 * 1024;
        const size_t bytes_needed = dataset_bytes + light_cache_bytes + kHeadroomBytes;
        if (bytes_needed > free_bytes)
        {
            error_out = "insufficient free VRAM for the full DAG: need ~" +
                std::to_string(bytes_needed / (1024 * 1024)) + " MiB, have " +
                std::to_string(free_bytes / (1024 * 1024)) + " MiB free";
            return false;
        }

        hash512* d_light_cache = nullptr;
        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_light_cache, light_cache_bytes));
        DEEPCORE_CUDA_CHECK(
            cudaMemcpy(d_light_cache, host_light_cache, light_cache_bytes, cudaMemcpyHostToDevice));

        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_dataset_, dataset_bytes));

        const uint32_t threads_per_block = 256;
        const uint32_t blocks = (num_items + threads_per_block - 1) / threads_per_block;
        progpowz_dag_generate_kernel<<<blocks, threads_per_block>>>(
            d_light_cache, light_cache_num_items, num_items, d_dataset_);

        DEEPCORE_CUDA_CHECK(cudaGetLastError());
        DEEPCORE_CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_light_cache);  // only needed during generation, not for mining lookups afterward

        uploaded_identity_ = host_light_cache;
        return true;
    }

    [[nodiscard]] const hash2048* device_dataset() const { return d_dataset_; }

private:
    void release()
    {
        if (d_dataset_) { cudaFree(d_dataset_); d_dataset_ = nullptr; }
        uploaded_identity_ = nullptr;
    }

    const hash512* uploaded_identity_ = nullptr;
    hash2048* d_dataset_ = nullptr;
};

// Launches progpowz_full_kernel against an already-built DeviceFullDataset,
// reusing a persistent (grow-only) output buffer - same pattern as
// PersistentGpuSearcher above, for light mode.
class PersistentFullDagSearcher {
public:
    PersistentFullDagSearcher() = default;
    ~PersistentFullDagSearcher() { if (d_out_) cudaFree(d_out_); }
    PersistentFullDagSearcher(const PersistentFullDagSearcher&) = delete;
    PersistentFullDagSearcher& operator=(const PersistentFullDagSearcher&) = delete;

    std::vector<GpuHashResult> search(DeviceFullDataset& dataset, const uint32_t* device_l1_cache_words,
        uint32_t full_dataset_num_items, int block_number, const hash256& header_hash, uint64_t start_nonce,
        uint32_t count)
    {
        ensure_out_capacity(count);

        const uint32_t threads_per_block = 256;
        const uint32_t blocks = (count + threads_per_block - 1) / threads_per_block;

        progpowz_full_kernel<<<blocks, threads_per_block>>>(device_l1_cache_words, full_dataset_num_items,
            dataset.device_dataset(), block_number, header_hash, start_nonce, count, d_out_);

        DEEPCORE_CUDA_CHECK(cudaGetLastError());
        DEEPCORE_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<GpuHashResult> results(count);
        DEEPCORE_CUDA_CHECK(cudaMemcpy(results.data(), d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult),
            cudaMemcpyDeviceToHost));
        return results;
    }

private:
    void ensure_out_capacity(uint32_t count)
    {
        if (count <= d_out_capacity_)
            return;
        if (d_out_)
            cudaFree(d_out_);
        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult)));
        d_out_capacity_ = count;
    }

    GpuHashResult* d_out_ = nullptr;
    uint32_t d_out_capacity_ = 0;
};

// ----------------------------------------------------------------------
// GPU throughput milestone 3: lane-cooperative (warp-shuffle) kernel,
// full-DAG mode only (light-cache mode keeps using progpowz_light_kernel/
// progpowz_full_kernel above, unmodified).
//
// progpowz_hash_light (the CPU-portable core every kernel above calls)
// maps ONE GPU THREAD to ALL 16 ProgPoW lanes, looping over them
// sequentially inside each round - the simplest, lowest-risk correctness
// baseline, but not how ProgPoW's "lanes" concept was actually designed
// to map onto a GPU: 16 threads are meant to cooperate via warp-shuffle,
// each holding only its own lane's register state, instead of one thread
// doing 16x the sequential work.
//
// Tracing through progpowz_hash_light's mix loop (progpowz_portable.hpp)
// shows the cross-lane data dependency is narrower than it might look:
//   - mix initialization: mix[l][*]'s seed depends on l but not on any
//     OTHER lane's state - each lane computes its own initial registers
//     independently (see the (jsr, jcong) derivation from `l`).
//   - the cache-access and math-operation steps within each round only
//     ever touch mix[l][*] for the SAME l throughout a round - the
//     src/dst/sel/rng sequence is identical across lanes (derived once
//     per round from `base_rng`, no lane-index input) - so these need NO
//     cross-lane communication at all.
//   - item_index = mix[r % num_lanes][0] % num_items - this DOES need one
//     specific lane's register 0 (whichever lane equals r % 16),
//     broadcast to every other lane - the only per-round cross-lane read.
//   - the item-application step's per-lane byte offset
//     ((l ^ r) % num_lanes) * num_words_per_lane only depends on this
//     lane's own l and the current r - no cross-lane data needed as long
//     as every lane holds its own copy of the (identical, broadcast-
//     index) 256-byte item. Every lane fetches it independently from
//     full_dataset; redundant per-lane reads of the same address are
//     cheap on real hardware (a warp's identical addresses coalesce/hit
//     cache) and far simpler/lower-risk than trying to have each lane
//     fetch only its own slice and shuffle the rest around.
//   - only the FINAL reduction (folding all 16 lanes' lane_hash into one
//     8-word mix_hash) genuinely gathers data across all 16 lanes, and
//     only once per hash, not once per round.
//
// So exactly two points need real cross-lane communication: the
// per-round item_index broadcast (__shfl_sync) and the once-per-hash
// final reduction (also __shfl_sync-based, see below). Everything else
// stays fully lane-parallel - the same algorithm, same values, just
// spread across 16 cooperating threads instead of computed sequentially
// by one.
//
// IMPORTANT: __shfl_sync is device-only - there is no CPU equivalent to
// test this specific mechanism against before real hardware. Every other
// GPU-facing change in this project could be at least partially proven
// on CPU first (the algorithm itself, a small synthetic full_dataset,
// etc.); this cross-lane orchestration cannot be. Correctness here rests
// entirely on: (a) careful, direct correspondence to the already-proven
// sequential algorithm (documented step by step above and in the
// function below), and (b) a direct GPU-side comparison against the
// already-validated progpowz_full_kernel for real nonces, in
// gpu_backend_selftest - not assumed, not "should work", checked.
// ----------------------------------------------------------------------

// Executed by all num_lanes (16) threads of one lane group simultaneously
// - every thread must reach every __shfl_sync call in lockstep (that is a
// hard CUDA requirement, not a style preference: a __shfl_sync whose
// participating-thread set doesn't match `mask` is undefined behavior).
// Only the thread at `lane == 0` returns a result callers should use;
// every other lane's returned struct is well-formed but not meaningful.
//
// `mask`: the __shfl_sync warp mask - the caller always uses the full
// 0xFFFFFFFF warp mask here (see progpowz_warp_kernel below for why that
// is always safe with this launch's thread layout).
// `warp_lane`: this thread's index within the 32-thread WARP - what
// __shfl_sync's source-lane argument is relative to.
// `lane`: this thread's ProgPoW lane index within its own 16-thread group
// (0..15).
// `base_rng`/`dst_seq`/`src_seq`: mix_rng_state{block_number / period_length}
// and its derived sequences - lane-independent (depend only on
// block_number, a launch-wide constant; base_rng itself is never mutated
// after dst_seq/src_seq are derived from it - see progpowz_portable.hpp's
// matching comment - so it is just as safe to share as they are), so the
// caller computes all three ONCE per block into shared memory (see
// progpowz_warp_kernel) instead of every one of this kernel's 256 threads
// redundantly recomputing and privately storing the identical values in
// registers. Real ptxas register-usage data (see src/cuda/README.md's
// launch-parameter-tuning section) showed dst_seq/src_seq alone accounted
// for 64 of this kernel's 79 registers/thread - the single largest
// register-pressure contributor by far.
__device__ inline progpowz_result progpowz_hash_warp_full_dag(
    const uint32_t* __restrict__ l1_cache_words, uint32_t full_dataset_num_items,
    const hash2048* __restrict__ full_dataset, kiss99_state base_rng, const uint32_t* __restrict__ dst_seq,
    const uint32_t* __restrict__ src_seq, const hash256& header_hash, uint64_t nonce, unsigned mask,
    int warp_lane, int lane)
{
    const int group_base_lane = warp_lane - lane;  // this group's first warp-lane index (0 or 16)

    const uint64_t seed = keccak_progpow_64(header_hash, nonce);

    // init_mix, this lane's slice only - see block comment above: jsr/
    // jcong depend on `lane`, z/w do not, matching mix[lane][*]'s
    // derivation in progpowz_hash_light exactly.
    uint32_t r[num_regs];
    {
        const uint32_t z = fnv1a(fnv_offset_basis, (uint32_t)seed);
        const uint32_t w = fnv1a(z, (uint32_t)(seed >> 32));
        const uint32_t jsr = fnv1a(w, (uint32_t)lane);
        const uint32_t jcong = fnv1a(jsr, (uint32_t)lane);
        kiss99_state rng{z, w, jsr, jcong};
        for (uint32_t i = 0; i < num_regs; ++i)
            r[i] = kiss99_next(rng);
    }

    constexpr int max_operations = num_cache_accesses > num_math_operations ? num_cache_accesses : num_math_operations;
    constexpr size_t num_words_per_lane = 256 / (4 * num_lanes);

    for (uint32_t round = 0; round < 64; ++round)
    {
        kiss99_state rng = base_rng;
        size_t dst_counter = 0, src_counter = 0;

        // item_index needs lane (round % num_lanes)'s CURRENT register 0 -
        // every lane in the group calls this same __shfl_sync in lockstep,
        // each specifying the identical source lane, so every lane
        // receives the identical broadcast value.
        const uint32_t reg0_broadcast =
            __shfl_sync(mask, r[0], group_base_lane + static_cast<int>(round % num_lanes));
        const uint32_t num_items = full_dataset_num_items / 2;
        const uint32_t item_index = reg0_broadcast % num_items;

        // Each of the 16 lanes reads ONLY its own 16-byte (num_words_per_lane
        // = 4 words) slice of the 256-byte item - together the group covers
        // the full item with ZERO redundant global memory traffic (an
        // earlier version had every lane independently read the full 256
        // bytes, i.e. 16x more DAG traffic than necessary; measured ~35%
        // SLOWER than the plain non-cooperative kernel on real hardware as
        // a direct result - see src/cuda/README.md's account of that
        // regression). Thread `lane`'s own slice lives at byte offset
        // `lane * num_words_per_lane * 4` within the item - this is a
        // FIXED identity mapping, independent of `round`; which slice each
        // lane actually NEEDS this round ((lane ^ round) % num_lanes) is
        // obtained below via __shfl_sync from whichever lane's identity
        // slice matches, not by re-reading memory.
        const hash2048* item_ptr = &full_dataset[item_index];
        uint32_t my_slice[num_words_per_lane];
        {
            const uint32_t my_slice_byte_offset = static_cast<uint32_t>(lane) * num_words_per_lane * 4;
            for (size_t i = 0; i < num_words_per_lane; ++i)
                my_slice[i] = load_le32(item_ptr->bytes + my_slice_byte_offset + i * 4);
        }

        for (int i = 0; i < max_operations; ++i)
        {
            if (i < num_cache_accesses)
            {
                const uint32_t src = src_seq[(src_counter++) % num_regs];
                const uint32_t dst = dst_seq[(dst_counter++) % num_regs];
                const uint32_t sel = kiss99_next(rng);
                const size_t offset = r[src] % l1_cache_num_items;
                random_merge(r[dst], l1_cache_words[offset], sel);
            }
            if (i < num_math_operations)
            {
                const auto src_rnd = kiss99_next(rng) % (num_regs * (num_regs - 1));
                const auto src1 = src_rnd % num_regs;
                auto src2 = src_rnd / num_regs;
                if (src2 >= src1) ++src2;
                const auto sel1 = kiss99_next(rng);
                const auto dst = dst_seq[(dst_counter++) % num_regs];
                const auto sel2 = kiss99_next(rng);
                const uint32_t data = random_math(r[src1], r[src2], sel1);
                random_merge(r[dst], data, sel2);
            }
        }

        uint32_t dsts[num_words_per_lane], sels[num_words_per_lane];
        for (size_t i = 0; i < num_words_per_lane; ++i)
        {
            dsts[i] = i == 0 ? 0 : dst_seq[(dst_counter++) % num_regs];
            sels[i] = kiss99_next(rng);
        }
        // The slice this lane actually needs this round, obtained by
        // shuffling from whichever lane's identity slice matches - every
        // lane in the group calls __shfl_sync in lockstep (required, same
        // as the item_index broadcast and final reduction above/below),
        // each with its own `needed_slice` as the source lane.
        const int needed_slice = static_cast<int>((static_cast<uint32_t>(lane) ^ round) % num_lanes);
        for (size_t i = 0; i < num_words_per_lane; ++i)
        {
            const uint32_t word = __shfl_sync(mask, my_slice[i], group_base_lane + needed_slice);
            random_merge(r[dsts[i]], word, sels[i]);
        }
    }

    uint32_t lane_hash = fnv_offset_basis;
    for (uint32_t i = 0; i < num_regs; ++i)
        lane_hash = fnv1a(lane_hash, r[i]);

    // Reduction: every lane in the group calls __shfl_sync for every l in
    // 0..15 in lockstep (required - see this function's header comment),
    // redundantly computing the identical mh[8] fold; only lane 0's copy
    // is actually used below.
    uint32_t mh[8];
    for (auto& w : mh) w = fnv_offset_basis;
    for (int l = 0; l < static_cast<int>(num_lanes); ++l)
    {
        const uint32_t lh = __shfl_sync(mask, lane_hash, group_base_lane + l);
        mh[l % 8] = fnv1a(mh[l % 8], lh);
    }

    progpowz_result out{};
    if (lane == 0)
    {
        hash256 mix_hash{};
        for (int i = 0; i < 8; ++i)
            store_le32(mix_hash.bytes + i * 4, mh[i]);
        out.mix_hash = mix_hash;
        out.final_hash = keccak_progpow_256(header_hash, seed, mix_hash);
    }
    return out;
}

// One 16-thread lane group computes one nonce's hash. Every launched
// thread executes progpowz_hash_warp_full_dag() in full, even threads
// whose nonce_group falls beyond `count` (padding to keep every launched
// warp fully, uniformly populated - required for __shfl_sync's lockstep
// requirement; a thread that returned early would desynchronize its
// warp-mates still trying to shuffle with it). Only the final write is
// guarded. blockDim.x must be a multiple of 32 (always true here: this
// project always launches 256) so warp-lane and lane-group boundaries
// never split a warp, which is what makes the full 0xFFFFFFFF mask always
// safe to use unconditionally.
__global__ void progpowz_warp_kernel(
    const uint32_t* __restrict__ l1_cache_words, uint32_t full_dataset_num_items,
    const hash2048* __restrict__ full_dataset, int block_number, hash256 header_hash, uint64_t start_nonce,
    uint32_t count, GpuHashResult* __restrict__ out)
{
    // __restrict__: l1_cache_words, full_dataset, and out are genuinely
    // non-overlapping buffers (never aliased) - this promise lets the
    // compiler use the read-only data cache path for the two read-only
    // pointers and reorder/cache loads more aggressively than it could
    // if it had to assume any of these might alias.
    __shared__ uint32_t s_l1_cache[l1_cache_num_items];
    for (uint32_t i = threadIdx.x; i < l1_cache_num_items; i += blockDim.x)
        s_l1_cache[i] = l1_cache_words[i];

    // mix_rng_state{block_number / period_length}'s dst_seq/src_seq are
    // identical for every thread in this launch (block_number is the same
    // for the whole batch) - computed ONCE per block by thread 0 into
    // shared memory rather than redundantly by all 256 threads in private
    // registers. See progpowz_hash_warp_full_dag's header comment for the
    // real register-usage data motivating this.
    __shared__ uint32_t s_dst_seq[num_regs], s_src_seq[num_regs];
    __shared__ kiss99_state s_base_rng;
    if (threadIdx.x == 0)
    {
        const uint64_t s = (uint64_t)(block_number / period_length);
        const uint32_t seed_lo = (uint32_t)s, seed_hi = (uint32_t)(s >> 32);
        const auto z = fnv1a(fnv_offset_basis, seed_lo);
        const auto w = fnv1a(z, seed_hi);
        const auto jsr = fnv1a(w, seed_lo);
        const auto jcong = fnv1a(jsr, seed_hi);
        s_base_rng = kiss99_state{z, w, jsr, jcong};
        init_dst_src_seq(s_base_rng, s_dst_seq, s_src_seq);
    }
    __syncthreads();

    const uint32_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nonce_group = global_thread_idx / num_lanes;
    const int lane = static_cast<int>(global_thread_idx % num_lanes);
    const int warp_lane = static_cast<int>(threadIdx.x % 32);

    const uint64_t nonce = start_nonce + nonce_group;
    constexpr unsigned kFullWarpMask = 0xFFFFFFFFu;

    progpowz_result r = progpowz_hash_warp_full_dag(
        s_l1_cache, full_dataset_num_items, full_dataset, s_base_rng, s_dst_seq, s_src_seq, header_hash,
        nonce, kFullWarpMask, warp_lane, lane);

    if (lane == 0 && nonce_group < count)
    {
        out[nonce_group].final_hash = r.final_hash;
        out[nonce_group].mix_hash = r.mix_hash;
        out[nonce_group].nonce = nonce;
    }
}

// Launches progpowz_warp_kernel against an already-built DeviceFullDataset
// - same reused-output-buffer pattern as PersistentFullDagSearcher, but
// sized/launched in units of num_lanes threads per nonce (blockDim.x=256
// = 16 complete 16-thread lane groups = 16 nonces per block, and always a
// multiple of 32 - see progpowz_warp_kernel's header comment for why that
// matters).
// Threads per block for progpowz_warp_kernel launches. Defaults to 256
// (16 lane-groups of 16 threads each, always warp-aligned - required,
// see progpowz_warp_kernel's header comment), but can be overridden via
// the DEEPCORE_WARP_THREADS_PER_BLOCK environment variable for empirical
// launch-parameter tuning without a rebuild (this algorithm's real
// register pressure - 32 ProgPoW registers plus src/dst sequences per
// lane - means the right occupancy/block-size tradeoff isn't obvious
// from theory alone on a register-heavy, memory-latency-bound kernel
// like this one; measuring real hashrate at a few different values is
// the only way to actually know). Silently ignores an invalid value
// (not a positive multiple of 32, the hard warp-alignment requirement)
// and falls back to the default rather than launching something unsafe.
inline uint32_t warp_kernel_threads_per_block()
{
    static const uint32_t value = [] {
        const uint32_t default_value = 256;
        const char* env = std::getenv("DEEPCORE_WARP_THREADS_PER_BLOCK");
        if (!env)
            return default_value;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || parsed <= 0 || parsed % 32 != 0)
        {
            std::fprintf(stderr,
                "warning: DEEPCORE_WARP_THREADS_PER_BLOCK=\"%s\" is invalid (must be a positive multiple "
                "of 32) - using default %u\n",
                env, default_value);
            return default_value;
        }
        return static_cast<uint32_t>(parsed);
    }();
    return value;
}

class PersistentWarpSearcher {
public:
    PersistentWarpSearcher() = default;
    ~PersistentWarpSearcher() { if (d_out_) cudaFree(d_out_); }
    PersistentWarpSearcher(const PersistentWarpSearcher&) = delete;
    PersistentWarpSearcher& operator=(const PersistentWarpSearcher&) = delete;

    std::vector<GpuHashResult> search(DeviceFullDataset& dataset, const uint32_t* device_l1_cache_words,
        uint32_t full_dataset_num_items, int block_number, const hash256& header_hash, uint64_t start_nonce,
        uint32_t count)
    {
        ensure_out_capacity(count);

        const uint32_t threads_per_block = warp_kernel_threads_per_block();
        const uint32_t total_threads = count * static_cast<uint32_t>(num_lanes);
        const uint32_t blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        progpowz_warp_kernel<<<blocks, threads_per_block>>>(device_l1_cache_words, full_dataset_num_items,
            dataset.device_dataset(), block_number, header_hash, start_nonce, count, d_out_);

        DEEPCORE_CUDA_CHECK(cudaGetLastError());
        DEEPCORE_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<GpuHashResult> results(count);
        DEEPCORE_CUDA_CHECK(cudaMemcpy(results.data(), d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult),
            cudaMemcpyDeviceToHost));
        return results;
    }

private:
    void ensure_out_capacity(uint32_t count)
    {
        if (count <= d_out_capacity_)
            return;
        if (d_out_)
            cudaFree(d_out_);
        DEEPCORE_CUDA_CHECK(cudaMalloc(&d_out_, static_cast<size_t>(count) * sizeof(GpuHashResult)));
        d_out_capacity_ = count;
    }

    GpuHashResult* d_out_ = nullptr;
    uint32_t d_out_capacity_ = 0;
};

#undef DEEPCORE_CUDA_CHECK

}  // namespace deepcore::progpowz
