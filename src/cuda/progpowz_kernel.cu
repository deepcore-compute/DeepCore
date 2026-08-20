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

#undef DEEPCORE_CUDA_CHECK

}  // namespace deepcore::progpowz
