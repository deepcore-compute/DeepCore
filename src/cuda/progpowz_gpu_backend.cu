// Implementation of deepcore::mining::GpuHashSearchBackend - see
// progpowz_gpu_backend.hpp for scope/status. Built with nvcc only (this
// file uses CUDA runtime calls); never compiled into a target unless
// DEEPCORE_WITH_CUDA is on and a CUDA compiler was actually found - see
// CMakeLists.txt.
//
// STATUS: the correctness-only version of this backend (always re-upload,
// no persistent device memory) was validated on a real Quadro GV100 - see
// src/cuda/README.md and src/network/README.md (real LuckyPool/Rigel
// cross-check). This version adds persistent VRAM caching (DeviceEpochCache
// / PersistentGpuSearcher in progpowz_kernel.cu) on top of that - the
// per-hash algorithm and kernel are completely unchanged, only the device
// memory lifecycle differs, but the change itself has NOT yet been run on
// real hardware. Do not treat it as validated until
// tools/gpu_backend_selftest has been rebuilt and rerun on real hardware
// and reported PASS.

#include "progpowz_gpu_backend.hpp"

#include <cuda_runtime.h>

// Reuses GpuHashResult / DeviceEpochCache / PersistentGpuSearcher
// unmodified - same #include-the-.cu-directly pattern
// tools/cuda_selftest/gpu_selftest.cu uses for run_progpowz_light_gpu, so
// there is exactly one implementation of each, not a second copy.
#include "progpowz_kernel.cu"

namespace deepcore::mining {

namespace {
// Chosen to amortize per-call overhead (kernel launch, and - before this
// change - the light_cache/l1_cache re-upload, now eliminated on repeat
// calls for the same epoch) across enough GPU work to be worthwhile,
// while keeping the cancellation-latency window (see header comment)
// bounded to a fraction of a second even on modest hardware. Not yet
// tuned against a real measured hashrate - revisit once this backend's
// real throughput is known (see src/cuda/README.md's roadmap: this is
// still light-cache mode, not yet the full-DAG kernel).
constexpr std::uint64_t kPreferredBatchSize = 65536;
}  // namespace

struct GpuHashSearchBackend::Impl {
    progpowz::DeviceEpochCache cache;
    progpowz::PersistentGpuSearcher searcher;
};

GpuHashSearchBackend::GpuHashSearchBackend(int device_index)
    : device_index_(device_index), impl_(std::make_unique<Impl>())
{
}

GpuHashSearchBackend::~GpuHashSearchBackend() = default;

std::uint64_t GpuHashSearchBackend::preferred_batch_size() const
{
    return kPreferredBatchSize;
}

std::optional<FoundShare> GpuHashSearchBackend::search(const ProgPowZJob& job,
    const ethash::epoch_context& ctx, std::uint64_t start_nonce, std::uint64_t count,
    const std::atomic<bool>& cancelled)
{
    // Only checked once per batch, not per-nonce - see header comment on
    // cancellation granularity.
    if (cancelled.load(std::memory_order_relaxed))
        return std::nullopt;

    cudaSetDevice(device_index_);

    const auto* host_light_cache = reinterpret_cast<const progpowz::hash512*>(ctx.light_cache);
    impl_->cache.ensure_uploaded(
        host_light_cache, ctx.light_cache_num_items, ctx.l1_cache, progpowz::l1_cache_num_items);

    auto results = impl_->searcher.search(impl_->cache, ctx.light_cache_num_items,
        static_cast<std::uint32_t>(ctx.full_dataset_num_items), job.block_number, job.pow_hash,
        start_nonce, static_cast<std::uint32_t>(count));

    for (const auto& r : results)
    {
        if (hash_meets_target(r.final_hash, job.target_boundary))
            return FoundShare{r.nonce, r.mix_hash};
    }
    return std::nullopt;
}

}  // namespace deepcore::mining
