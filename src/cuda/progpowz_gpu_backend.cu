// Implementation of deepcore::mining::GpuHashSearchBackend - see
// progpowz_gpu_backend.hpp for scope/status. Built with nvcc only (this
// file uses CUDA runtime calls); never compiled into a target unless
// DEEPCORE_WITH_CUDA is on and a CUDA compiler was actually found - see
// CMakeLists.txt.
//
// STATUS: the light-cache-mode version of this backend (persistent VRAM
// caching, no full DAG) was validated on a real Quadro GV100 - see
// src/cuda/README.md. This version adds the full-DAG kernel on top - the
// dataset-generation and full-DAG mining kernels are new, unvalidated
// surface (their mechanism was proven correct on CPU with a small
// synthetic dataset in tools/cuda_selftest/cuda_selftest.cpp, but
// real-scale, real-hardware validation has NOT been done yet). Do not
// treat this as working until tools/gpu_backend_selftest has been
// rebuilt and rerun on real hardware and reported PASS.

#include "progpowz_gpu_backend.hpp"

#include <cstdio>

#include <cuda_runtime.h>

// Reuses GpuHashResult / DeviceEpochCache / PersistentGpuSearcher /
// DeviceFullDataset / PersistentFullDagSearcher unmodified - same
// #include-the-.cu-directly pattern tools/cuda_selftest/gpu_selftest.cu
// uses for run_progpowz_light_gpu, so there is exactly one implementation
// of each, not a second copy.
#include "progpowz_kernel.cu"

namespace deepcore::mining {

namespace {
// See progpowz_gpu_backend.cu's previous (light-only) revision for why
// this value was chosen; unchanged here - not yet retuned for full-DAG
// mode's real (currently unmeasured) throughput.
constexpr std::uint64_t kPreferredBatchSize = 65536;
}  // namespace

struct GpuHashSearchBackend::Impl {
    // l1_cache (needed in BOTH modes - the mix loop's l1_cache lookups are
    // unrelated to whether dataset items are recomputed or read from a
    // precomputed array) and, harmlessly unused in full-DAG mode, the
    // light_cache itself (DeviceEpochCache uploads both together; the
    // light_cache upload is comparatively tiny next to a multi-GB full
    // dataset, not worth a separate class to avoid).
    progpowz::DeviceEpochCache epoch_cache;
    progpowz::PersistentGpuSearcher light_searcher;

    progpowz::DeviceFullDataset full_dataset;
    progpowz::PersistentFullDagSearcher full_searcher;

    // Deduplicates the fallback warning so it prints once per epoch that
    // doesn't fit, not once per search() call.
    const void* last_fallback_warned_identity = nullptr;

    bool last_used_full_dag = false;
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

bool GpuHashSearchBackend::last_search_used_full_dag() const
{
    return impl_->last_used_full_dag;
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
    const auto full_dataset_num_items = static_cast<std::uint32_t>(ctx.full_dataset_num_items);

    impl_->epoch_cache.ensure_uploaded(
        host_light_cache, ctx.light_cache_num_items, ctx.l1_cache, progpowz::l1_cache_num_items);

    std::vector<progpowz::GpuHashResult> results;

    std::string dag_error;
    if (impl_->full_dataset.ensure_built(
            host_light_cache, ctx.light_cache_num_items, full_dataset_num_items, dag_error))
    {
        impl_->last_used_full_dag = true;
        results = impl_->full_searcher.search(impl_->full_dataset, impl_->epoch_cache.device_l1_cache(),
            full_dataset_num_items, job.block_number, job.pow_hash, start_nonce,
            static_cast<std::uint32_t>(count));
    }
    else
    {
        impl_->last_used_full_dag = false;
        if (impl_->last_fallback_warned_identity != host_light_cache)
        {
            std::fprintf(stderr,
                "GpuHashSearchBackend: full-DAG mode unavailable for this epoch (%s) - "
                "falling back to light-cache mode (correct, but far slower)\n",
                dag_error.c_str());
            impl_->last_fallback_warned_identity = host_light_cache;
        }

        results = impl_->light_searcher.search(impl_->epoch_cache, ctx.light_cache_num_items,
            full_dataset_num_items, job.block_number, job.pow_hash, start_nonce,
            static_cast<std::uint32_t>(count));
    }

    for (const auto& r : results)
    {
        if (hash_meets_target(r.final_hash, job.target_boundary))
            return FoundShare{r.nonce, r.mix_hash};
    }
    return std::nullopt;
}

}  // namespace deepcore::mining
