// Implementation of deepcore::mining::GpuHashSearchBackend - see
// progpowz_gpu_backend.hpp for scope/status. Built with nvcc only (this
// file uses CUDA runtime calls); never compiled into a target unless
// DEEPCORE_WITH_CUDA is on and a CUDA compiler was actually found - see
// CMakeLists.txt.
//
// STATUS: light-cache mode, plain full-DAG mode, the lane-cooperative
// warp-shuffle full-DAG kernel, and now the per-period NVRTC-compiled
// kernel (NvrtcWarpSearcher - see progpowz_nvrtc_kernel.hpp/.cpp and
// src/cuda/README.md's step 10) have all been validated on a real Quadro
// GV100 - deepcore-nvrtc-kernel-selftest passed 18/18 checks on the FIRST
// real-hardware attempt. This revision makes the NVRTC path the primary
// full-DAG search path (real production miners' actual technique - see
// that README section for the sourced research), falling back to the
// interpreted warp kernel (PersistentWarpSearcher, itself already
// real-hardware-validated) if the NVRTC path fails at runtime for any
// reason (e.g. a compile error on a period this project's own testing
// didn't happen to exercise) - mining should degrade to a slower but
// still-correct path, not stall, on an unexpected NVRTC failure. Not yet
// re-measured for real hashrate through this backend/the CLI
// specifically - do not treat the ~1.03 MH/s figure recorded for the
// interpreted warp kernel as this path's number until measured.

#include "progpowz_gpu_backend.hpp"

#include <cstdio>

#include <cuda_runtime.h>

// Reuses GpuHashResult / DeviceEpochCache / PersistentGpuSearcher /
// DeviceFullDataset / PersistentFullDagSearcher unmodified - same
// #include-the-.cu-directly pattern tools/cuda_selftest/gpu_selftest.cu
// uses for run_progpowz_light_gpu, so there is exactly one implementation
// of each, not a second copy.
#include "progpowz_kernel.cu"

#include "progpowz_nvrtc_kernel.hpp"

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
    // Primary full-DAG path: per-period NVRTC-compiled kernel (real
    // production miners' technique - see this file's header comment).
    progpowz::NvrtcWarpSearcher nvrtc_warp_searcher;
    // Fallback if the NVRTC path fails at runtime, and the cross-check
    // baseline for tools/nvrtc_kernel_selftest and
    // tools/warp_kernel_selftest - itself already real-hardware-validated
    // (interpreted per-hash, no NVRTC/Driver-API dependency to fail).
    progpowz::PersistentWarpSearcher warp_searcher;

    // Deduplicates the fallback warnings so each prints once per
    // condition, not once per search() call.
    const void* last_fallback_warned_identity = nullptr;
    bool last_nvrtc_failure_warned = false;

    bool last_used_full_dag = false;
    bool last_used_nvrtc = false;
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

bool GpuHashSearchBackend::last_search_used_nvrtc() const
{
    return impl_->last_used_nvrtc;
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

        std::string nvrtc_error;
        const bool nvrtc_ok = impl_->nvrtc_warp_searcher.search(impl_->epoch_cache.device_l1_cache(),
            full_dataset_num_items, impl_->full_dataset.device_dataset(), job.block_number, job.pow_hash,
            start_nonce, static_cast<std::uint32_t>(count), results, nvrtc_error);

        if (nvrtc_ok)
        {
            impl_->last_used_nvrtc = true;
            impl_->last_nvrtc_failure_warned = false;
        }
        else
        {
            impl_->last_used_nvrtc = false;
            if (!impl_->last_nvrtc_failure_warned)
            {
                std::fprintf(stderr,
                    "GpuHashSearchBackend: NVRTC kernel path failed (%s) - falling back to the "
                    "interpreted warp kernel (correct, but slower - see src/cuda/README.md)\n",
                    nvrtc_error.c_str());
                impl_->last_nvrtc_failure_warned = true;
            }
            results = impl_->warp_searcher.search(impl_->full_dataset, impl_->epoch_cache.device_l1_cache(),
                full_dataset_num_items, job.block_number, job.pow_hash, start_nonce,
                static_cast<std::uint32_t>(count));
        }
    }
    else
    {
        impl_->last_used_full_dag = false;
        impl_->last_used_nvrtc = false;
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
