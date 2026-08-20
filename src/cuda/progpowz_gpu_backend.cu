// Implementation of deepcore::mining::GpuHashSearchBackend - see
// progpowz_gpu_backend.hpp for scope/status. Built with nvcc only (this
// file uses CUDA runtime calls); never compiled into a target unless
// DEEPCORE_WITH_CUDA is on and a CUDA compiler was actually found - see
// CMakeLists.txt.
//
// STATUS: written but NOT YET VALIDATED on real hardware - unlike
// progpowz_kernel.cu (validated on a Quadro GV100, see src/cuda/README.md),
// this integration layer has not yet been built/run for real. Do not treat
// it as working until tools/gpu_backend_selftest has actually been built
// with nvcc and run on real hardware and reported PASS.

#include "progpowz_gpu_backend.hpp"

#include <cuda_runtime.h>

// Reuses GpuHashResult / run_progpowz_light_gpu unmodified - same
// #include-the-.cu-directly pattern tools/cuda_selftest/gpu_selftest.cu
// already uses, so there is exactly one implementation of the kernel
// wrapper, not a second copy.
#include "progpowz_kernel.cu"

namespace deepcore::mining {

namespace {
// Chosen to amortize the light_cache/l1_cache re-upload (the dominant
// per-call cost in this milestone - see header comment) across enough
// GPU work to be worthwhile, while keeping the cancellation-latency
// window (see header comment) bounded to a fraction of a second even on
// modest hardware. Not yet tuned against a real measured hashrate -
// revisit once this backend is validated and its actual throughput is
// known.
constexpr std::uint64_t kPreferredBatchSize = 65536;
}  // namespace

GpuHashSearchBackend::GpuHashSearchBackend(int device_index) : device_index_(device_index) {}

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

    auto results = progpowz::run_progpowz_light_gpu(
        reinterpret_cast<const progpowz::hash512*>(ctx.light_cache), ctx.light_cache_num_items,
        ctx.l1_cache, static_cast<std::uint32_t>(ctx.full_dataset_num_items), job.block_number,
        job.pow_hash, start_nonce, static_cast<std::uint32_t>(count));

    for (const auto& r : results)
    {
        if (hash_meets_target(r.final_hash, job.target_boundary))
            return FoundShare{r.nonce, r.mix_hash};
    }
    return std::nullopt;
}

}  // namespace deepcore::mining
