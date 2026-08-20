#pragma once

// deepcore::mining::GpuHashSearchBackend - a GPU-backed IHashSearchBackend
// (see src/mining/mining_loop.hpp) wrapping the already real-hardware-
// validated progpowz_light_kernel (progpowz_kernel.cu; see
// src/cuda/README.md's "GPU milestone 1" account - all 4 recorded test
// vectors passed byte-for-byte on a real Quadro GV100).
//
// SCOPE OF THIS MILESTONE: correctness/integration only, matching the
// kernel wrapper it calls (run_progpowz_light_gpu) - light-cache mode
// (each GPU thread recomputes dataset items on demand), and NO persistent
// device memory: the epoch's light_cache and l1_cache are re-uploaded to
// the GPU on every search() call, exactly as gpu_selftest.cu already
// validated on real hardware. This is real, working GPU compute, not a
// stub - but it is not yet competitive throughput. VRAM-resident caching
// (upload once per epoch, not once per batch) and a full-DAG kernel are
// separate, later performance milestones - see src/cuda/README.md. Do not
// treat this class as "the fast GPU backend"; it is "the first GPU backend
// proven correct end-to-end through MiningLoop."
//
// Coarser cancellation than CpuHashSearchBackend: a CUDA kernel launch
// cannot be interrupted mid-flight, so `cancelled` is only checked once,
// before launching a batch - not per-nonce. A batch already in flight when
// a new job supersedes the current one will run to completion before the
// next job's work begins. If that batch happens to contain a winning
// nonce, MiningLoop will still submit it; this is not a new correctness
// problem - it lands on the same "share for a superseded job" path that
// ZanoStratumClient already handles (see src/network/README.md: only the
// current job's header hash is retained, so a stale submission is
// recognized and dropped/counted, not silently miscounted as valid).
// Keeping the batch size (preferred_batch_size()) modest keeps this
// window short.

#include <cstdint>

#include "../mining/mining_loop.hpp"

namespace deepcore::mining {

class GpuHashSearchBackend final : public IHashSearchBackend {
public:
    // `device_index` selects which CUDA device to run on (as would be
    // passed to cudaSetDevice). Multi-GPU orchestration (one backend
    // instance per device feeding one MiningLoop worker thread each, or
    // per-device MiningLoop instances) is not implemented anywhere yet -
    // see src/hardware/gpu_manager.hpp, still just an interface. This
    // class only drives a single device.
    explicit GpuHashSearchBackend(int device_index = 0);

    std::optional<FoundShare> search(const ProgPowZJob& job, const ethash::epoch_context& ctx,
        std::uint64_t start_nonce, std::uint64_t count, const std::atomic<bool>& cancelled) override;

    [[nodiscard]] std::uint64_t preferred_batch_size() const override;

private:
    int device_index_;
};

}  // namespace deepcore::mining
