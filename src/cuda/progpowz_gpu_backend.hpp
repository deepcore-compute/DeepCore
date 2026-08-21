#pragma once

// deepcore::mining::GpuHashSearchBackend - a GPU-backed IHashSearchBackend
// (see src/mining/mining_loop.hpp) wrapping the already real-hardware-
// validated progpowz_light_kernel (progpowz_kernel.cu; see
// src/cuda/README.md's "GPU milestone 1" account - all 4 recorded test
// vectors passed byte-for-byte on a real Quadro GV100).
//
// SCOPE: uses the lane-cooperative warp-shuffle full-DAG kernel
// (progpowz_warp_kernel / PersistentWarpSearcher in progpowz_kernel.cu,
// 16 threads cooperating per hash via __shfl_sync) when there is enough
// free VRAM to build the epoch's complete dataset once (checked for real
// via cudaMemGetInfo, never assumed); falls back to light-cache mode
// (recompute each dataset item on demand, one thread per hash) otherwise,
// with a clearly printed reason - this is a real capacity constraint on
// some hardware, not something to silently degrade past. Real, measured
// progress against a competing miner (Rigel, ~38 MH/s on a Quadro
// GV100): light mode alone (~13 KH/s) was about 3000x behind; the plain
// (non-cooperative) full-DAG kernel measured ~524 KH/s (~40x faster,
// ~72x behind); the warp-shuffle kernel first measured ~642 KH/s (~59x
// behind) but that build was later found to be compiling for the wrong
// GPU architecture (sm_52 instead of sm_70 - a real CMakeLists.txt bug,
// see src/cuda/README.md's "Next steps" step 9) - after fixing it and
// rebuilding for genuine sm_70/sm_80, the SAME kernel (identical register
// usage) measures ~1.03 MH/s (~37x behind Rigel) - see
// src/network/README.md's cross-check account and src/cuda/README.md for
// the full history, including a real regression this milestone measured
// and fixed along the way.
//
// Both the epoch's l1_cache (DeviceEpochCache) and, when in full-DAG
// mode, the full dataset (DeviceFullDataset) are kept VRAM-resident
// across search() calls rather than re-uploaded/rebuilt every batch.
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
#include <memory>

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
    ~GpuHashSearchBackend() override;

    GpuHashSearchBackend(const GpuHashSearchBackend&) = delete;
    GpuHashSearchBackend& operator=(const GpuHashSearchBackend&) = delete;

    std::optional<FoundShare> search(const ProgPowZJob& job, const ethash::epoch_context& ctx,
        std::uint64_t start_nonce, std::uint64_t count, const std::atomic<bool>& cancelled) override;

    [[nodiscard]] std::uint64_t preferred_batch_size() const override;

    // True iff the most recent search() call used the full-DAG kernel
    // rather than falling back to light-cache mode. Exists mainly so
    // gpu_backend_selftest can assert full-DAG mode was actually
    // exercised, not just that *some* correct backend happened to answer
    // (light mode is also correct - passing tests alone don't prove which
    // path ran). Undefined (returns false) before the first search() call.
    [[nodiscard]] bool last_search_used_full_dag() const;

private:
    int device_index_;

    // Hides the CUDA-specific persistent device state (DeviceEpochCache,
    // PersistentGpuSearcher - see progpowz_kernel.cu) behind a pImpl so
    // this header stays plain C++ and can be included from non-CUDA
    // translation units (e.g. deepcore_miner_main.cpp, which is compiled
    // by the regular C++ compiler even in a CUDA-enabled build - only
    // progpowz_gpu_backend.cu itself is compiled by nvcc).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace deepcore::mining
