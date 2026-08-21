#pragma once

// deepcore::progpowz::NvrtcWarpSearcher - drives the per-period,
// NVRTC-compiled ProgPowZ warp kernel (see progpowz_codegen.hpp's
// generate_progpowz_nvrtc_kernel_source() for what gets compiled, and
// that file's header comment for WHY: real production ProgPoW miners
// resolve the entire 64-round mix program once per period_length blocks
// and compile THAT fixed program, instead of interpreting
// dst_seq/src_seq/kiss99 on every single hash the way
// progpowz_kernel.cu's progpowz_warp_kernel does).
//
// STATUS: written, NOT YET COMPILED OR RUN. This is the first code in
// this project to use the CUDA Driver API (cuModuleLoadDataEx,
// cuLaunchKernel, ...) and NVRTC (nvrtcCreateProgram,
// nvrtcCompileProgram, ...) - every other GPU-facing file uses only the
// CUDA Runtime API. No CUDA toolkit is available in the environment this
// was written in (not even headers - see src/cuda/README.md), so this
// could not be compiled here even for a syntax check, let alone run.
// Higher first-draft risk than any prior GPU change in this project for
// that reason: the underlying algorithm (via progpowz_codegen.hpp) is
// already CPU-validated (see tools/codegen_selftest), but the NVRTC
// compile/load/launch plumbing itself is unverified. See
// tools/nvrtc_kernel_selftest for the real-hardware validation gate this
// needs before being trusted, and be ready for at least one real-hardware
// debug round-trip - unlike prior kernel milestones, this genuinely
// exercises new-to-this-project API surface, not just new kernel logic
// on already-proven plumbing.
//
// Runtime/Driver API interop: this class uses the Driver API
// (cuModuleLoadDataEx/cuLaunchKernel) for the JIT-compiled kernel, while
// the rest of the project (DeviceEpochCache, DeviceFullDataset, etc. in
// progpowz_kernel.cu) uses the Runtime API (cudaMalloc/cudaMemcpy/...)
// for buffer management - this mixing is standard, documented CUDA usage
// (the Runtime API's implicit per-device "primary context" is exactly
// what cuDevicePrimaryCtxRetain below attaches the Driver API calls to),
// not a new or risky pattern in itself.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "progpowz_portable.hpp"

namespace deepcore::progpowz {

class NvrtcWarpSearcher
{
public:
    explicit NvrtcWarpSearcher(int device_index = 0);
    ~NvrtcWarpSearcher();

    NvrtcWarpSearcher(const NvrtcWarpSearcher&) = delete;
    NvrtcWarpSearcher& operator=(const NvrtcWarpSearcher&) = delete;

    // Compiles (or reuses an already-compiled, cached) kernel for
    // `block_number`'s period, then launches it against the given
    // already-VRAM-resident full dataset and l1_cache. `full_dataset` and
    // `l1_cache_words` are DEVICE pointers (already uploaded - see
    // DeviceFullDataset/DeviceEpochCache in progpowz_kernel.cu; this
    // class does not manage that upload itself, matching
    // PersistentWarpSearcher's own division of responsibility).
    //
    // Returns false (with `error` set) if compilation or launch failed -
    // the caller must not trust `out_results` in that case. Compilation
    // only happens when the period changes from the previous call (or on
    // the first call); repeated calls within the same period reuse the
    // cached compiled kernel, matching real miners' amortize-compile-
    // cost-over-a-whole-period design.
    bool search(const void* device_l1_cache_words, uint32_t full_dataset_num_items, const void* device_full_dataset,
        int block_number, const hash256& header_hash, uint64_t start_nonce, uint32_t count,
        std::vector<GpuHashResult>& out_results, std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace deepcore::progpowz
