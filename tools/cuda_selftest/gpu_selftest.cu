// GPU correctness self-test for progpowz_kernel.cu / progpowz_portable.hpp.
//
// THIS IS THE ACTUAL VALIDATION GATE for GPU milestone 1 (see
// src/cuda/README.md). Everything up to this point (progpowz_portable.hpp
// passing tools/cuda_selftest/cuda_selftest.cpp) only proves the algorithm
// is correct as ordinary CPU code - it says nothing about whether it
// executes correctly as compiled CUDA device code on a real GPU. This
// program is what actually answers that question. It must be built with
// nvcc and run on real hardware (this project's target: Tesla V100-SXM2
// [Volta, sm_70] and Tesla A100 [Ampere, sm_80]) - it has not been done in
// the environment that wrote this file, which has neither the CUDA
// Toolkit nor an NVIDIA GPU.
//
// Method: build a real epoch-0 context via the unmodified reference
// library (same as tools/cuda_selftest/cuda_selftest.cpp), upload it to the
// GPU, run the same 4 fixed test-vector nonces through
// progpowz_light_kernel, copy results back, and compare byte-for-byte
// against the reference implementation's own CPU output.
#include <cstdio>
#include <cstring>

#include <ethash/progpow.hpp>

#include "../../src/cuda/progpowz_kernel.cu"

namespace {

int g_failures = 0;

deepcore::progpowz::hash256 to_portable(const ethash::hash256& h)
{
    deepcore::progpowz::hash256 out;
    std::memcpy(out.bytes, h.bytes, 32);
    return out;
}

void print_hex(const char* label, const uint8_t* bytes, size_t n)
{
    std::printf("%s: 0x", label);
    for (size_t i = 0; i < n; ++i)
        std::printf("%02x", bytes[i]);
    std::printf("\n");
}

void run_case(const ethash::epoch_context& ctx, int block_number,
    const ethash::hash256& header_hash, uint64_t nonce)
{
    std::printf("--- block_number=%d nonce=0x%016llx ---\n", block_number,
        static_cast<unsigned long long>(nonce));

    auto ref = progpow::hash(ctx, block_number, header_hash, nonce);

    auto gpu_results = deepcore::progpowz::run_progpowz_light_gpu(
        reinterpret_cast<const deepcore::progpowz::hash512*>(ctx.light_cache),
        ctx.light_cache_num_items, ctx.l1_cache,
        static_cast<uint32_t>(ctx.full_dataset_num_items), block_number,
        to_portable(header_hash), nonce, /*count=*/1);

    const auto& gpu = gpu_results[0];

    bool final_ok = std::memcmp(ref.final_hash.bytes, gpu.final_hash.bytes, 32) == 0;
    bool mix_ok = std::memcmp(ref.mix_hash.bytes, gpu.mix_hash.bytes, 32) == 0;

    print_hex("reference (CPU) final_hash", ref.final_hash.bytes, 32);
    print_hex("GPU            final_hash", gpu.final_hash.bytes, 32);
    print_hex("reference (CPU) mix_hash  ", ref.mix_hash.bytes, 32);
    print_hex("GPU            mix_hash  ", gpu.mix_hash.bytes, 32);

    if (final_ok && mix_ok)
    {
        std::printf("RESULT: PASS\n\n");
    }
    else
    {
        std::printf("RESULT: FAIL (final_hash %s, mix_hash %s)\n\n",
            final_ok ? "match" : "MISMATCH", mix_ok ? "match" : "MISMATCH");
        ++g_failures;
    }
}

}  // namespace

int main()
{
    auto ctx = ethash::create_epoch_context(0);
    if (!ctx)
    {
        std::fprintf(stderr, "FATAL: create_epoch_context(0) failed\n");
        return 1;
    }
    std::printf("epoch 0: light_cache_num_items=%d full_dataset_num_items=%d\n\n",
        ctx->light_cache_num_items, ctx->full_dataset_num_items);

    ethash::hash256 zero_hash{};
    std::memset(&zero_hash, 0, sizeof(zero_hash));

    ethash::hash256 pattern_hash{};
    for (int i = 0; i < 32; ++i)
        pattern_hash.bytes[i] = static_cast<uint8_t>(i);

    run_case(*ctx, 0, zero_hash, 0);
    run_case(*ctx, 0, zero_hash, 1);
    run_case(*ctx, 0, pattern_hash, 0x123456789abcdef0ULL);
    run_case(*ctx, 12345, pattern_hash, 0x123456789abcdef0ULL);

    if (g_failures == 0)
    {
        std::printf("ALL VECTORS PASSED ON GPU: progpowz_kernel.cu matches the CPU "
                     "reference implementation exactly.\n");
        return 0;
    }
    std::printf("%d VECTOR(S) FAILED ON GPU - do not trust progpowz_kernel.cu until "
                "fixed.\n", g_failures);
    return 1;
}
