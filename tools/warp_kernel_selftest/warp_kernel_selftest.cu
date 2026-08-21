// Correctness gate for progpowz_warp_kernel (the lane-cooperative,
// warp-shuffle full-DAG kernel - see its own header comment in
// progpowz_kernel.cu for the full design rationale).
//
// THIS IS THE VALIDATION GATE for GPU throughput milestone 3. Unlike
// every earlier kernel in this project, __shfl_sync has no CPU
// equivalent at all - there is no way to pre-check this specific
// mechanism before real hardware. Correctness here rests entirely on
// this test: comparing progpowz_warp_kernel's output against the
// already-validated progpowz_full_kernel (GPU throughput milestone 2,
// proven correct against the CPU reference and against a live pool) for
// real nonces, across batch sizes deliberately chosen to exercise
// multiple lane groups, multiple blocks, and non-block-aligned batch
// sizes (the padding/tail-handling logic).
//
// Must be built with nvcc and run on real hardware; it has not been done
// in the environment that wrote this file (no CUDA Toolkit, no GPU).
// GpuHashSearchBackend does NOT use progpowz_warp_kernel yet - that
// wiring is a deliberately separate, later step, only after this gate
// passes for real - see src/cuda/README.md.
#include <cstdio>
#include <cstring>
#include <string>

#include <ethash/progpow.hpp>

#include "../../src/cuda/progpowz_kernel.cu"

using namespace deepcore::progpowz;

namespace {

int g_failures = 0;
void check(bool cond, const char* what)
{
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

hash256 to_portable(const ethash::hash256& h)
{
    hash256 out;
    std::memcpy(out.bytes, h.bytes, 32);
    return out;
}

// Runs a batch of `count` consecutive nonces starting at `start_nonce`
// through both the already-validated full-DAG kernel and the new warp
// kernel, comparing every result byte-for-byte, and also cross-checks
// against the real reference implementation directly for the first nonce
// in the batch (so a bug that happened to affect BOTH GPU kernels
// identically - e.g. a shared, wrong full_dataset - would still be
// caught, not just kernel-vs-kernel agreement on a shared bug).
void run_batch_case(DeviceEpochCache& epoch_cache, DeviceFullDataset& dataset, const ethash::epoch_context& ctx,
    int block_number, const ethash::hash256& header_hash_eth, uint64_t start_nonce, uint32_t count,
    const char* case_name)
{
    std::printf("--- %s: block_number=%d start_nonce=0x%016llx count=%u ---\n", case_name, block_number,
        static_cast<unsigned long long>(start_nonce), count);

    auto header_hash = to_portable(header_hash_eth);
    auto full_dataset_num_items = static_cast<uint32_t>(ctx.full_dataset_num_items);

    PersistentFullDagSearcher full_searcher;
    PersistentWarpSearcher warp_searcher;

    auto full_results = full_searcher.search(dataset, epoch_cache.device_l1_cache(), full_dataset_num_items,
        block_number, header_hash, start_nonce, count);
    auto warp_results = warp_searcher.search(dataset, epoch_cache.device_l1_cache(), full_dataset_num_items,
        block_number, header_hash, start_nonce, count);

    bool all_match = true;
    for (uint32_t i = 0; i < count; ++i)
    {
        bool final_ok = std::memcmp(full_results[i].final_hash.bytes, warp_results[i].final_hash.bytes, 32) == 0;
        bool mix_ok = std::memcmp(full_results[i].mix_hash.bytes, warp_results[i].mix_hash.bytes, 32) == 0;
        bool nonce_ok = full_results[i].nonce == warp_results[i].nonce && full_results[i].nonce == start_nonce + i;
        if (!final_ok || !mix_ok || !nonce_ok)
        {
            all_match = false;
            std::printf("  MISMATCH at index %u (nonce=0x%016llx): final_hash %s, mix_hash %s, nonce %s\n", i,
                static_cast<unsigned long long>(start_nonce + i), final_ok ? "match" : "MISMATCH",
                mix_ok ? "match" : "MISMATCH", nonce_ok ? "match" : "MISMATCH");
        }
    }
    check(all_match, "warp kernel matches full-DAG kernel for every nonce in the batch");

    // Independent cross-check of the first nonce against the real
    // reference implementation directly - catches a bug that might
    // affect both GPU kernels identically (e.g. a shared wrong dataset).
    auto ref = progpow::hash(ctx, block_number, header_hash_eth, start_nonce);
    bool ref_final_ok = std::memcmp(ref.final_hash.bytes, warp_results[0].final_hash.bytes, 32) == 0;
    bool ref_mix_ok = std::memcmp(ref.mix_hash.bytes, warp_results[0].mix_hash.bytes, 32) == 0;
    check(ref_final_ok && ref_mix_ok,
        "warp kernel's first result also matches the real reference implementation directly");

    std::printf("\n");
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
    std::printf("epoch 0: light_cache_num_items=%d full_dataset_num_items=%d\n\n", ctx->light_cache_num_items,
        ctx->full_dataset_num_items);

    const auto* host_light_cache = reinterpret_cast<const hash512*>(ctx->light_cache);

    DeviceEpochCache epoch_cache;
    epoch_cache.ensure_uploaded(host_light_cache, ctx->light_cache_num_items, ctx->l1_cache, l1_cache_num_items);

    DeviceFullDataset dataset;
    std::string dag_error;
    if (!dataset.ensure_built(
            host_light_cache, ctx->light_cache_num_items, static_cast<uint32_t>(ctx->full_dataset_num_items),
            dag_error))
    {
        std::fprintf(stderr, "FATAL: could not build the full DAG for epoch 0: %s\n", dag_error.c_str());
        return 1;
    }

    ethash::hash256 zero_hash{};
    std::memset(&zero_hash, 0, sizeof(zero_hash));
    ethash::hash256 pattern_hash{};
    for (int i = 0; i < 32; ++i)
        pattern_hash.bytes[i] = static_cast<uint8_t>(i);

    // count=1: minimal case, a single lane group, no padding needed.
    run_batch_case(epoch_cache, dataset, *ctx, 0, zero_hash, 0, 1, "single nonce");
    run_batch_case(epoch_cache, dataset, *ctx, 0, zero_hash, 1, 1, "single nonce (different)");
    run_batch_case(
        epoch_cache, dataset, *ctx, 12345, pattern_hash, 0x123456789abcdef0ULL, 1, "single nonce, real block/header");

    // count=17: odd, not block/warp-aligned (17*16=272 threads, not a
    // multiple of 256) - exercises the padding/tail-handling logic.
    run_batch_case(epoch_cache, dataset, *ctx, 0, pattern_hash, 1000, 17, "odd, non-block-aligned batch");

    // count=256: multiple full blocks (256*16=4096 threads = 16 blocks of
    // 256), exercises multiple lane groups across multiple blocks.
    run_batch_case(epoch_cache, dataset, *ctx, 0, zero_hash, 500000, 256, "large multi-block batch");

    if (g_failures == 0)
    {
        std::printf(
            "ALL CHECKS PASSED: progpowz_warp_kernel matches the full-DAG kernel and the reference "
            "implementation exactly.\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED - do not trust progpowz_warp_kernel until fixed.\n", g_failures);
    return 1;
}
