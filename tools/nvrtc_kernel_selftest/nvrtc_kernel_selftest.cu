// Correctness gate for NvrtcWarpSearcher (progpowz_nvrtc_kernel.hpp/.cpp)
// - the per-period, NVRTC-compiled ProgPowZ warp kernel. See that
// header's comment for scope/status and WHY this exists (real production
// miners resolve the mix program once per period instead of interpreting
// it on every hash - see src/cuda/README.md's sourced research).
//
// THIS IS THE REAL-HARDWARE VALIDATION GATE for that work. Unlike every
// prior kernel milestone in this project, this is the FIRST to exercise
// entirely new-to-the-project API surface (NVRTC + the CUDA Driver API,
// not just new kernel logic on already-proven Runtime-API plumbing), and
// it depends on progpowz_codegen.hpp's CUDA source TEXT EMITTER, which
// (unlike the trace generator/interpreter it's built from - see
// tools/codegen_selftest, already validated on CPU) could not be
// compile-tested at all before real hardware. Expect this to plausibly
// need at least one real-hardware debug round-trip, unlike prior
// kernel-only changes.
//
// Compares NvrtcWarpSearcher's output against the already-validated
// PersistentWarpSearcher (progpowz_kernel.cu's interpreted warp kernel -
// itself already proven against the reference implementation, see
// tools/warp_kernel_selftest) for the exact same (block_number, nonce)
// inputs, AND cross-checks the first nonce of each batch against the real
// reference implementation directly - so a bug shared by both GPU paths
// (e.g. a wrong full dataset) would still be caught, not just
// kernel-vs-kernel agreement on a shared bug.
#include <cstdio>
#include <cstring>
#include <string>

#include <ethash/progpow.hpp>

#include "../../src/cuda/progpowz_kernel.cu"
#include "../../src/cuda/progpowz_nvrtc_kernel.hpp"

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

void run_batch_case(NvrtcWarpSearcher& nvrtc_searcher, DeviceEpochCache& epoch_cache, DeviceFullDataset& dataset,
    const ethash::epoch_context& ctx, int block_number, const ethash::hash256& header_hash_eth,
    uint64_t start_nonce, uint32_t count, const char* case_name)
{
    std::printf("--- %s: block_number=%d start_nonce=0x%016llx count=%u ---\n", case_name, block_number,
        static_cast<unsigned long long>(start_nonce), count);

    auto header_hash = to_portable(header_hash_eth);
    auto full_dataset_num_items = static_cast<uint32_t>(ctx.full_dataset_num_items);

    PersistentWarpSearcher interpreted_searcher;
    auto interpreted_results = interpreted_searcher.search(dataset, epoch_cache.device_l1_cache(),
        full_dataset_num_items, block_number, header_hash, start_nonce, count);

    std::vector<GpuHashResult> nvrtc_results;
    std::string nvrtc_error;
    bool nvrtc_ok = nvrtc_searcher.search(epoch_cache.device_l1_cache(), full_dataset_num_items,
        dataset.device_dataset(), block_number, header_hash, start_nonce, count, nvrtc_results, nvrtc_error);

    if (!nvrtc_ok)
    {
        std::printf("  NVRTC path FAILED: %s\n", nvrtc_error.c_str());
        check(false, "NVRTC kernel compiled and launched successfully");
        std::printf("\n");
        return;
    }
    check(true, "NVRTC kernel compiled and launched successfully");

    bool all_match = true;
    for (uint32_t i = 0; i < count; ++i)
    {
        bool final_ok =
            std::memcmp(interpreted_results[i].final_hash.bytes, nvrtc_results[i].final_hash.bytes, 32) == 0;
        bool mix_ok = std::memcmp(interpreted_results[i].mix_hash.bytes, nvrtc_results[i].mix_hash.bytes, 32) == 0;
        bool nonce_ok =
            interpreted_results[i].nonce == nvrtc_results[i].nonce && interpreted_results[i].nonce == start_nonce + i;
        if (!final_ok || !mix_ok || !nonce_ok)
        {
            all_match = false;
            std::printf("  MISMATCH at index %u (nonce=0x%016llx): final_hash %s, mix_hash %s, nonce %s\n", i,
                static_cast<unsigned long long>(start_nonce + i), final_ok ? "match" : "MISMATCH",
                mix_ok ? "match" : "MISMATCH", nonce_ok ? "match" : "MISMATCH");
        }
    }
    check(all_match, "NVRTC-compiled kernel matches the interpreted warp kernel for every nonce in the batch");

    auto ref = progpow::hash(ctx, block_number, header_hash_eth, start_nonce);
    bool ref_final_ok = std::memcmp(ref.final_hash.bytes, nvrtc_results[0].final_hash.bytes, 32) == 0;
    bool ref_mix_ok = std::memcmp(ref.mix_hash.bytes, nvrtc_results[0].mix_hash.bytes, 32) == 0;
    check(ref_final_ok && ref_mix_ok,
        "NVRTC-compiled kernel's first result also matches the real reference implementation directly");

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

    NvrtcWarpSearcher nvrtc_searcher;

    ethash::hash256 zero_hash{};
    std::memset(&zero_hash, 0, sizeof(zero_hash));
    ethash::hash256 pattern_hash{};
    for (int i = 0; i < 32; ++i)
        pattern_hash.bytes[i] = static_cast<uint8_t>(i);

    // count=1: minimal case, single lane group, exercises the first
    // real NVRTC compile (block_number=0, period 0).
    run_batch_case(nvrtc_searcher, epoch_cache, dataset, *ctx, 0, zero_hash, 0, 1, "single nonce");
    run_batch_case(nvrtc_searcher, epoch_cache, dataset, *ctx, 0, zero_hash, 1, 1, "single nonce (different)");

    // Same period (0), different block_number (49) and different
    // header/nonce - must reuse the CACHED compiled kernel (no
    // recompile), and must still produce correct results with it.
    run_batch_case(
        nvrtc_searcher, epoch_cache, dataset, *ctx, 49, pattern_hash, 0x123456789abcdef0ULL, 1,
        "same period (49), cached-kernel reuse");

    // New period (50) - must trigger a real recompile, and the newly
    // compiled kernel must be correct.
    run_batch_case(
        nvrtc_searcher, epoch_cache, dataset, *ctx, 50, pattern_hash, 0x123456789abcdef0ULL, 1,
        "new period (50), forces recompile");

    // count=17: odd, not block/warp-aligned - exercises the padding/tail
    // handling in the generated kernel's launch wrapper.
    run_batch_case(
        nvrtc_searcher, epoch_cache, dataset, *ctx, 0, pattern_hash, 1000, 17, "odd, non-block-aligned batch");

    // count=256: multiple full blocks.
    run_batch_case(
        nvrtc_searcher, epoch_cache, dataset, *ctx, 0, zero_hash, 500000, 256, "large multi-block batch");

    if (g_failures == 0)
    {
        std::printf(
            "ALL CHECKS PASSED: the NVRTC-compiled per-period kernel matches the interpreted warp kernel "
            "and the reference implementation exactly.\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED - do not trust NvrtcWarpSearcher until fixed.\n", g_failures);
    return 1;
}
