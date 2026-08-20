// CPU-only correctness self-test for the portable ProgPowZ core in
// src/cuda/progpowz_portable.hpp.
//
// This does NOT require CUDA, nvcc, or a GPU - it links the vendored
// reference library (third_party/ethash-progpow-ref, unmodified) to build a
// real epoch context (light cache + l1 cache + item counts), calls the
// reference's own progpow::hash() as ground truth, then calls our portable
// port with the exact same context data and compares outputs byte-for-byte.
//
// This is intentionally the ONLY thing that gets trusted before any CUDA
// code is written: src/cuda/progpowz_portable.hpp is written with
// __host__ __device__ so the exact same source that passes here is what a
// CUDA kernel later calls per-thread - only the kernel launch / memory
// transfer wrapper around it is new, untested surface at that point.
//
// No CUDA toolchain is available in the environment this was written in, so
// the actual .cu kernel wrapper has not been compiled or run anywhere yet -
// see src/cuda/README.md for current status and what still needs
// validating on real hardware.
#include <cstdio>
#include <cstring>
#include <vector>

#include <ethash/progpow.hpp>
#include "ethash-internal.hpp"

#include "../../src/cuda/progpowz_portable.hpp"

namespace {

int g_failures = 0;

deepcore::progpowz::hash256 to_portable(const ethash::hash256& h)
{
    deepcore::progpowz::hash256 out;
    std::memcpy(out.bytes, h.bytes, 32);
    return out;
}

bool bytes_equal(const uint8_t* a, const uint8_t* b, size_t n)
{
    return std::memcmp(a, b, n) == 0;
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

    // Ground truth: the real, unmodified reference implementation.
    auto ref = progpow::hash(ctx, block_number, header_hash, nonce);

    // Our portable port, given the exact same epoch context data.
    auto our_header = to_portable(header_hash);
    auto ours = deepcore::progpowz::progpowz_hash_light(
        reinterpret_cast<const deepcore::progpowz::hash512*>(ctx.light_cache),
        ctx.light_cache_num_items,
        ctx.l1_cache,
        static_cast<uint32_t>(ctx.full_dataset_num_items),
        block_number, our_header, nonce);

    // Diagnostic: isolate the keccak_progpow_64/256 pipeline from the
    // round()/mix pipeline by recomputing final_hash using OUR seed/keccak
    // functions but the REFERENCE's own mix_hash. If this matches
    // ref.final_hash, our keccak_progpow_* primitives are correct and the
    // bug is confined to the round()/mix logic; if not, the bug is in
    // keccak_progpow_64/256 (keccakf800) itself.
    {
        uint64_t our_seed = deepcore::progpowz::keccak_progpow_64(our_header, nonce);
        auto ref_mix_portable = to_portable(ref.mix_hash);
        auto recombined = deepcore::progpowz::keccak_progpow_256(our_header, our_seed, ref_mix_portable);
        bool keccak_pipeline_ok = bytes_equal(ref.final_hash.bytes, recombined.bytes, 32);
        std::printf("diag: keccak_progpow_64/256 pipeline (using ref mix_hash) %s\n",
            keccak_pipeline_ok ? "MATCHES reference" : "DOES NOT MATCH reference");
    }

    // Diagnostic 2: isolate calculate_dataset_item_2048 by comparing our
    // port's output directly against the reference's own (internal but
    // includable) implementation, for a fixed item index.
    {
        uint32_t index = 12345 % (uint32_t)(ctx.full_dataset_num_items / 2);
        ethash::hash2048 ref_item = ethash::calculate_dataset_item_2048(ctx, index);
        auto our_item = deepcore::progpowz::calculate_dataset_item_2048(
            reinterpret_cast<const deepcore::progpowz::hash512*>(ctx.light_cache),
            ctx.light_cache_num_items, index);
        bool item_ok = bytes_equal(ref_item.bytes, our_item.bytes, 256);
        std::printf("diag: calculate_dataset_item_2048(index=%u) %s\n", index,
            item_ok ? "MATCHES reference" : "DOES NOT MATCH reference");
    }

    bool final_ok = bytes_equal(ref.final_hash.bytes, ours.final_hash.bytes, 32);
    bool mix_ok = bytes_equal(ref.mix_hash.bytes, ours.mix_hash.bytes, 32);

    print_hex("reference final_hash", ref.final_hash.bytes, 32);
    print_hex("ours      final_hash", ours.final_hash.bytes, 32);
    print_hex("reference mix_hash  ", ref.mix_hash.bytes, 32);
    print_hex("ours      mix_hash  ", ours.mix_hash.bytes, 32);

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

// Correctness check for progpowz_hash_light's new optional `full_dataset`
// parameter (see that function's header comment in progpowz_portable.hpp)
// - the mechanism the real full-DAG GPU kernel will use. Does NOT build a
// real epoch's full dataset (millions of items, gigabytes - only
// realistically feasible on a GPU, see progpowz_kernel.cu's DAG-generation
// kernel and src/cuda/README.md); instead uses a small, artificial
// full_dataset_num_items against the REAL epoch-0 light_cache (only
// light_cache/light_cache_num_items matter for computing an individual
// item - full_dataset_num_items just sets the item_index range, so a
// small fake value here is a legitimate, fast, CPU-only way to test the
// array-read branch's correctness without needing a real-scale dataset).
//
// Method: for each of several nonces, run progpowz_hash_light once with
// full_dataset=nullptr (light/recompute mode, the path every existing
// caller already uses and this project has already proven correct) and
// once with full_dataset pointing at a small array pre-populated by
// calling calculate_dataset_item_2048 directly for every index in range -
// i.e. the array holds exactly what the recompute path would have
// computed anyway. The two calls MUST produce byte-identical output; any
// difference means the new array-read branch's indexing/threading is
// wrong, independent of whether calculate_dataset_item_2048 itself is
// correct (already proven separately, above).
void run_full_dataset_mode_case(const ethash::epoch_context& ctx, int block_number,
    const ethash::hash256& header_hash, uint64_t nonce,
    const std::vector<deepcore::progpowz::hash2048>& small_full_dataset, uint32_t fake_full_dataset_num_items)
{
    auto our_header = to_portable(header_hash);
    const auto* light_cache = reinterpret_cast<const deepcore::progpowz::hash512*>(ctx.light_cache);

    auto light_mode = deepcore::progpowz::progpowz_hash_light(light_cache, ctx.light_cache_num_items,
        ctx.l1_cache, fake_full_dataset_num_items, block_number, our_header, nonce, /*full_dataset=*/nullptr);

    auto full_dag_mode = deepcore::progpowz::progpowz_hash_light(light_cache, ctx.light_cache_num_items,
        ctx.l1_cache, fake_full_dataset_num_items, block_number, our_header, nonce, small_full_dataset.data());

    bool final_ok = bytes_equal(light_mode.final_hash.bytes, full_dag_mode.final_hash.bytes, 32);
    bool mix_ok = bytes_equal(light_mode.mix_hash.bytes, full_dag_mode.mix_hash.bytes, 32);

    std::printf("full_dataset-mode check (nonce=0x%016llx): %s\n", static_cast<unsigned long long>(nonce),
        (final_ok && mix_ok) ? "PASS (light mode and full_dataset-array mode agree exactly)"
                              : "FAIL (final_hash/mix_hash mismatch)");
    if (!final_ok || !mix_ok)
        ++g_failures;
}

}  // namespace

int main()
{
    // Build the real epoch-0 light context via the unmodified reference
    // library - this reuses already-proven code for cache/DAG-size setup,
    // so only the per-nonce hash logic in progpowz_portable.hpp is what
    // this test is actually validating.
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

    // full_dataset-mode (full-DAG kernel mechanism) correctness - see that
    // function's header comment above. Small, artificial dataset size
    // (128 -> 64 hash2048 items), built from the real epoch-0 light_cache.
    {
        std::printf("--- full_dataset-mode (full-DAG kernel mechanism) checks ---\n");
        const uint32_t fake_full_dataset_num_items = 128;
        const uint32_t num_small_items = fake_full_dataset_num_items / 2;

        std::vector<deepcore::progpowz::hash2048> small_full_dataset(num_small_items);
        const auto* light_cache = reinterpret_cast<const deepcore::progpowz::hash512*>(ctx->light_cache);
        for (uint32_t i = 0; i < num_small_items; ++i)
        {
            small_full_dataset[i] =
                deepcore::progpowz::calculate_dataset_item_2048(light_cache, ctx->light_cache_num_items, i);
        }

        run_full_dataset_mode_case(*ctx, 0, zero_hash, 0, small_full_dataset, fake_full_dataset_num_items);
        run_full_dataset_mode_case(*ctx, 0, zero_hash, 1, small_full_dataset, fake_full_dataset_num_items);
        run_full_dataset_mode_case(
            *ctx, 0, pattern_hash, 0x123456789abcdef0ULL, small_full_dataset, fake_full_dataset_num_items);
        run_full_dataset_mode_case(
            *ctx, 12345, pattern_hash, 0x123456789abcdef0ULL, small_full_dataset, fake_full_dataset_num_items);
        std::printf("\n");
    }

    if (g_failures == 0)
    {
        std::printf("ALL VECTORS PASSED: portable ProgPowZ core matches the reference "
                     "implementation exactly.\n");
        return 0;
    }
    std::printf("%d VECTOR(S) FAILED - do not trust progpowz_portable.hpp until fixed.\n",
        g_failures);
    return 1;
}
