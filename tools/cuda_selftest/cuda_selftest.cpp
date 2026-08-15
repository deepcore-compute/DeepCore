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
