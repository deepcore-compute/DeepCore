// CPU-only correctness self-test for src/cuda/progpowz_codegen.hpp's
// per-period program trace generator and trace interpreter.
//
// Does NOT require CUDA, nvcc, or a GPU - same pattern as
// tools/cuda_selftest/cuda_selftest.cpp: build a real epoch context via
// the vendored reference library, then compare against
// progpowz_hash_light() (already proven correct against the reference
// implementation by cuda_selftest) byte-for-byte.
//
// What this actually proves: generate_progpowz_trace(block_number) +
// trace_interpret_hash(trace, ...) together compute the EXACT SAME hash
// as progpowz_hash_light(..., block_number, ...) - i.e. resolving the
// per-round program once (independent of nonce) and then replaying it is
// truly equivalent to interpreting dst_seq/src_seq/kiss99 fresh on every
// hash. This is the necessary foundation before trusting any CUDA source
// text generated from the same trace (see
// generate_progpowz_cuda_round_source() - NOT validated here, since that
// requires an actual CUDA compile/run this environment cannot do; see
// src/cuda/README.md for that gate).
//
// Runs many nonces against a FIXED trace per block_number to specifically
// exercise the "trace is nonce-independent" property (the whole point of
// per-period compilation): the SAME trace object, generated once, must
// correctly reproduce every nonce's hash for that block_number.
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include <ethash/progpow.hpp>
#include "ethash-internal.hpp"

#include "../../src/cuda/progpowz_codegen.hpp"

namespace {

int g_failures = 0;

bool bytes_equal(const uint8_t* a, const uint8_t* b, size_t n) { return std::memcmp(a, b, n) == 0; }

deepcore::progpowz::hash256 to_portable(const ethash::hash256& h)
{
    deepcore::progpowz::hash256 out;
    std::memcpy(out.bytes, h.bytes, 32);
    return out;
}

// Compares trace_interpret_hash() against progpowz_hash_light() for one
// (block_number, nonce) pair, given a trace already generated for that
// block_number (the caller reuses the same trace across many nonces -
// see main() - to specifically exercise trace reuse, not just a single
// lucky case).
void run_case(const ethash::epoch_context& ctx, const std::vector<deepcore::progpowz::TraceOp>& trace,
    int block_number, const ethash::hash256& header_hash, uint64_t nonce)
{
    const auto* light_cache = reinterpret_cast<const deepcore::progpowz::hash512*>(ctx.light_cache);
    const auto our_header = to_portable(header_hash);

    auto expected = deepcore::progpowz::progpowz_hash_light(light_cache, ctx.light_cache_num_items, ctx.l1_cache,
        static_cast<uint32_t>(ctx.full_dataset_num_items), block_number, our_header, nonce);

    auto traced = deepcore::progpowz::trace_interpret_hash(trace, light_cache, ctx.light_cache_num_items,
        ctx.l1_cache, static_cast<uint32_t>(ctx.full_dataset_num_items), our_header, nonce);

    bool final_ok = bytes_equal(expected.final_hash.bytes, traced.final_hash.bytes, 32);
    bool mix_ok = bytes_equal(expected.mix_hash.bytes, traced.mix_hash.bytes, 32);

    std::printf("  nonce=0x%016llx: %s\n", static_cast<unsigned long long>(nonce),
        (final_ok && mix_ok) ? "PASS" : "FAIL");
    if (!final_ok || !mix_ok)
    {
        std::printf("    expected final_hash: 0x");
        for (auto b : expected.final_hash.bytes) std::printf("%02x", b);
        std::printf("\n    traced   final_hash: 0x");
        for (auto b : traced.final_hash.bytes) std::printf("%02x", b);
        std::printf("\n");
        ++g_failures;
    }
}

void run_block(const ethash::epoch_context& ctx, int block_number, const ethash::hash256& header_hash)
{
    std::printf("--- block_number=%d (period=%d) ---\n", block_number,
        block_number / deepcore::progpowz::period_length);

    auto trace = deepcore::progpowz::generate_progpowz_trace(block_number);
    std::printf("  trace size: %zu ops\n", trace.size());

    // Same trace, many different nonces - this is the property that
    // makes per-period compilation valid: the trace must not depend on
    // nonce at all.
    run_case(ctx, trace, block_number, header_hash, 0);
    run_case(ctx, trace, block_number, header_hash, 1);
    run_case(ctx, trace, block_number, header_hash, 0x123456789abcdef0ULL);
    run_case(ctx, trace, block_number, header_hash, 0xffffffffffffffffULL);
    run_case(ctx, trace, block_number, header_hash, 42);
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

    ethash::hash256 zero_hash{};
    std::memset(&zero_hash, 0, sizeof(zero_hash));
    ethash::hash256 pattern_hash{};
    for (int i = 0; i < 32; ++i) pattern_hash.bytes[i] = static_cast<uint8_t>(i);

    // block_number=0 and block_number=49 are the same period
    // (period_length=50, 0/50 == 49/50 == 0) - the trace for both MUST be
    // byte-identical (block_number only affects the mix program via
    // block_number/period_length; progpowz_hash_light has no other
    // block_number dependence), and trace_interpret_hash() using that one
    // shared trace must still correctly reproduce progpowz_hash_light()'s
    // own output when called with either block_number.
    run_block(*ctx, 0, zero_hash);
    run_block(*ctx, 49, zero_hash);
    // block_number=50 starts a new period - trace must differ from the
    // block_number=0/49 trace (checked implicitly: if it didn't, and the
    // derivation had a bug making it period-independent, run_case here
    // would still incidentally pass since it always re-derives block_number's
    // own period internally - the real cross-check for "does period
    // actually change the trace" is done explicitly below).
    run_block(*ctx, 50, pattern_hash);
    run_block(*ctx, 12345, pattern_hash);

    // Explicit check: traces for two different periods must differ (if
    // they were accidentally identical for all block_numbers, every
    // run_case check above could still spuriously pass while the
    // derivation itself was broken - e.g. always deriving with s=0).
    {
        auto trace_period0 = deepcore::progpowz::generate_progpowz_trace(0);
        auto trace_period1 = deepcore::progpowz::generate_progpowz_trace(50);
        bool same_size = trace_period0.size() == trace_period1.size();
        bool identical = same_size &&
            std::memcmp(trace_period0.data(), trace_period1.data(), trace_period0.size() * sizeof(deepcore::progpowz::TraceOp)) == 0;
        std::printf("--- cross-period distinctness check ---\n");
        std::printf("  period 0 trace size=%zu, period 1 trace size=%zu, identical=%s: %s\n",
            trace_period0.size(), trace_period1.size(), identical ? "true" : "false",
            !identical ? "PASS (distinct periods really do produce distinct traces)"
                       : "FAIL (traces for different periods must not be identical)");
        if (identical)
            ++g_failures;

        // Same-period distinctness: block_number=0 and block_number=49
        // (both period 0) MUST produce byte-identical traces.
        auto trace_same_period = deepcore::progpowz::generate_progpowz_trace(49);
        bool same_period_identical = trace_period0.size() == trace_same_period.size() &&
            std::memcmp(trace_period0.data(), trace_same_period.data(),
                trace_period0.size() * sizeof(deepcore::progpowz::TraceOp)) == 0;
        std::printf("  block_number=0 vs block_number=49 (same period) traces identical=%s: %s\n",
            same_period_identical ? "true" : "false",
            same_period_identical ? "PASS" : "FAIL (same-period block numbers must share one trace)");
        if (!same_period_identical)
            ++g_failures;
        std::printf("\n");
    }

    // Sanity (not correctness) check on the CUDA source text emitter:
    // just confirm it runs without crashing and produces non-trivial
    // output containing the expected round markers. This does NOT
    // validate the generated CUDA is correct C++/CUDA or that it computes
    // the right values - that requires a real nvcc/NVRTC compile and an
    // on-GPU comparison, which this CPU-only tool cannot do. See
    // src/cuda/README.md for that still-pending real-hardware gate.
    {
        auto trace = deepcore::progpowz::generate_progpowz_trace(0);
        std::string src = deepcore::progpowz::generate_progpowz_cuda_round_source(trace);
        bool has_round0 = src.find("round 0") != std::string::npos;
        bool has_round63 = src.find("round 63") != std::string::npos;
        std::printf("--- CUDA source text emitter sanity (NOT a correctness check) ---\n");
        std::printf("  generated %zu bytes, contains round 0 marker=%s, round 63 marker=%s\n", src.size(),
            has_round0 ? "yes" : "no", has_round63 ? "yes" : "no");
        std::printf("  NOTE: this only confirms the emitter runs and looks structurally plausible.\n");
        std::printf("  It does NOT prove the generated CUDA compiles or computes correct values -\n");
        std::printf("  that needs a real nvcc/NVRTC compile + on-GPU comparison (not available in\n");
        std::printf("  this environment). Do not treat this as a correctness gate.\n\n");
        if (!has_round0 || !has_round63)
            ++g_failures;
    }

    // Same disclaimer as above: structural sanity only, not a
    // correctness check. Verifies generate_progpowz_nvrtc_kernel_source()
    // (the full standalone .cu source string handed to NVRTC - see
    // progpowz_nvrtc_kernel.cpp) runs, contains the expected entry-point
    // symbol, and has balanced braces (a real, if weak, signal that the
    // splicing in generate_progpowz_nvrtc_kernel_source didn't produce
    // truncated or malformed text) - NOT that it compiles or computes
    // correct values.
    {
        std::string full_src = deepcore::progpowz::generate_progpowz_nvrtc_kernel_source(0);
        bool has_entry_point = full_src.find("progpowz_nvrtc_warp_kernel") != std::string::npos;
        bool has_extern_c = full_src.find("extern \"C\"") != std::string::npos;
        int depth = 0;
        bool balanced = true;
        for (char c : full_src)
        {
            if (c == '{') ++depth;
            else if (c == '}') { --depth; if (depth < 0) { balanced = false; break; } }
        }
        balanced = balanced && depth == 0;
        std::printf("--- full NVRTC kernel source structural sanity (NOT a correctness check) ---\n");
        std::printf("  generated %zu bytes, entry point present=%s, extern \"C\" present=%s, braces balanced=%s\n",
            full_src.size(), has_entry_point ? "yes" : "no", has_extern_c ? "yes" : "no", balanced ? "yes" : "no");
        std::printf("  This is a structural smoke test only - real validation is "
                     "tools/nvrtc_kernel_selftest (real hardware required).\n\n");
        if (!has_entry_point || !has_extern_c || !balanced)
            ++g_failures;
    }

    // random_math has 11 cases (selector % 11) and random_merge has 4
    // (selector % 4) - generate_progpowz_cuda_round_source() has a
    // separate emission branch for each. A single block_number's trace
    // does NOT necessarily exercise every case (case distribution depends
    // on that period's specific kiss99 sequence - block_number=0's trace
    // was found, during manual review, to exercise only 7 of 11 math
    // cases), so checking only one trace would leave some emission
    // branches completely uncovered by any check in this file. Scan
    // across many periods and confirm every case is hit by at least one
    // of them.
    {
        std::set<uint32_t> math_cases_seen, merge_cases_seen;
        int last_block_number_needed = 0;
        for (int bn = 0; bn < 50 * 200; bn += 50)
        {
            auto trace = deepcore::progpowz::generate_progpowz_trace(bn);
            for (const auto& op : trace)
            {
                if (op.kind == deepcore::progpowz::TraceOpKind::Math)
                    math_cases_seen.insert(op.math_case);
                merge_cases_seen.insert(op.merge_case);
            }
            last_block_number_needed = bn;
            if (math_cases_seen.size() == 11 && merge_cases_seen.size() == 4)
                break;
        }
        std::printf("--- math/merge case coverage across periods ---\n");
        std::printf("  all 11 random_math cases and all 4 random_merge cases observed by block_number=%d: %s\n",
            last_block_number_needed,
            (math_cases_seen.size() == 11 && merge_cases_seen.size() == 4) ? "PASS" : "FAIL");
        std::printf("  (math cases seen: %zu/11, merge cases seen: %zu/4)\n\n", math_cases_seen.size(),
            merge_cases_seen.size());
        if (math_cases_seen.size() != 11 || merge_cases_seen.size() != 4)
            ++g_failures;
    }

    if (g_failures == 0)
    {
        std::printf(
            "ALL CHECKS PASSED: per-period trace generation + interpretation exactly matches\n"
            "progpowz_hash_light() across multiple periods and nonces. This validates the\n"
            "DECISION LOGIC (which register, which operation, per round) that any CUDA source\n"
            "generated from the same trace would encode. The generated CUDA text itself is\n"
            "still NOT validated - that requires a real compile and on-GPU run.\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED - do not trust progpowz_codegen.hpp until fixed.\n", g_failures);
    return 1;
}
