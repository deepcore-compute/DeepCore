// Correctness/integration self-test for GpuHashSearchBackend (see
// src/cuda/progpowz_gpu_backend.hpp for scope/status).
//
// THIS IS THE VALIDATION GATE for GPU milestone 2 (the mining_loop-facing
// GPU backend), same role gpu_selftest.cu played for the raw kernel
// wrapper (GPU milestone 1). Must be built with nvcc and run on real
// hardware; it has not been done in the environment that wrote this file.
//
// Method: build a real epoch-0 context via the unmodified reference
// library, compute one known nonce's hash via the reference implementation
// directly (progpow::hash), then drive GpuHashSearchBackend through
// IHashSearchBackend::search() with targets constructed from that known
// hash so the expected outcome (found / not found, and exactly which
// nonce/mix_hash) is known ahead of time - not just "looks plausible".
// Also cross-checks GpuHashSearchBackend against the already-validated
// CpuHashSearchBackend on the same job/range, since both implement the
// same interface and must agree.
#include <atomic>
#include <cstdio>
#include <cstring>

#include <ethash/progpow.hpp>

#include "../../src/cuda/progpowz_gpu_backend.hpp"

using namespace deepcore;
using namespace deepcore::mining;

namespace {

int g_failures = 0;
void check(bool cond, const char* what)
{
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

progpowz::hash256 to_portable(const ethash::hash256& h)
{
    progpowz::hash256 out;
    std::memcpy(out.bytes, h.bytes, 32);
    return out;
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

    const int block_number = 0;
    ethash::hash256 header_hash_eth{};
    std::memset(&header_hash_eth, 0, sizeof(header_hash_eth));
    const std::uint64_t known_nonce = 0;

    // Ground truth, computed once via the unmodified reference implementation.
    auto ref = progpow::hash(*ctx, block_number, header_hash_eth, known_nonce);

    ProgPowZJob job;
    job.block_number = block_number;
    job.pow_hash = to_portable(header_hash_eth);
    job.seed_hash = {};  // not read by search()
    job.target_boundary = to_portable(ref.final_hash);  // == known_nonce's hash -> trivially met

    GpuHashSearchBackend backend(/*device_index=*/0);

    check(backend.preferred_batch_size() > 0, "preferred_batch_size() returns a positive value");

    // ---- A: exact single-nonce search finds the known winner ----
    {
        std::atomic<bool> cancelled{false};
        auto found = backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
        check(found.has_value(), "A: count=1 search at the known winning nonce finds a share");
        if (found)
        {
            check(found->nonce == known_nonce, "A: found nonce matches the known winning nonce");
            check(std::memcmp(found->mix_hash.bytes, ref.mix_hash.bytes, 32) == 0,
                "A: found mix_hash matches the reference implementation's mix_hash");
        }
    }

    // ---- B: a full preferred-size batch starting at the winner still finds it (index 0) ----
    {
        std::atomic<bool> cancelled{false};
        auto found = backend.search(job, *ctx, /*start_nonce=*/known_nonce, backend.preferred_batch_size(),
            cancelled);
        check(found.has_value(), "B: a full preferred-size batch containing the winner finds it");
        if (found)
            check(found->nonce == known_nonce, "B: found nonce is still the known winning nonce");
    }

    // ---- C: search actually honors [start_nonce, start_nonce+count) ----
    // Uses an always-satisfied (all-0xFF) target rather than "a batch that
    // excludes a specific known nonce should find nothing", which was
    // this test's original (buggy) approach: target_boundary equal to one
    // nonce's exact hash is NOT a "hard to meet" target in general - its
    // magnitude as a fraction of the full 256-bit space is whatever that
    // one hash happens to be (here, roughly 1-in-5, since its leading
    // byte is 0x32), so a 65536-nonce batch had a near-certain chance of
    // a coincidental extra match with no bug involved. An all-0xFF target
    // matches literally every hash, so which nonce comes back is fully
    // deterministic and actually tests window correctness.
    {
        ProgPowZJob easy_job = job;
        std::memset(easy_job.target_boundary.bytes, 0xFF, sizeof(easy_job.target_boundary.bytes));
        std::atomic<bool> cancelled{false};

        auto found_at_0 = backend.search(easy_job, *ctx, /*start_nonce=*/0, /*count=*/1, cancelled);
        check(found_at_0.has_value() && found_at_0->nonce == 0,
            "C1: an always-satisfied target at start_nonce=0 finds nonce 0");

        auto found_at_5 = backend.search(easy_job, *ctx, /*start_nonce=*/5, /*count=*/1, cancelled);
        check(found_at_5.has_value() && found_at_5->nonce == 5,
            "C2: an always-satisfied target at start_nonce=5 finds nonce 5 (not nonce 0) - "
            "confirms the search window is actually respected");
    }

    // ---- D: an impossible target (zero) is never met ----
    {
        ProgPowZJob hard_job = job;
        hard_job.target_boundary = progpowz::hash256{};  // all-zero: virtually nothing qualifies
        std::atomic<bool> cancelled{false};
        auto found = backend.search(hard_job, *ctx, /*start_nonce=*/known_nonce,
            backend.preferred_batch_size(), cancelled);
        check(!found.has_value(), "D: an all-zero (impossible) target finds nothing");
    }

    // ---- E: cancellation is honored before launch, even for an otherwise-winning range ----
    {
        std::atomic<bool> cancelled{true};
        auto found = backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
        check(!found.has_value(), "E: a pre-cancelled search returns nothing despite a winning nonce in range");
    }

    // ---- F: repeated calls with identical inputs are deterministic ----
    {
        std::atomic<bool> cancelled{false};
        auto first = backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
        auto second = backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
        bool consistent = first.has_value() && second.has_value() && first->nonce == second->nonce &&
            std::memcmp(first->mix_hash.bytes, second->mix_hash.bytes, 32) == 0;
        check(consistent, "F: repeated identical search() calls return identical results");
    }

    // ---- H: agrees with the already-validated CPU backend on the same job/range ----
    {
        CpuHashSearchBackend cpu_backend;
        std::atomic<bool> cancelled{false};
        auto gpu_found = backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/64, cancelled);
        auto cpu_found = cpu_backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/64, cancelled);
        bool agree = gpu_found.has_value() && cpu_found.has_value() && gpu_found->nonce == cpu_found->nonce &&
            std::memcmp(gpu_found->mix_hash.bytes, cpu_found->mix_hash.bytes, 32) == 0;
        check(agree, "H: GpuHashSearchBackend and CpuHashSearchBackend agree on the same job/range");
    }

    // ---- I: persistent VRAM cache correctly re-uploads for a genuinely
    // different epoch_context object, not just a different epoch number -
    // targets DeviceEpochCache's "same host pointer = skip re-upload"
    // identity check directly. Builds a second, INDEPENDENT epoch-0
    // context (same epoch, different object/pointer - as if MiningLoop's
    // own epoch cache had been evicted and rebuilt), interleaves calls
    // against the original and the new one, and confirms neither call
    // picks up stale results left over from the other. If the identity
    // check ever failed to detect the switch (e.g. compared the wrong
    // pointer, or a real host allocator reused the same address), a
    // wrong/stale light_cache could silently be used, corrupting results
    // without any error - this is the test that would catch that. ----
    {
        auto ctx2 = ethash::create_epoch_context(0);
        if (!ctx2)
        {
            check(false, "I: could not build a second independent epoch-0 context");
        }
        else
        {
            std::atomic<bool> cancelled{false};

            // Same known-answer setup as A/B, but against the fresh
            // context - correct results here require DeviceEpochCache to
            // have actually re-uploaded for ctx2's (different) light_cache
            // pointer, not silently kept serving ctx's data.
            auto found_ctx2 = backend.search(job, *ctx2, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
            check(found_ctx2.has_value() && found_ctx2->nonce == known_nonce &&
                    std::memcmp(found_ctx2->mix_hash.bytes, ref.mix_hash.bytes, 32) == 0,
                "I1: search against a second, independent epoch-0 context gives correct results");

            // Switch back to the original context - confirms the cache
            // re-uploads AGAIN on the way back, rather than e.g. getting
            // stuck always re-uploading ctx2's data after the first switch.
            auto found_ctx1_again =
                backend.search(job, *ctx, /*start_nonce=*/known_nonce, /*count=*/1, cancelled);
            check(found_ctx1_again.has_value() && found_ctx1_again->nonce == known_nonce &&
                    std::memcmp(found_ctx1_again->mix_hash.bytes, ref.mix_hash.bytes, 32) == 0,
                "I2: switching back to the original context after a different one still gives correct results");
        }
    }

    // ---- J: full-DAG mode was actually used, not silently falling back ----
    // Every check above already ran through GpuHashSearchBackend, which
    // prefers full-DAG mode whenever the epoch's dataset fits in free VRAM
    // (checked for real via cudaMemGetInfo - never assumed). Since
    // light-cache mode is ALSO correct, tests A/B/C/D/E/F/H/I passing does
    // NOT by itself prove full-DAG mode ran - this check closes that gap
    // directly, so a real, no-hardware-degradation full-DAG validation
    // requires this to be true (a silent fallback would need investigating
    // separately, not just accepted because everything else passed).
    {
        check(backend.last_search_used_full_dag(),
            "J: the most recent search() used full-DAG mode, not a silent light-cache fallback");
    }

    if (g_failures == 0)
    {
        std::printf("\nALL CHECKS PASSED: GpuHashSearchBackend is correctly wired to the GPU kernel "
                     "and agrees with the CPU reference backend.\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED - do not trust GpuHashSearchBackend until fixed.\n", g_failures);
    return 1;
}
