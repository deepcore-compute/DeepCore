// Self-test for src/mining/progpowz_work.{hpp,cpp}.
//
// Cross-validates keccak256_32() directly against the real vendored
// ethash_keccak256_32() (third_party/ethash-progpow-ref/keccak.c) - the
// same reference-library-as-ground-truth approach used everywhere else in
// this project - rather than trusting a from-scratch port on inspection
// alone. epoch_from_seed_hash() is not separately cross-checked against
// zanod's own epoch_by_seedhash() (that function pulls in most of Zano's
// codebase as dependencies - epee, currency_core - impractical to link
// here); it reuses the already-cross-validated keccak256_32 primitive and
// implements the exact same trivial search zanod's own function does, so
// this is judged sufficient without pulling in that dependency graph.
#include <cstdio>
#include <cstring>

#include <ethash/keccak.h>

#include "../../src/mining/progpowz_work.hpp"
#include "../../src/network/zano_stratum_protocol.hpp"

using namespace deepcore;

namespace {

int g_failures = 0;
void check(bool cond, const char* what)
{
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

}  // namespace

int main()
{
    // ---- keccak256_32 vs the real reference implementation ----
    {
        progpowz::hash256 zero{};
        auto ours = mining::keccak256_32(zero);
        ethash_hash256 ref = ethash_keccak256_32(zero.bytes);
        check(std::memcmp(ours.bytes, ref.bytes, 32) == 0, "keccak256_32(zero) matches ethash_keccak256_32 exactly");
    }
    {
        progpowz::hash256 pattern{};
        for (int i = 0; i < 32; ++i) pattern.bytes[i] = static_cast<uint8_t>(i);
        auto ours = mining::keccak256_32(pattern);
        ethash_hash256 ref = ethash_keccak256_32(pattern.bytes);
        check(std::memcmp(ours.bytes, ref.bytes, 32) == 0, "keccak256_32(0x00..1f) matches ethash_keccak256_32 exactly");
    }
    {
        // A couple of iterations chained (as epoch_from_seed_hash does
        // internally) still match the reference at each step.
        progpowz::hash256 seed{};
        ethash_hash256 ref{};
        for (int i = 0; i < 5; ++i)
        {
            seed = mining::keccak256_32(seed);
            ref = ethash_keccak256_32(ref.bytes);
        }
        check(std::memcmp(seed.bytes, ref.bytes, 32) == 0, "5 chained keccak256_32 calls still match the reference");
    }

    // ---- epoch_from_seed_hash ----
    {
        progpowz::hash256 zero{};
        check(mining::epoch_from_seed_hash(zero) == 0, "epoch 0's seed hash (all zero) resolves to epoch 0");

        progpowz::hash256 epoch3_seed = zero;
        for (int i = 0; i < 3; ++i) epoch3_seed = mining::keccak256_32(epoch3_seed);
        check(mining::epoch_from_seed_hash(epoch3_seed) == 3, "epoch 3's seed hash resolves to epoch 3");

        progpowz::hash256 garbage{};
        for (auto& b : garbage.bytes) b = 0xAB;
        check(mining::epoch_from_seed_hash(garbage) == -1, "an unrecognized seed hash returns -1, not a false match");
    }

    // ---- hash_meets_target ----
    {
        progpowz::hash256 small{}, big{}, target{};
        small.bytes[0] = 0x01;
        big.bytes[0] = 0xFF;
        target.bytes[0] = 0x80;
        check(mining::hash_meets_target(small, target), "a numerically smaller hash meets a larger target");
        check(!mining::hash_meets_target(big, target), "a numerically larger hash does not meet a smaller target");
        check(mining::hash_meets_target(target, target), "a hash exactly equal to the target meets it (<=)");
    }

    // ---- parse_job: hand-written literal, same approach as the protocol self-test ----
    {
        network::JobNotification job;
        job.job_id = "0x" + std::string(64, '1');
        std::string pow_hex = "0x" + std::string(64, '1');
        std::string seed_hex = "0x" + std::string(64, '2');
        std::string target_hex = "0x" + std::string(64, '3');
        std::string height_hex = "0x0000000000002710";  // 10000
        job.raw_payload = "{\"jsonrpc\":\"2.0\",\"result\":[\"" + pow_hex + "\",\"" + seed_hex + "\",\"" +
            target_hex + "\",\"" + height_hex + "\"]}";

        auto parsed = mining::parse_job(job);
        check(parsed.has_value(), "parse_job succeeds on a real-shaped work notification");
        if (parsed)
        {
            check(parsed->block_number == 10000, "parse_job extracts height correctly");
            bool pow_ok = true;
            for (int i = 0; i < 32; ++i) if (parsed->pow_hash.bytes[i] != 0x11) pow_ok = false;
            check(pow_ok, "parse_job extracts pow_hash correctly (straight decode)");
        }

        network::JobNotification not_work;
        not_work.raw_payload = R"({"jsonrpc":"2.0","id":1,"result":true})";
        check(!mining::parse_job(not_work).has_value(), "parse_job rejects a non-work message (e.g. a login ack)");
    }

    // ---- build_share_submission ----
    {
        network::JobNotification job;
        job.job_id = "0x" + std::string(64, 'a');
        progpowz::hash256 mix{};
        for (int i = 0; i < 32; ++i) mix.bytes[i] = static_cast<uint8_t>(0xC0 + i);

        auto sub = mining::build_share_submission(job, 0x123456789abcdef0ULL, mix, 2);
        check(sub.job_id == job.job_id, "build_share_submission carries job_id through unchanged");
        check(sub.nonce_hex == "0x123456789abcdef0", "build_share_submission formats nonce_hex correctly");
        check(sub.result_hex == network::zano_stratum::hex_encode(mix.bytes, 32),
            "build_share_submission formats result_hex as straight-order hex of mix_hash");
        check(sub.device_index == 2, "build_share_submission carries device_index through");
    }

    if (g_failures == 0)
    {
        std::printf("\nALL CHECKS PASSED: progpowz_work matches the real reference implementation "
                     "and produces correctly-shaped submissions.\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED.\n", g_failures);
    return 1;
}
