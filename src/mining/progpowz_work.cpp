#include "progpowz_work.hpp"

#include <cstdio>
#include <cstring>

#include "../network/zano_stratum_protocol.hpp"

namespace deepcore::mining {

namespace {

progpowz::hash256 to_progpowz_hash(const std::array<std::uint8_t, 32>& b)
{
    progpowz::hash256 h;
    std::memcpy(h.bytes, b.data(), 32);
    return h;
}

}  // namespace

std::optional<ProgPowZJob> parse_job(const network::JobNotification& job)
{
    auto parsed = network::zano_stratum::parse_message(job.raw_payload);
    if (!parsed || parsed->kind != network::zano_stratum::MessageKind::Work)
        return std::nullopt;

    ProgPowZJob out;
    out.block_number = static_cast<int>(parsed->work.height);
    out.pow_hash = to_progpowz_hash(parsed->work.pow_hash);
    out.seed_hash = to_progpowz_hash(parsed->work.seed_hash);
    out.target_boundary = to_progpowz_hash(parsed->work.target_boundary);
    return out;
}

bool hash_meets_target(const progpowz::hash256& final_hash, const progpowz::hash256& target_boundary)
{
    // Equivalent to ethash::is_less_or_equal() - see this file's header
    // comment for the derivation of why a plain byte-lexicographic compare
    // (bytes read as a big-endian 256-bit integer, byte[0] most
    // significant) matches that function's word64-by-word64 comparison.
    return std::memcmp(final_hash.bytes, target_boundary.bytes, 32) <= 0;
}

progpowz::hash256 keccak256_32(const progpowz::hash256& input)
{
    // Single-block Keccak-f[1600] sponge absorb, rate = (1600 - 2*256)/8 =
    // 136 bytes = 17 words. Padding derivation and word positions are
    // explained in this file's header comment / the commit that added
    // this function - verified against the real ethash_keccak256_32 in
    // tools/mining_selftest.
    uint64_t state[25] = {};
    for (int i = 0; i < 4; ++i)
        state[i] = progpowz::load_le64(input.bytes + i * 8);
    state[4] ^= 0x1ULL;
    state[16] ^= 0x8000000000000000ULL;
    progpowz::keccakf1600(state);

    progpowz::hash256 out;
    for (int i = 0; i < 4; ++i)
        progpowz::store_le64(out.bytes + i * 8, state[i]);
    return out;
}

int epoch_from_seed_hash(const progpowz::hash256& seed_hash)
{
    progpowz::hash256 epoch_seed{};
    for (int i = 0; i < 2016; ++i)
    {
        if (std::memcmp(seed_hash.bytes, epoch_seed.bytes, 32) == 0)
            return i;
        epoch_seed = keccak256_32(epoch_seed);
    }
    return -1;
}

network::ShareSubmission build_share_submission(const network::JobNotification& job,
    std::uint64_t nonce, const progpowz::hash256& mix_hash, std::uint32_t device_index)
{
    network::ShareSubmission sub;
    sub.job_id = job.job_id;

    char nonce_buf[19];  // "0x" + 16 hex digits + NUL
    std::snprintf(nonce_buf, sizeof(nonce_buf), "0x%016llx", static_cast<unsigned long long>(nonce));
    sub.nonce_hex = nonce_buf;

    sub.result_hex = network::zano_stratum::hex_encode(mix_hash.bytes, 32);
    sub.device_index = device_index;
    return sub;
}

}  // namespace deepcore::mining
