#pragma once

// deepcore::mining - glue between a pool job (deepcore::network::
// JobNotification, as delivered by ZanoStratumClient) and the ProgPowZ
// hashing core (deepcore::progpowz, src/cuda/progpowz_portable.hpp).
//
// This is coin-specific logic deliberately kept separate from both the
// generic pool-client interface (stratum_client.hpp doesn't know what a
// "job" contains) and the generic hashing core (progpowz_portable.hpp
// doesn't know about pool wire formats). Nothing here talks to a socket or
// a GPU directly - it is pure data transformation, testable in isolation
// (see tools/mining_selftest), same discipline as every other layer in
// this project.

#include <cstdint>
#include <optional>

#include "../cuda/progpowz_portable.hpp"
#include "../network/stratum_client.hpp"

namespace deepcore::mining {

// A job as it matters to the ProgPowZ algorithm - the fields extracted
// from a JobNotification's raw_payload (see zano_stratum_protocol.hpp's
// WorkPayload, which this is built from).
struct ProgPowZJob {
    int block_number;
    progpowz::hash256 pow_hash;          // == the header_hash progpowz_hash_light expects
    progpowz::hash256 seed_hash;         // identifies the DAG epoch
    progpowz::hash256 target_boundary;   // a share is valid iff final_hash <= this
};

// Parses a JobNotification's raw_payload (the raw JSON text
// ZanoStratumClient passes through opaquely) back into a ProgPowZJob.
// Returns std::nullopt if the payload isn't a recognized work message.
std::optional<ProgPowZJob> parse_job(const network::JobNotification& job);

// True iff `final_hash` <= `target_boundary`, both read as big-endian
// 256-bit integers (byte[0] most significant) - this is the exact
// share-acceptance rule zanod itself uses (ethash::is_less_or_equal in
// third_party/ethash-progpow-ref/ethash-internal.hpp), confirmed
// equivalent to a plain lexicographic byte compare before being
// implemented this way.
bool hash_meets_target(const progpowz::hash256& final_hash, const progpowz::hash256& target_boundary);

// Keccak-256 (sponge, 32-byte input, 32-byte output) - a DIFFERENT
// construction from the Keccak-512 already in progpowz_portable.hpp
// (different rate/padding), needed only for epoch_from_seed_hash below.
// Matches ethash_keccak256_32 in the vendored reference exactly (verified
// in tools/mining_selftest by calling that real function directly, since
// it's a header-only inline in stratum_helpers.h).
progpowz::hash256 keccak256_32(const progpowz::hash256& input);

// Determines the Ethash/ProgPoW epoch number for a given seed_hash, by the
// same method zanod itself uses (stratum::epoch_by_seedhash in
// stratum_helpers.h): epoch 0's seed is all-zero bytes, and each next
// epoch's seed is keccak256 of the previous one; search up to 2016 epochs.
// Returns -1 if no match is found in that range (should not happen for any
// real, current job).
int epoch_from_seed_hash(const progpowz::hash256& seed_hash);

// Formats a found nonce/mix_hash pair into a ShareSubmission ready for
// IStratumClient::submit_share(). nonce_hex is plain "0x" + 16 lowercase
// hex digits in natural (big-endian-style) numeric notation; result_hex is
// mix_hash in straight byte order - both match exactly what
// ZanoStratumClient::submit_share expects to parse (see that file).
network::ShareSubmission build_share_submission(const network::JobNotification& job,
    std::uint64_t nonce, const progpowz::hash256& mix_hash, std::uint32_t device_index = 0);

}  // namespace deepcore::mining
