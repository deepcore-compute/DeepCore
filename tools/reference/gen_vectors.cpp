// ProgPowZ reference test-vector generator.
//
// Links against the vendored, unmodified ProgPoW/Ethash reference code in
// third_party/ethash-progpow-ref (see that directory's README.md for
// provenance). Running this program is how the values recorded in
// tests/reference/progpowz_vectors.txt were produced: they are the
// authoritative "correct answers" that any future independent ProgPowZ
// implementation (including GPU kernels) must reproduce for the same
// inputs.
//
// This tool is not part of deepcore-miner - it has no pool/network code and
// is never linked into the miner binary. It only exists to (re)generate the
// checked-in reference vectors, e.g. after auditing/updating the vendored
// reference code.
#include <cstdio>
#include <cstring>
#include <ethash/progpow.hpp>

namespace {

void print_hex(std::FILE* out, const char* label, const uint8_t* bytes, size_t n)
{
    std::fprintf(out, "%s: 0x", label);
    for (size_t i = 0; i < n; ++i)
        std::fprintf(out, "%02x", bytes[i]);
    std::fprintf(out, "\n");
}

void run_vector(std::FILE* out, const ethash::epoch_context& ctx, int block_number,
    const ethash::hash256& header_hash, uint64_t nonce)
{
    std::fprintf(out, "--- block_number=%d nonce=0x%016llx ---\n", block_number,
        static_cast<unsigned long long>(nonce));
    print_hex(out, "header_hash", header_hash.bytes, 32);
    auto r = progpow::hash(ctx, block_number, header_hash, nonce);
    print_hex(out, "final_hash ", r.final_hash.bytes, 32);
    print_hex(out, "mix_hash   ", r.mix_hash.bytes, 32);
    std::fprintf(out, "\n");
}

}  // namespace

int main()
{
    // Epoch 0 light context: covers block_number 0..29999
    // (ETHASH_EPOCH_LENGTH=30000). Light context builds only the cache, not
    // the full multi-gigabyte dataset - sufficient for generating a small,
    // fixed set of reference vectors.
    auto ctx = ethash::create_epoch_context(0);

    ethash::hash256 zero_hash{};
    std::memset(&zero_hash, 0, sizeof(zero_hash));

    ethash::hash256 pattern_hash{};
    for (int i = 0; i < 32; ++i)
        pattern_hash.bytes[i] = static_cast<uint8_t>(i);

    run_vector(stdout, *ctx, 0, zero_hash, 0);
    run_vector(stdout, *ctx, 0, zero_hash, 1);
    run_vector(stdout, *ctx, 0, pattern_hash, 0x123456789abcdef0ULL);
    run_vector(stdout, *ctx, 12345, pattern_hash, 0x123456789abcdef0ULL);

    return 0;
}
