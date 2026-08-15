#pragma once

// ProgPowZ algorithm core, ported to a portable (host- and device-callable)
// form from the vendored Apache-2.0 reference in
// third_party/ethash-progpow-ref (ethash.cpp, progpow.cpp, keccak.c,
// keccakf1600.c, keccakf800.c - see that directory's LICENSE). Per the
// Apache License's requirement to mark modified files: this is a MODIFIED,
// derivative rendering of that code - restructured into free functions
// callable from both host C++ and CUDA device code, with no algorithmic
// changes. Every function here is deliberately single-threaded / sequential
// (including the per-lane mix loop, which loops over all 16 lanes rather
// than assuming 16 cooperating threads) so it:
//   1. can be compiled and validated on the CPU alone, with zero CUDA
//      toolchain dependency, against tests/reference/progpowz_vectors.txt;
//   2. can then be called from a CUDA kernel with one thread computing one
//      full nonce's hash (the simplest, lowest-risk correctness-first GPU
//      mapping). A lane-cooperative (16 threads/warp via shuffle) version
//      is a later PERFORMANCE milestone, not attempted here - see
//      third_party/ethash-progpow-ref-cuda/README.md.
//
// NOT wired into deepcore-miner. Used only by tools/cuda_selftest.

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define PPZ_HD __host__ __device__
#else
#define PPZ_HD
#endif

namespace deepcore::progpowz {

// ----------------------------------------------------------------------
// Fixed-size hash types (byte-compatible with ethash::hash256/hash512 etc.)
// ----------------------------------------------------------------------

struct hash256 { uint8_t bytes[32]; };
struct hash512 { uint8_t bytes[64]; };

PPZ_HD inline uint32_t load_le32(const uint8_t* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
PPZ_HD inline uint64_t load_le64(const uint8_t* p)
{
    uint64_t lo = load_le32(p);
    uint64_t hi = load_le32(p + 4);
    return lo | (hi << 32);
}
PPZ_HD inline void store_le32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
PPZ_HD inline void store_le64(uint8_t* p, uint64_t v)
{
    store_le32(p, (uint32_t)v);
    store_le32(p + 4, (uint32_t)(v >> 32));
}

// ----------------------------------------------------------------------
// Keccak-f[1600] permutation.
// Direct port of third_party/ethash-progpow-ref/keccakf1600.c
// ("simple" implementation by Ronny Van Keer, as noted in that file).
// ----------------------------------------------------------------------

PPZ_HD inline uint64_t rol64(uint64_t x, unsigned s) { return (x << s) | (x >> (64 - s)); }

PPZ_HD inline void keccakf1600(uint64_t state[25])
{
    static const uint64_t round_constants[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
    };

    uint64_t Aba=state[0],Abe=state[1],Abi=state[2],Abo=state[3],Abu=state[4];
    uint64_t Aga=state[5],Age=state[6],Agi=state[7],Ago=state[8],Agu=state[9];
    uint64_t Aka=state[10],Ake=state[11],Aki=state[12],Ako=state[13],Aku=state[14];
    uint64_t Ama=state[15],Ame=state[16],Ami=state[17],Amo=state[18],Amu=state[19];
    uint64_t Asa=state[20],Ase=state[21],Asi=state[22],Aso=state[23],Asu=state[24];

    uint64_t Eba,Ebe,Ebi,Ebo,Ebu, Ega,Ege,Egi,Ego,Egu, Eka,Eke,Eki,Eko,Eku;
    uint64_t Ema,Eme,Emi,Emo,Emu, Esa,Ese,Esi,Eso,Esu;
    uint64_t Ba,Be,Bi,Bo,Bu, Da,De,Di,Do,Du;

    for (int round = 0; round < 24; round += 2)
    {
        Ba=Aba^Aga^Aka^Ama^Asa; Be=Abe^Age^Ake^Ame^Ase; Bi=Abi^Agi^Aki^Ami^Asi;
        Bo=Abo^Ago^Ako^Amo^Aso; Bu=Abu^Agu^Aku^Amu^Asu;
        Da=Bu^rol64(Be,1); De=Ba^rol64(Bi,1); Di=Be^rol64(Bo,1); Do=Bi^rol64(Bu,1); Du=Bo^rol64(Ba,1);

        Ba=Aba^Da; Be=rol64(Age^De,44); Bi=rol64(Aki^Di,43); Bo=rol64(Amo^Do,21); Bu=rol64(Asu^Du,14);
        Eba=Ba^(~Be&Bi)^round_constants[round]; Ebe=Be^(~Bi&Bo); Ebi=Bi^(~Bo&Bu); Ebo=Bo^(~Bu&Ba); Ebu=Bu^(~Ba&Be);

        Ba=rol64(Abo^Do,28); Be=rol64(Agu^Du,20); Bi=rol64(Aka^Da,3); Bo=rol64(Ame^De,45); Bu=rol64(Asi^Di,61);
        Ega=Ba^(~Be&Bi); Ege=Be^(~Bi&Bo); Egi=Bi^(~Bo&Bu); Ego=Bo^(~Bu&Ba); Egu=Bu^(~Ba&Be);

        Ba=rol64(Abe^De,1); Be=rol64(Agi^Di,6); Bi=rol64(Ako^Do,25); Bo=rol64(Amu^Du,8); Bu=rol64(Asa^Da,18);
        Eka=Ba^(~Be&Bi); Eke=Be^(~Bi&Bo); Eki=Bi^(~Bo&Bu); Eko=Bo^(~Bu&Ba); Eku=Bu^(~Ba&Be);

        Ba=rol64(Abu^Du,27); Be=rol64(Aga^Da,36); Bi=rol64(Ake^De,10); Bo=rol64(Ami^Di,15); Bu=rol64(Aso^Do,56);
        Ema=Ba^(~Be&Bi); Eme=Be^(~Bi&Bo); Emi=Bi^(~Bo&Bu); Emo=Bo^(~Bu&Ba); Emu=Bu^(~Ba&Be);

        Ba=rol64(Abi^Di,62); Be=rol64(Ago^Do,55); Bi=rol64(Aku^Du,39); Bo=rol64(Ama^Da,41); Bu=rol64(Ase^De,2);
        Esa=Ba^(~Be&Bi); Ese=Be^(~Bi&Bo); Esi=Bi^(~Bo&Bu); Eso=Bo^(~Bu&Ba); Esu=Bu^(~Ba&Be);

        Ba=Eba^Ega^Eka^Ema^Esa; Be=Ebe^Ege^Eke^Eme^Ese; Bi=Ebi^Egi^Eki^Emi^Esi;
        Bo=Ebo^Ego^Eko^Emo^Eso; Bu=Ebu^Egu^Eku^Emu^Esu;
        Da=Bu^rol64(Be,1); De=Ba^rol64(Bi,1); Di=Be^rol64(Bo,1); Do=Bi^rol64(Bu,1); Du=Bo^rol64(Ba,1);

        Ba=Eba^Da; Be=rol64(Ege^De,44); Bi=rol64(Eki^Di,43); Bo=rol64(Emo^Do,21); Bu=rol64(Esu^Du,14);
        Aba=Ba^(~Be&Bi)^round_constants[round+1]; Abe=Be^(~Bi&Bo); Abi=Bi^(~Bo&Bu); Abo=Bo^(~Bu&Ba); Abu=Bu^(~Ba&Be);

        Ba=rol64(Ebo^Do,28); Be=rol64(Egu^Du,20); Bi=rol64(Eka^Da,3); Bo=rol64(Eme^De,45); Bu=rol64(Esi^Di,61);
        Aga=Ba^(~Be&Bi); Age=Be^(~Bi&Bo); Agi=Bi^(~Bo&Bu); Ago=Bo^(~Bu&Ba); Agu=Bu^(~Ba&Be);

        Ba=rol64(Ebe^De,1); Be=rol64(Egi^Di,6); Bi=rol64(Eko^Do,25); Bo=rol64(Emu^Du,8); Bu=rol64(Esa^Da,18);
        Aka=Ba^(~Be&Bi); Ake=Be^(~Bi&Bo); Aki=Bi^(~Bo&Bu); Ako=Bo^(~Bu&Ba); Aku=Bu^(~Ba&Be);

        Ba=rol64(Ebu^Du,27); Be=rol64(Ega^Da,36); Bi=rol64(Eke^De,10); Bo=rol64(Emi^Di,15); Bu=rol64(Eso^Do,56);
        Ama=Ba^(~Be&Bi); Ame=Be^(~Bi&Bo); Ami=Bi^(~Bo&Bu); Amo=Bo^(~Bu&Ba); Amu=Bu^(~Ba&Be);

        Ba=rol64(Ebi^Di,62); Be=rol64(Ego^Do,55); Bi=rol64(Eku^Du,39); Bo=rol64(Ema^Da,41); Bu=rol64(Ese^De,2);
        Asa=Ba^(~Be&Bi); Ase=Be^(~Bi&Bo); Asi=Bi^(~Bo&Bu); Aso=Bo^(~Bu&Ba); Asu=Bu^(~Ba&Be);
    }

    state[0]=Aba; state[1]=Abe; state[2]=Abi; state[3]=Abo; state[4]=Abu;
    state[5]=Aga; state[6]=Age; state[7]=Agi; state[8]=Ago; state[9]=Agu;
    state[10]=Aka; state[11]=Ake; state[12]=Aki; state[13]=Ako; state[14]=Aku;
    state[15]=Ama; state[16]=Ame; state[17]=Ami; state[18]=Amo; state[19]=Amu;
    state[20]=Asa; state[21]=Ase; state[22]=Asi; state[23]=Aso; state[24]=Asu;
}

// keccak512() over a <=64-byte input (the only size ProgPowZ ever needs:
// hashing a 64-byte "mix" value in dataset-item generation). Single-block
// sponge absorb, matching third_party/ethash-progpow-ref/keccak.c's
// generic keccak() for this fixed size (block_size for rate=512 is 72
// bytes, so a <=64-byte input always fits in exactly one block).
PPZ_HD inline hash512 keccak512_64(const uint8_t data[64])
{
    uint64_t state[25];
    for (int i = 0; i < 25; ++i) state[i] = 0;
    for (int i = 0; i < 8; ++i) state[i] ^= load_le64(data + i * 8);
    // Padding: last byte of the block gets 0x01 (domain separator), then
    // the final bit of the block is set (0x80...) per keccak.c's fixed-size
    // path (size < word_size after consuming whole words -> last_word logic
    // degenerates to: state_iter already advanced past all 8 words, so the
    // pad byte 0x01 XORs into word 8, and the block-end bit XORs into the
    // last word of the block (index 8 for rate 72 bytes / 8 = 9 words -> index 8)).
    state[8] ^= 0x0000000000000001ULL;
    state[8] ^= 0x8000000000000000ULL;
    keccakf1600(state);
    hash512 out;
    for (int i = 0; i < 8; ++i) store_le64(out.bytes + i * 8, state[i]);
    return out;
}

// ----------------------------------------------------------------------
// Keccak-f[800] permutation (32-bit words), used only for ProgPowZ's
// header/final hashing (keccak_progpow_256). Direct port of
// third_party/ethash-progpow-ref/keccakf800.c.
// ----------------------------------------------------------------------

PPZ_HD inline uint32_t rol32(uint32_t x, unsigned s) { return (x << s) | (x >> (32 - s)); }

// Direct, line-for-line port of
// third_party/ethash-progpow-ref/keccakf800.c (fully unrolled, same
// structure as keccakf1600 above but with 32-bit words, 22 rounds, and its
// own rotation-offset/round-constant tables). An earlier draft of this
// function used a generic pi/rho/chi triple-loop reconstruction instead of
// porting the actual file - that version was WRONG (caught by
// tools/cuda_selftest, which is exactly why that self-test exists). This
// replaces it with a faithful transcription.
PPZ_HD inline void keccakf800(uint32_t state[25])
{
    static const uint32_t round_constants[22] = {
        0x00000001,0x00008082,0x0000808A,0x80008000,0x0000808B,0x80000001,0x80008081,0x00008009,
        0x0000008A,0x00000088,0x80008009,0x8000000A,0x8000808B,0x0000008B,0x00008089,0x00008003,
        0x00008002,0x00000080,0x0000800A,0x8000000A,0x80008081,0x00008080,
    };

    uint32_t Aba=state[0],Abe=state[1],Abi=state[2],Abo=state[3],Abu=state[4];
    uint32_t Aga=state[5],Age=state[6],Agi=state[7],Ago=state[8],Agu=state[9];
    uint32_t Aka=state[10],Ake=state[11],Aki=state[12],Ako=state[13],Aku=state[14];
    uint32_t Ama=state[15],Ame=state[16],Ami=state[17],Amo=state[18],Amu=state[19];
    uint32_t Asa=state[20],Ase=state[21],Asi=state[22],Aso=state[23],Asu=state[24];

    uint32_t Eba,Ebe,Ebi,Ebo,Ebu, Ega,Ege,Egi,Ego,Egu, Eka,Eke,Eki,Eko,Eku;
    uint32_t Ema,Eme,Emi,Emo,Emu, Esa,Ese,Esi,Eso,Esu;
    uint32_t Ba,Be,Bi,Bo,Bu, Da,De,Di,Do,Du;

    for (int round = 0; round < 22; round += 2)
    {
        Ba=Aba^Aga^Aka^Ama^Asa; Be=Abe^Age^Ake^Ame^Ase; Bi=Abi^Agi^Aki^Ami^Asi;
        Bo=Abo^Ago^Ako^Amo^Aso; Bu=Abu^Agu^Aku^Amu^Asu;
        Da=Bu^rol32(Be,1); De=Ba^rol32(Bi,1); Di=Be^rol32(Bo,1); Do=Bi^rol32(Bu,1); Du=Bo^rol32(Ba,1);

        Ba=Aba^Da; Be=rol32(Age^De,12); Bi=rol32(Aki^Di,11); Bo=rol32(Amo^Do,21); Bu=rol32(Asu^Du,14);
        Eba=Ba^(~Be&Bi)^round_constants[round]; Ebe=Be^(~Bi&Bo); Ebi=Bi^(~Bo&Bu); Ebo=Bo^(~Bu&Ba); Ebu=Bu^(~Ba&Be);

        Ba=rol32(Abo^Do,28); Be=rol32(Agu^Du,20); Bi=rol32(Aka^Da,3); Bo=rol32(Ame^De,13); Bu=rol32(Asi^Di,29);
        Ega=Ba^(~Be&Bi); Ege=Be^(~Bi&Bo); Egi=Bi^(~Bo&Bu); Ego=Bo^(~Bu&Ba); Egu=Bu^(~Ba&Be);

        Ba=rol32(Abe^De,1); Be=rol32(Agi^Di,6); Bi=rol32(Ako^Do,25); Bo=rol32(Amu^Du,8); Bu=rol32(Asa^Da,18);
        Eka=Ba^(~Be&Bi); Eke=Be^(~Bi&Bo); Eki=Bi^(~Bo&Bu); Eko=Bo^(~Bu&Ba); Eku=Bu^(~Ba&Be);

        Ba=rol32(Abu^Du,27); Be=rol32(Aga^Da,4); Bi=rol32(Ake^De,10); Bo=rol32(Ami^Di,15); Bu=rol32(Aso^Do,24);
        Ema=Ba^(~Be&Bi); Eme=Be^(~Bi&Bo); Emi=Bi^(~Bo&Bu); Emo=Bo^(~Bu&Ba); Emu=Bu^(~Ba&Be);

        Ba=rol32(Abi^Di,30); Be=rol32(Ago^Do,23); Bi=rol32(Aku^Du,7); Bo=rol32(Ama^Da,9); Bu=rol32(Ase^De,2);
        Esa=Ba^(~Be&Bi); Ese=Be^(~Bi&Bo); Esi=Bi^(~Bo&Bu); Eso=Bo^(~Bu&Ba); Esu=Bu^(~Ba&Be);

        Ba=Eba^Ega^Eka^Ema^Esa; Be=Ebe^Ege^Eke^Eme^Ese; Bi=Ebi^Egi^Eki^Emi^Esi;
        Bo=Ebo^Ego^Eko^Emo^Eso; Bu=Ebu^Egu^Eku^Emu^Esu;
        Da=Bu^rol32(Be,1); De=Ba^rol32(Bi,1); Di=Be^rol32(Bo,1); Do=Bi^rol32(Bu,1); Du=Bo^rol32(Ba,1);

        Ba=Eba^Da; Be=rol32(Ege^De,12); Bi=rol32(Eki^Di,11); Bo=rol32(Emo^Do,21); Bu=rol32(Esu^Du,14);
        Aba=Ba^(~Be&Bi)^round_constants[round+1]; Abe=Be^(~Bi&Bo); Abi=Bi^(~Bo&Bu); Abo=Bo^(~Bu&Ba); Abu=Bu^(~Ba&Be);

        Ba=rol32(Ebo^Do,28); Be=rol32(Egu^Du,20); Bi=rol32(Eka^Da,3); Bo=rol32(Eme^De,13); Bu=rol32(Esi^Di,29);
        Aga=Ba^(~Be&Bi); Age=Be^(~Bi&Bo); Agi=Bi^(~Bo&Bu); Ago=Bo^(~Bu&Ba); Agu=Bu^(~Ba&Be);

        Ba=rol32(Ebe^De,1); Be=rol32(Egi^Di,6); Bi=rol32(Eko^Do,25); Bo=rol32(Emu^Du,8); Bu=rol32(Esa^Da,18);
        Aka=Ba^(~Be&Bi); Ake=Be^(~Bi&Bo); Aki=Bi^(~Bo&Bu); Ako=Bo^(~Bu&Ba); Aku=Bu^(~Ba&Be);

        Ba=rol32(Ebu^Du,27); Be=rol32(Ega^Da,4); Bi=rol32(Eke^De,10); Bo=rol32(Emi^Di,15); Bu=rol32(Eso^Do,24);
        Ama=Ba^(~Be&Bi); Ame=Be^(~Bi&Bo); Ami=Bi^(~Bo&Bu); Amo=Bo^(~Bu&Ba); Amu=Bu^(~Ba&Be);

        Ba=rol32(Ebi^Di,30); Be=rol32(Ego^Do,23); Bi=rol32(Eku^Du,7); Bo=rol32(Ema^Da,9); Bu=rol32(Ese^De,2);
        Asa=Ba^(~Be&Bi); Ase=Be^(~Bi&Bo); Asi=Bi^(~Bo&Bu); Aso=Bo^(~Bu&Ba); Asu=Bu^(~Ba&Be);
    }

    state[0]=Aba; state[1]=Abe; state[2]=Abi; state[3]=Abo; state[4]=Abu;
    state[5]=Aga; state[6]=Age; state[7]=Agi; state[8]=Ago; state[9]=Agu;
    state[10]=Aka; state[11]=Ake; state[12]=Aki; state[13]=Ako; state[14]=Aku;
    state[15]=Ama; state[16]=Ame; state[17]=Ami; state[18]=Amo; state[19]=Amu;
    state[20]=Asa; state[21]=Ase; state[22]=Asi; state[23]=Aso; state[24]=Asu;
}

// Packs header_hash(32B) + nonce(8B) + mix_hash(32B) into a 25x32-bit
// Keccak-f[800] state, permutes once, and returns the first 32 output
// bytes - matching progpow.cpp's keccak_progpow_256 exactly (no sponge
// padding: ProgPowZ always feeds exactly 72 bytes = 18 words here).
PPZ_HD inline hash256 keccak_progpow_256(const hash256& header_hash, uint64_t nonce, const hash256& mix_hash)
{
    uint32_t state[25];
    for (int i = 0; i < 25; ++i) state[i] = 0;

    for (int i = 0; i < 8; ++i)
        state[i] = load_le32(header_hash.bytes + i * 4);
    state[8] = (uint32_t)nonce;
    state[9] = (uint32_t)(nonce >> 32);
    for (int i = 0; i < 8; ++i)
        state[10 + i] = load_le32(mix_hash.bytes + i * 4);

    keccakf800(state);

    hash256 out;
    for (int i = 0; i < 8; ++i)
        store_le32(out.bytes + i * 4, state[i]);
    return out;
}

PPZ_HD inline uint64_t keccak_progpow_64(const hash256& header_hash, uint64_t nonce)
{
    hash256 null_mix{};
    for (auto& b : null_mix.bytes) b = 0;
    hash256 h = keccak_progpow_256(header_hash, nonce, null_mix);
    // be::uint64(h.word64s[0]) in the reference: big-endian read of the
    // first 8 output bytes.
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v = (v << 8) | h.bytes[i];
    return v;
}

// ----------------------------------------------------------------------
// KISS99 PRNG - direct port of third_party/ethash-progpow-ref/kiss99.hpp.
// ----------------------------------------------------------------------

struct kiss99_state { uint32_t z, w, jsr, jcong; };

PPZ_HD inline uint32_t kiss99_next(kiss99_state& s)
{
    s.z = 36969 * (s.z & 65535) + (s.z >> 16);
    s.w = 18000 * (s.w & 65535) + (s.w >> 16);
    uint32_t mwc = ((s.z << 16) + s.w);
    s.jsr ^= (s.jsr << 17);
    s.jsr ^= (s.jsr >> 13);
    s.jsr ^= (s.jsr << 5);
    s.jcong = 69069 * s.jcong + 1234567;
    return (mwc ^ s.jcong) + s.jsr;
}

// ----------------------------------------------------------------------
// FNV1/FNV1a mixing - from bit_manipulation.h.
// ----------------------------------------------------------------------

constexpr uint32_t fnv_prime = 0x01000193;
constexpr uint32_t fnv_offset_basis = 0x811c9dc5;

PPZ_HD inline uint32_t fnv1(uint32_t u, uint32_t v) { return (u * fnv_prime) ^ v; }
PPZ_HD inline uint32_t fnv1a(uint32_t u, uint32_t v) { return (u ^ v) * fnv_prime; }

// ----------------------------------------------------------------------
// ProgPowZ constants (must match ethash/progpow.hpp exactly).
// ----------------------------------------------------------------------

constexpr int period_length = 50;
constexpr uint32_t num_regs = 32;
constexpr size_t num_lanes = 16;
constexpr int num_cache_accesses = 12;
constexpr int num_math_operations = 20;
constexpr size_t l1_cache_num_items = (16 * 1024) / sizeof(uint32_t);

// ----------------------------------------------------------------------
// Light-cache dataset-item generation - port of ethash.cpp's item_state /
// calculate_dataset_item_2048. `cache` points to the epoch's light cache
// (array of hash512, length num_cache_items), exactly as built by
// ethash::create_epoch_context() on the host.
// ----------------------------------------------------------------------

constexpr int full_dataset_item_parents = 256;

struct item_state_t
{
    const hash512* cache;
    int64_t num_cache_items;
    uint32_t seed;
    hash512 mix;
};

PPZ_HD inline void item_state_init(item_state_t& s, const hash512* cache, int64_t num_cache_items, int64_t index)
{
    s.cache = cache;
    s.num_cache_items = num_cache_items;
    s.seed = (uint32_t)index;
    s.mix = cache[index % num_cache_items];
    uint32_t w0 = load_le32(s.mix.bytes);
    store_le32(s.mix.bytes, w0 ^ s.seed);
    s.mix = keccak512_64(s.mix.bytes);
}

PPZ_HD inline void item_state_update(item_state_t& s, uint32_t round)
{
    constexpr size_t num_words = 16; // 64 bytes / 4
    uint32_t mix_word = load_le32(s.mix.bytes + (round % num_words) * 4);
    uint32_t t = fnv1(s.seed ^ round, mix_word);
    int64_t parent_index = t % s.num_cache_items;
    const hash512& parent = s.cache[parent_index];
    hash512 new_mix;
    for (int i = 0; i < 16; ++i)
    {
        uint32_t a = load_le32(s.mix.bytes + i * 4);
        uint32_t b = load_le32(parent.bytes + i * 4);
        store_le32(new_mix.bytes + i * 4, fnv1(a, b));
    }
    s.mix = new_mix;
}

PPZ_HD inline hash512 item_state_final(item_state_t& s) { return keccak512_64(s.mix.bytes); }

// A 2048-bit (256-byte) dataset item = 4 interleaved 512-bit sub-items.
struct hash2048 { uint8_t bytes[256]; };

// `light_cache_num_items` here is specifically the LIGHT cache's item count
// (ethash_calculate_light_cache_num_items(epoch)) - a different, smaller
// number than the full dataset's item count. Both are needed by callers but
// only the light-cache count is used inside this function, for the
// cache[index % light_cache_num_items] lookups in item_state_init/update.
PPZ_HD inline hash2048 calculate_dataset_item_2048(const hash512* light_cache, int64_t light_cache_num_items, uint32_t index)
{
    item_state_t item0, item1, item2, item3;
    item_state_init(item0, light_cache, light_cache_num_items, (int64_t)index * 4 + 0);
    item_state_init(item1, light_cache, light_cache_num_items, (int64_t)index * 4 + 1);
    item_state_init(item2, light_cache, light_cache_num_items, (int64_t)index * 4 + 2);
    item_state_init(item3, light_cache, light_cache_num_items, (int64_t)index * 4 + 3);

    for (uint32_t j = 0; j < full_dataset_item_parents; ++j)
    {
        item_state_update(item0, j);
        item_state_update(item1, j);
        item_state_update(item2, j);
        item_state_update(item3, j);
    }

    hash2048 out;
    hash512 h0 = item_state_final(item0), h1 = item_state_final(item1);
    hash512 h2 = item_state_final(item2), h3 = item_state_final(item3);
    std::memcpy(out.bytes, h0.bytes, 64);
    std::memcpy(out.bytes + 64, h1.bytes, 64);
    std::memcpy(out.bytes + 128, h2.bytes, 64);
    std::memcpy(out.bytes + 192, h3.bytes, 64);
    return out;
}

// ----------------------------------------------------------------------
// ProgPoW mix RNG state - port of progpow.cpp's mix_rng_state.
// ----------------------------------------------------------------------

PPZ_HD inline void init_dst_src_seq(kiss99_state& rng, uint32_t* dst_seq, uint32_t* src_seq)
{
    for (uint32_t i = 0; i < num_regs; ++i) { dst_seq[i] = i; src_seq[i] = i; }
    for (uint32_t i = num_regs; i > 1; --i)
    {
        uint32_t j = kiss99_next(rng) % i;
        uint32_t tmp = dst_seq[i-1]; dst_seq[i-1] = dst_seq[j]; dst_seq[j] = tmp;
        j = kiss99_next(rng) % i;
        tmp = src_seq[i-1]; src_seq[i-1] = src_seq[j]; src_seq[j] = tmp;
    }
}

// Safe (arbitrary, unmasked shift amount) 32-bit rotates - direct port of
// third_party/ethash-progpow-ref/bit_manipulation.h's rotl32/rotr32. Unlike
// rol32() above (only ever called with fixed, in-range compile-time
// constants inside the Keccak permutations), random_math() below calls
// these with a fully arbitrary runtime uint32_t as the shift amount, which
// would be undefined behavior (shift >= width) without this masking.
PPZ_HD inline uint32_t safe_rotl32(uint32_t n, unsigned c)
{
    c &= 31u;
    unsigned neg_c = (unsigned)(-(int)c);
    return (n << c) | (n >> (neg_c & 31u));
}
PPZ_HD inline uint32_t safe_rotr32(uint32_t n, unsigned c)
{
    c &= 31u;
    unsigned neg_c = (unsigned)(-(int)c);
    return (n >> c) | (n << (neg_c & 31u));
}

PPZ_HD inline uint32_t random_math(uint32_t a, uint32_t b, uint32_t selector)
{
    switch (selector % 11)
    {
        default:
        case 2: return a + b;
        case 3: return a * b;
        case 4: { uint64_t p = (uint64_t)a * (uint64_t)b; return (uint32_t)(p >> 32); }
        case 5: return a < b ? a : b;
        case 6: return safe_rotl32(a, b);
        case 7: return safe_rotr32(a, b);
        case 8: return a & b;
        case 9: return a | b;
        case 10: return a ^ b;
        case 0: {
#if defined(__CUDA_ARCH__)
            int za = __clz((int)a);
            int zb = __clz((int)b);
#else
            int za = a == 0 ? 32 : __builtin_clz(a);
            int zb = b == 0 ? 32 : __builtin_clz(b);
#endif
            return (uint32_t)(za + zb);
        }
        case 1: {
#if defined(__CUDA_ARCH__)
            int pa = __popc(a);
            int pb = __popc(b);
#else
            int pa = __builtin_popcount(a);
            int pb = __builtin_popcount(b);
#endif
            return (uint32_t)(pa + pb);
        }
    }
}

PPZ_HD inline void random_merge(uint32_t& a, uint32_t b, uint32_t selector)
{
    const uint32_t x = (selector >> 16) % 31 + 1;
    switch (selector % 4)
    {
        case 0: a = (a * 33) + b; break;
        case 1: a = (a ^ b) * 33; break;
        case 2: a = rol32(a, x) ^ b; break;
        case 3: a = ((a >> x) | (a << (32 - x))) ^ b; break; // rotr32(a, x) ^ b
    }
}

// ----------------------------------------------------------------------
// Full ProgPowZ hash for one (header_hash, nonce) pair. Single-threaded:
// `mix` is [num_lanes][num_regs], processed lane-by-lane in a loop (no
// cross-thread cooperation) - byte-for-byte equivalent to the CPU
// reference's mix_array, just not vectorized. `light_cache` /
// `num_cache_items` come from the host-built epoch context (light mode:
// dataset items are recomputed on every lookup, matching how
// tests/reference/progpowz_vectors.txt was generated).
// ----------------------------------------------------------------------

struct progpowz_result { hash256 final_hash; hash256 mix_hash; };

// `light_cache` / `light_cache_num_items`: the epoch's light cache, used to
// (re)compute dataset items on demand (light/verification-mode - matches
// how tests/reference/progpowz_vectors.txt was generated).
// `l1_cache_words` / its length is always l1_cache_num_items (4096 u32
// words = 16KiB): the precomputed fast-access cache, i.e. dataset items
// 0..63 (2048 bits each) computed once per epoch - see
// third_party/ethash-progpow-ref/ethash.cpp create_epoch_context(), which
// builds this by calling calculate_dataset_item_2048 for i in
// [0, l1_cache_size/sizeof(hash2048)). The caller (tools/cuda_selftest)
// builds this the same way, reusing the vendored reference library so this
// setup step carries zero re-implementation risk.
// `full_dataset_num_items`: ethash_calculate_full_dataset_num_items(epoch) -
// a distinct value from light_cache_num_items, used only for the main
// per-round item_index modulus.
PPZ_HD inline progpowz_result progpowz_hash_light(
    const hash512* light_cache, int64_t light_cache_num_items,
    const uint32_t* l1_cache_words, uint32_t full_dataset_num_items,
    int block_number, const hash256& header_hash, uint64_t nonce)
{
    const uint64_t seed = keccak_progpow_64(header_hash, nonce);

    // init_mix
    uint32_t mix[num_lanes][num_regs];
    {
        const uint32_t z = fnv1a(fnv_offset_basis, (uint32_t)seed);
        const uint32_t w = fnv1a(z, (uint32_t)(seed >> 32));
        for (uint32_t l = 0; l < num_lanes; ++l)
        {
            const uint32_t jsr = fnv1a(w, l);
            const uint32_t jcong = fnv1a(jsr, l);
            kiss99_state rng{z, w, jsr, jcong};
            for (uint32_t r = 0; r < num_regs; ++r)
                mix[l][r] = kiss99_next(rng);
        }
    }

    // mix_rng_state{block_number / period_length}
    //
    // IMPORTANT (see caller-facing comment above progpowz_hash_light):
    // the reference's round() takes its mix_rng_state parameter BY VALUE
    // (progpow.cpp: `void round(..., mix_rng_state state, ...)`), so every
    // one of the 64 rounds below gets a FRESH COPY of this base state -
    // the RNG/counters do NOT carry over from one round to the next, only
    // within a single round's own operations. `base_rng`/`dst_seq`/
    // `src_seq` are therefore computed once here and never mutated again;
    // each round below copies `base_rng` into a local `round_rng` and
    // starts its own counters at 0.
    kiss99_state base_rng;
    uint32_t dst_seq[num_regs], src_seq[num_regs];
    {
        const uint64_t s = (uint64_t)(block_number / period_length);
        const uint32_t seed_lo = (uint32_t)s, seed_hi = (uint32_t)(s >> 32);
        const auto z = fnv1a(fnv_offset_basis, seed_lo);
        const auto w = fnv1a(z, seed_hi);
        const auto jsr = fnv1a(w, seed_lo);
        const auto jcong = fnv1a(jsr, seed_hi);
        base_rng = kiss99_state{z, w, jsr, jcong};
        init_dst_src_seq(base_rng, dst_seq, src_seq);
    }

    constexpr int max_operations = num_cache_accesses > num_math_operations ? num_cache_accesses : num_math_operations;
    constexpr size_t num_words_per_lane = 256 / (4 * num_lanes); // hash2048 bytes / (4 * lanes)

    for (uint32_t r = 0; r < 64; ++r)
    {
        kiss99_state rng = base_rng;
        size_t dst_counter = 0, src_counter = 0;

        const uint32_t num_items = full_dataset_num_items / 2;
        const uint32_t item_index = mix[r % num_lanes][0] % num_items;
        hash2048 item = calculate_dataset_item_2048(light_cache, light_cache_num_items, item_index);

        for (int i = 0; i < max_operations; ++i)
        {
            if (i < num_cache_accesses)
            {
                const uint32_t src = src_seq[(src_counter++) % num_regs];
                const uint32_t dst = dst_seq[(dst_counter++) % num_regs];
                const uint32_t sel = kiss99_next(rng);
                for (size_t l = 0; l < num_lanes; ++l)
                {
                    const size_t offset = mix[l][src] % l1_cache_num_items;
                    random_merge(mix[l][dst], l1_cache_words[offset], sel);
                }
            }
            if (i < num_math_operations)
            {
                const auto src_rnd = kiss99_next(rng) % (num_regs * (num_regs - 1));
                const auto src1 = src_rnd % num_regs;
                auto src2 = src_rnd / num_regs;
                if (src2 >= src1) ++src2;
                const auto sel1 = kiss99_next(rng);
                const auto dst = dst_seq[(dst_counter++) % num_regs];
                const auto sel2 = kiss99_next(rng);
                for (size_t l = 0; l < num_lanes; ++l)
                {
                    const uint32_t data = random_math(mix[l][src1], mix[l][src2], sel1);
                    random_merge(mix[l][dst], data, sel2);
                }
            }
        }

        uint32_t dsts[num_words_per_lane], sels[num_words_per_lane];
        for (size_t i = 0; i < num_words_per_lane; ++i)
        {
            dsts[i] = i == 0 ? 0 : dst_seq[(dst_counter++) % num_regs];
            sels[i] = kiss99_next(rng);
        }
        for (size_t l = 0; l < num_lanes; ++l)
        {
            const auto offset = ((l ^ r) % num_lanes) * num_words_per_lane;
            for (size_t i = 0; i < num_words_per_lane; ++i)
            {
                const uint32_t word = load_le32(item.bytes + (offset + i) * 4);
                random_merge(mix[l][dsts[i]], word, sels[i]);
            }
        }
    }

    // Reduce mix to final result.
    uint32_t lane_hash[num_lanes];
    for (size_t l = 0; l < num_lanes; ++l)
    {
        lane_hash[l] = fnv_offset_basis;
        for (uint32_t i = 0; i < num_regs; ++i)
            lane_hash[l] = fnv1a(lane_hash[l], mix[l][i]);
    }
    hash256 mix_hash{};
    uint32_t mh[8];
    for (auto& w : mh) w = fnv_offset_basis;
    for (size_t l = 0; l < num_lanes; ++l)
        mh[l % 8] = fnv1a(mh[l % 8], lane_hash[l]);
    for (int i = 0; i < 8; ++i) store_le32(mix_hash.bytes + i * 4, mh[i]);

    hash256 final_hash = keccak_progpow_256(header_hash, seed, mix_hash);
    return {final_hash, mix_hash};
}

}  // namespace deepcore::progpowz
