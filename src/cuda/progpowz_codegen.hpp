#pragma once

// ProgPowZ per-period program trace + CUDA source generator.
//
// WHY THIS EXISTS: real production ProgPoW miners (see the vendored-
// algorithm-matching hyle-team/progminer CUDA implementation, and the
// ProgPoW spec itself) do NOT interpret the mix program at runtime on
// every hash. Instead, because `base_rng`/`dst_seq`/`src_seq` (see
// progpowz_portable.hpp's progpowz_hash_light) depend ONLY on
// `block_number` - not on nonce, lane, or any mix register value - the
// entire sequence of "which register reads which register, via which
// literal operation" for all 64 rounds is fully determined once per
// `period_length` (50) blocks. Real miners resolve that sequence on the
// CPU once per period and compile a kernel with the sequence baked in as
// literal, unrolled code - no dst_seq/src_seq array indexing, no per-hash
// kiss99 draws for op selection, no runtime switch on selector - so the
// only per-hash runtime work left is the actual data flow (register
// values, DAG lookups, item-index shuffles), and the compiler can
// register-allocate a short, FIXED program instead of a generic
// interpreter loop. This is understood to be a substantial real
// contributor to why mature miners outperform a generic interpreter
// kernel like the one in progpowz_kernel.cu - see src/cuda/README.md's
// "next steps" section for the sourced research behind this file.
//
// SCOPE / STATUS: this file is host-only C++ (no CUDA toolchain
// dependency) and has two purposes, in order:
//   1. `generate_progpowz_trace()` + `trace_interpret_hash()`: prove, on
//      the CPU alone (no GPU needed), that the per-period operation
//      sequence this file resolves is EXACTLY the sequence
//      progpowz_hash_light() would have executed for that block_number -
//      byte-for-byte hash equality across many (block_number, nonce)
//      pairs is the correctness gate (see tools/codegen_selftest). This
//      must pass before trusting anything generated from the same trace
//      for the GPU.
//   2. `generate_progpowz_cuda_round_source()`: emit literal CUDA device
//      source text for one period's resolved program, meant to be
//      compiled at runtime via NVRTC (see the (separate, GPU-toolchain-
//      dependent, NOT YET REAL-HARDWARE-VALIDATED) NVRTC integration this
//      unlocks). Do not treat generated CUDA source as correct merely
//      because it was derived from a validated trace - it still needs a
//      real compile + a real on-GPU comparison against the existing,
//      already-validated warp kernel before being trusted, exactly like
//      every other GPU-facing change in this project.

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "progpowz_portable.hpp"

namespace deepcore::progpowz {

enum class TraceOpKind : uint8_t { Cache, Math, ItemMerge };

// One resolved mix-program operation. All fields except `kind`/`round`
// are only meaningful for the relevant kind; kept as a flat struct
// (rather than a tagged union) for simplicity - this is a host-only,
// generated-once-per-period, small (~2304 entries) structure, not
// something performance-sensitive itself.
struct TraceOp
{
    TraceOpKind kind = TraceOpKind::Cache;
    uint32_t round = 0;

    // Cache: mix[l][dst] = merge(mix[l][dst], l1_cache_words[mix[l][src] % l1_cache_num_items], ...)
    // Math:  data = math(mix[l][src1], mix[l][src2]); mix[l][dst] = merge(mix[l][dst], data, ...)
    // ItemMerge: mix[l][dst] = merge(mix[l][dst], item_word(l, round, word_index), ...)
    uint32_t dst = 0;
    uint32_t src = 0;   // Cache only
    uint32_t src1 = 0;  // Math only
    uint32_t src2 = 0;  // Math only
    uint32_t word_index = 0;  // ItemMerge only, 0..3

    uint32_t math_case = 0;   // Math only: random_math's `selector % 11`, already resolved
    uint32_t merge_case = 0;  // Cache/Math/ItemMerge: random_merge's `selector % 4`, already resolved
    uint32_t merge_x = 0;     // Cache/Math/ItemMerge: random_merge's rotate amount, already resolved
};

// Derives base_rng/dst_seq/src_seq exactly as progpowz_hash_light does,
// then walks the SAME loop structure (same iteration counts, same
// dst_counter/src_counter interleaving) - but instead of applying each
// operation to live register data, records the fully-resolved decision.
// Deterministic and pure: same block_number always yields the same
// trace, independent of nonce/lane - this is the whole point.
inline std::vector<TraceOp> generate_progpowz_trace(int block_number)
{
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
    constexpr size_t num_words_per_lane = 256 / (4 * num_lanes);

    std::vector<TraceOp> trace;
    trace.reserve(64 * (num_cache_accesses + num_math_operations + num_words_per_lane));

    for (uint32_t r = 0; r < 64; ++r)
    {
        kiss99_state rng = base_rng;
        size_t dst_counter = 0, src_counter = 0;

        for (int i = 0; i < max_operations; ++i)
        {
            if (i < num_cache_accesses)
            {
                const uint32_t src = src_seq[(src_counter++) % num_regs];
                const uint32_t dst = dst_seq[(dst_counter++) % num_regs];
                const uint32_t sel = kiss99_next(rng);
                TraceOp op;
                op.kind = TraceOpKind::Cache;
                op.round = r;
                op.dst = dst;
                op.src = src;
                op.merge_case = sel % 4;
                op.merge_x = (sel >> 16) % 31 + 1;
                trace.push_back(op);
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
                TraceOp op;
                op.kind = TraceOpKind::Math;
                op.round = r;
                op.src1 = src1;
                op.src2 = src2;
                op.math_case = sel1 % 11;
                op.dst = dst;
                op.merge_case = sel2 % 4;
                op.merge_x = (sel2 >> 16) % 31 + 1;
                trace.push_back(op);
            }
        }

        for (size_t i = 0; i < num_words_per_lane; ++i)
        {
            const uint32_t dst = i == 0 ? 0 : dst_seq[(dst_counter++) % num_regs];
            const uint32_t sel = kiss99_next(rng);
            TraceOp op;
            op.kind = TraceOpKind::ItemMerge;
            op.round = r;
            op.dst = dst;
            op.word_index = (uint32_t)i;
            op.merge_case = sel % 4;
            op.merge_x = (sel >> 16) % 31 + 1;
            trace.push_back(op);
        }
    }

    return trace;
}

// random_math, but pre-resolved to a known case (0..10) instead of a
// runtime selector - same switch body as progpowz_portable.hpp's
// random_math(), just keyed directly on the already-computed case.
inline uint32_t trace_apply_math(uint32_t a, uint32_t b, uint32_t math_case)
{
    switch (math_case)
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
        case 0: return (uint32_t)((a == 0 ? 32 : __builtin_clz(a)) + (b == 0 ? 32 : __builtin_clz(b)));
        case 1: return (uint32_t)(__builtin_popcount(a) + __builtin_popcount(b));
    }
}

// random_merge, but pre-resolved to a known case (0..3) and rotate amount
// instead of a runtime selector - same switch body as
// progpowz_portable.hpp's random_merge().
inline void trace_apply_merge(uint32_t& a, uint32_t b, uint32_t merge_case, uint32_t x)
{
    switch (merge_case)
    {
        case 0: a = (a * 33) + b; break;
        case 1: a = (a ^ b) * 33; break;
        case 2: a = rol32(a, x) ^ b; break;
        case 3: a = ((a >> x) | (a << (32 - x))) ^ b; break;
    }
}

// Executes a trace from generate_progpowz_trace() to compute a full
// ProgPowZ hash - CPU-only, used ONLY to validate that the trace really
// does capture progpowz_hash_light()'s exact behavior (see
// tools/codegen_selftest). Mirrors progpowz_hash_light()'s structure
// exactly except the inner per-round operation selection is replaced by
// consuming pre-resolved TraceOp entries instead of re-deriving them from
// dst_seq/src_seq/kiss99 - this IS the same computation, restructured, not
// an independent reimplementation, since it delegates the resolution
// logic to generate_progpowz_trace() (which uses the exact same
// derivation as progpowz_hash_light). The value of this function is
// proving those two independently-invoked pieces (trace generation, then
// trace application) recombine to the identical byte-for-byte result -
// exactly the property real per-period-compiled kernels also need to
// hold, before any GPU work is trusted.
inline progpowz_result trace_interpret_hash(
    const std::vector<TraceOp>& trace, const hash512* light_cache, int64_t light_cache_num_items,
    const uint32_t* l1_cache_words, uint32_t full_dataset_num_items, const hash256& header_hash, uint64_t nonce,
    const hash2048* full_dataset = nullptr)
{
    const uint64_t seed = keccak_progpow_64(header_hash, nonce);

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

    constexpr size_t num_words_per_lane = 256 / (4 * num_lanes);

    size_t cursor = 0;
    for (uint32_t r = 0; r < 64; ++r)
    {
        const uint32_t num_items = full_dataset_num_items / 2;
        const uint32_t item_index = mix[r % num_lanes][0] % num_items;
        hash2048 item = full_dataset ? full_dataset[item_index]
                                      : calculate_dataset_item_2048(light_cache, light_cache_num_items, item_index);

        while (cursor < trace.size() && trace[cursor].round == r)
        {
            const TraceOp& op = trace[cursor];
            if (op.kind == TraceOpKind::Cache)
            {
                for (size_t l = 0; l < num_lanes; ++l)
                {
                    const size_t offset = mix[l][op.src] % l1_cache_num_items;
                    trace_apply_merge(mix[l][op.dst], l1_cache_words[offset], op.merge_case, op.merge_x);
                }
            }
            else if (op.kind == TraceOpKind::Math)
            {
                for (size_t l = 0; l < num_lanes; ++l)
                {
                    const uint32_t data = trace_apply_math(mix[l][op.src1], mix[l][op.src2], op.math_case);
                    trace_apply_merge(mix[l][op.dst], data, op.merge_case, op.merge_x);
                }
            }
            else
            {
                for (size_t l = 0; l < num_lanes; ++l)
                {
                    const size_t item_offset = ((l ^ r) % num_lanes) * num_words_per_lane;
                    const uint32_t word = load_le32(item.bytes + (item_offset + op.word_index) * 4);
                    trace_apply_merge(mix[l][op.dst], word, op.merge_case, op.merge_x);
                }
            }
            ++cursor;
        }
    }

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

// Emits a literal, fully-unrolled CUDA __device__ function body
// implementing one lane's per-round mix program for this trace, with
// every dst/src register index and every merge/math case already
// resolved to a literal - no runtime dst_seq/src_seq array indexing, no
// per-hash kiss99 draws for op selection, no runtime switch on a
// selector. Meant to be spliced into a full kernel source (header
// boilerplate + this body + launch code) and compiled via NVRTC once per
// period. This function ONLY emits text - it does not compile or run
// anything, and generated text is NOT validated merely by having been
// derived from a validated trace (see this file's header comment).
//
// Mirrors progpowz_hash_warp_full_dag's per-lane structure: operates on a
// single lane's `uint32_t r[num_regs]` register array, with the item
// fetch/shuffle machinery left as genuine runtime code (round is a
// compile-time-known literal per unrolled block, but the actual DAG
// lookup and cross-lane data exchange are real per-hash, per-round
// runtime work that cannot be precomputed).
inline std::string generate_progpowz_cuda_round_source(const std::vector<TraceOp>& trace)
{
    std::ostringstream out;
    out << "// Generated by progpowz_codegen.hpp::generate_progpowz_cuda_round_source -\n"
           "// DO NOT EDIT BY HAND. One period's fully-resolved ProgPowZ mix program,\n"
           "// unrolled per round with literal register indices and literal\n"
           "// merge/math cases baked in (see progpowz_codegen.hpp's header comment).\n";

    uint32_t current_round = UINT32_MAX;
    size_t item_merge_seen_this_round = 0;

    auto emit_merge = [&](const char* dst_expr, const std::string& value_expr, uint32_t merge_case, uint32_t x) {
        switch (merge_case)
        {
            case 0: out << dst_expr << " = (" << dst_expr << " * 33u) + (" << value_expr << ");\n"; break;
            case 1: out << dst_expr << " = ((" << dst_expr << " ^ (" << value_expr << ")) * 33u);\n"; break;
            case 2:
                out << dst_expr << " = (__funnelshift_l(" << dst_expr << ", " << dst_expr << ", " << x
                    << ") ^ (" << value_expr << "));\n";
                break;
            case 3:
                out << dst_expr << " = (__funnelshift_r(" << dst_expr << ", " << dst_expr << ", " << x
                    << ") ^ (" << value_expr << "));\n";
                break;
        }
    };

    for (const auto& op : trace)
    {
        if (op.round != current_round)
        {
            if (current_round != UINT32_MAX)
                out << "    }\n";
            current_round = op.round;
            item_merge_seen_this_round = 0;
            out << "    {  // round " << current_round << "\n";
            out << "        const uint32_t reg0_broadcast = __shfl_sync(mask, r[0], group_base_lane + "
                << (current_round % num_lanes) << ");\n";
            out << "        const uint32_t item_index = reg0_broadcast % (full_dataset_num_items / 2);\n";
            out << "        const hash2048* item_ptr = &full_dataset[item_index];\n";
            out << "        uint32_t my_slice[4];\n";
            out << "        {\n";
            out << "            const uint32_t my_slice_byte_offset = (uint32_t)lane * 16u;\n";
            out << "            for (int _i = 0; _i < 4; ++_i)\n";
            out << "                my_slice[_i] = load_le32(item_ptr->bytes + my_slice_byte_offset + _i * "
                   "4);\n";
            out << "        }\n";
        }

        std::ostringstream dst_expr;
        dst_expr << "r[" << op.dst << "]";
        const std::string dst = dst_expr.str();

        if (op.kind == TraceOpKind::Cache)
        {
            std::ostringstream value_expr;
            value_expr << "l1_cache_words[r[" << op.src << "] % l1_cache_num_items]";
            emit_merge(dst.c_str(), value_expr.str(), op.merge_case, op.merge_x);
        }
        else if (op.kind == TraceOpKind::Math)
        {
            out << "        uint32_t _math = ";
            switch (op.math_case)
            {
                case 2: out << "r[" << op.src1 << "] + r[" << op.src2 << "];\n"; break;
                case 3: out << "r[" << op.src1 << "] * r[" << op.src2 << "];\n"; break;
                case 4:
                    out << "(uint32_t)(((uint64_t)r[" << op.src1 << "] * (uint64_t)r[" << op.src2
                        << "]) >> 32);\n";
                    break;
                case 5: out << "r[" << op.src1 << "] < r[" << op.src2 << "] ? r[" << op.src1 << "] : r["
                             << op.src2 << "];\n"; break;
                case 6: out << "__funnelshift_l(r[" << op.src1 << "], r[" << op.src1 << "], r[" << op.src2
                             << "] & 31u);\n"; break;
                case 7: out << "__funnelshift_r(r[" << op.src1 << "], r[" << op.src1 << "], r[" << op.src2
                             << "] & 31u);\n"; break;
                case 8: out << "r[" << op.src1 << "] & r[" << op.src2 << "];\n"; break;
                case 9: out << "r[" << op.src1 << "] | r[" << op.src2 << "];\n"; break;
                case 10: out << "r[" << op.src1 << "] ^ r[" << op.src2 << "];\n"; break;
                case 0: out << "(uint32_t)(__clz((int)r[" << op.src1 << "]) + __clz((int)r[" << op.src2
                             << "]));\n"; break;
                case 1: out << "(uint32_t)(__popc(r[" << op.src1 << "]) + __popc(r[" << op.src2
                             << "]));\n"; break;
                default: out << "r[" << op.src1 << "] + r[" << op.src2 << "];\n"; break;
            }
            emit_merge(dst.c_str(), "_math", op.merge_case, op.merge_x);
        }
        else  // ItemMerge
        {
            if (item_merge_seen_this_round == 0)
            {
                out << "        const int _needed_slice = (int)(((uint32_t)lane ^ " << current_round
                    << "u) % " << num_lanes << "u);\n";
            }
            std::ostringstream value_expr;
            value_expr << "__shfl_sync(mask, my_slice[" << op.word_index << "], group_base_lane + _needed_slice)";
            emit_merge(dst.c_str(), value_expr.str(), op.merge_case, op.merge_x);
            ++item_merge_seen_this_round;
        }
    }
    if (current_round != UINT32_MAX)
        out << "    }\n";

    return out.str();
}

}  // namespace deepcore::progpowz
