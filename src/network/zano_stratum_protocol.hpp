#pragma once

// deepcore::network::zano_stratum - wire-format encode/decode for Zano's
// daemon-native "stratum" protocol (the eth_getWork/eth_submitWork/
// eth_submitLogin/eth_submitHashrate JSON-RPC-over-TCP dialect implemented
// by zanod's own src/stratum/stratum_server.cpp - NOT Bitcoin-style
// Stratum V1). Every message shape, hex-encoding rule, and framing detail
// here was read directly from that file (see comments below for exact
// line-level provenance) rather than assumed from a generic spec, since
// getting these details wrong produces silent protocol mismatches, not
// compile errors.
//
// This header is pure encode/decode - it has NO socket/networking code and
// makes NO assumption about how bytes arrive or get sent. It is the layer
// tools/protocol_selftest validates without any live daemon or network
// access, the same "prove the logic before adding I/O" approach used for
// the ProgPowZ CUDA kernel (see src/cuda/README.md). A real IStratumClient
// implementation (src/network/stratum_client.hpp) is expected to be a thin
// socket-handling wrapper around these functions, not to re-implement the
// wire format itself.
//
// Important asymmetry to know before using this: zanod's own
// send_response_default() (result:true) is used for BOTH a genuinely
// accepted share AND a silently-dropped stale share (see handle_work() in
// stratum_server.cpp) - the wire response cannot distinguish the two.
// Detecting "stale" is therefore the caller's responsibility (compare the
// job_id a share was computed against to the current job at submit time),
// not something this parser can report from the response alone.

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

namespace deepcore::network::zano_stratum {

// -----------------------------------------------------------------------------
// Hex encoding - direct behavioral port of zanod's src/stratum/stratum_helpers.h
// pod_to_net_format() / pod_to_net_format_reverse() / pod_from_net_format() /
// pod_from_net_format_reverse(): "0x" + lowercase hex, and a decoder that
// accepts the "0x" prefix optionally and both upper/lowercase hex digits
// (matching that file's hexmap_backward table, which maps both 'A'-'F' and
// 'a'-'f' to 10-15).
// -----------------------------------------------------------------------------

// Straight byte order: byte[0] becomes the first two hex characters.
std::string hex_encode(const uint8_t* data, size_t len);

// Reversed byte order: byte[len-1] becomes the first two hex characters.
// Used for fields zanod encodes with pod_to_net_format_reverse (nonce,
// target boundary, block height).
std::string hex_encode_reversed(const uint8_t* data, size_t len);

// Decodes into exactly `len` bytes, straight order. Returns false on wrong
// length (after stripping an optional "0x") or a non-hex character.
bool hex_decode(const std::string& s, uint8_t* out, size_t len);

// Reversed-order counterpart of hex_decode.
bool hex_decode_reversed(const std::string& s, uint8_t* out, size_t len);

// -----------------------------------------------------------------------------
// Request builders
//
// Each returns a complete JSON-RPC 2.0 request, terminated with "\n" - the
// trailing LF is not part of the JSON-RPC spec but zanod's own
// send_response()/send_notification() always emit one "REQUIRED by
// ethminer 0.12 to work" (stratum_server.cpp), so this client does the same
// for consistency with what the daemon itself expects to parse (its
// json_helper frames by brace-counting, not by newlines, but emitting the
// LF costs nothing and matches known-working miner behavior).
// -----------------------------------------------------------------------------

// stratum_server.cpp handle_method_eth_submitLogin(): params = [user, pass],
// worker sent as a top-level "worker" field (not in params). `user` must
// not itself contain '-' unless intentionally encoding a start-difficulty
// suffix (zanod's own convention: "address-1000000" sets start difficulty
// 1000000) - this function does not add or strip that suffix for you.
std::string build_submit_login(int64_t id, const std::string& user, const std::string& pass, const std::string& worker);

// stratum_server.cpp handle_method_eth_getWork(): no params.
std::string build_get_work(int64_t id);

// stratum_server.cpp handle_method_eth_submitWork(): params = [nonce
// (reversed hex), header_hash (straight hex), mix_hash (straight hex)],
// worker as a top-level field. Note: zanod reads but does NOT use mix_hash
// server-side (it recomputes the mix hash itself during verification) -
// this function still sends it since the wire format requires 3 params,
// but its correctness is not load-bearing for share acceptance.
std::string build_submit_work(int64_t id, const std::string& worker, uint64_t nonce,
    const std::array<uint8_t, 32>& header_hash, const std::array<uint8_t, 32>& mix_hash);

// stratum_server.cpp handle_method_eth_submitHashrate(): params =
// [hashrate (reversed hex, any length up to 256 bits but must fit in the
// low 64 bits or zanod rejects it as overflow), rate_submit_id (straight
// hex, an arbitrary caller-chosen identifying hash)].
std::string build_submit_hashrate(int64_t id, uint64_t hashrate_hps, const std::array<uint8_t, 32>& rate_submit_id);

// -----------------------------------------------------------------------------
// Framing
//
// zanod's own incoming-message framing (stratum::json_helper in
// stratum_helpers.h) extracts complete top-level JSON objects from a
// streaming buffer by counting '{'/'}' characters - explicitly NOT by
// splitting on newlines, and explicitly NOT handling braces that appear
// inside a JSON string value ("does not handle curly brackets within
// strings to make things simpler" per that file's own comment). This class
// mirrors that exact behavior for parsing what the daemon sends back: safe
// here because none of this protocol's actual string values (hex, method
// names) ever contain '{' or '}'.
// -----------------------------------------------------------------------------

class MessageFramer {
public:
    // Appends newly-received bytes to the internal buffer.
    void feed(const char* data, size_t len);
    void feed(const std::string& s) { feed(s.data(), s.size()); }

    // Pops the oldest complete JSON object extracted so far, if any.
    // Returns false (leaving `out` untouched) if none is available yet.
    bool pop_object(std::string& out);

private:
    std::string buffer_;
    std::deque<std::string> objects_;
};

// -----------------------------------------------------------------------------
// Response / notification parsing
// -----------------------------------------------------------------------------

// The 4-element array zanod sends both as the eth_getWork response AND as
// an unsolicited notification when the work changes (get_work_json() in
// stratum_server.cpp - the SAME payload shape is used for both, see that
// function's two callers). A caller must handle both delivery paths, not
// just poll eth_getWork - the daemon proactively pushes new work.
struct WorkPayload {
    std::array<uint8_t, 32> pow_hash;        // straight hex
    std::array<uint8_t, 32> seed_hash;       // straight hex
    std::array<uint8_t, 32> target_boundary; // reversed hex
    uint64_t height;                         // reversed hex, 8 bytes
};

enum class MessageKind {
    Unknown,
    // A "result" field holding the 4-element work array - could be either
    // a direct eth_getWork response or an unsolicited work notification;
    // this parser does not and cannot distinguish them (same payload
    // shape), matching the daemon's own dual-use of get_work_json().
    Work,
    // A plain "result":true response (login ack, or share
    // accepted-or-silently-stale - see this header's top-level comment).
    Ok,
    // A JSON-RPC error object.
    Error,
};

struct ParsedMessage {
    MessageKind kind = MessageKind::Unknown;
    std::optional<int64_t> id;      // absent for notifications
    WorkPayload work{};             // valid only when kind == Work
    int64_t error_code = 0;         // valid only when kind == Error
    std::string error_message;      // valid only when kind == Error
};

// Parses one complete JSON object (as produced by MessageFramer::pop_object)
// into a ParsedMessage. Returns std::nullopt if the object does not parse
// as valid JSON or does not match any recognized response shape.
std::optional<ParsedMessage> parse_message(const std::string& json_object);

}  // namespace deepcore::network::zano_stratum
