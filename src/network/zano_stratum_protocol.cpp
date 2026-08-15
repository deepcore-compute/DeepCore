#include "zano_stratum_protocol.hpp"

#include <cstring>

#include <nlohmann/json.hpp>

using nlohmann::json;

namespace deepcore::network::zano_stratum {

namespace {

constexpr char kHexMap[] = "0123456789abcdef";

// Matches stratum_helpers.h's hexmap_backward: '0'-'9' -> 0-9, 'A'-'F' and
// 'a'-'f' -> 10-15, everything else invalid (represented here as -1).
int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Matches stratum_helpers.h's trim_0x() exactly: only a lowercase "0x"
// prefix is recognized and stripped.
std::string trim_0x(const std::string& s)
{
    if (s.size() >= 2 && s[0] == '0' && s[1] == 'x')
        return s.substr(2);
    return s;
}

}  // namespace

std::string hex_encode(const uint8_t* data, size_t len)
{
    std::string s;
    s.reserve(2 + len * 2);
    s += "0x";
    for (size_t i = 0; i < len; ++i)
    {
        s += kHexMap[(data[i] >> 4) & 0xF];
        s += kHexMap[data[i] & 0xF];
    }
    return s;
}

std::string hex_encode_reversed(const uint8_t* data, size_t len)
{
    std::string s;
    s.reserve(2 + len * 2);
    s += "0x";
    for (size_t i = 0; i < len; ++i)
    {
        uint8_t b = data[len - i - 1];
        s += kHexMap[(b >> 4) & 0xF];
        s += kHexMap[b & 0xF];
    }
    return s;
}

bool hex_decode(const std::string& s_in, uint8_t* out, size_t len)
{
    std::string s = trim_0x(s_in);
    if (s.size() != len * 2)
        return false;
    for (size_t i = 0; i < len; ++i)
    {
        int hi = hex_nibble(s[2 * i]);
        int lo = hex_nibble(s[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

bool hex_decode_reversed(const std::string& s_in, uint8_t* out, size_t len)
{
    std::string s = trim_0x(s_in);
    if (s.size() != len * 2)
        return false;
    for (size_t i = 0; i < len; ++i)
    {
        int hi = hex_nibble(s[2 * i]);
        int lo = hex_nibble(s[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return false;
        out[len - i - 1] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

namespace {

uint64_t bytes8_to_u64_native(const std::array<uint8_t, 8>& b)
{
    // hex_decode_reversed already un-reverses network byte order into
    // memory order; on a little-endian host (the only platform this
    // project targets - x86_64/ARM64 Linux and Windows) memory order for a
    // uint64_t IS native order, so a plain memcpy-equivalent read is
    // correct here without a further byte-order conversion step.
    static_assert(sizeof(uint64_t) == 8);
    uint64_t v;
    std::memcpy(&v, b.data(), sizeof(v));
    return v;
}

std::array<uint8_t, 8> u64_to_bytes8_native(uint64_t v)
{
    std::array<uint8_t, 8> b;
    std::memcpy(b.data(), &v, sizeof(v));
    return b;
}

}  // namespace

std::string build_submit_login(int64_t id, const std::string& user, const std::string& pass, const std::string& worker)
{
    json j;
    j["id"] = id;
    j["jsonrpc"] = "2.0";
    j["method"] = "eth_submitLogin";
    j["params"] = json::array({user, pass});
    j["worker"] = worker;
    return j.dump() + "\n";
}

std::string build_get_work(int64_t id)
{
    json j;
    j["id"] = id;
    j["jsonrpc"] = "2.0";
    j["method"] = "eth_getWork";
    j["params"] = json::array();
    return j.dump() + "\n";
}

std::string build_submit_work(int64_t id, const std::string& worker, uint64_t nonce,
    const std::array<uint8_t, 32>& header_hash, const std::array<uint8_t, 32>& mix_hash)
{
    auto nonce_bytes = u64_to_bytes8_native(nonce);
    json j;
    j["id"] = id;
    j["jsonrpc"] = "2.0";
    j["method"] = "eth_submitWork";
    j["params"] = json::array({
        hex_encode_reversed(nonce_bytes.data(), nonce_bytes.size()),
        hex_encode(header_hash.data(), header_hash.size()),
        hex_encode(mix_hash.data(), mix_hash.size()),
    });
    j["worker"] = worker;
    return j.dump() + "\n";
}

std::string build_submit_hashrate(int64_t id, uint64_t hashrate_hps, const std::array<uint8_t, 32>& rate_submit_id)
{
    auto rate_bytes = u64_to_bytes8_native(hashrate_hps);
    json j;
    j["id"] = id;
    j["jsonrpc"] = "2.0";
    j["method"] = "eth_submitHashrate";
    j["params"] = json::array({
        hex_encode_reversed(rate_bytes.data(), rate_bytes.size()),
        hex_encode(rate_submit_id.data(), rate_submit_id.size()),
    });
    return j.dump() + "\n";
}

void MessageFramer::feed(const char* data, size_t len)
{
    buffer_.append(data, len);

    // Mirrors stratum::json_helper::feed() exactly: count '{'/'}' and
    // extract each complete top-level object as soon as its closing brace
    // brings the depth back to zero. Does not special-case braces inside
    // string values - see this file's header comment for why that is safe
    // for this specific protocol's message shapes.
    int brace_count = 0;
    size_t i = 0;
    while (i < buffer_.size())
    {
        char c = buffer_[i];
        if (c == '{')
        {
            ++brace_count;
        }
        else if (c == '}')
        {
            if (--brace_count == 0)
            {
                objects_.push_back(buffer_.substr(0, i + 1));
                buffer_.erase(0, i + 1);
                i = 0;
                continue;
            }
        }
        ++i;
    }
}

bool MessageFramer::pop_object(std::string& out)
{
    if (objects_.empty())
        return false;
    out = std::move(objects_.front());
    objects_.pop_front();
    return true;
}

std::optional<ParsedMessage> parse_message(const std::string& json_object)
{
    json j;
    try
    {
        j = json::parse(json_object);
    }
    catch (const json::parse_error&)
    {
        return std::nullopt;
    }

    if (!j.is_object())
        return std::nullopt;

    ParsedMessage msg;
    if (j.contains("id") && !j["id"].is_null())
    {
        if (j["id"].is_number_integer())
            msg.id = j["id"].get<int64_t>();
        else if (j["id"].is_string())
        {
            // zanod's jsonrpc_id_t permits a string id; parse leniently.
            try { msg.id = std::stoll(j["id"].get<std::string>()); }
            catch (...) { /* leave id unset rather than fail the whole parse */ }
        }
    }

    if (j.contains("error") && j["error"].is_object())
    {
        msg.kind = MessageKind::Error;
        const auto& err = j["error"];
        msg.error_code = err.value("code", int64_t{0});
        msg.error_message = err.value("message", std::string{});
        return msg;
    }

    if (j.contains("result"))
    {
        const auto& result = j["result"];
        if (result.is_boolean())
        {
            msg.kind = MessageKind::Ok;
            return msg;
        }
        if (result.is_array() && result.size() == 4)
        {
            // [pow_hash(straight), seed_hash(straight),
            //  target_boundary(reversed), height(reversed)] - see
            // get_work_json() in stratum_server.cpp.
            bool ok = true;
            ok = ok && result[0].is_string() && hex_decode(result[0].get<std::string>(), msg.work.pow_hash.data(), 32);
            ok = ok && result[1].is_string() && hex_decode(result[1].get<std::string>(), msg.work.seed_hash.data(), 32);
            ok = ok && result[2].is_string() && hex_decode_reversed(result[2].get<std::string>(), msg.work.target_boundary.data(), 32);
            if (ok && result[3].is_string())
            {
                std::array<uint8_t, 8> height_bytes{};
                ok = hex_decode_reversed(result[3].get<std::string>(), height_bytes.data(), 8);
                if (ok)
                    msg.work.height = bytes8_to_u64_native(height_bytes);
            }
            else
            {
                ok = false;
            }
            if (!ok)
                return std::nullopt;
            msg.kind = MessageKind::Work;
            return msg;
        }
        // Some other result shape (e.g. login ack may be a plain `true`,
        // already handled above) - not a shape this parser recognizes.
        return std::nullopt;
    }

    return std::nullopt;
}

}  // namespace deepcore::network::zano_stratum
