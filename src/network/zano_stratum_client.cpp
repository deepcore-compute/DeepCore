#include "zano_stratum_client.hpp"
#include "zano_stratum_protocol.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <system_error>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace deepcore::network::zano_stratum {

namespace {

#if defined(_WIN32)
using raw_socket_t = SOCKET;
constexpr raw_socket_t kInvalidRawSocket = INVALID_SOCKET;

// Winsock requires a process-wide WSAStartup() before any socket call.
// Meyers-singleton init: thread-safe and runs exactly once, on first use,
// per the C++11 "magic statics" guarantee - no explicit synchronization
// needed here.
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
    }
    ~WinsockInit() { WSACleanup(); }
};
void ensure_winsock_initialized() { static WinsockInit init; }

void close_raw_socket(raw_socket_t s) { closesocket(s); }
int send_raw(raw_socket_t s, const char* data, int len) { return ::send(s, data, len, 0); }
int recv_raw(raw_socket_t s, char* buf, int cap) { return ::recv(s, buf, cap, 0); }
void set_recv_timeout(raw_socket_t s, int ms) {
    DWORD timeout = static_cast<DWORD>(ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
}
#else
using raw_socket_t = int;
constexpr raw_socket_t kInvalidRawSocket = -1;
void ensure_winsock_initialized() {}
void close_raw_socket(raw_socket_t s) { ::close(s); }
long send_raw(raw_socket_t s, const char* data, size_t len) { return ::send(s, data, len, 0); }
long recv_raw(raw_socket_t s, char* buf, size_t cap) { return ::recv(s, buf, cap, 0); }
void set_recv_timeout(raw_socket_t s, int ms) {
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}
#endif

// Resolves host:port and connects, trying every address getaddrinfo
// returns until one succeeds. Sets a receive timeout so the caller's read
// loop can periodically check for a shutdown request instead of blocking
// forever on recv().
raw_socket_t connect_tcp(const std::string& host, std::uint16_t port, int recv_timeout_ms)
{
    ensure_winsock_initialized();

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0 || result == nullptr)
        return kInvalidRawSocket;

    raw_socket_t sock = kInvalidRawSocket;
    for (struct addrinfo* p = result; p != nullptr; p = p->ai_next)
    {
        sock = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (sock == kInvalidRawSocket)
            continue;
        if (::connect(sock, p->ai_addr, static_cast<int>(p->ai_addrlen)) == 0)
        {
            set_recv_timeout(sock, recv_timeout_ms);
            break;
        }
        close_raw_socket(sock);
        sock = kInvalidRawSocket;
    }
    freeaddrinfo(result);
    return sock;
}

bool send_all(raw_socket_t sock, const std::string& data)
{
    size_t sent = 0;
    while (sent < data.size())
    {
#if defined(_WIN32)
        int n = send_raw(sock, data.data() + sent, static_cast<int>(data.size() - sent));
#else
        long n = send_raw(sock, data.data() + sent, data.size() - sent);
#endif
        if (n <= 0)
            return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

std::string bytes_to_hex_lower(const std::array<std::uint8_t, 32>& b)
{
    return zano_stratum::hex_encode(b.data(), b.size());
}

// Diagnostic-only: dumps every raw wire message sent/received to stderr
// when the DEEPCORE_DEBUG_WIRE environment variable is set to anything.
// Not part of the public interface - purely for inspecting real pool
// traffic when something needs cross-checking against actual bytes rather
// than assumptions (see src/network/README.md's real-daemon validation
// notes for why this project treats "assumed correct" and "verified
// against real bytes" as different things).
bool wire_debug_enabled()
{
    static const bool enabled = std::getenv("DEEPCORE_DEBUG_WIRE") != nullptr;
    return enabled;
}

void log_wire(const char* direction, const std::string& message)
{
    if (wire_debug_enabled())
        std::fprintf(stderr, "[wire %s] %s\n", direction, message.c_str());
}

}  // namespace

ZanoStratumClient::ZanoStratumClient() = default;

ZanoStratumClient::~ZanoStratumClient()
{
    shutdown();
}

void ZanoStratumClient::set_state(ConnectionState s, IStratumEventSink* sink)
{
    state_.store(s);
    if (sink)
        sink->on_state_changed(s);
}

std::error_code ZanoStratumClient::connect(const PoolConfig& config, IStratumEventSink& sink)
{
    if (config.endpoints.empty())
        return std::make_error_code(std::errc::invalid_argument);

    // TLS and multi-endpoint failover are not implemented in this
    // milestone - see this class's header comment. Fail synchronously and
    // clearly rather than silently connecting insecurely or to the wrong
    // endpoint.
    if (config.endpoints.front().use_tls)
        return std::make_error_code(std::errc::not_supported);

    worker_ = std::thread(&ZanoStratumClient::worker_main, this, config, &sink);
    return {};
}

void ZanoStratumClient::worker_main(PoolConfig config, IStratumEventSink* sink)
{
    set_state(ConnectionState::Connecting, sink);

    const PoolEndpoint& ep = config.endpoints.front();
    raw_socket_t sock = connect_tcp(ep.host, ep.port, /*recv_timeout_ms=*/500);
    if (sock == kInvalidRawSocket)
    {
        sink->on_error(StratumError{"failed to connect to " + ep.host + ":" + std::to_string(ep.port), true});
        set_state(ConnectionState::Disconnected, sink);
        return;
    }
    socket_handle_.store(static_cast<std::intptr_t>(sock));

    set_state(ConnectionState::Authorizing, sink);
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        std::string login_msg = build_submit_login(1, config.user, config.password, config.worker_name);
        log_wire("send", login_msg);
        if (!send_all(sock, login_msg))
        {
            sink->on_error(StratumError{"failed to send login", true});
            close_raw_socket(sock);
            socket_handle_.store(-1);
            set_state(ConnectionState::Disconnected, sink);
            return;
        }
        // Explicitly pull the current job right after login rather than
        // waiting for the next unsolicited work-change notification, so
        // the caller gets a job immediately instead of only on the next
        // block/difficulty change.
        std::string get_work_msg = build_get_work(2);
        log_wire("send", get_work_msg);
        send_all(sock, get_work_msg);
    }

    bool logged_in = false;
    MessageFramer framer;
    char recv_buffer[4096];

    while (!shutdown_requested_.load())
    {
#if defined(_WIN32)
        int n = recv_raw(sock, recv_buffer, static_cast<int>(sizeof(recv_buffer)));
#else
        long n = recv_raw(sock, recv_buffer, sizeof(recv_buffer));
#endif
        if (n > 0)
        {
            framer.feed(recv_buffer, static_cast<size_t>(n));

            std::string obj;
            while (framer.pop_object(obj))
            {
                log_wire("recv", obj);
                auto parsed = parse_message(obj);
                if (!parsed)
                    continue;  // malformed/unrecognized - ignore rather than tear down the connection

                if (parsed->kind == MessageKind::Work)
                {
                    {
                        std::lock_guard<std::mutex> lock(job_mutex_);
                        current_header_hash_ = parsed->work.pow_hash;
                        have_job_ = true;
                    }
                    JobNotification job;
                    job.job_id = bytes_to_hex_lower(parsed->work.pow_hash);
                    job.raw_payload = obj;  // opaque to us; the coin-specific mining loop re-parses this
                    job.received_at = std::chrono::steady_clock::now();
                    job.clean_jobs = true;
                    sink->on_job(job);
                }
                else if (parsed->kind == MessageKind::Ok)
                {
                    if (!logged_in)
                    {
                        logged_in = true;
                        set_state(ConnectionState::Ready, sink);
                    }
                    else if (parsed->id.has_value())
                    {
                        std::lock_guard<std::mutex> lock(counters_mutex_);
                        ++counters_.accepted;
                        sink->on_share_result(static_cast<std::uint64_t>(*parsed->id),
                            ShareResult{ShareResultStatus::Accepted, ""});
                    }
                }
                else if (parsed->kind == MessageKind::Error)
                {
                    if (!logged_in)
                    {
                        sink->on_error(StratumError{"login rejected: " + parsed->error_message, false});
                        // Fall through to disconnect below.
                    }
                    else if (parsed->id.has_value())
                    {
                        std::lock_guard<std::mutex> lock(counters_mutex_);
                        ++counters_.rejected;
                        sink->on_share_result(static_cast<std::uint64_t>(*parsed->id),
                            ShareResult{ShareResultStatus::Rejected, parsed->error_message});
                    }
                }
            }
        }
        else if (n == 0)
        {
            break;  // peer closed the connection
        }
        else
        {
            // Either the recv timeout elapsed (expected - loop back and
            // check shutdown_requested_) or a real socket error. There is
            // no portable, allocation-free way to distinguish EWOULDBLOCK/
            // WSAETIMEDOUT from a hard error here without more platform
            // branching than this milestone scopes in; treating a timeout
            // indistinguishably from "nothing happened yet" is safe either
            // way since the loop just re-checks shutdown and retries.
            continue;
        }
    }

    close_raw_socket(sock);
    socket_handle_.store(-1);
    set_state(ConnectionState::Disconnected, sink);
}

std::uint64_t ZanoStratumClient::submit_share(const ShareSubmission& submission)
{
    std::uint64_t id = next_submission_id_.fetch_add(1);

    if (shutdown_requested_.load())
        return id;  // fail fast, per the interface's shutdown contract - no callback fired

    std::array<std::uint8_t, 32> header_hash{};
    bool job_matches;
    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        job_matches = have_job_ && bytes_to_hex_lower(current_header_hash_) == submission.job_id;
        header_hash = current_header_hash_;
    }

    if (!job_matches)
    {
        // This milestone only retains the CURRENT job's header hash (see
        // this class's header comment) - a submission for any other
        // job_id is necessarily stale from this client's point of view,
        // regardless of PoolConfig::send_stale_shares, since we have
        // nothing to submit it against.
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.stale;
        return id;  // no on_share_result callback: nothing was sent, caller can infer from counters()
    }

    {
        std::lock_guard<std::mutex> lock(submitted_mutex_);
        for (const auto& [job_id, nonce_hex] : submitted_shares_)
        {
            if (job_id == submission.job_id && nonce_hex == submission.nonce_hex)
                return id;  // duplicate: silently absorbed, not re-sent
        }
        submitted_shares_.emplace_back(submission.job_id, submission.nonce_hex);
    }

    std::uint64_t nonce = 0;
    try
    {
        std::string s = submission.nonce_hex;
        if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
            s = s.substr(2);
        nonce = std::stoull(s, nullptr, 16);
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.invalid;
        return id;
    }

    std::array<std::uint8_t, 32> mix_hash{};
    if (!hex_decode(submission.result_hex, mix_hash.data(), mix_hash.size()))
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.invalid;
        return id;
    }

    raw_socket_t sock = static_cast<raw_socket_t>(socket_handle_.load());
    if (sock == kInvalidRawSocket)
        return id;  // not connected: nothing to send

    std::string worker_name;  // worker name isn't part of ShareSubmission; sent once at login already
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        std::string submit_msg =
            build_submit_work(static_cast<std::int64_t>(id), worker_name, nonce, header_hash, mix_hash);
        log_wire("send", submit_msg);
        send_all(sock, submit_msg);
    }
    return id;
}

StratumCounters ZanoStratumClient::counters() const
{
    std::lock_guard<std::mutex> lock(counters_mutex_);
    return counters_;
}

ConnectionState ZanoStratumClient::state() const
{
    return state_.load();
}

void ZanoStratumClient::shutdown()
{
    shutdown_requested_.store(true);
    if (worker_.joinable())
        worker_.join();
}

}  // namespace deepcore::network::zano_stratum
