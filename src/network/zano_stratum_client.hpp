#pragma once

// deepcore::network::zano_stratum::ZanoStratumClient - a concrete
// IStratumClient (see stratum_client.hpp) implementation for Zano's
// daemon-native stratum protocol, built on the wire-format layer in
// zano_stratum_protocol.hpp.
//
// SCOPE OF THIS MILESTONE (see src/network/README.md for the full picture):
//   - Plain TCP only. TLS (PoolEndpoint::use_tls) is not implemented yet -
//     connect() returns an error if the first configured endpoint requests it.
//   - Only the FIRST endpoint in PoolConfig::endpoints is tried. Ordered
//     failover across multiple endpoints is not implemented yet.
//   - No reconnect/backoff loop yet: on disconnect, the client reports
//     ConnectionState::Disconnected and stops - it does not retry.
//   - No keepalive pings, no job-timeout-triggered failover.
// These are real, tracked gaps, not silently skipped - see the README for
// why they're sequenced after this milestone rather than attempted here.
//
// What DOES work in this milestone: connect, login, receive work (both as
// a direct eth_getWork response and as an unsolicited work-changed
// notification - zanod sends both, see zano_stratum_protocol.hpp), submit
// a share, receive accept/reject, clean shutdown. Validated end-to-end
// against a local mock server in tools/network_selftest (not yet against a
// real zanod - see that tool's header comment).

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "stratum_client.hpp"

namespace deepcore::network::zano_stratum {

class ZanoStratumClient final : public IStratumClient {
public:
    ZanoStratumClient();
    ~ZanoStratumClient() override;

    ZanoStratumClient(const ZanoStratumClient&) = delete;
    ZanoStratumClient& operator=(const ZanoStratumClient&) = delete;

    std::error_code connect(const PoolConfig& config, IStratumEventSink& sink) override;
    std::uint64_t submit_share(const ShareSubmission& submission) override;
    [[nodiscard]] StratumCounters counters() const override;
    [[nodiscard]] ConnectionState state() const override;
    void shutdown() override;

private:
    void worker_main(PoolConfig config, IStratumEventSink* sink);
    void set_state(ConnectionState s, IStratumEventSink* sink);

    // Platform socket handle, stored as an integer so this header never
    // needs to include <winsock2.h> or <sys/socket.h>. -1 means "no
    // socket" on every platform this project targets (POSIX fds and
    // Winsock SOCKETs are both representable this way in practice here -
    // see the .cpp for the exact platform-specific handling).
    std::atomic<std::intptr_t> socket_handle_{-1};

    std::mutex send_mutex_;   // serializes writes to the socket
    std::thread worker_;
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<ConnectionState> state_{ConnectionState::Disconnected};

    mutable std::mutex counters_mutex_;
    StratumCounters counters_;

    std::atomic<std::uint64_t> next_submission_id_{1};

    mutable std::mutex job_mutex_;
    std::array<std::uint8_t, 32> current_header_hash_{};
    bool have_job_{false};

    mutable std::mutex submitted_mutex_;
    // (job header_hash hex, nonce_hex) pairs already sent, so we never
    // submit the same share twice - required by the interface contract in
    // stratum_client.hpp.
    std::vector<std::pair<std::string, std::string>> submitted_shares_;
};

}  // namespace deepcore::network::zano_stratum
