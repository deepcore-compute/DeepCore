#pragma once

// deepcore::network - asynchronous pool client interface.
//
// This header defines a transport-and-protocol-agnostic contract for
// connecting to a mining pool, receiving job notifications, and submitting
// shares. It does not assume:
//   - a specific coin, PoW algorithm, or job payload shape (JobNotification
//     carries an opaque payload parsed by the coin-specific proof module)
//   - that Stratum V1 (or any particular protocol dialect) is the wire
//     format - a concrete implementation targets whatever protocol the
//     coin's pool documentation specifies, and only needs to satisfy this
//     interface
//   - a specific networking/JSON library. IIoExecutor is a minimal seam so
//     this header compiles without one; a production implementation will
//     sit on top of a real event loop (e.g. standalone/Boost Asio, or
//     platform IOCP/epoll) and a real JSON library (e.g. nlohmann::json),
//     both to be selected and wired in when StratumClient is implemented.
//
// No pool URLs, wallet addresses, credentials, or secrets are hard-coded
// anywhere in this file.

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace deepcore::network {

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct PoolEndpoint {
    std::string host;
    std::uint16_t port{};
    bool use_tls{false};
};

// A fully-resolved pool configuration. `endpoints` is an ordered failover
// list: the client tries them in order, advancing to the next on connect
// failure or job timeout and wrapping back to the first after exhausting the
// list (repeatable pool URLs).
struct PoolConfig {
    std::vector<PoolEndpoint> endpoints;

    // Public payout address / pool login identity. This type never carries a
    // private key, seed phrase, or signing credential - callers must not
    // populate it with one.
    std::string user;
    std::string worker_name;
    std::string password;

    bool verify_tls_certificates{true};

    std::chrono::seconds job_timeout{120};
    std::optional<std::chrono::seconds> keepalive_interval;

    // Shares for a job that has already been superseded are dropped locally
    // by default rather than submitted, per default-safe behavior; set true
    // only for explicit diagnostics/testing.
    bool send_stale_shares{false};

    std::chrono::milliseconds initial_reconnect_backoff{1000};
    std::chrono::milliseconds max_reconnect_backoff{60000};
};

// -----------------------------------------------------------------------------
// Connection / session state
// -----------------------------------------------------------------------------

enum class ConnectionState {
    Disconnected,
    Resolving,
    Connecting,
    TlsHandshake,
    Subscribing,
    Authorizing,
    Ready,
    Reconnecting,
    ShuttingDown,
};

struct StratumError {
    std::string message;
    bool is_transport_error{false}; // false => protocol/application-level error (e.g. auth rejected)
};

// -----------------------------------------------------------------------------
// Jobs and shares
// -----------------------------------------------------------------------------

// A job notification from the pool. `raw_payload` is the algorithm-specific
// job body (e.g. header blob, seed hash, target, height for a ProgPoW-family
// algorithm) as opaque, not-yet-parsed text; the coin-specific module that
// builds GPU work from a job owns the actual schema and parsing.
struct JobNotification {
    std::string job_id;
    std::string raw_payload;
    std::chrono::steady_clock::time_point received_at;

    // true (the default assumption): the miner must cancel all in-flight
    // and queued GPU work for previous jobs immediately upon receiving this
    // notification. false: this job may be worked on alongside the previous
    // one (protocol-dependent; most jobs are clean_jobs=true in practice).
    bool clean_jobs{true};
};

struct ShareSubmission {
    std::string job_id;
    std::string nonce_hex;
    std::string result_hex; // proof/solution payload; format owned by the coin module
    std::uint32_t device_index{};
};

enum class ShareResultStatus {
    Accepted,
    Rejected,
    Stale,
    Invalid,
    TransportError,
};

struct ShareResult {
    ShareResultStatus status{ShareResultStatus::TransportError};
    std::string message; // pool-provided reason text, if any
};

struct StratumCounters {
    std::uint64_t accepted{};
    std::uint64_t rejected{};
    std::uint64_t stale{};
    std::uint64_t invalid{};
};

// -----------------------------------------------------------------------------
// Minimal async execution seam
// -----------------------------------------------------------------------------

// A production IStratumClient implementation needs an event loop to drive
// non-blocking socket I/O, timers (job timeout, reconnect backoff,
// keepalive), and callback dispatch. Rather than requiring a specific
// networking library to make this starter header compile, IIoExecutor
// captures the minimal primitives such an implementation needs; a real
// backend can implement this directly on top of Asio/IOCP/epoll, or adapt
// an existing executor to it.
class IIoExecutor {
public:
    virtual ~IIoExecutor() = default;

    // Schedules `task` to run on the executor as soon as possible.
    virtual void post(std::function<void()> task) = 0;

    // Schedules `task` to run after `delay`. Used for reconnect backoff,
    // job-timeout detection, and keepalive pings.
    virtual void schedule_after(std::chrono::milliseconds delay, std::function<void()> task) = 0;
};

// -----------------------------------------------------------------------------
// Event sink
// -----------------------------------------------------------------------------

// Callback interface a caller implements to receive client events.
//
// Thread-safety: methods on IStratumEventSink may be invoked from a thread
// owned by the IStratumClient implementation (e.g. the thread driving its
// IIoExecutor), not necessarily the thread that called connect(). An
// implementer must not assume these callbacks run on any particular thread
// and must synchronize any shared state it touches from them. Callbacks
// must not block for a long time or perform another blocking call back into
// the IStratumClient from within the callback.
class IStratumEventSink {
public:
    virtual ~IStratumEventSink() = default;

    virtual void on_state_changed(ConnectionState new_state) = 0;
    virtual void on_job(const JobNotification& job) = 0;

    // Reports the outcome of a previously submitted share. `submission_id`
    // is the value returned by IStratumClient::submit_share() for the share
    // this result corresponds to.
    virtual void on_share_result(std::uint64_t submission_id, const ShareResult& result) = 0;

    virtual void on_error(const StratumError& error) = 0;
};

// -----------------------------------------------------------------------------
// Client interface
// -----------------------------------------------------------------------------

// Abstract asynchronous pool client.
//
// Thread-safety: implementations MUST make submit_share(), counters(),
// state(), and shutdown() safe to call concurrently from multiple threads
// (typically one worker thread per GPU calling submit_share(), plus a
// monitoring/API thread calling counters()/state()). connect() is expected
// to be called once per client instance before any other method.
//
// Cancellation/shutdown: shutdown() must cancel any in-flight connection
// attempt or request, stop reconnect/keepalive timers, and return only once
// the client has fully released its I/O resources (or completed an
// asynchronous teardown and signaled ConnectionState::Disconnected via the
// event sink - implementations must document which of these two shapes they
// use). After shutdown() begins, further submit_share() calls must fail
// fast rather than block or hang.
//
// Duplicate-share prevention and stale-work handling are the implementation's
// responsibility: it must not submit the same (job_id, nonce_hex) pair to
// the pool twice, and must drop shares for a superseded job unless
// PoolConfig::send_stale_shares is set.
class IStratumClient {
public:
    virtual ~IStratumClient() = default;

    // Begins connecting using `config`, trying endpoints in order with
    // exponential backoff between attempts (per config's backoff settings)
    // and failing over to the next endpoint on connect failure or job
    // timeout. This call is non-blocking: progress and terminal outcomes are
    // reported through `sink` (on_state_changed, on_error), not through this
    // call's return value. The caller must keep `sink` alive until after
    // shutdown() returns / completes.
    //
    // Returns a non-success error_code only for immediate, synchronous
    // configuration problems (e.g. an empty endpoint list) that make
    // connecting impossible without ever attempting I/O.
    virtual std::error_code connect(const PoolConfig& config, IStratumEventSink& sink) = 0;

    // Submits a share asynchronously. Returns a submission id that will be
    // passed to a later IStratumEventSink::on_share_result() call; does not
    // block waiting for the pool's response. Safe to call from any thread,
    // including concurrently from multiple GPU worker threads.
    virtual std::uint64_t submit_share(const ShareSubmission& submission) = 0;

    [[nodiscard]] virtual StratumCounters counters() const = 0;
    [[nodiscard]] virtual ConnectionState state() const = 0;

    // Gracefully cancels all in-flight work and disconnects. See the
    // class-level comment for the exact cancellation contract.
    virtual void shutdown() = 0;
};

} // namespace deepcore::network
