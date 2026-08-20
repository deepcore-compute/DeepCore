#pragma once

// deepcore::mining::MiningLoop - the persistent, continuously-running
// component that ties everything else in this project together: receives
// jobs from an IStratumClient, searches for a passing nonce via a
// pluggable IHashSearchBackend, and submits any share found back through
// the same client. Cancels in-flight work promptly when a new job
// supersedes the current one.
//
// The search strategy is intentionally pluggable (IHashSearchBackend)
// rather than hard-wired to either CPU or GPU:
//   - CpuHashSearchBackend (below) is a real, working reference
//     implementation using progpowz_hash_light - correct but far too slow
//     for real mining throughput (see src/cuda/README.md). It exists so
//     this loop's ARCHITECTURE (job handling, cancellation, submission,
//     counters) can be validated for real, independent of GPU hardware
//     access.
//   - A GPU-backed IHashSearchBackend wrapping progpowz_kernel.cu's
//     run_progpowz_light_gpu (and, later, a real full-DAG kernel) is the
//     intended production backend, and can be swapped in without any
//     change to MiningLoop itself.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ethash/ethash.hpp>

#include "../network/stratum_client.hpp"
#include "progpowz_work.hpp"

namespace deepcore::mining {

struct FoundShare {
    std::uint64_t nonce;
    progpowz::hash256 mix_hash;
};

// Searches nonces [start_nonce, start_nonce + count) for `job` using the
// light epoch context `ctx` (see progpowz_hash_light's own documentation
// for what "light" means here). Implementations MUST check `cancelled`
// frequently (at least every few dozen nonces) so a superseding job can
// interrupt the search promptly - MiningLoop relies on this for responsive
// job cancellation and clean shutdown.
class IHashSearchBackend {
public:
    virtual ~IHashSearchBackend() = default;
    virtual std::optional<FoundShare> search(const ProgPowZJob& job, const ethash::epoch_context& ctx,
        std::uint64_t start_nonce, std::uint64_t count, const std::atomic<bool>& cancelled) = 0;
};

// Reference CPU backend (progpowz_hash_light). Real and correct, not a
// stub - but slow, see this file's header comment. Used to validate
// MiningLoop's architecture independent of GPU access.
class CpuHashSearchBackend final : public IHashSearchBackend {
public:
    std::optional<FoundShare> search(const ProgPowZJob& job, const ethash::epoch_context& ctx,
        std::uint64_t start_nonce, std::uint64_t count, const std::atomic<bool>& cancelled) override;
};

// Thread-safety: on_job_notification(), start(), stop(), and counters()
// are safe to call from any thread. MiningLoop runs its own worker
// thread(s) internally.
class MiningLoop {
public:
    // `client` must outlive this MiningLoop and must already be connected
    // (or connecting) - MiningLoop only calls submit_share() on it, never
    // connect(). Takes ownership of `backend`. `worker_thread_count`
    // should generally match however many parallel search streams make
    // sense for the backend (e.g. one per GPU for a GPU backend); defaults
    // to 1, appropriate for the CPU reference backend.
    MiningLoop(network::IStratumClient& client, std::unique_ptr<IHashSearchBackend> backend,
        unsigned worker_thread_count = 1);
    ~MiningLoop();

    MiningLoop(const MiningLoop&) = delete;
    MiningLoop& operator=(const MiningLoop&) = delete;

    // Feed a job into the loop - call this from an IStratumEventSink's
    // on_job() (directly, or have your sink delegate to it). If parsing
    // fails or the epoch can't be determined, the notification is
    // discarded (counted, not silently lost - see counters()) rather than
    // crashing or leaving the loop stuck on stale work.
    void on_job_notification(const network::JobNotification& job);

    // Starts/stops the worker thread(s). Safe to call start() before any
    // job has arrived - workers idle-wait until on_job_notification()
    // provides one. stop() blocks until all worker threads have exited;
    // safe to call from the destructor path (also called there).
    void start();
    void stop();

    struct Counters {
        std::uint64_t hashes_computed{};
        std::uint64_t shares_found{};
        std::uint64_t jobs_received{};
        std::uint64_t jobs_discarded{};  // failed to parse / unknown epoch
    };
    [[nodiscard]] Counters counters() const;

private:
    struct JobState {
        ProgPowZJob job;
        network::JobNotification notification;
        std::shared_ptr<const ethash::epoch_context> ctx;
        std::shared_ptr<std::atomic<bool>> cancel_token;
        std::atomic<std::uint64_t> next_nonce{0};
    };

    void worker_main();
    std::shared_ptr<const ethash::epoch_context> epoch_context_for(int epoch);
    std::shared_ptr<JobState> current_job_snapshot() const;

    network::IStratumClient& client_;
    std::unique_ptr<IHashSearchBackend> backend_;
    unsigned worker_thread_count_;

    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    mutable std::mutex job_mutex_;
    std::shared_ptr<JobState> current_job_;

    std::mutex epoch_cache_mutex_;
    std::unordered_map<int, std::shared_ptr<const ethash::epoch_context>> epoch_cache_;

    mutable std::mutex counters_mutex_;
    Counters counters_;
};

}  // namespace deepcore::mining
