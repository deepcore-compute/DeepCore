// Self-test for src/mining/mining_loop.{hpp,cpp} - validates the loop's
// ARCHITECTURE (job handling, cancellation, share submission, counters)
// using a fake in-memory IStratumClient and the real CpuHashSearchBackend.
// No real network or daemon involved (see src/network/README.md for the
// separate real-daemon validation of the underlying pieces this loop
// wires together).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "../../src/mining/mining_loop.hpp"
#include "../../src/network/zano_stratum_protocol.hpp"

using namespace deepcore;
using namespace deepcore::network;
using namespace deepcore::network::zano_stratum;

namespace {

int g_failures = 0;
void check(bool cond, const char* what)
{
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

class FakeStratumClient : public IStratumClient {
public:
    std::error_code connect(const PoolConfig&, IStratumEventSink&) override { return {}; }

    std::uint64_t submit_share(const ShareSubmission& s) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        submissions.push_back(s);
        return next_id++;
    }

    [[nodiscard]] StratumCounters counters() const override { return {}; }
    [[nodiscard]] ConnectionState state() const override { return ConnectionState::Ready; }
    void shutdown() override {}

    mutable std::mutex mutex;
    std::vector<ShareSubmission> submissions;
    std::uint64_t next_id = 1;
};

// Builds a synthetic, correctly-shaped work JobNotification, same wire
// format used throughout this project's self-tests. `fill_byte` makes the
// pow_hash (and hence job_id) distinguishable between jobs; seed_hash is
// always all-zero (epoch 0), so tests only ever need one cached epoch
// context.
JobNotification make_job(std::uint8_t pow_fill_byte, std::uint8_t target_fill_byte)
{
    std::array<std::uint8_t, 32> pow{}, seed{}, target{};
    pow.fill(pow_fill_byte);
    seed.fill(0);
    target.fill(target_fill_byte);

    std::string pow_hex = hex_encode(pow.data(), 32);
    std::string seed_hex = hex_encode(seed.data(), 32);
    std::string target_hex = hex_encode(target.data(), 32);

    JobNotification job;
    job.job_id = pow_hex;
    job.raw_payload = "{\"jsonrpc\":\"2.0\",\"result\":[\"" + pow_hex + "\",\"" + seed_hex + "\",\"" +
        target_hex + "\",\"0x0000000000000001\"]}";
    job.received_at = std::chrono::steady_clock::now();
    job.clean_jobs = true;
    return job;
}

template <class Predicate>
bool wait_for(Predicate pred, std::chrono::milliseconds timeout)
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return pred();
}

}  // namespace

int main()
{
    // ---- Test 1: an easy job (max target) is found and submitted quickly ----
    {
        FakeStratumClient client;
        mining::MiningLoop loop(client, std::make_unique<mining::CpuHashSearchBackend>(), 2);
        loop.start();

        loop.on_job_notification(make_job(0x11, 0xFF));  // target=0xff...ff: virtually any hash qualifies

        bool found = wait_for([&] {
            std::lock_guard<std::mutex> lock(client.mutex);
            return !client.submissions.empty();
        }, std::chrono::milliseconds(5000));
        check(found, "an easy (max-target) job produces a submitted share within 5s");

        auto counters = loop.counters();
        check(counters.jobs_received == 1, "counters().jobs_received == 1 after one job");
        check(counters.shares_found >= 1, "counters().shares_found >= 1 after finding a share");

        loop.stop();
    }

    // ---- Test 2: cancellation - a hard job never yields a share, and the
    // loop promptly picks up an easy job sent afterward -------------------
    {
        FakeStratumClient client;
        mining::MiningLoop loop(client, std::make_unique<mining::CpuHashSearchBackend>(), 2);
        loop.start();

        auto hard_job = make_job(0x22, 0x00);  // target=0x00...00: no hash will ever qualify
        loop.on_job_notification(hard_job);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));  // let it churn on the hard job
        check(loop.counters().hashes_computed > 0, "loop is actively searching the hard job (hashes_computed > 0)");

        auto easy_job = make_job(0x33, 0xFF);
        loop.on_job_notification(easy_job);

        bool found = wait_for([&] {
            std::lock_guard<std::mutex> lock(client.mutex);
            return !client.submissions.empty();
        }, std::chrono::milliseconds(5000));
        check(found, "after cancellation, the easy job produces a submitted share within 5s");

        {
            std::lock_guard<std::mutex> lock(client.mutex);
            bool any_for_hard_job = false;
            bool any_for_easy_job = false;
            for (auto& s : client.submissions)
            {
                if (s.job_id == hard_job.job_id) any_for_hard_job = true;
                if (s.job_id == easy_job.job_id) any_for_easy_job = true;
            }
            check(!any_for_hard_job, "no share was ever submitted for the (impossible) hard job");
            check(any_for_easy_job, "at least one share was submitted for the easy job");
        }

        check(loop.counters().jobs_received == 2, "counters().jobs_received == 2 after two jobs");

        loop.stop();
    }

    // ---- Test 3: a malformed job notification is discarded, not crashed on ----
    {
        FakeStratumClient client;
        mining::MiningLoop loop(client, std::make_unique<mining::CpuHashSearchBackend>(), 1);
        loop.start();

        JobNotification garbage;
        garbage.raw_payload = R"({"jsonrpc":"2.0","id":1,"result":true})";  // not a work message
        loop.on_job_notification(garbage);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto counters = loop.counters();
        check(counters.jobs_discarded == 1, "a malformed job notification is counted as discarded");
        check(counters.jobs_received == 0, "a malformed job notification does not count as received");

        loop.stop();
    }

    // ---- Test 4: stop() actually stops (doesn't hang, no further progress) ----
    {
        FakeStratumClient client;
        mining::MiningLoop loop(client, std::make_unique<mining::CpuHashSearchBackend>(), 2);
        loop.start();
        loop.on_job_notification(make_job(0x44, 0x00));  // hard job: keeps workers busy
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto before_stop = std::chrono::steady_clock::now();
        loop.stop();
        auto stop_duration = std::chrono::steady_clock::now() - before_stop;
        check(stop_duration < std::chrono::seconds(3), "stop() returns promptly (worker threads are responsive to cancellation)");

        auto after_stop_hashes = loop.counters().hashes_computed;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        check(loop.counters().hashes_computed == after_stop_hashes, "no further hashing happens after stop()");
    }

    if (g_failures == 0)
    {
        std::printf("\nALL CHECKS PASSED: MiningLoop correctly handles jobs, cancellation, "
                     "submission, and shutdown.\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED.\n", g_failures);
    return 1;
}
