// End-to-end test for ZanoStratumClient (src/network/zano_stratum_client.*)
// against a local mock server - NOT against a real zanod daemon.
//
// This validates the transport mechanics (connect, login, receive work,
// submit a share, receive a result, shutdown) using a minimal mock server
// built with the same wire-format helpers already proven correct by
// deepcore-protocol-selftest. It deliberately does NOT prove that a real
// zanod accepts this client - only that the client correctly speaks the
// protocol it's supposed to. Running against a real zanod --stratum
// testnet node is the next validation gate, not yet done (see
// src/network/README.md).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "../../src/network/zano_stratum_client.hpp"
#include "../../src/network/zano_stratum_protocol.hpp"

using namespace deepcore::network;
using namespace deepcore::network::zano_stratum;

namespace {

#if defined(_WIN32)
using raw_socket_t = SOCKET;
constexpr raw_socket_t kInvalidRawSocket = INVALID_SOCKET;
void close_raw_socket(raw_socket_t s) { closesocket(s); }
#else
using raw_socket_t = int;
constexpr raw_socket_t kInvalidRawSocket = -1;
void close_raw_socket(raw_socket_t s) { ::close(s); }
#endif

int g_failures = 0;
void check(bool cond, const char* what)
{
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

// ---- Minimal mock zanod stratum server -------------------------------------

struct MockServer {
    raw_socket_t listen_sock = kInvalidRawSocket;
    std::uint16_t port = 0;
    std::thread thread;
    std::atomic<bool> stop{false};

    // What the mock server observed, for the test to assert on afterward.
    std::atomic<bool> saw_login{false};
    std::atomic<bool> saw_get_work{false};
    std::atomic<bool> saw_submit_work{false};
    std::atomic<bool> submitted_header_matched_job{false};
    std::string observed_user;

    std::array<std::uint8_t, 32> job_pow_hash{};

    bool start()
    {
#if defined(_WIN32)
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        listen_sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock == kInvalidRawSocket) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = 0;  // ask the OS for an ephemeral port
        if (::bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) return false;

        socklen_t len = sizeof(addr);
        if (::getsockname(listen_sock, reinterpret_cast<sockaddr*>(&addr), &len) != 0) return false;
        port = ntohs(addr.sin_port);

        if (::listen(listen_sock, 1) != 0) return false;

        // Fill the job pow_hash with a recognizable pattern.
        for (int i = 0; i < 32; ++i) job_pow_hash[static_cast<size_t>(i)] = static_cast<std::uint8_t>(0x50 + i);

        thread = std::thread([this] { run(); });
        return true;
    }

    void run()
    {
        // Accept with a timeout so the thread can exit if the test never connects.
#if !defined(_WIN32)
        struct timeval tv { 5, 0 };
        setsockopt(listen_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        raw_socket_t conn = ::accept(listen_sock, nullptr, nullptr);
        if (conn == kInvalidRawSocket) return;

#if defined(_WIN32)
        DWORD timeout_ms = 5000;
        setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
        struct timeval tv2 { 5, 0 };
        setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &tv2, sizeof(tv2));
#endif

        MessageFramer framer;
        char buf[4096];

        auto send_line = [&](const std::string& s) {
#if defined(_WIN32)
            ::send(conn, s.data(), static_cast<int>(s.size()), 0);
#else
            ::send(conn, s.data(), s.size(), 0);
#endif
        };

        std::string work_hex_pow = hex_encode(job_pow_hash.data(), job_pow_hash.size());
        std::array<std::uint8_t, 32> seed{}, target{};
        for (int i = 0; i < 32; ++i) { seed[static_cast<size_t>(i)] = 0x22; target[static_cast<size_t>(i)] = static_cast<std::uint8_t>(i); }
        std::string work_hex_seed = hex_encode(seed.data(), seed.size());
        std::string work_hex_target = hex_encode(target.data(), target.size());
        std::string work_result_array = "[\"" + work_hex_pow + "\",\"" + work_hex_seed + "\",\"" +
            work_hex_target + "\",\"0x0000000000000001\"]";

        while (!stop.load())
        {
#if defined(_WIN32)
            int n = ::recv(conn, buf, static_cast<int>(sizeof(buf)), 0);
#else
            long n = ::recv(conn, buf, sizeof(buf), 0);
#endif
            if (n <= 0) break;
            framer.feed(buf, static_cast<size_t>(n));

            std::string obj;
            while (framer.pop_object(obj))
            {
                if (obj.find("eth_submitLogin") != std::string::npos)
                {
                    saw_login = true;
                    // We don't have a JSON parser handy for the mock side
                    // beyond substring checks - good enough to confirm the
                    // client actually sent the configured user string.
                    if (obj.find("test_user") != std::string::npos)
                        observed_user = "test_user";
                    send_line(R"({"jsonrpc":"2.0","id":1,"result":true})" "\n");
                }
                else if (obj.find("eth_getWork") != std::string::npos)
                {
                    saw_get_work = true;
                    send_line(R"({"jsonrpc":"2.0","id":2,"result":)" + work_result_array + "}\n");
                }
                else if (obj.find("eth_submitWork") != std::string::npos)
                {
                    saw_submit_work = true;
                    if (obj.find(work_hex_pow) != std::string::npos)
                        submitted_header_matched_job = true;

                    // Extract the request id to answer with the same id
                    // (a real parse_message would do this properly; a
                    // quick scan is sufficient for a mock).
                    auto pos = obj.find("\"id\":");
                    std::string id_str = "0";
                    if (pos != std::string::npos)
                    {
                        pos += 5;
                        auto end = obj.find_first_of(",}", pos);
                        id_str = obj.substr(pos, end - pos);
                    }
                    // Object-shaped accept response (result:{"status":"OK"}),
                    // not a plain boolean - this is LuckyPool's real observed
                    // shape for a share accept (captured via
                    // DEEPCORE_DEBUG_WIRE against the live pool, and
                    // independently via Rigel's --log-network hitting the
                    // same pool), distinct from its own login response
                    // (result:true, matching zanod's shape - see the login
                    // mock response above, also captured for real). Using
                    // the real-world shape here, not the older boolean one,
                    // is what makes this test actually exercise the
                    // object-shape fix through the full client rather than
                    // just the isolated parser (see protocol_selftest.cpp).
                    send_line(R"({"jsonrpc":"2.0","id":)" + id_str + R"(,"result":{"status":"OK"}})" "\n");
                }
            }
        }
        close_raw_socket(conn);
    }

    void stop_and_join()
    {
        stop.store(true);
        close_raw_socket(listen_sock);
        if (thread.joinable()) thread.join();
    }
};

// ---- Test event sink --------------------------------------------------------

class TestSink : public IStratumEventSink {
public:
    std::mutex mutex;
    std::vector<ConnectionState> states;
    std::vector<JobNotification> jobs;
    std::vector<std::pair<std::uint64_t, ShareResult>> share_results;
    std::vector<StratumError> errors;

    void on_state_changed(ConnectionState s) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        states.push_back(s);
    }
    void on_job(const JobNotification& job) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        jobs.push_back(job);
    }
    void on_share_result(std::uint64_t id, const ShareResult& result) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        share_results.emplace_back(id, result);
    }
    void on_error(const StratumError& err) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        errors.push_back(err);
    }
};

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
    MockServer server;
    check(server.start(), "mock server starts and binds an ephemeral loopback port");
    std::printf("mock server listening on 127.0.0.1:%u\n", server.port);

    TestSink sink;
    ZanoStratumClient client;

    PoolConfig config;
    config.endpoints.push_back(PoolEndpoint{"127.0.0.1", server.port, false});
    config.user = "test_user";
    config.worker_name = "test_worker";
    config.password = "x";

    auto err = client.connect(config, sink);
    check(!err, "connect() returns success synchronously (real work happens in background)");

    bool got_job = wait_for([&] {
        std::lock_guard<std::mutex> lock(sink.mutex);
        return !sink.jobs.empty();
    }, std::chrono::milliseconds(3000));
    check(got_job, "client receives a job within 3s");

    check(server.saw_login.load(), "mock server observed a login request");
    check(server.observed_user == "test_user", "login request carried the configured user string");
    check(server.saw_get_work.load(), "mock server observed a getWork request");

    bool reached_ready = wait_for([&] {
        std::lock_guard<std::mutex> lock(sink.mutex);
        for (auto s : sink.states) if (s == ConnectionState::Ready) return true;
        return false;
    }, std::chrono::milliseconds(1000));
    check(reached_ready, "client reached ConnectionState::Ready after login");

    std::string job_id;
    {
        std::lock_guard<std::mutex> lock(sink.mutex);
        if (!sink.jobs.empty()) job_id = sink.jobs.front().job_id;
    }

    ShareSubmission submission;
    submission.job_id = job_id;
    submission.nonce_hex = "0x123456789abcdef0";
    submission.result_hex = "0x" + std::string(64, '7');  // fake mix_hash
    submission.device_index = 0;

    std::uint64_t submission_id = client.submit_share(submission);
    check(submission_id != 0, "submit_share returns a nonzero submission id");

    bool got_result = wait_for([&] {
        std::lock_guard<std::mutex> lock(sink.mutex);
        return !sink.share_results.empty();
    }, std::chrono::milliseconds(3000));
    check(got_result, "client receives a share result within 3s");

    check(server.saw_submit_work.load(), "mock server observed a submitWork request");
    check(server.submitted_header_matched_job.load(), "submitted header_hash matched the job's pow_hash");

    {
        std::lock_guard<std::mutex> lock(sink.mutex);
        bool found = false;
        for (auto& [id, result] : sink.share_results)
        {
            if (id == submission_id && result.status == ShareResultStatus::Accepted) found = true;
        }
        check(found, "share result for our submission id was Accepted");
    }

    auto counters = client.counters();
    check(counters.accepted == 1, "counters().accepted == 1 after one accepted share");

    client.shutdown();
    check(client.state() == ConnectionState::Disconnected, "client reports Disconnected after shutdown()");

    server.stop_and_join();

    if (g_failures == 0)
    {
        std::printf("\nALL CHECKS PASSED: ZanoStratumClient correctly speaks the wire protocol "
                     "end-to-end against a mock server.\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED.\n", g_failures);
    return 1;
}
