// deepcore-miner - CLI entry point.
//
// Assembles ZanoStratumClient + MiningLoop into an actual runnable program:
// connects to a Zano-protocol pool/daemon, mines (CPU reference backend by
// default; --gpu selects GpuHashSearchBackend when this build was compiled
// with CUDA - see src/cuda/README.md), and prints periodic status.
//
// Deliberately does NOT expose flags for features that don't exist yet
// (multi-GPU selection beyond a single --gpu-device index, API port,
// power/temp limits, auto-tune, keepalive, stale-share submission, job
// timeout) - see README files under src/ for what's tracked as a real,
// honest gap rather than silently unsupported. --gpu itself only appears
// in --help/is accepted when DEEPCORE_HAVE_GPU_BACKEND was defined at
// build time (i.e. this build actually has CUDA) - never a flag for
// something this particular binary can't do.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <thread>

#include "../mining/mining_loop.hpp"
#include "../network/zano_stratum_client.hpp"

#ifdef DEEPCORE_HAVE_GPU_BACKEND
#include "../cuda/progpowz_gpu_backend.hpp"
#endif

using namespace deepcore;
using namespace deepcore::network;

namespace {

std::atomic<bool> g_stop_requested{false};

void handle_signal(int) { g_stop_requested.store(true); }

struct Args {
    std::string url;      // host:port
    std::string user;     // public payout address (pool login)
    std::string worker;   // worker name, optional
    std::string password{"x"};
    unsigned threads{0};  // 0 => hardware_concurrency(); ignored when --gpu is set (forced to 1)
    bool threads_explicit{false};
    unsigned status_interval_seconds{10};
    bool dry_run{false};
    bool quiet{false};
    bool show_help{false};
#ifdef DEEPCORE_HAVE_GPU_BACKEND
    bool gpu{false};
    int gpu_device{0};
#endif
};

void print_usage(const char* argv0)
{
    std::printf(
        "Usage: %s --url <host:port> --user <address> [options]\n"
        "\n"
        "Required:\n"
        "  --url <host:port>       Pool/daemon stratum endpoint (plain TCP only for now)\n"
        "  --user <address>        Public payout address (never a private key/seed)\n"
        "\n"
        "Options:\n"
        "  --worker <name>         Worker name reported to the pool (default: none)\n"
        "  --password <value>      Stratum password (default: \"x\")\n"
        "  --threads <n>           CPU search worker threads (default: hardware_concurrency())\n"
        "                          ignored if --gpu is set (one worker drives one GPU)\n"
        "  --status-interval <s>   Seconds between status lines, 0 to disable (default: 10)\n"
        "  --dry-run               Connect, wait for the first job, print result, then exit\n"
        "  --quiet                 Suppress per-event log lines (status lines still print)\n"
#ifdef DEEPCORE_HAVE_GPU_BACKEND
        "  --gpu                   Mine with the GPU backend instead of the CPU one\n"
        "  --gpu-device <n>        CUDA device index to use with --gpu (default: 0)\n"
#endif
        "  --help, -h              Show this help text\n"
        "\n"
#ifdef DEEPCORE_HAVE_GPU_BACKEND
        "Note: --gpu uses the light-cache kernel (correctness-validated, not yet\n"
        "throughput-optimized - see src/cuda/README.md) and drives a single CUDA\n"
        "device; multi-GPU selection is not implemented yet.\n"
#else
        "Note: mining currently runs on the CPU reference backend, which is\n"
        "correct but not competitive throughput - see src/mining/README.md. This\n"
        "build was not compiled with CUDA, so --gpu is not available - see\n"
        "src/cuda/README.md.\n"
#endif
        ,
        argv0);
}

bool parse_host_port(const std::string& url, std::string& host, std::uint16_t& port)
{
    auto colon = url.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= url.size())
        return false;

    host = url.substr(0, colon);
    std::string port_str = url.substr(colon + 1);
    if (port_str.empty() || port_str.find_first_not_of("0123456789") != std::string::npos)
        return false;

    long parsed = std::strtol(port_str.c_str(), nullptr, 10);
    if (parsed <= 0 || parsed > 65535)
        return false;

    port = static_cast<std::uint16_t>(parsed);
    return true;
}

bool parse_args(int argc, char** argv, Args& out, std::string& error)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc)
            {
                error = std::string(flag) + " requires a value";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") { out.show_help = true; }
        else if (arg == "--url") { const char* v = next("--url"); if (!v) return false; out.url = v; }
        else if (arg == "--user") { const char* v = next("--user"); if (!v) return false; out.user = v; }
        else if (arg == "--worker") { const char* v = next("--worker"); if (!v) return false; out.worker = v; }
        else if (arg == "--password") { const char* v = next("--password"); if (!v) return false; out.password = v; }
        else if (arg == "--threads")
        {
            const char* v = next("--threads");
            if (!v) return false;
            out.threads = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
            out.threads_explicit = true;
        }
        else if (arg == "--status-interval")
        {
            const char* v = next("--status-interval");
            if (!v) return false;
            out.status_interval_seconds = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        }
        else if (arg == "--dry-run") { out.dry_run = true; }
        else if (arg == "--quiet") { out.quiet = true; }
#ifdef DEEPCORE_HAVE_GPU_BACKEND
        else if (arg == "--gpu") { out.gpu = true; }
        else if (arg == "--gpu-device")
        {
            const char* v = next("--gpu-device");
            if (!v) return false;
            out.gpu_device = std::atoi(v);
        }
#else
        else if (arg == "--gpu" || arg == "--gpu-device")
        {
            error = arg + " requires a build compiled with CUDA (DEEPCORE_HAVE_GPU_BACKEND) - "
                           "see src/cuda/README.md";
            return false;
        }
#endif
        else { error = "unrecognized argument: " + arg; return false; }
    }
    return true;
}

const char* state_name(ConnectionState s)
{
    switch (s)
    {
        case ConnectionState::Disconnected: return "Disconnected";
        case ConnectionState::Resolving: return "Resolving";
        case ConnectionState::Connecting: return "Connecting";
        case ConnectionState::TlsHandshake: return "TlsHandshake";
        case ConnectionState::Subscribing: return "Subscribing";
        case ConnectionState::Authorizing: return "Authorizing";
        case ConnectionState::Ready: return "Ready";
        case ConnectionState::Reconnecting: return "Reconnecting";
        case ConnectionState::ShuttingDown: return "ShuttingDown";
    }
    return "Unknown";
}

const char* share_status_name(ShareResultStatus s)
{
    switch (s)
    {
        case ShareResultStatus::Accepted: return "Accepted";
        case ShareResultStatus::Rejected: return "Rejected";
        case ShareResultStatus::Stale: return "Stale";
        case ShareResultStatus::Invalid: return "Invalid";
        case ShareResultStatus::TransportError: return "TransportError";
    }
    return "Unknown";
}

// Forwards job notifications into MiningLoop and logs events. Also tracks
// whether at least one job has been seen, for --dry-run.
class CliSink final : public IStratumEventSink {
public:
    mining::MiningLoop* loop = nullptr;
    bool quiet = false;
    std::atomic<bool> ready{false};
    std::atomic<bool> got_job{false};

    void on_state_changed(ConnectionState new_state) override
    {
        if (new_state == ConnectionState::Ready)
            ready.store(true);
        if (!quiet)
            std::printf("[state] %s\n", state_name(new_state));
    }

    void on_job(const JobNotification& job) override
    {
        got_job.store(true);
        if (!quiet)
            std::printf("[job] id=%s\n", job.job_id.c_str());
        if (loop)
            loop->on_job_notification(job);
    }

    void on_share_result(std::uint64_t submission_id, const ShareResult& result) override
    {
        if (!quiet)
        {
            std::printf("[share] id=%llu status=%s%s%s\n", static_cast<unsigned long long>(submission_id),
                share_status_name(result.status), result.message.empty() ? "" : " reason=",
                result.message.c_str());
        }
    }

    void on_error(const StratumError& error) override
    {
        if (!quiet)
            std::printf("[error] %s%s\n", error.message.c_str(), error.is_transport_error ? " (transport)" : "");
    }
};

}  // namespace

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args;
    std::string parse_error;
    if (!parse_args(argc, argv, args, parse_error))
    {
        std::fprintf(stderr, "error: %s\n\n", parse_error.c_str());
        print_usage(argv[0]);
        return 2;
    }

    if (args.show_help)
    {
        print_usage(argv[0]);
        return 0;
    }

    if (args.url.empty() || args.user.empty())
    {
        std::fprintf(stderr, "error: --url and --user are required\n\n");
        print_usage(argv[0]);
        return 2;
    }

    std::string host;
    std::uint16_t port{};
    if (!parse_host_port(args.url, host, port))
    {
        std::fprintf(stderr, "error: --url must be host:port (got \"%s\")\n", args.url.c_str());
        return 2;
    }

    unsigned thread_count = args.threads != 0 ? args.threads : std::thread::hardware_concurrency();
    if (thread_count == 0)
        thread_count = 1;

    std::unique_ptr<mining::IHashSearchBackend> backend;
    const char* backend_name = "cpu";
#ifdef DEEPCORE_HAVE_GPU_BACKEND
    if (args.gpu)
    {
        if (args.threads_explicit && args.threads != 1 && !args.quiet)
        {
            std::printf("note: --threads is ignored when --gpu is set (one worker drives one "
                        "GPU); requested %u, using 1\n", args.threads);
        }
        thread_count = 1;
        backend = std::make_unique<mining::GpuHashSearchBackend>(args.gpu_device);
        backend_name = "gpu";
    }
    else
    {
        backend = std::make_unique<mining::CpuHashSearchBackend>();
    }
#else
    backend = std::make_unique<mining::CpuHashSearchBackend>();
#endif

    PoolConfig config;
    config.endpoints.push_back(PoolEndpoint{host, port, false});
    config.user = args.user;
    config.worker_name = args.worker;
    config.password = args.password;

    zano_stratum::ZanoStratumClient client;
    CliSink sink;
    sink.quiet = args.quiet;

    mining::MiningLoop loop(client, std::move(backend), thread_count);
    sink.loop = &loop;

    auto connect_err = client.connect(config, sink);
    if (connect_err)
    {
        std::fprintf(stderr, "error: connect() failed: %s\n", connect_err.message().c_str());
        return 1;
    }

    if (args.dry_run)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (sink.ready.load() && sink.got_job.load())
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        bool ok = sink.ready.load() && sink.got_job.load();
        std::printf("%s: dry-run %s (ready=%s, got_job=%s)\n", ok ? "PASS" : "FAIL",
            ok ? "connected and received a job" : "timed out",
            sink.ready.load() ? "true" : "false", sink.got_job.load() ? "true" : "false");

        client.shutdown();
        return ok ? 0 : 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::printf("deepcore-miner: connecting to %s:%u as %s (backend=%s, threads=%u)\n", host.c_str(),
        static_cast<unsigned>(port), args.user.c_str(), backend_name, thread_count);

    loop.start();

    auto last_status = std::chrono::steady_clock::now();
    while (!g_stop_requested.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        if (args.status_interval_seconds == 0)
            continue;

        auto now = std::chrono::steady_clock::now();
        if (now - last_status >= std::chrono::seconds(args.status_interval_seconds))
        {
            last_status = now;
            auto mc = loop.counters();
            auto sc = client.counters();
            std::printf(
                "[status] state=%s hashes=%llu shares_found=%llu jobs=%llu accepted=%llu rejected=%llu "
                "stale=%llu invalid=%llu\n",
                state_name(client.state()), static_cast<unsigned long long>(mc.hashes_computed),
                static_cast<unsigned long long>(mc.shares_found),
                static_cast<unsigned long long>(mc.jobs_received),
                static_cast<unsigned long long>(sc.accepted), static_cast<unsigned long long>(sc.rejected),
                static_cast<unsigned long long>(sc.stale), static_cast<unsigned long long>(sc.invalid));
        }
    }

    std::printf("deepcore-miner: shutting down...\n");
    loop.stop();
    client.shutdown();
    return 0;
}
