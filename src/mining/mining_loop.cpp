#include "mining_loop.hpp"

#include <chrono>

namespace deepcore::mining {

std::optional<FoundShare> CpuHashSearchBackend::search(const ProgPowZJob& job,
    const ethash::epoch_context& ctx, std::uint64_t start_nonce, std::uint64_t count,
    const std::atomic<bool>& cancelled)
{
    for (std::uint64_t i = 0; i < count; ++i)
    {
        if (cancelled.load(std::memory_order_relaxed))
            return std::nullopt;

        std::uint64_t nonce = start_nonce + i;
        auto r = progpowz::progpowz_hash_light(
            reinterpret_cast<const progpowz::hash512*>(ctx.light_cache), ctx.light_cache_num_items,
            ctx.l1_cache, static_cast<std::uint32_t>(ctx.full_dataset_num_items), job.block_number,
            job.pow_hash, nonce);

        if (hash_meets_target(r.final_hash, job.target_boundary))
            return FoundShare{nonce, r.mix_hash};
    }
    return std::nullopt;
}

MiningLoop::MiningLoop(network::IStratumClient& client, std::unique_ptr<IHashSearchBackend> backend,
    unsigned worker_thread_count)
    : client_(client), backend_(std::move(backend)),
      worker_thread_count_(worker_thread_count == 0 ? 1 : worker_thread_count)
{
}

MiningLoop::~MiningLoop()
{
    stop();
}

std::shared_ptr<const ethash::epoch_context> MiningLoop::epoch_context_for(int epoch)
{
    std::lock_guard<std::mutex> lock(epoch_cache_mutex_);
    auto it = epoch_cache_.find(epoch);
    if (it != epoch_cache_.end())
        return it->second;

    ethash::epoch_context_ptr owned = ethash::create_epoch_context(epoch);
    if (!owned)
        return nullptr;

    ethash::epoch_context* raw = owned.release();
    std::shared_ptr<const ethash::epoch_context> shared(
        raw, [](const ethash::epoch_context* p) { ethash_destroy_epoch_context(const_cast<ethash::epoch_context*>(p)); });

    epoch_cache_.emplace(epoch, shared);
    return shared;
}

void MiningLoop::on_job_notification(const network::JobNotification& job)
{
    auto parsed = parse_job(job);
    if (!parsed)
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.jobs_discarded;
        return;
    }

    int epoch = epoch_from_seed_hash(parsed->seed_hash);
    if (epoch < 0)
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.jobs_discarded;
        return;
    }

    auto ctx = epoch_context_for(epoch);
    if (!ctx)
    {
        std::lock_guard<std::mutex> lock(counters_mutex_);
        ++counters_.jobs_discarded;
        return;
    }

    auto new_state = std::make_shared<JobState>();
    new_state->job = *parsed;
    new_state->notification = job;
    new_state->ctx = ctx;
    new_state->cancel_token = std::make_shared<std::atomic<bool>>(false);

    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        // job.clean_jobs is always true for every job ZanoStratumClient
        // currently produces (see that class), so unconditionally
        // cancelling the previous job's in-flight search here matches its
        // actual behavior; a future protocol/client where clean_jobs can
        // be false would need to gate this on that flag instead of always
        // cancelling.
        if (current_job_)
            current_job_->cancel_token->store(true, std::memory_order_relaxed);
        current_job_ = new_state;
    }

    std::lock_guard<std::mutex> lock(counters_mutex_);
    ++counters_.jobs_received;
}

std::shared_ptr<MiningLoop::JobState> MiningLoop::current_job_snapshot() const
{
    std::lock_guard<std::mutex> lock(job_mutex_);
    return current_job_;
}

void MiningLoop::worker_main()
{
    constexpr std::uint64_t kDefaultBatchSize = 64;  // CPU-tuned: keeps cancellation/shutdown responsive

    const std::uint64_t preferred = backend_->preferred_batch_size();
    const std::uint64_t batch_size = preferred != 0 ? preferred : kDefaultBatchSize;

    while (running_.load(std::memory_order_relaxed))
    {
        auto snapshot = current_job_snapshot();
        if (!snapshot)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        std::uint64_t start_nonce = snapshot->next_nonce.fetch_add(batch_size, std::memory_order_relaxed);
        auto found = backend_->search(snapshot->job, *snapshot->ctx, start_nonce, batch_size,
            *snapshot->cancel_token);

        {
            std::lock_guard<std::mutex> lock(counters_mutex_);
            counters_.hashes_computed += batch_size;
        }

        if (found)
        {
            auto submission = build_share_submission(snapshot->notification, found->nonce, found->mix_hash);
            client_.submit_share(submission);
            std::lock_guard<std::mutex> lock(counters_mutex_);
            ++counters_.shares_found;
        }
    }
}

void MiningLoop::start()
{
    if (running_.exchange(true))
        return;  // already running

    workers_.reserve(worker_thread_count_);
    for (unsigned i = 0; i < worker_thread_count_; ++i)
        workers_.emplace_back(&MiningLoop::worker_main, this);
}

void MiningLoop::stop()
{
    running_.store(false);
    for (auto& t : workers_)
        if (t.joinable())
            t.join();
    workers_.clear();
}

MiningLoop::Counters MiningLoop::counters() const
{
    std::lock_guard<std::mutex> lock(counters_mutex_);
    return counters_;
}

}  // namespace deepcore::mining
