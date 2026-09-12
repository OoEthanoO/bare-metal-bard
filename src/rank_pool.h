#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// One persistent worker per rank for both the trainer and its regression test.
// CUDA's current device and the model's scratch caches are thread-local. Each
// worker must keep the same device across jobs; recreating workers every step
// reallocates and leaks those caches and serializes cudaMallocHost calls.
struct RankPool {
    std::vector<std::thread> threads;
    std::function<void(int)> job;
    std::mutex mu;
    std::condition_variable cv_start, cv_done;
    unsigned long long epoch = 0;
    int pending = 0;
    bool quit = false;

    void start(int n) {
        for (int r = 0; r < n; ++r)
            threads.emplace_back([this, r] {
                unsigned long long seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_start.wait(lk, [&] { return quit || epoch != seen; });
                    if (quit) return;
                    seen = epoch;
                    auto fn = job;
                    lk.unlock();
                    fn(r);
                    lk.lock();
                    if (--pending == 0) cv_done.notify_one();
                }
            });
    }
    void run(const std::function<void(int)> &fn) {
        std::unique_lock<std::mutex> lk(mu);
        job = fn;
        pending = (int)threads.size();
        ++epoch;
        cv_start.notify_all();
        cv_done.wait(lk, [&] { return pending == 0; });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv_start.notify_all();
        for (auto &t : threads) t.join();
        threads.clear();
    }
    ~RankPool() { if (!threads.empty()) stop(); }
};
