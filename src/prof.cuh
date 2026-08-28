#pragma once
// Where does a training step actually go, without a profiler?
//
// `ncu` answers this precisely and this repo already leans on it -- three times
// it has overruled an intuition of mine, and tools/step_profile.py exists to
// aggregate its output. But it needs elevated GPU performance-counter access,
// which on WSL means a registry change and a reboot, and on the free Colab or
// Kaggle T4 this project is supposed to be reproducible on it is simply not
// available. A measurement you cannot take on the target hardware is not much
// of a measurement.
//
// So: one `cudaEvent` pair around each region, every record issued on the same
// stream during the step, and exactly ONE synchronize at the end to read them
// all back. An event record is a marker rather than a barrier, so the step
// being measured is still the step that would have run -- which is the whole
// difference between this and wrapping every kernel in a sync.
//
// WHAT IT CANNOT DO, stated up front so it is not mistaken for ncu. It
// attributes wall time to regions, not kernels, and it cannot say WHY a region
// is slow -- no occupancy, no memory throughput, no stall reasons. It tells you
// where to point the real profiler. On a machine where the real profiler will
// not run, it is the difference between measuring and guessing.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace prof {

constexpr int MAXCAT = 24;
constexpr int MAXREG = 2048;  // regions per step; ~150 at the default depth

struct State {
    const char *name[MAXCAT] = {};
    double ms[MAXCAT] = {};
    int ncat = 0;

    cudaEvent_t beg[MAXREG], end[MAXREG];
    int cat[MAXREG];
    int n = 0;

    bool events_made = false;
    bool overflowed = false;
    int steps = 0;
    bool on = false;
};

inline State &S() {
    static State s;
    return s;
}

inline void set_enabled(bool on) { S().on = on; }
inline bool enabled() { return S().on; }

// strcmp rather than pointer identity: the region names are string literals in
// several translation units, and identical literals are not required to share
// an address across them.
inline int cat_id(const char *name) {
    State &s = S();
    for (int i = 0; i < s.ncat; ++i)
        if (std::strcmp(s.name[i], name) == 0) return i;
    if (s.ncat >= MAXCAT) return 0;
    s.name[s.ncat] = name;
    s.ms[s.ncat] = 0.0;
    return s.ncat++;
}

inline void begin(const char *name) {
    State &s = S();
    if (!s.on) return;
    if (!s.events_made) {
        for (int i = 0; i < MAXREG; ++i) {
            cudaEventCreate(&s.beg[i]);
            cudaEventCreate(&s.end[i]);
        }
        s.events_made = true;
    }
    if (s.n >= MAXREG) { s.overflowed = true; return; }
    s.cat[s.n] = cat_id(name);
    cudaEventRecord(s.beg[s.n]);
}

inline void end() {
    State &s = S();
    if (!s.on || s.n >= MAXREG) return;
    cudaEventRecord(s.end[s.n]);
    ++s.n;
}

// Called once per step, after the step's work is queued. The single sync here
// is the only one the instrumentation adds.
inline void flush() {
    State &s = S();
    if (!s.on) return;
    cudaDeviceSynchronize();
    for (int i = 0; i < s.n; ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s.beg[i], s.end[i]);
        s.ms[s.cat[i]] += ms;
    }
    s.n = 0;
    ++s.steps;
}

// Drop everything measured so far. Used to discard warm-up steps, whose
// allocator and cache behaviour is not representative.
inline void reset() {
    State &s = S();
    for (int i = 0; i < s.ncat; ++i) s.ms[i] = 0.0;
    s.steps = 0;
}

inline void report() {
    State &s = S();
    if (!s.on || s.steps == 0) return;
    double total = 0.0;
    for (int i = 0; i < s.ncat; ++i) total += s.ms[i];
    if (total <= 0.0) return;

    // Descending by share, which is the order the question is asked in.
    int order[MAXCAT];
    for (int i = 0; i < s.ncat; ++i) order[i] = i;
    for (int i = 1; i < s.ncat; ++i)
        for (int j = i; j > 0 && s.ms[order[j]] > s.ms[order[j - 1]]; --j) {
            const int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
        }

    printf("\nstep profile  (%d steps, cudaEvent regions, one sync per step)\n",
           s.steps);
    printf("%-28s %10s %8s\n", "region", "ms/step", "share");
    printf("-------------------------------------------------\n");
    for (int k = 0; k < s.ncat; ++k) {
        const int i = order[k];
        printf("%-28s %10.3f %7.1f%%\n", s.name[i], s.ms[i] / s.steps,
               100.0 * s.ms[i] / total);
    }
    printf("-------------------------------------------------\n");
    printf("%-28s %10.3f\n", "measured total", total / s.steps);
    if (s.overflowed)
        printf("  NOTE: region buffer overflowed -- numbers are incomplete\n");
    printf("  Regions nest none deeper than one level, so shares sum to the\n"
           "  instrumented part of a step, not to the step. The gap against\n"
           "  the reported ms/step is uninstrumented work plus launch overhead.\n");
}

// The instrumentation is compiled in unconditionally but costs a predictable
// branch when off; --profile is what arms it. Making it a macro keeps the call
// sites readable and lets them disappear entirely if that ever matters.
#define PROF_BEGIN(name) ::prof::begin(name)
#define PROF_END() ::prof::end()

}  // namespace prof
