// Which dispatch branches did this process actually take?
//
// WHY THIS EXISTS. The layernorm backward was selected by `if (C == 384)` --
// this model's n_embd -- while the gradient check ran at C=128, which took the
// other branch. So the kernel the model trains with had never been
// gradient-checked; the check validated its neighbour, and had done since the
// fast path was written. Nothing about that was visible by reading either
// file, because the two numbers live in different files.
//
// That is not a layernorm problem, it is a dispatch problem, and this repo is
// full of dispatches: the GEMM picks a narrow or wide tile, a split count, one
// of ten epilogue masks and one of four transpose cases; the fused attention
// picks a tile config per context length and head size. Any of them can have
// the same shape of hole.
//
// So instead of arguing about it, record it. Every dispatch site calls
// BMB_COVER("tag") with the branch it took, and at exit the process prints the
// set. Run the model, run the tests, diff the two sets: anything the model
// reaches that the tests do not is a kernel running untested in training.
//
// COST WHEN OFF IS ONE PREDICTED BRANCH on a host function that is already
// launching a kernel -- the enabled() check reads a static bool. It is off
// unless BMB_COVER is set in the environment, so it cannot affect a timed run
// by accident. It deliberately records only that a branch was REACHED, not how
// often: a count would vary with batch size and step count and make two runs
// impossible to diff.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

namespace cover {

inline bool enabled() {
    static const bool on = getenv("BMB_COVER") != nullptr;
    return on;
}

inline std::set<std::string> &tags() {
    static std::set<std::string> s;
    return s;
}

inline std::mutex &lock() {
    static std::mutex m;
    return m;
}

// Printed at exit rather than on demand, so a tool that forgets to call a dump
// function still reports. One line per branch, sorted, so two runs diff.
inline void dump() {
    const std::set<std::string> &s = tags();
    fprintf(stderr, "=== BMB_COVER %zu branches ===\n", s.size());
    for (const std::string &t : s) fprintf(stderr, "COVER %s\n", t.c_str());
}

// Dumped by a static destructor rather than atexit(), and the difference is
// not cosmetic. The first version registered atexit(dump) on the first note(),
// which happens BEFORE the tag set is first constructed -- so the set was
// destroyed first and dump() then walked freed strings. It printed
// "=== BMB_COVER 14 branches ===" followed by fourteen empty tags: the count
// lived in the set's own storage and survived, the string buffers did not.
//
// A static object whose destructor dumps has the ordering the right way round,
// because destruction is the reverse of construction: touch tags() first so it
// is constructed first, and this dumper is then destroyed before it.
struct Dumper {
    ~Dumper() { dump(); }
};

inline void note(const char *tag) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> g(lock());
    tags();            // constructed before the dumper, so destroyed after it
    static Dumper d;   // its destructor runs while tags() is still alive
    (void)d;
    tags().insert(tag);
}

// Formatted variant, for tags that carry a tile shape or a split count. The
// enabled() check comes first so a disabled run does not pay for the vsnprintf.
inline void notef(const char *fmt, ...) {
    if (!enabled()) return;
    char buf[192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    note(buf);
}

}  // namespace cover

#define BMB_COVER(tag) ::cover::note(tag)
#define BMB_COVERF(...) ::cover::notef(__VA_ARGS__)
