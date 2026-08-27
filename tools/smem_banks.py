#!/usr/bin/env python3
"""Derive and check kernel 10's shared-memory layout, away from the GPU.

Kernel 10 stores each operand tile in the order the tensor core's registers
want it, so that a fragment load is one 128-bit access instead of four 32-bit
ones. That layout has three properties it must have, and none of them is
obvious by inspection:

  1. it is a bijection -- every element of the block tile lands somewhere, and
     nothing is overwritten;
  2. the fragment load a warp issues for unit u picks up, in register order,
     exactly the elements the `mma.m16n8k8` register mapping demands;
  3. neither the staging stores nor the fragment loads collide in the 32
     shared-memory banks.

Property 3 is why the swizzle exists. Without it the staging stores are 8-way
conflicted on A and 16-way on B: a warp of staging threads varies only in bits
that the slot index scales by 4, so 32 lanes land on 2-4 banks. Checking that
here rather than in `ncu` is the point -- it is a property of the index
arithmetic, so it can be settled before the kernel is built, and it stays
checked afterwards.

    python3 tools/smem_banks.py

Prints a table and exits nonzero if anything is wrong.
"""
import sys

BM, BN, BK = 128, 128, 32
THREADS = 256
MMA_M, MMA_K = 16, 8
UNIT = 128  # floats per operand tile: 32 lanes x 4

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


# --------------------------------------------------------------- the maps
def a_swz(unit):
    return (unit // (BM // MMA_M)) & 7


def b_swz(unit):
    return unit % (BN // 16)


def a_off(m, k, swz=True):
    unit = (k // MMA_K) * (BM // MMA_M) + m // MMA_M
    slot = (m % 8) * 4 + (k % 4)
    if swz:
        slot ^= a_swz(unit)
    elem = ((k % MMA_K) // 4) * 2 + ((m % MMA_M) // 8)
    return unit * UNIT + slot * 4 + elem


def b_off(k, n, swz=True):
    unit = (k // MMA_K) * (BN // 16) + n // 16
    slot = (n % 8) * 4 + (k % 4)
    if swz:
        slot ^= b_swz(unit)
    elem = ((k % MMA_K) // 4) + ((n % 16) // 8) * 2
    return unit * UNIT + slot * 4 + elem


# ------------------------------------------------------- 1. bijections
for name, n_elem, fn, dims in (
    ("As", BM * BK, a_off, [(m, k) for m in range(BM) for k in range(BK)]),
    ("Bs", BK * BN, b_off, [(k, n) for k in range(BK) for n in range(BN)]),
):
    seen = {}
    for x, y in dims:
        o = fn(x, y)
        if o in seen:
            fails.append("%s: %s and %s both map to %d" % (name, seen[o], (x, y), o))
            break
        seen[o] = (x, y)
    check(len(seen) == n_elem, "%s: covered %d of %d elements" % (name, len(seen), n_elem))

# ------------------------------------- 2. the fragment load picks up a0..a3
# The register mapping of mma.m16n8k8 with .tf32, confirmed on hardware by
# scripts/probe.bat rather than recalled:
#   g = lane>>2, t = lane&3
#   A: a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4)
#   B: b0=(t,g) b1=(t+4,g)                       (n8 tile; x2 for the n16 unit)
bad = 0
for kt in range(BK // MMA_K):
    for mi in range(BM // MMA_M):
        unit = kt * (BM // MMA_M) + mi
        for l in range(32):
            g, t = l >> 2, l & 3
            want = [(mi * 16 + g, kt * 8 + t), (mi * 16 + g + 8, kt * 8 + t),
                    (mi * 16 + g, kt * 8 + t + 4), (mi * 16 + g + 8, kt * 8 + t + 4)]
            base = unit * UNIT + (l ^ a_swz(unit)) * 4
            for e, (m, k) in enumerate(want):
                bad += a_off(m, k) != base + e
check(bad == 0, "A fragment load: %d register slots wrong" % bad)

bad = 0
for kt in range(BK // MMA_K):
    for nj in range(BN // 16):
        unit = kt * (BN // 16) + nj
        for l in range(32):
            g, t = l >> 2, l & 3
            want = [(kt * 8 + t, nj * 16 + g), (kt * 8 + t + 4, nj * 16 + g),
                    (kt * 8 + t, nj * 16 + 8 + g), (kt * 8 + t + 4, nj * 16 + 8 + g)]
            base = unit * UNIT + (l ^ b_swz(unit)) * 4
            for e, (k, n) in enumerate(want):
                bad += b_off(k, n) != base + e
check(bad == 0, "B fragment load: %d register slots wrong" % bad)

# --------------------------- 2b. the kernel's hoisted staging address form
# The kernel does not call the map four times per float4; it computes one base
# and one XOR per element, which is only valid because j occupies bits the
# base leaves clear. That algebra is easy to get wrong, so check it.
bad = 0
for m in range(BM):
    for c in range(BK // 4):
        k0 = c * 4
        unit = (k0 // MMA_K) * (BM // MMA_M) + m // MMA_M
        base = unit * UNIT + (((k0 % MMA_K) // 4) * 2 + ((m % MMA_M) // 8))
        s0 = ((m % 8) * 4) ^ a_swz(unit)
        for j in range(4):
            bad += base + ((s0 ^ j) * 4) != a_off(m, k0 + j)
check(bad == 0, "A hoisted staging address: %d mismatches" % bad)

bad = 0
for k in range(BK):
    for c in range(BN // 4):
        n0 = c * 4
        unit = (k // MMA_K) * (BN // 16) + n0 // 16
        base = unit * UNIT + (((k % MMA_K) // 4) + ((n0 % 16) // 8) * 2)
        s0 = ((n0 % 8) * 4 + (k % 4)) ^ b_swz(unit)
        for j in range(4):
            bad += base + ((s0 ^ (j * 4)) * 4) != b_off(k, n0 + j)
check(bad == 0, "B hoisted staging address: %d mismatches" % bad)


# ------------------- 2c. the transpose-aware staging in src/gemm.cu
# The model GEMM stages the same layout from four operand orientations. The map
# does not change -- it is a function of the logical (m,k), not of how the
# operand is stored -- but two things do:
#
#   * which axis the global float4 runs along, and therefore which bits of the
#     destination slot the four staged values step through (1 for a read along
#     k, 4 for a read along m or n);
#   * which tile coordinate the SWIZZLE must key on. It has to be uniform
#     across a warp doing a fragment load and varying across a warp doing a
#     staging store, and only the coordinate the staging warp actually walks is
#     both. Key it on the other one and it is warp-uniform during staging, so
#     it does nothing -- which is not a correctness bug and therefore does not
#     show up in any test. It showed up as the transposed cases running 20%
#     slower than the WMMA path they replaced.
#
# So this block checks both, for all four cases.
NT = 256  # threads in the model kernel


def swz(unit, kind, along_k):
    """`along_k` is whether the staging read for this operand runs along k."""
    per = (BM // MMA_M) if kind == "a" else (BN // 16)
    return ((unit // per) if along_k else (unit % per)) & 7


def off_a(m, k, along_k):
    unit = (k // MMA_K) * (BM // MMA_M) + m // MMA_M
    slot = ((m % 8) * 4 + (k % 4)) ^ swz(unit, "a", along_k)
    return unit * UNIT + slot * 4 + ((k % MMA_K) // 4) * 2 + ((m % MMA_M) // 8)


def off_b(k, n, along_k):
    unit = (k // MMA_K) * (BN // 16) + n // 16
    slot = ((n % 8) * 4 + (k % 4)) ^ swz(unit, "b", along_k)
    return unit * UNIT + slot * 4 + ((k % MMA_K) // 4) + ((n % 16) // 8) * 2


def stage(kind, trans):
    """Replay one operand's staging loop; return (address errors, worst ways)."""
    is_a = kind == "a"
    # transA=True means A is stored K x M, so the read runs along m, not k.
    along_k = (not trans) if is_a else trans
    wide = BM if (is_a and trans) else (BK if (is_a or trans) else BN)
    per = wide // 4
    stride = NT // per
    outer = (BK if is_a else BN) if trans else (BM if is_a else BK)
    step = 1 if along_k else 4
    bad, worst = 0, 0
    for off in range(0, outer, stride):
        for warp in range(NT // 32):
            for j in range(4):
                banks = {}
                for l in range(32):
                    tid = warp * 32 + l
                    row, col = tid // per, tid % per
                    if is_a:
                        m = col * 4 + j if trans else row + off
                        k = row + off if trans else col * 4 + j
                        base_m = col * 4 if trans else m
                        base_k = k if trans else col * 4
                        unit = (base_k // MMA_K) * (BM // MMA_M) + base_m // MMA_M
                        base = unit * UNIT + (((base_k % MMA_K) // 4) * 2 +
                                              ((base_m % MMA_M) // 8))
                        slot = ((base_m % 8) * 4 + (base_k % 4)) ^ swz(unit, "a", along_k)
                        got = base + ((slot ^ (j * step)) * 4)
                        bad += got != off_a(m, k, along_k)
                    else:
                        k = col * 4 + j if trans else row + off
                        n = row + off if trans else col * 4 + j
                        base_k = col * 4 if trans else k
                        base_n = n if trans else col * 4
                        unit = (base_k // MMA_K) * (BN // 16) + base_n // 16
                        base = unit * UNIT + (((base_k % MMA_K) // 4) +
                                              ((base_n % 16) // 8) * 2)
                        slot = ((base_n % 8) * 4 + (base_k % 4)) ^ swz(unit, "b", along_k)
                        got = base + ((slot ^ (j * step)) * 4)
                        bad += got != off_b(k, n, along_k)
                    banks[got % 32] = banks.get(got % 32, 0) + 1
                worst = max(worst, max(banks.values()))
    return bad, worst


print("model GEMM staging, all four operand orientations:")
print("%-24s %10s %14s" % ("", "addr errors", "worst bank way"))
for kind, trans, label in (("a", False, "A  transA=false"), ("a", True, "A  transA=true"),
                           ("b", False, "B  transB=false"), ("b", True, "B  transB=true")):
    bad, ways = stage(kind, trans)
    print("%-24s %10d %13dx" % (label, bad, ways))
    check(bad == 0, "%s: %d staging addresses wrong" % (label, bad))
    check(ways <= 2, "%s: staging is %d-way bank conflicted" % (label, ways))
print()


# ---------------------------------------------- 2b. the cp.async staging map
#
# `cp.async` copies 4 bytes at a time here (cp_async.cuh says why the size is
# forced), so a staging thread no longer writes four consecutive elements. It
# writes ONE, and consecutive lanes take consecutive positions along whichever
# axis the operand is contiguous in. That keeps each individual copy a coalesced
# 128-byte warp read on the GLOBAL side -- and completely changes which bits
# vary across a warp on the SHARED side.
#
# WHICH BREAKS TWO OF THE FOUR CASES ON THE STORE SIDE. A logical slot is
# (m%8)*4 + (k%4): the low two bits come from k, the top three from m. An
# address is slot*4 + elem floats, so the k bits stride banks by 4 and the m
# bits stride them by 16 -- which is 2 banks, not 8. Kernel 10 never meets this
# because each of its threads writes four elements and steps the slot itself,
# moving both bit groups. With one element per thread, only the axis the warp
# walks moves:
#
#     A transA=false  walks k  ->  2-way
#     A transA=true   walks m  ->  8-way      <- and the XOR swizzle cannot
#     B transB=false  walks n  ->  8-way         help, because it is uniform
#     B transB=true   walks k  ->  2-way         inside a unit
#
# Rotating the slot so the varying bits land where they stride banks fixes that
# exactly, and is invisible to correctness -- the fragment load reads whole
# units, so any bijection on slots still has 32 lanes covering 32 distinct
# 16-byte chunks.
#
# AND IT IS 7% SLOWER ON THE GPU (11994 -> 11152 GF/s, kernel 11 at N=4096).
# The reason is the column this table did not have when the rotation was
# written: a permutation of slots applies to the LOAD as well, and a 128-bit
# load is serviced eight lanes at a time, so each octet must cover 32 banks by
# itself. The rotation gives the store side 32 lanes over 32 banks and takes the
# load side from 1-way to 4-way. It does not remove a conflict, it MOVES one.
#
# Which is why both columns are printed and neither is asserted alone. The
# store side is asynchronous under `cp.async` and has a whole k-chunk of
# arithmetic to hide behind; the load side stalls the warp that issued it. A
# tool that had only ever counted stores would have recommended the slower
# kernel with total confidence -- it did, and this comment is the repair.
CP_BK = 16  # the pipelined kernels trade BK for stages; see gemm.cu


def rot(s, on):
    return (((s & 3) << 3) | (s >> 2)) if on else s


def cp_off_a(m, k, along_k):
    unit = (k // MMA_K) * (BM // MMA_M) + m // MMA_M
    slot = rot((m % 8) * 4 + (k % 4), not along_k) ^ swz(unit, "a", along_k)
    return unit * UNIT + slot * 4 + ((k % MMA_K) // 4) * 2 + ((m % MMA_M) // 8)


def cp_off_b(k, n, along_k):
    unit = (k // MMA_K) * (BN // 16) + n // 16
    slot = rot((n % 8) * 4 + (k % 4), not along_k) ^ swz(unit, "b", along_k)
    return unit * UNIT + slot * 4 + ((k % MMA_K) // 4) + ((n % 16) // 8) * 2


def cp_elems(kind, trans, bk):
    """The (element index) -> (m,k) or (k,n) map the cp.async staging uses."""
    is_a = kind == "a"
    along_k = (not trans) if is_a else trans
    inner = bk if along_k else (BM if is_a else BN)
    total = (BM if is_a else BN) * bk
    for e in range(total):
        if is_a:
            yield e, ((e % inner) if trans else (e // inner),
                      (e // inner) if trans else (e % inner))
        else:
            yield e, ((e // inner) if not trans else (e % inner),
                      (e % inner) if not trans else (e // inner))


def cp_stage(kind, trans, nt, bk=CP_BK, rotate=True):
    """Worst simultaneous hits on one bank across a warp of cp.async stores."""
    is_a = kind == "a"
    along_k = (not trans) if is_a else trans
    ak = along_k if rotate else True  # rotate=False models the unfixed version
    worst, seen = 0, {}
    banks = {}
    for e, (x, y) in cp_elems(kind, trans, bk):
        o = cp_off_a(x, y, ak) if is_a else cp_off_b(x, y, ak)
        seen[o] = seen.get(o, 0) + 1
        banks.setdefault(e // 32, {}).setdefault(o % 32, 0)
        banks[e // 32][o % 32] += 1
    for w in banks.values():
        worst = max(worst, max(w.values()))
    total = (BM if is_a else BN) * bk
    return worst, len(seen) == total


def cp_load_ways(rotate):
    """A fragment load is 128-bit: eight lanes per phase, 32 banks per phase."""
    worst = 0
    for unit in range(BK // MMA_K * (BN // 16)):
        f = unit % 8
        for phase in range(4):
            banks = {}
            for l in range(phase * 8, phase * 8 + 8):
                p = rot(l, rotate) ^ f
                for e in range(4):
                    banks[(p * 4 + e) % 32] = banks.get((p * 4 + e) % 32, 0) + 1
            worst = max(worst, max(banks.values()))
    return worst


print("cp.async staging (BK=%d), all four operand orientations:" % CP_BK)
print("%-24s %7s %12s %10s %8s" % ("", "walks", "stores u/r", "loads u/r", "onto"))
for kind, trans, label in (("a", False, "A  transA=false"), ("a", True, "A  transA=true"),
                           ("b", False, "B  transB=false"), ("b", True, "B  transB=true")):
    is_a = kind == "a"
    along_k = (not trans) if is_a else trans
    axis = "k" if along_k else ("m" if is_a else "n")
    unrot, _ = cp_stage(kind, trans, NT, rotate=False)
    rotd, bij = cp_stage(kind, trans, NT, rotate=True)
    # The load side depends only on whether the permutation is applied.
    lu, lr = cp_load_ways(False), cp_load_ways(not along_k)
    print("%-24s %7s %6dx /%3dx %5dx /%3dx %8s"
          % (label, axis, unrot, rotd, lu, lr, "1:1" if bij else "COLLIDES"))
    check(bij, "cp.async %s: staging map is not a bijection" % label)
    # What IS asserted: whichever variant ships, the exposed side stays clean.
    # The kernels ship unrotated, so the load column is the one that binds.
    check(lu <= 2, "cp.async %s: fragment load is %d-way conflicted" % (label, lu))
print("(u = unrotated, r = rotated. The kernels ship unrotated: the store")
print(" column is worse and the load column, which is the exposed one, is not.)")
print()

# --------------------------------------------------------- 3. bank conflicts
def store_ways(is_a, swz):
    """Worst simultaneous hits on one bank across a warp of staging threads."""
    worst = 0
    per = (BK // 4) if is_a else (BN // 4)
    stride = THREADS // per
    outer = range(0, BM, stride) if is_a else range(0, BK, stride)
    for off in outer:
        for warp in range(THREADS // 32):
            for j in range(4):
                banks = {}
                for l in range(32):
                    tid = warp * 32 + l
                    if is_a:
                        o = a_off(tid // per + off, (tid % per) * 4 + j, swz)
                    else:
                        o = b_off(tid // per + off, (tid % per) * 4 + j, swz)
                    banks[o % 32] = banks.get(o % 32, 0) + 1
                worst = max(worst, max(banks.values()))
    return worst


def load_ways(swz_fn, nunits):
    """128-bit loads are serviced 8 lanes at a time; each octet needs 32 banks."""
    worst = 0
    for unit in range(nunits):
        f = swz_fn(unit) if swz_fn else 0
        for phase in range(4):
            banks = {}
            for l in range(phase * 8, phase * 8 + 8):
                for e in range(4):
                    b = ((l ^ f) * 4 + e) % 32
                    banks[b] = banks.get(b, 0) + 1
            worst = max(worst, max(banks.values()))
    return worst


print("%-28s %10s %10s" % ("", "no swizzle", "swizzled"))
rows = [
    ("A staging stores", store_ways(True, False), store_ways(True, True)),
    ("B staging stores", store_ways(False, False), store_ways(False, True)),
    ("A fragment loads", load_ways(None, 32), load_ways(a_swz, 32)),
    ("B fragment loads", load_ways(None, 32), load_ways(b_swz, 32)),
]
for name, a, b in rows:
    print("%-28s %9dx %9dx" % (name, a, b))

for _, _, after in rows:
    check(after <= 2, "a swizzled access is still %d-way conflicted" % after)

print()
if fails:
    for f in fails:
        print("FAIL:", f)
    sys.exit(1)
print("layout ok: bijective, fragment-exact, at most 2-way conflicted")
