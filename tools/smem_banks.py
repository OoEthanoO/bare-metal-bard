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
