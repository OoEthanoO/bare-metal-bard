import { data } from './data';

// Slow -> fast. Matches the ramp used by tools/plot_results.py so the inline
// bars and the SVG charts read as one system.
const RAMP = [
  '#b3543f', '#c07f3c', '#c9a63c', '#9aa845', '#6a9f5c',
  '#3f8f74', '#2f7d8a', '#2f6f9a', '#5a5aa8',
];
const CUBLAS_C = '#8a6fb0';

const byName = (n: string) => data.kernels.find((k) => k.name === n)!;

function KernelBars() {
  // Kernels 8 and 9 pass the cuBLAS line, so the axis cannot be scaled to it.
  const fastest = Math.max(...data.kernels.map((k) => k.gflops));
  const max = Math.max(data.cublas, fastest) * 1.02;
  return (
    <div className="bars">
      {data.kernels.map((k, i) => (
        <div className="bar" key={k.name}>
          <div className="name">{k.name}</div>
          <div className="track">
            <div
              className="fill"
              style={{
                width: `${Math.max((k.gflops / max) * 100, 0.4)}%`,
                background: RAMP[Math.min(i, RAMP.length - 1)],
              }}
            />
          </div>
          <div className="val">
            <b>{k.gflops.toFixed(0)}</b> <span>{k.pct.toFixed(0)}%</span>
          </div>
        </div>
      ))}
      <div className="bar">
        <div className="name" style={{ color: 'var(--fg)' }}>
          cuBLAS
        </div>
        <div className="track">
          <div
            className="fill"
            style={{ width: `${(data.cublas / max) * 100}%`, background: CUBLAS_C, opacity: 0.42 }}
          />
        </div>
        <div className="val">
          <b>{data.cublas.toFixed(0)}</b> <span>100%</span>
        </div>
      </div>
    </div>
  );
}

export default function Page() {
  const t = data.training;
  const f = data.flash as any;
  const first = data.kernels[0];
  const dbuf = byName('dbuffer');
  const tc = byName('tensorcore');
  const warp = byName('warptile');
  const vec = byName('vectorized');
  const speedup = tc.gflops / first.gflops;

  const tcTf32 = (data.tf32 as any)?.tensorcore?.[String(data.benchSize)];
  const dbTf32 = (data.tf32 as any)?.dbuffer?.[String(data.benchSize)];

  return (
    <main className="wrap">
      <header className="hero">
        <p className="eyebrow">CUDA · from scratch</p>
        <h1>Writing a CUDA matmul that catches cuBLAS — then training a GPT on it</h1>
        <p className="lede">
          Nine rewrites of a single kernel, from the naive version everyone writes first to a
          double-buffered, warp-tiled one that matches NVIDIA&rsquo;s hand-tuned library — and a
          tensor-core version that passes it. Then a language model trained end to end on those
          kernels, with no PyTorch and no vendor BLAS anywhere in the training path.
        </p>
        <div className="statgrid">
          <div className="stat">
            <span className="v">{speedup.toFixed(0)}×</span>
            <span className="k">faster than the naive kernel</span>
          </div>
          <div className="stat">
            <span className="v">{dbuf.pct.toFixed(0)}%</span>
            <span className="k">of cuBLAS in fp32, like for like</span>
          </div>
          <div className="stat">
            <span className="v">{(t.tokPerSec / 1000).toFixed(1)}k</span>
            <span className="k">tokens/s training a 10.8M GPT</span>
          </div>
          <div className="stat">
            <span className="v">{t.bestVal.toFixed(3)}</span>
            <span className="k">val loss, nats/char</span>
          </div>
        </div>
      </header>

      <p>
        The hardware is an <strong>RTX 4070 Laptop</strong> (Ada, sm_89): 36 SMs, 256 GB/s of memory
        bandwidth, 8 GB, and a 55 W power budget. Two numbers from that spec sheet explain
        everything that follows.
      </p>
      <p>
        At the clock these benchmarks run at, the card can do about 11.1 TFLOP/s of fp32 and move
        256 GB/s. Divide them and you get the <strong>ridge point: 43 FLOP/byte</strong>. Every byte
        read from memory has to feed 43 floating-point operations before the arithmetic units stop
        waiting on memory. A naive matmul manages <strong>0.25</strong>.
      </p>
      <p>
        That gap — a factor of ~170 — is the entire project. Almost every optimization below is the
        same move applied at a different level of the memory hierarchy:{' '}
        <em>load a value once, then spend it on as much arithmetic as possible before letting it
        go.</em>
      </p>

      <h2>The nine kernels</h2>
      <KernelBars />
      <p className="cap">
        GFLOP/s at N={data.benchSize}, SM clock pinned to 1200 MHz. The cuBLAS baseline is{' '}
        <strong>fp32</strong>, which is what cuBLAS does by default. Kernel 9 uses tensor cores and
        is therefore not computing the same thing — see below.
      </p>

      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>kernel</th>
              <th className="n">GFLOP/s</th>
              <th className="n">% cuBLAS</th>
              <th className="n">FLOP/byte</th>
              <th>what changed</th>
            </tr>
          </thead>
          <tbody>
            {data.kernels.map((k) => (
              <tr key={k.name} className={k.name === 'dbuffer' || k.name === 'tensorcore' ? 'hi' : undefined}>
                <td className="n">{k.id}</td>
                <td>
                  <code>{k.name}</code>
                </td>
                <td className="n">{k.gflops.toFixed(1)}</td>
                <td className="n">{k.pct.toFixed(1)}%</td>
                <td className="n">{k.intensity}</td>
                <td>{k.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Kernel 1 → 2: one index, 8× faster</h3>
      <p>
        The naive kernel maps <code>threadIdx.x</code> to the row of the output. That is the natural
        thing to write and exactly wrong. Lanes 0–31 of a warp then read addresses K floats apart,
        so each lane needs its own memory transaction — 32 transactions for what could be one.
      </p>
      <p>
        Swapping the mapping so <code>threadIdx.x</code> is the <em>column</em> makes the warp read
        32 consecutive floats: one 128-byte transaction. The arithmetic is byte-for-byte identical.
        Only the address pattern moved, and throughput went from {first.gflops.toFixed(0)} to{' '}
        {data.kernels[1].gflops.toFixed(0)} GFLOP/s.
      </p>

      <h3>Kernels 3–5: reuse, at three levels</h3>
      <p>
        Shared memory tiling gets each value loaded from global memory once per block instead of
        once per thread (0.25 → 8 FLOP/byte). Then register tiling attacks the next bottleneck:
        every fused multiply-add in the shared-memory kernel needs two SMEM loads, and shared
        memory cannot feed the FMA pipes that fast. Giving each thread an 8×8 patch of output means
        16 loads serve 64 FMAs instead of 128.
      </p>

      <h2>The measurement was nearly wrong — in my favor</h2>
      <div className="note">
        <p style={{ margin: 0 }}>
          <strong>The problem.</strong> Left alone, this 55 W laptop GPU boosts to 3105 MHz and then
          sags as it hits the power cap. Measured cuBLAS at N=2048 swung between{' '}
          <strong>7.6 and 12.7 TFLOP/s</strong> across runs on thermal state alone — a 60% swing in
          the <em>denominator</em> of every &ldquo;% of cuBLAS&rdquo; claim. I could have reported
          almost any number I wanted.
        </p>
      </div>
      <p>
        The fix is to pin the SM clock. 1500 MHz does not hold — the power cap drags it to ~1320 and
        it wanders. 1200 MHz holds rock steady through a sustained dense-GEMM load (39–47 W, 68–83
        °C, clock never moved). With clocks pinned, best-vs-median across runs agrees to under 1%.
      </p>
      <p>
        Correctness needed the same care. Elementwise relative error is the wrong metric for a
        matmul: each output is a sum of K products, so where cancellation drives an entry near zero
        the rounding noise of the other terms remains. An entry of magnitude 1e-3 can carry 1e-4 of
        absolute error while every input was computed correctly. Comparing worst absolute error
        against <code>max|ref|</code> separates cleanly — reordered fp32 summation lands around
        1e-6, a real bug lands at 1e-1.
      </p>

      <h2>The profiler found what reading could not</h2>
      <p>
        At kernel 6 I was at {vec.pct.toFixed(0)}% and out of ideas. Nsight Compute gave the answer
        in three lines:
      </p>
      <pre>
        <code>{`DRAM Throughput          15.1%   <- global memory is not the problem
L1/TEX Cache Throughput  81.4%   <- saturated
Compute (SM) Throughput  54.2%`}</code>
      </pre>
      <p>
        Shared memory and L1 share the same LSU/MIO datapath on NVIDIA hardware. 81% L1 against 15%
        DRAM means the kernel was bound on <strong>shared-memory-to-register traffic</strong>. The
        FMA units were idle waiting for operands. No further work on global memory access — the
        thing I had spent five kernels on — would have bought anything.
      </p>
      <p>
        Warp tiling fixes exactly that, by adding a blocking level between the block and the thread
        so each value pulled from SMEM feeds more arithmetic. Loads per FMA drop from 0.25 to
        0.1875. Measured after the change:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>metric</th>
              <th className="n">kernel 6</th>
              <th className="n">kernel 7</th>
              <th className="n">kernel 8</th>
            </tr>
          </thead>
          <tbody>
            <tr className="hi">
              <td>L1/TEX throughput</td>
              <td className="n">81.4%</td>
              <td className="n">50.0%</td>
              <td className="n">52.6%</td>
            </tr>
            <tr>
              <td>Memory throughput</td>
              <td className="n">73.8%</td>
              <td className="n">45.3%</td>
              <td className="n">46.1%</td>
            </tr>
            <tr>
              <td>Compute (SM)</td>
              <td className="n">54.2%</td>
              <td className="n">54.8%</td>
              <td className="n">55.8%</td>
            </tr>
            <tr className="hi">
              <td>Warp cycles per issued instruction</td>
              <td className="n">5.75</td>
              <td className="n">3.08</td>
              <td className="n">2.85</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Kernel 8: the loop was shaped wrong</h2>
      <p>
        Kernel 7 left the kernel in a genuinely different regime — neither memory (45%) nor compute
        (55%) saturated. It was <strong>latency bound</strong>, and the reason was visible in the
        loop structure rather than in any counter:
      </p>
      <pre>
        <code>{`load global -> shared        <- ~500 cycle latency
__syncthreads()              <- every warp blocks until it lands
compute on the tile
__syncthreads()              <- and blocks again before overwriting`}</code>
      </pre>
      <p>
        The load is immediately followed by a barrier, so nothing overlaps it. Every K-chunk pays a
        full round trip to DRAM with the arithmetic units idle. Double buffering keeps two tiles in
        shared memory: while the warps compute on one, the next chunk&rsquo;s loads are already in
        flight. It also halves the barriers, because reading one buffer and writing the other
        cannot conflict.
      </p>
      <p>
        That took kernel 8 to <strong>{dbuf.pct.toFixed(1)}% of cuBLAS</strong> at N=
        {data.benchSize}, and past it at sizes that divide evenly into the 128×128 block tile —
        104.0% at N=1536 and 104.2% at N=6144, reproducible across runs. The cost was 219 registers
        against 186, and 32 KiB of shared memory against 16.
      </p>

      <h2>Kernel 9: the silicon I had not touched</h2>
      <p>
        Everything to this point runs on the SM&rsquo;s fp32 FMA pipes. The card also has tensor
        cores, which had been idle for eight kernels. The honest way to see how much that matters
        is to let cuBLAS use them too:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>at N={data.benchSize}</th>
              <th className="n">GFLOP/s</th>
              <th className="n">vs fp32 cuBLAS</th>
              <th className="n">vs TF32 cuBLAS</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>cuBLAS, true fp32 (default)</td>
              <td className="n">{data.cublas.toFixed(0)}</td>
              <td className="n">100%</td>
              <td className="n">—</td>
            </tr>
            <tr>
              <td>cuBLAS, TF32 tensor cores</td>
              <td className="n">{tcTf32 ? tcTf32.cublasTf32.toFixed(0) : '—'}</td>
              <td className="n">
                {tcTf32 ? ((tcTf32.cublasTf32 / data.cublas) * 100).toFixed(0) : '—'}%
              </td>
              <td className="n">100%</td>
            </tr>
            <tr>
              <td>
                <code>dbuffer</code> (mine, fp32)
              </td>
              <td className="n">{dbuf.gflops.toFixed(0)}</td>
              <td className="n">{dbuf.pct.toFixed(1)}%</td>
              <td className="n">{dbTf32 ? dbTf32.pct.toFixed(1) : '—'}%</td>
            </tr>
            <tr className="hi">
              <td>
                <code>tensorcore</code> (mine, TF32)
              </td>
              <td className="n">{tc.gflops.toFixed(0)}</td>
              <td className="n">{tc.pct.toFixed(1)}%</td>
              <td className="n">{tcTf32 ? tcTf32.pct.toFixed(1) : '—'}%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        So the tensor-core kernel beats fp32 cuBLAS by {(tc.pct - 100).toFixed(0)}%, and trails
        cuBLAS&rsquo;s own tensor-core path by about{' '}
        {tcTf32 ? (100 - tcTf32.pct).toFixed(0) : '23'}%. Both of those are worth saying out loud;
        quoting only the first would be the flattering half.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          <strong>TF32 is not free speed.</strong> Despite the name it has fp32&rsquo;s 8-bit
          exponent but only 10 mantissa bits — fp32&rsquo;s range at roughly fp16&rsquo;s precision.
          Measured normwise error against an fp32 reference goes from <strong>6.2e-08</strong> for
          the fp32 kernels to <strong>2.4e-04</strong> for this one. That is ~4000× more error, and
          it is a deliberate trade, not a bug: it is the trade that makes neural network training
          fast, and it is why this kernel carries its own tolerance in the test suite rather than
          quietly loosening the bar for everyone.
        </p>
      </div>
      <p>
        Tuning it was also a reminder that intuition is not a substitute for measurement. The first
        version gave each thread a 4×4 grid of accumulator fragments and hit{' '}
        <strong>255 registers</strong> — the hardware ceiling — which throttled occupancy. Spreading
        the same block tile over 8 warps instead of 4 cut that to 128 registers. And of two
        arrangements with <em>identical</em> register counts and identical instruction counts, one
        was 27% faster than the other, purely from how the warps&rsquo; fragment loads land in
        shared memory.
      </p>

      <figure>
        <img src="/sgemm_scaling.svg" alt="Fraction of cuBLAS achieved, by matrix size" />
        <figcaption>
          Small matrices fall off because a 128×128 block tile leaves most of the 36 SMs idle — at
          N=512 the grid is only 4×4 blocks.
        </figcaption>
      </figure>

      <h2>Four ways not to speed up a tensor-core kernel</h2>
      <p>
        Kernel 9 trails cuBLAS&rsquo;s own TF32 path by about 21%. The obvious place to look is the
        profiler, which says: DRAM 64.6%, compute 36.9%, L2 hit rate 58%, occupancy 32.8%. The
        tempting reading is <em>bandwidth bound</em> — and three of the four attempts below came
        from taking that at face value. All four made it slower. Three runs each, N=
        {data.benchSize}:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>attempt</th>
              <th className="n">GFLOP/s</th>
              <th className="n">delta</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>grouped block scheduling</td>
              <td className="n">8290 → 7677</td>
              <td className="n">−7.4%</td>
            </tr>
            <tr className="hi">
              <td>bigger tile, 256×128</td>
              <td className="n">8290 → 6930</td>
              <td className="n">−16.4%</td>
            </tr>
            <tr>
              <td>
                <code>BK</code> 32 → 16
              </td>
              <td className="n">8290 → 7899</td>
              <td className="n">−4.7%</td>
            </tr>
            <tr>
              <td>double buffering</td>
              <td className="n">7899 → 7518</td>
              <td className="n">−4.8%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The <strong>bigger tile is the one that settles it</strong>. A block tile&rsquo;s arithmetic
        intensity is <code>BM·BN / (2(BM+BN))</code> — 32 FLOP/byte at 128×128, and 42.7 at 256×128,
        which clears this card&rsquo;s 43 ridge point. If the kernel were really bandwidth bound,
        that is the fix. It cost 16%. So it is <em>not</em> bandwidth bound, however much the DRAM
        counter looks like it.
      </p>
      <p>
        And double buffering is kernel 8&rsquo;s cure for exactly this
        neither-counter-is-saturated signature, which earlier in this project was worth 6%. Here it
        does nothing, and not for want of registers — 128 either way, twelve bytes of spill. So the
        latency is not on the global-memory path either.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          Ruling things out is the useful part. Global bandwidth, L2 reuse, tile size and global
          latency are all eliminated here by measurement rather than by argument, which leaves the
          abstraction itself: WMMA fixes the fragment layout and forces four separate shared-memory
          reads per fragment where raw <code>mma.sync</code> would need one. That is a specific,
          falsifiable claim, so the next kernel tests it — and it turns out to be wrong.
        </p>
      </div>

      <h2>Kernel 10: the hypothesis was wrong and the kernel got faster anyway</h2>
      <p>
        <strong>Shared memory does not have to hold a matrix.</strong> It only has to hold whatever
        makes the next read cheap. The <code>mma.m16n8k8</code> TF32 instruction requires each of
        the 32 lanes to hold four particular elements of A —{' '}
        <code>a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4)</code>, where <code>g=laneid&gt;&gt;2</code>{' '}
        and <code>t=laneid&amp;3</code>. In a row-major tile those sit at four unrelated addresses.
        So kernel 10 stores each 16×8 tile <em>lane-major</em> — lane <code>L</code>&rsquo;s four
        elements at <code>L*4</code> — and the fragment load becomes one 128-bit access at{' '}
        <code>base + laneid*16</code>, which is also the ideal shared-memory pattern: 32 lanes over
        512 contiguous bytes, no bank conflict possible. Per warp per k-step, 24 shared-load
        instructions become 6, moving exactly the same 3072 bytes.
      </p>
      <p>
        <strong>It bought 2.7%.</strong> Instruction issue on the shared path was never the
        constraint. The bytes were — and WMMA moves exactly as many.
      </p>
      <p>
        The other 8% came from somewhere kernel 9 could not reach. What this kernel is actually
        limited by is shared-memory <em>reuse</em>: bytes read from shared per{' '}
        <code>mma</code> issued is <code>4096·(WM+WN)/(WM·WN)</code>, which falls only when both
        warp-tile dimensions grow — and the accumulator costs <code>WM·WN/32</code> registers per
        thread, so reuse is a register-budget problem in disguise. Going from a 32×64 warp tile to
        64×64 halves it, and is worth <strong>8.7%</strong>, three times what hand-written PTX was
        worth on its own.
      </p>
      <p>
        And that exact shape is already in kernel 9&rsquo;s tuning table, at 7783 GF/s — <em>slower</em>{' '}
        than the shape it settled on. WMMA at a 64×64 warp tile needs 255 registers and spills; the
        hand-written version spills four bytes. <strong>So raw PTX did matter — indirectly, by
        making the tile affordable rather than by making the loads cheaper.</strong> The stated
        hypothesis was wrong and the conclusion drawn from it was right, which is not the same
        thing as being right.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>at N={data.benchSize}</th>
              <th className="n">GFLOP/s</th>
              <th className="n">vs cuBLAS TF32</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>kernel 9, WMMA</td>
              <td className="n">8318</td>
              <td className="n">79.2%</td>
            </tr>
            <tr>
              <td>kernel 10, raw mma.sync, same tile</td>
              <td className="n">8540</td>
              <td className="n">81.3%</td>
            </tr>
            <tr className="hi">
              <td>kernel 10, 64×64 warp tile</td>
              <td className="n">9210</td>
              <td className="n">87.7%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <h3>Two failures on the way, both worth more than the result</h3>
      <p>
        <strong>A fragment layout is a fact to measure, not to recall.</strong> I wrote the A
        register order from memory and swapped <code>a1</code> and <code>a2</code> — correct for the
        analogous f16 shape, wrong for TF32, which concatenates two k4 chunks rather than two row
        halves. The kernel compiled, ran at full speed, and returned garbage. Guessing produced a{' '}
        <em>silent</em> wrong answer, so there is now a one-hot probe that discovers the mapping on
        the hardware: set one element of A to 1, make B the identity, and see which lane and
        register light up.
      </p>
      <p>
        <strong>Fewer instructions is worth nothing if the bytes arrive four at a time.</strong> The
        first correct version ran 8% <em>slower</em> than the WMMA kernel it was meant to beat, on a
        layout whose entire justification was cheaper shared access. Lane-major staging is 8-way
        bank conflicted on A and 16-way on B, because a warp of staging threads varies only in bits
        the slot index scales by 4. The fix is an XOR swizzle keyed on the tile index: staging sees
        it vary across the warp and spreads out, while a fragment load reads one whole tile per warp
        so it is uniform there and the permutation is invisible. 8-way and 16-way become 2-way, and
        the load stays perfectly conflict-free.
      </p>
      <p>
        That is index arithmetic, not a hardware mystery, so it can be settled without a GPU.{' '}
        <code>tools/smem_banks.py</code> derives both layouts, proves they are bijections, checks
        that a fragment load picks up exactly the elements the instruction demands in exactly the
        right register slots, and simulates the bank pattern of every access.
      </p>

      <h3>Putting it in the model, where it behaves differently</h3>
      <p>
        The ladder kernel only does <code>NN</code>; the model needs all four transpose cases. The
        layout ports cleanly, because it is a function of the <em>logical</em> element{' '}
        <code>(m,k)</code> rather than of how the operand is stored — all four cases share one map,
        and only the axis the global <code>float4</code> runs along changes.{' '}
        <strong>67.7 → 65.2 ms per training step, 3.7%</strong>, replicated three times each way at
        a pinned clock, with identical loss to four decimals.
      </p>
      <p>
        <strong>But the swizzle has to key on the axis staging walks.</strong> It must be uniform
        across a warp doing a fragment load and varying across a warp doing a staging store, and
        only one tile coordinate is both — the one the staging warp walks, which the transpose flag
        decides. I ported the untransposed key to all four cases. For the transposed ones that is
        warp-uniform during staging, so the swizzle does nothing and the stores go back to 16-way
        conflicted. <strong>It is not a correctness bug</strong>, so every test passed. It showed up
        only as the transposed cases running 20% slower than the WMMA path they replaced: geomean
        over the model&rsquo;s twenty shape/transpose combinations was <strong>0.887×</strong>, and{' '}
        <strong>1.044×</strong> with the key fixed. What identified it was the pattern —{' '}
        <code>NN</code> and <code>NT</code> near parity, <code>TN</code> and <code>TT</code> at
        0.80–0.84, which is exactly the half where <code>transA</code> is true.
      </p>
      <p>
        <strong>And the best tile on a square benchmark is not the best tile in the model.</strong>{' '}
        The 64×64 warp tile that is worth 8.7% at N=4096 <em>loses</em> in situ, 1.034× against the
        narrower shape&rsquo;s 1.044×. The model&rsquo;s GEMMs are 4096×384×384 and friends, so a
        128×128 block tile gives 96 blocks against 36 SMs — the machine is not full, and a
        128-thread block brings half as many warps per SM to hide latency with. The extra reuse is
        real and there is nothing to spend it on. The ladder and the model deliberately run
        different tiles.
      </p>

      <h2>The compiler optimized the thing it could see</h2>
      <p>
        Kernel 9 sat at 8064 GF/s until a question about a version number. The writeup said CUDA
        12.5; 13.3 was also installed. Building the same source with both — same machine, same
        pinned clock, N={data.benchSize} — eight of the nine kernels came out indistinguishable, and
        one did not:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>kernel</th>
              <th className="n">CUDA 12.5</th>
              <th className="n">CUDA 13.3</th>
              <th className="n">delta</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>naive … dbuffer</td>
              <td className="n">—</td>
              <td className="n">—</td>
              <td className="n">within 1%</td>
            </tr>
            <tr className="hi">
              <td>
                <code>tensorcore</code>
              </td>
              <td className="n">8064</td>
              <td className="n">6499</td>
              <td className="n">−19.4%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        A fifth of the throughput, reproducibly, at <em>identical</em> numerical error — so it was
        still genuinely TF32, just slower. <code>ptxas -v</code> gives the whole story in two lines:
      </p>
      <pre>
        <code>{`12.5:  128 registers, 12 bytes spill stores
13.3:  142 registers,  0 spills`}</code>
      </pre>
      <p>
        nvcc 13.3 spent 14 more registers to eliminate a 12-byte spill. In isolation that is a good
        trade. Here it crosses an occupancy cliff, because registers are allocated per warp in
        multiples of eight and this kernel runs 256 threads per block:
      </p>
      <ul>
        <li>
          128 regs → 4096/warp → 65536/4096 = <strong>16 warps per SM</strong> → 2 resident blocks
        </li>
        <li>
          142 regs → rounds to 144 → 4608/warp → 65536/4608 = <strong>14 warps per SM</strong> → 1
          block
        </li>
      </ul>
      <p>
        Halving the resident blocks to avoid twelve bytes of spill. The compiler optimized what it
        could see — the spill — and could not see what it cost.
      </p>
      <p>
        The fix is to state what the kernel needs instead of hoping the register allocator infers
        it. The second argument to <code>__launch_bounds__</code> is minimum blocks per SM:
      </p>
      <pre>
        <code>{`__global__ __launch_bounds__(NUM_THREADS, 2) void tensorcore_kernel(...)`}</code>
      </pre>
      <p>
        Both toolkits then allocate 128 registers and accept the spill. This is not a 13.3
        workaround — it is faster on both, and 8290 GF/s is the fastest this kernel has ever run:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>toolkit</th>
              <th className="n">before</th>
              <th className="n">after</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>CUDA 12.5</td>
              <td className="n">8064 (113.9%)</td>
              <td className="n">8250 (116.7%)</td>
            </tr>
            <tr>
              <td>CUDA 13.3</td>
              <td className="n">6499 (91.6%)</td>
              <td className="n">8290 (117.3%)</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        A spilled byte is cheap; a resident block is not. <code>__launch_bounds__</code> is how you
        tell the compiler which one you are buying. With the fix in place the two toolkits agree
        across the board, so the build now takes whichever is newest and prints which one it used.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          The lasting lesson is not that one compiler release regressed. It is that a{' '}
          <strong>19% loss passed every correctness test, every gradient check and every loss
          curve</strong> without a murmur, and surfaced only because someone asked why a page said
          12.5. Performance regressions are invisible to correctness testing by construction. The
          only thing that catches them is measuring on purpose.
        </p>
      </div>

      <h2>Then: a language model on top of it</h2>
      <p>
        A matmul is only interesting if something uses it. So the second half was a GPT — 6 layers,
        6 heads, 384 embedding, 256 context, weight-tied head, 10.8M parameters — trained on
        character-level Shakespeare. Layernorm, GELU, causal multi-head attention, softmax,
        cross-entropy and AdamW are all hand-written; every matmul routes through the kernels above.
      </p>
      <p>
        The backward pass needs <code>dX = dY·Wᵀ</code> and <code>dW = dYᵀ·X</code>. Rather than
        materializing transposed copies — an extra bandwidth-bound pass over the data — the
        transpose folds into the shared-memory staging. The A tile was <em>already</em> stored
        transposed, because that is what makes the per-thread register reads contiguous, so each of
        the four transpose cases is just a different index map.
      </p>

      <div className="statgrid" style={{ margin: '26px 0' }}>
        <div className="stat">
          <span className="v">{t.msPerStep.toFixed(1)} ms</span>
          <span className="k">per step (4096 tokens)</span>
        </div>
        <div className="stat">
          <span className="v">{t.gflops.toLocaleString()}</span>
          <span className="k">GFLOP/s end to end</span>
        </div>
        <div className="stat">
          <span className="v">{t.bestVal.toFixed(4)}</span>
          <span className="k">best val loss (step {t.bestStep})</span>
        </div>
        <div className="stat">
          <span className="v">
            {f?.memory?.length ? `${f.memory[0].fused.toFixed(2)} GB` : '0.75 GB'}
          </span>
          <span className="k">resident, everything</span>
        </div>
      </div>

      <figure>
        <img src="/training_curve.svg" alt="Training and validation loss" />
        <figcaption>
          Starting at ln(65) = 4.174 nats/char, which is what a uniform guess over the vocabulary
          scores. Validation bottoms at {t.bestVal.toFixed(4)} around step {t.bestStep} and then
          rises — 10.8M parameters on 1 MB of text overfits, so the best-validation checkpoint is
          the one kept.
        </figcaption>
      </figure>

      <h3>A falling loss does not verify a backward pass</h3>
      <p>
        This is the part most from-scratch projects skip. A dropped correction term in layernorm
        still produces a descending loss curve — just a worse model. So the gradients are checked
        against finite differences directly.
      </p>
      <p>
        The naive check perturbs one scalar and watches the loss, but in fp32 that barely works:
        individual gradients here are ~1e-4, so the loss moves by about as much as the forward
        pass&rsquo;s own rounding noise. Instead each parameter tensor is stepped along{' '}
        <code>u = g/‖g‖</code>, so every element contributes coherently and the predicted change is
        exactly <code>‖g‖</code> — three orders of magnitude above the noise floor.
      </p>
      <p>
        All 16 parameter tensors agree to 1e-5…2e-3. And the check is <em>sensitive</em>:
        deliberately deleting the <code>x̂·mean(dx̂·x̂)</code> term from layernorm backward makes 14
        of 16 tensors fail immediately. (The two that still pass are the final layernorm&rsquo;s
        weight and bias — their gradients do not flow through the path that was broken, which is
        exactly right.)
      </p>

      <h3>What it writes</h3>
      <p>Sampled from the best checkpoint at temperature 0.8:</p>
      <div className="sample">{t.sample}</div>

      <h2>What the profiler found that I never would have</h2>
      <p>
        This repo had the tooling to profile a training step for two sessions before it ever ran
        one, because reading GPU performance counters needs administrator on Windows and it never
        seemed worth the interruption. It was worth the interruption. One click, thirty seconds,
        and it said something I had not guessed &mdash; twice.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>share of a training step</th>
              <th className="n">before</th>
              <th className="n">after</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>GEMM</td><td className="n">62.4%</td><td className="n">64.9%</td></tr>
            <tr><td>attention backward</td><td className="n">14.4%</td><td className="n">15.7%</td></tr>
            <tr className="hi"><td>bias add / column reduce</td><td className="n">8.2%</td><td className="n">3.3%</td></tr>
            <tr><td>GELU</td><td className="n">4.4%</td><td className="n">4.8%</td></tr>
            <tr><td>layernorm</td><td className="n">3.2%</td><td className="n">3.4%</td></tr>
            <tr><td>attention forward</td><td className="n">3.1%</td><td className="n">3.4%</td></tr>
            <tr><td>optimizer</td><td className="n">2.4%</td><td className="n">2.6%</td></tr>
            <tr><td>residual add</td><td className="n">1.2%</td><td className="n">1.3%</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        <strong>67.7 &rarr; 59.9 ms per step, 11.5%.</strong> Everything that grew as a share grew
        because the total shrank.
      </p>
      <h3>The bias add did not need to exist</h3>
      <p>
        Second biggest thing in the step, for one add per element. As its own kernel it reads the
        entire output tensor and writes it back to do that; in the GEMM epilogue &mdash; which is
        already holding the value in a register, about to store it &mdash; the same add costs two
        floats per lane out of L1 and no global traffic at all.
      </p>
      <p>
        There is a reason it was a separate kernel, and it is not laziness. Adding a
        per-<em>column</em> value in an epilogue requires knowing which accumulator register holds
        which column, and that is exactly what WMMA&rsquo;s fragment type hides. Kernel 9 could not
        have done this; the raw-PTX kernel can, because the register mapping is the thing it was
        built around. <strong>The abstraction that turned out not to be the bottleneck for
        arithmetic turned out to be a real constraint on what could be fused</strong> &mdash; a
        better argument for writing the PTX than the 2.7% was.
      </p>
      <h3>The column reduction read memory the wrong way round</h3>
      <p>
        What was left after fusing the forward was the bias <em>backward</em>, at 5.4% of a step for
        an operation whose floor is a single streaming read. It ran{' '}
        <strong>one block per column</strong>, striding down the rows &mdash; under a comment of
        mine asserting the reads were coalesced <em>because consecutive blocks own consecutive
        columns</em>. They were not. Coalescing happens within a warp, and in that arrangement a
        warp&rsquo;s 32 threads read 32 different <strong>rows</strong> at the same column: 32
        addresses <code>C</code> floats apart, so 32 separate transactions fetching 32 bytes each to
        use 4.
      </p>
      <p>
        Neighbouring blocks do re-use those sectors out of L2, which is why it was bad rather than
        catastrophic &mdash; and why it survived. Nothing about the source looks wrong. It has the
        shape of a coalesced kernel and a comment explaining why it is one. The fix is the first
        thing anyone learns about CUDA.
      </p>
      <h3>The residual and GELU go the same way, and cost three detours</h3>
      <p>
        Same waste, same fix &mdash; and two activation buffers that then had nothing left to hold,
        since <code>attproj</code> and <code>fcproj</code> are never read again, not even by the
        backward. <strong>fp32 70.1 &rarr; 69.1 ms, TF32 59.8 &rarr; 58.9 ms, 0.91 &rarr; 0.84 GB
        resident.</strong> About 1.5% &mdash; and getting to an honest 1.5% took three detours that
        are worth more than the number.
      </p>
      <p>
        <strong>It measured as 37% first.</strong> The profile said total kernel time had barely
        moved, which is the only reason I looked: <code>ncu</code> resets the application clock when
        it detaches, so an earlier clock lock had been silently undone and the card was boosting.
        59.9 &rarr; 38 ms is 1.58&times;; it was a clock ratio wearing a speedup&rsquo;s clothes.
        This project already had a rule about never quoting a ratio on an unpinned clock. What it
        did not have was a way to notice the pin coming undone <em>mid-session</em>.
      </p>
      <p>
        <strong>Then the loss started varying in the fourth decimal</strong>, which looked exactly
        like a race I had just introduced. It is not mine and it is not new: two backward kernels
        accumulate through global atomics, so floating-point summation order varies between runs.
        The parent commit does it too &mdash; 1 run in 14. I was one plausible story away from
        attributing a pre-existing property of the model to my own change, and what prevented it was
        building the parent and running it fourteen times.
      </p>
      <p>
        <strong>And the fp32 path came out 6% slower.</strong> Not register pressure &mdash; 219
        against 221, essentially unchanged. The epilogue tested <code>if (ep.gelu_out)</code> at
        runtime, and <code>tanhf</code> expands to a substantial block of code that sits in the
        kernel whether or not the branch is taken. Making the feature set a template parameter and
        testing it with <code>if constexpr</code> turned a 6% regression into a 1.4% gain.{' '}
        <strong>Code you do not execute is not free.</strong>
      </p>

      <h3>One kernel was never going to be enough</h3>
      <p>
        Before starting on the attention backward I noticed the GEMM benchmark had never covered
        three of the model&rsquo;s own shapes. Adding them showed the attention-projection weight
        gradient running at <strong>1846 GF/s where every other shape reaches 7000&ndash;8000</strong>.
        A 128&times;128 tile cuts 384&times;384 into <strong>nine blocks</strong> on a 36-SM card;
        three quarters of the machine is idle and no amount of inner-loop work fixes that.
      </p>
      <p>
        Two fixes, both shape-dependent &mdash; which is the point, and is why cuBLAS ships dozens
        of kernels rather than one good one. A <strong>second, half-height tile</strong> (64&times;128,
        4 blocks/SM) for grids that cannot fill the machine once: +25% on the nine-block shape,
        &minus;3% on the 96-block ones, so it is chosen per shape. And <strong>split-K when the
        output is small and K is long</strong>.
      </p>
      <p>
        I had costed split-K before, for 4096&times;384&times;384, and correctly rejected it: the
        output there is 6.3 MB, so every extra partial is another 6.3 MB of traffic to recover a
        third of a wave. The weight gradients are the <em>opposite</em> shape &mdash; 590 KB of
        output against K=4096 &mdash; so the partials are nearly free.{' '}
        <strong>Same technique, opposite verdict, and the deciding quantity is K against M&middot;N,
        not the wave count I had been staring at.</strong>
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th>weight-gradient shape</th><th className="n">before</th><th className="n">after</th></tr>
          </thead>
          <tbody>
            <tr className="hi"><td>384&times;384&times;4096</td><td className="n">1846</td><td className="n">6899</td></tr>
            <tr><td>384&times;1152&times;4096</td><td className="n">4223</td><td className="n">7727</td></tr>
            <tr><td>384&times;1536&times;4096</td><td className="n">5591</td><td className="n">7673</td></tr>
            <tr><td>1536&times;384&times;4096</td><td className="n">5578</td><td className="n">7623</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        <strong>58.9 &rarr; 54.0 ms per step.</strong> The part worth dwelling on is that three of
        the four largest matmuls in the backward pass had never been measured individually, so the
        one running at a quarter speed was invisible &mdash; averaged into a GEMM category that
        looked healthy at 65%.
      </p>
      <h3>The clock lock does not stay locked</h3>
      <p>
        Twice now a change has measured as a large speedup and been a clock ratio. <code>ncu</code>
        {' '}resets the application clock when it detaches; the lock has also lapsed on its own. Both
        times the number looked plausible &mdash; 59.9 &rarr; 38 ms is 1.58&times;, and so is
        1900/1200. This project already had a rule saying never to quote a ratio on an unpinned
        clock, and that rule does not survive a pin coming undone <em>silently, between the pinning
        and the measurement</em>. Every timing run now reads the clock on both sides of itself and
        labels the output UNPINNED if either reading is wrong.{' '}
        <strong>A discipline that depends on remembering to check is not a discipline.</strong>
      </p>

      <div className="note">
        <p style={{ margin: 0 }}>
          The two biggest wins available in a step I had spent months optimizing were a pass that
          did not need to exist and a reduction that read memory backwards. Both were in code I
          wrote, read, and <em>commented</em>. Neither is subtle once seen, and I did not see either
          one &mdash; I spent the preceding sessions chasing 2&ndash;8% inside a matmul that was
          already at 88% of cuBLAS, because that was the part I found interesting. The profiler does
          not care what is interesting.
        </p>
      </div>

      <h2>Fusing attention: the score matrix never exists</h2>
      <p>
        The table above says attention costs about 21% of a training step across three kernels.
        None of them is badly written — the batched GEMM is the same code that reaches 90% of
        cuBLAS. The cost is <em>structural</em>. A (B, NH, T, T) score matrix gets written to global
        memory and read back, and the softmax over it does roughly 5 flops per 8 bytes on a card
        whose ridge point is 43 FLOP/byte. That is 0.6% of peak no matter how good the kernel is.
        The only fix is to not have the intermediate.
      </p>
      <p>
        A softmax normally needs the whole row before it can emit anything, because it needs the row
        max and the row sum. But both are <em>running</em> statistics. After seeing part of a row you
        hold a partial max <code>m</code> and partial sum <code>l</code>, and when a later block
        raises the max to <code>m&prime;</code>, everything computed so far is corrected by one
        factor — <code>exp(m - m&prime;)</code>:
      </p>
      <pre>
        <code>{`m' = max(m, rowmax(S_j))
l' = l * exp(m - m') + rowsum(exp(S_j - m'))
O' = O * exp(m - m') + exp(S_j - m') @ V_j`}</code>
      </pre>
      <p>
        with <code>O</code> divided by <code>l</code> only at the end. Each score tile lives in
        registers for the few instructions it takes to consume it and is then gone. This is
        FlashAttention. Note it does <em>more</em> arithmetic than the unfused version, not less —
        the whole win is in what never gets written.
      </p>
      <p>
        Two more wins fall out of the structure, and on this hardware they matter as much as the
        famous one. <strong>Causality becomes a loop bound rather than a mask</strong>: the unfused
        path computes all T×T scores and discards the upper triangle inside the softmax, while a
        query block here simply never visits key blocks past its diagonal — so both attention
        matmuls do half the work. And <strong>the head permute disappears</strong>: the unfused path
        pays a bandwidth-bound pass over 3·B·T·C to make each head&rsquo;s slice contiguous because
        a batched GEMM needs uniform strides, while the fused kernel indexes q/k/v straight out of
        the (B, T, 3C) projection — one block owns one head, so the head offset is a constant on the
        row pointer.
      </p>
      <p>
        Backward never stores the score matrix either. It rebuilds the probabilities from the saved
        log-sum-exp, which is exact and needs no reductions at all:{' '}
        <code>P[i,j] = exp(S[i,j] − lse[i])</code>. Recomputing S costs one matmul; reading a stored
        score matrix back costs 25 MB of DRAM per layer. On this card that trade is not close. It is
        two kernels rather than one, because dQ reduces over keys while dK and dV reduce over
        queries — fusing them would push one of the three through global atomics on a tensor the
        size of the activations. <strong>Recompute beats communication</strong>, which is this
        project&rsquo;s whole lesson restated one level up.
      </p>

      {f.fwd?.best && f.bwd?.best && (
        <div className="statgrid" style={{ margin: '26px 0' }}>
          <div className="stat">
            <span className="v">{f.fwd.best.speedup.toFixed(2)}×</span>
            <span className="k">attention forward</span>
          </div>
          <div className="stat">
            <span className="v">{f.bwd.best.speedup.toFixed(2)}×</span>
            <span className="k">attention backward</span>
          </div>
          <div className="stat">
            <span className="v">{f.savedMbPerLayer.toFixed(0)} MB</span>
            <span className="k">saved per layer</span>
          </div>
          <div className="stat">
            <span className="v">{f.err.out.toExponential(1)}</span>
            <span className="k">normwise error vs unfused</span>
          </div>
        </div>
      )}

      <h3>The backward is only {f.bwd?.best ? f.bwd.best.speedup.toFixed(2) : ''}×, and why</h3>
      <p>
        The forward result is most of the story; the backward is nearly a wash, and it is worth
        saying why rather than quoting the forward alone. My first answer was a guess: the fastest
        backward config ran 64 threads per block at 46 KB of shared memory — two blocks per SM, 8%
        occupancy. So I added a config with twice the threads. It gained 2.4%. The profiler explains
        why:
      </p>

      {f.profile && (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>kernel</th>
                <th className="n">L1/TEX</th>
                <th className="n">DRAM</th>
                <th className="n">compute</th>
                <th className="n">occupancy</th>
              </tr>
            </thead>
            <tbody>
              {f.profile.map((p: any) => (
                <tr key={p.name}>
                  <td>
                    <code>{p.name}</code>
                  </td>
                  <td className="n">{p.l1.toFixed(1)}%</td>
                  <td className="n">{p.dram.toFixed(1)}%</td>
                  <td className="n">{p.compute.toFixed(1)}%</td>
                  <td className="n">{p.occupancy.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p>
        L1/TEX saturated near 70% with DRAM at ~21% is a signature this project has met before —
        kernel 6 showed 81% against 15%. Shared memory and L1 share the LSU/MIO datapath, so this is
        shared-memory→register traffic and the FMA pipes are starved. <strong>The same disease as
        kernel 6, one level up the hierarchy.</strong> Counting loads per FMA agrees: the forward
        spends 12 shared loads on 32 FMAs in its P@V loop, the backward accumulation spends 16 on
        32. That is also why more threads did not help — a narrower column tile buys occupancy by
        loading <em>more</em> per unit of arithmetic, so the two effects nearly cancel.
      </p>

      <h3>What the memory actually buys: twice the context</h3>
      <p>
        The speedup is nice; the memory is the part that changes what the card can do. Unfused
        attention is <em>quadratic</em> in context length, fused is <em>linear</em>. Total resident
        memory at batch 16 — parameters, gradients, Adam state, activations and backward scratch:
      </p>

      {f.memory && (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th className="n">context</th>
                <th className="n">fused</th>
                <th className="n">unfused</th>
              </tr>
            </thead>
            <tbody>
              {f.memory.map((m: any) => (
                <tr key={m.ctx} className={m.ctx === 2048 ? 'hi' : undefined}>
                  <td className="n">{m.ctx}</td>
                  <td className="n">{m.fused === null ? 'out of memory' : `${m.fused.toFixed(2)} GB`}</td>
                  <td className="n">
                    {m.unfused === null ? 'out of memory' : `${m.unfused.toFixed(2)} GB`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p>
        This card has 8 GB. The unfused path tops out at context 1024; the fused path trains at
        context 2048, and the unfused path would need nearly three times the card to do the same.{' '}
        <strong>Fusing attention doubled the context this GPU can train.</strong>
      </p>
      <div className="note">
        <p>
          A thing worth knowing on Windows: a successful <code>cudaMalloc</code> is not evidence
          that a model fits. WDDM will oversubscribe VRAM into system memory and page, so the 17.94
          GB allocation above <em>succeeded</em> and then ran at a crawl. The number that means
          something is the total against the card&rsquo;s memory.
        </p>
      </div>

      <h2>Twenty percent that was hiding behind a square benchmark</h2>
      <p>
        The benchmark above measures square matrices. The model does not run square matrices — its
        matmuls are 4096×384×384, 4096×1536×384, 384×384×4096. I had written that sentence in the
        README months earlier as an explanation for why end-to-end throughput sits below the
        headline GFLOP/s, and never measured it. So <code>./bench/sgemm</code> got{' '}
        <code>--mnk M,N,K</code>, and two pieces of free performance fell out immediately.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>shape (M×N×K)</th>
              <th>what it is</th>
              <th className="n">k7 (in use)</th>
              <th className="n">k8</th>
              <th className="n">k9 (TF32)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="n">4096×1152×384</td>
              <td>qkv projection</td>
              <td className="n">6112</td>
              <td className="n">6517</td>
              <td className="n">7482</td>
            </tr>
            <tr>
              <td className="n">4096×384×384</td>
              <td>attention out</td>
              <td className="n">4068</td>
              <td className="n">4353</td>
              <td className="n">6176</td>
            </tr>
            <tr>
              <td className="n">4096×1536×384</td>
              <td>MLP up</td>
              <td className="n">5651</td>
              <td className="n">6026</td>
              <td className="n">7396</td>
            </tr>
            <tr>
              <td className="n">4096×384×1536</td>
              <td>MLP down</td>
              <td className="n">4377</td>
              <td className="n">4658</td>
              <td className="n">6929</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        First: <strong>the model&rsquo;s GEMM was built on kernel 7</strong>, and kernel 8 beats it
        at every one of these shapes. The transpose-aware GEMM was written against the kernel-7
        structure, kernel 8 added double buffering afterwards, and the model simply never got the
        prefetch — on about 80% of a training step. Porting it across took <strong>11.4%</strong>{' '}
        off the step, with the arithmetic untouched: loss and gradient norm identical to every
        printed digit.
      </p>
      <p>
        Nothing was broken. No bug, no regression, every test passing the whole time. It was
        invisible because the benchmark measured a shape the model does not run.
      </p>

      <h3>The speedup that did not survive integration</h3>
      <p>
        Second: the tensor-core kernel is 1.22–1.58× faster than kernel 7 at these shapes — a{' '}
        <em>wider</em> margin than the 1.30× it manages on square matrices, because skinny matmuls
        starve the fp32 pipes harder than they starve the tensor cores. So I built a TF32 path for
        the model&rsquo;s GEMM. Wired into training it was <strong>2.4% slower</strong>.
      </p>
      <p>
        A speedup that does not survive integration is a measurement that has not finished. Timing
        the same entry point both ways, per shape <em>and per transpose case</em>, gave a split so
        clean it named the cause by itself:
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>case</th>
              <th className="n">TF32 vs fp32</th>
              <th />
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>NN, NT — A not transposed</td>
              <td className="n">1.10–1.47×</td>
              <td>tensor cores win</td>
            </tr>
            <tr className="hi">
              <td>TN, TT — A transposed</td>
              <td className="n">0.64–0.87×</td>
              <td>tensor cores lose</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        WMMA wants its A fragment stored m-major. Holding <code>As[m][k]</code> forces the
        transposed case to scatter four scalars for every <code>float4</code> read, where the fp32
        kernel stores <code>As[k][m]</code> and gets a straight vector copy. And{' '}
        <strong>the backward pass computes every weight gradient as TN</strong> — so the forward
        won, and the backward handed the winnings straight back.
      </p>
      <p>
        The fix is not to pick a better tile. It is to stop insisting on one storage orientation:
        keep each operand in whichever layout makes staging a vector copy, and choose the{' '}
        <em>fragment layout</em> to match — <code>col_major</code> when the operand arrived
        transposed. Same bytes, same <code>mma</code> instruction, no scatter.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>case</th>
              <th className="n">before</th>
              <th className="n">after</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>TN</td>
              <td className="n">0.64–0.84×</td>
              <td className="n">0.96–1.33×</td>
            </tr>
            <tr>
              <td>TT</td>
              <td className="n">0.69–0.87×</td>
              <td className="n">1.13–1.43×</td>
            </tr>
            <tr className="hi">
              <td>end to end</td>
              <td className="n">2.4% slower</td>
              <td className="n">9.8% faster</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h3>Does TF32 cost the model anything?</h3>
      <p>Full 5000-step runs, same seed, same configuration, clock pinned:</p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th />
              <th className="n">step</th>
              <th className="n">tokens/s</th>
              <th className="n">best val</th>
              <th className="n">at step</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>fp32</td>
              <td className="n">75.1 ms</td>
              <td className="n">54,516</td>
              <td className="n">1.5138</td>
              <td className="n">2400</td>
            </tr>
            <tr className="hi">
              <td>TF32</td>
              <td className="n">67.6 ms</td>
              <td className="n">60,568</td>
              <td className="n">1.5178</td>
              <td className="n">2800</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The validation losses differ by 0.004 nats. That is comfortably inside the run-to-run spread
        this setup shows from fp32 summation order alone — across configurations on the same data
        and seed I have measured 1.5035, 1.5056, 1.5138, 1.5147, 1.5178 and 1.5194. So TF32 buys 10%
        of training time and costs nothing I can distinguish from noise, which is why every
        framework defaults to it on Ampere and later.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          It stays opt-in here rather than becoming the default, because it changes what the model
          computes and one run on 1 MB of Shakespeare is not enough evidence to make that choice
          silently for someone else. Together the two changes took a training step from{' '}
          <strong>85.0 ms to 67.6 ms</strong> — and both were found by measuring the shapes the
          model actually runs instead of the ones the benchmark happened to default to.
        </p>
      </div>

      <h2>Two GPUs, and the lesson that turned out to be wrong</h2>
      <p>
        The received wisdom about multi-GPU training is that communication becomes the bottleneck.
        Split the work across two cards and the interconnect, not the arithmetic, is what limits
        you. I rented two A40s to see it happen.
      </p>
      <p>
        First, the collective itself. No NCCL — for the same reason there is no cuBLAS here. A ring
        all-reduce splits the gradient buffer into <em>n</em> chunks and runs two phases of{' '}
        <em>n−1</em> steps, reduce-scatter then all-gather, with every device sending and receiving
        at once. Each device moves <code>2(n−1)/n · S</code> bytes, which stops growing with{' '}
        <em>n</em>, and every link stays busy. That is why bandwidth-optimal collectives are rings.
      </p>

      <h3>The driver said peer-to-peer worked. It did not.</h3>
      <p>
        <code>cudaDeviceCanAccessPeer</code> returned true both directions.{' '}
        <code>cudaDeviceEnablePeerAccess</code> succeeded. Every <code>cudaMemcpyPeerAsync</code>{' '}
        returned success, every stream synchronise returned success — and the bytes never arrived.
        PCIe ACS or IOMMU misconfiguration on a virtualised host, which is most rented hardware.
      </p>
      <p>
        It failed <em>silently</em>. The all-reduce reported error exactly 1.00 — the result was
        exactly zero — at 0.2 GB/s. And before the test caught it, the training log had already
        said so in a way I nearly missed: one rank reported a gradient norm of{' '}
        <strong>15.023</strong>, two ranks reported <strong>7.672</strong>. Exactly half, because
        each rank kept its own shard&rsquo;s gradient and then divided by the rank count. No sum
        ever happened, and the loss curve looked entirely reasonable while it did not.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          So <code>ddp_init</code> now sends four bytes across every enabled pair and reads them
          back before trusting the flag, falling back to host staging for any pair that fails.{' '}
          <strong>A capability bit is a claim, not a measurement.</strong> With staging the
          collective is correct: 43 MB in 8.65 ms, 5.0 GB/s — a round trip through host memory,
          roughly a quarter of what the gen4 x16 link should manage directly.
        </p>
      </div>

      <h3>Communication was 6%. Two GPUs were still slower than one.</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="n">ranks</th>
              <th className="n">global batch</th>
              <th className="n">per GPU</th>
              <th className="n">step</th>
              <th className="n">tokens/s</th>
              <th className="n">comm</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="n">1</td>
              <td className="n">16</td>
              <td className="n">16</td>
              <td className="n">40.9 ms</td>
              <td className="n">100,206</td>
              <td className="n">—</td>
            </tr>
            <tr>
              <td className="n">1</td>
              <td className="n">32</td>
              <td className="n">32</td>
              <td className="n">77.5 ms</td>
              <td className="n">105,707</td>
              <td className="n">—</td>
            </tr>
            <tr className="hi">
              <td className="n">2</td>
              <td className="n">32</td>
              <td className="n">16</td>
              <td className="n">149.1 ms</td>
              <td className="n">54,951</td>
              <td className="n">8.9 ms (6.0%)</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Two A40s deliver roughly <strong>half</strong> the throughput of one, and the interconnect
        accounts for six percent of it. The wire is not the problem. The host loop is: one process
        driving both devices, with blocking calls inside the step, so the two GPUs barely overlap.
        Giving each rank its own thread — CUDA&rsquo;s current device is per-thread — moved it from
        159 to 149 ms. Real, and nowhere near the ~85 ms two genuinely overlapped ranks should
        reach.
      </p>
      <p>
        Chasing that further turned up a bug worth the trip on its own. The loss reduction cached
        its output scalar in a <code>static</code> device pointer, so every rank got the same
        pointer, allocated on whichever device called first — one rank reducing into another
        rank&rsquo;s memory, both racing to read one scalar. It produced plausible losses the whole
        time.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          I went looking for the communication wall and found a scheduling problem wearing its
          coat. That is worth more than confirming the slogan would have been:{' '}
          <strong>&ldquo;communication is the bottleneck&rdquo; is a claim about a ratio</strong>,
          and at 10.8M parameters on two cards the ratio is not what the slogan assumes. Knowing
          which side of it you are on requires measuring both — and the measurement said the
          collective was cheap and correct while my orchestration was neither. Full log:{' '}
          <code>bench/logs/multigpu_a40.txt</code>.
        </p>
      </div>

      <h2>Attention on the tensor cores, and a round trip that vanishes</h2>
      <p>
        Every matmul in the project ran on tensor cores under <code>--tf32</code> except the ones
        inside attention, which stayed on fp32 FMAs with 4×2 and 4×4 register tiles while the
        model&rsquo;s own GEMM ran 8×8. That gap was most of why the fused backward beat the
        unfused path by only 1.06×. Attention was the last consumer in the repo computing a matmul
        the slow way.
      </p>
      <p>
        Porting it turns on one fact. <code>S = Q@Kᵀ</code> comes out of an{' '}
        <code>mma</code> as an <em>accumulator</em>, and <code>P = softmax(S)</code> then has to go
        back in as the <em>A operand</em> of <code>P@V</code>. If those layouts agree the fragment
        is reused in place; if they do not, P goes through shared memory — and that round trip is
        the structural reason the backward was slow.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          They do not agree, and I measured rather than recalled — a fragment layout guessed wrong
          is silent, which kernel 10 had already learned once about the <code>a1</code>/
          <code>a2</code> order. The m16n8k8 TF32 accumulator holds{' '}
          <code>reg i of lane L = (row = L/4 + 8·(i/2), col = 2·(L%4) + i%2)</code>, verified
          against all 128 entries of a matrix built so each decodes by inspection. The A operand
          wants columns <code>&#123;t, t+4&#125;</code> where the accumulator holds{' '}
          <code>&#123;2t, 2t+1&#125;</code>. The f16 shapes happen to agree, which is why
          FlashAttention-2 gets the fused form for free and TF32 does not.
        </p>
      </div>
      <p>
        But the disagreement is confined to the four lanes that share a row group, so{' '}
        <strong>eight <code>__shfl_sync</code>es convert one to the other exactly</strong>, and P
        never reaches shared memory at all. One trap on the way, worth stating because it is
        silent: shuffling the already-selected register —{' '}
        <code>__shfl_sync(m, par ? d[1] : d[0], src)</code> — reads whichever register the{' '}
        <em>source</em> lane&rsquo;s own <code>par</code> chose. It lands on the neighbouring
        column and stays entirely plausible. Both registers have to be shuffled and the selection
        applied afterwards.
      </p>
      <h3>The backward needed a transpose, and did not get one</h3>
      <p>
        The forward ports directly and is 1.81× the best fp32 tile. The backward is two kernels and
        only one of them is that easy: <code>dQ</code>&rsquo;s second matmul reduces over keys, so
        its A operand is <code>dS</code> in the accumulator&rsquo;s own orientation. But{' '}
        <code>dK/dV</code> reduces over <em>queries</em> — <code>dV[c][j] += Σ P[i][c] dO[i][j]</code>{' '}
        — so its A operand is P <em>transposed</em>, and transposing an accumulator is exactly the
        round trip being deleted.
      </p>
      <p>
        So that kernel does not compute S. It computes <code>Sᵀ = K@Qᵀ</code>{' '}
        directly, key as M, and its accumulator is <code>[key][query]</code> from the start — the
        orientation the second matmul wants. The reorientation is free, and the reason is the part
        worth keeping: <strong>the backward needs no row reductions at all</strong>. The forward
        must reduce along a query&rsquo;s keys for the running max and sum; here lse and D are
        already known, so P and dS are elementwise with a per-query scalar.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>fused backward</th>
              <th className="n">best fp32</th>
              <th className="n">both mma</th>
              <th className="n"></th>
              <th className="n">vs unfused</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>ctx 256</td><td className="n">1.335 ms</td><td className="n">0.881 ms</td><td className="n">−34.0%</td><td className="n">1.06× → 1.59×</td></tr>
            <tr><td>ctx 512</td><td className="n">2.416 ms</td><td className="n">1.515 ms</td><td className="n">−37.3%</td><td className="n">1.06× → 1.70×</td></tr>
            <tr><td>ctx 1024</td><td className="n">4.411 ms</td><td className="n">2.779 ms</td><td className="n">−37.0%</td><td className="n">1.11× → 1.77×</td></tr>
            <tr className="hi"><td>ctx 2048</td><td className="n">8.407 ms</td><td className="n">5.206 ms</td><td className="n">−38.1%</td><td className="n">1.15× → 1.84×</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        End to end, both arms with <code>--tf32</code> matmuls and attention pinned each way:{' '}
        <strong>−6.9%</strong> of a training step at ctx 256, <strong>−18.5%</strong> at 1024,{' '}
        <strong>−24.3%</strong> at 2048. The win grows with context because attention&rsquo;s share
        of the step does.
      </p>
      <div className="note">
        <p style={{ margin: 0 }}>
          How you tell a port from a corruption: one config ports <code>dK/dV</code> and leaves{' '}
          <code>dQ</code> in fp32, and its error signature is{' '}
          <code>dq 3.00e-07, dk 6.01e-04, dv 3.54e-04</code> — fp32 precision exactly where the
          fp32 kernel still runs, TF32 where the new one does. Only <code>dQ</code>&rsquo;s
          tolerance is relaxed, per config. A blanket tolerance would have hidden corruption in the
          other two. Errors stay in a 1.8e-04 to 9.6e-04 band across ctx 1, 33, 63, 100, 256, 512,
          777 and 1024 — ragged shapes being where a transposed kernel&rsquo;s indexing breaks
          first.
        </p>
      </div>

      <h2>A stale constant, and two wrong explanations for it</h2>
      <p>
        With attention ported the matmuls are back to roughly 72% of a step, so the next thing is
        the tensor-core dispatch&rsquo;s choice between a 128×128 tile and a 64×128 one. Its rule
        was <code>TILE_SWITCH_BLOCKS = 72</code> — two blocks per SM across the <strong>36</strong>{' '}
        SMs of the 4070 it was written on. The card it now runs on has 46. Re-measuring is worth{' '}
        <strong>7.6% of a training step</strong>, 43.6 → 40.2 ms, and lifts{' '}
        <code>mlp up</code> to 96.5% of the ladder&rsquo;s own square-N peak.
      </p>
      <p>
        The interesting part is that I explained it wrong twice, and both times building the
        explanation is what exposed it.
      </p>
      <p>
        <strong>Wave quantisation</strong> was the first: 96 blocks against 46 SMs × 2 = 92 slots is
        one full wave plus four stragglers. The control that killed it was forcing the narrow tile
        on <em>every</em> shape — it wins at 384 blocks as decisively as at 96 — and the arithmetic
        agrees it was never the story, since 96-into-92 and 192-into-184 are the same 52%
        efficiency.
      </p>
      <p>
        <strong>The roofline</strong> was the second, and wrong more interestingly. A{' '}
        <code>BM×BN</code> tile reads <code>(BM+BN)·BK·4</code> bytes per <code>2·BM·BN·BK</code>{' '}
        flops, so its arithmetic intensity is <code>BM·BN / 2(BM+BN)</code>: 32.0 FLOP/byte wide,
        21.3 narrow. This card&rsquo;s ridge point is 21.0 — so the narrow tile clears it and the
        wide one does not, which explains the measurement <em>and</em> the 4070&rsquo;s opposite
        result. A tidy story.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>evaluated at</th>
              <th className="n">peak fp32</th>
              <th className="n">ridge point</th>
              <th>narrow (21.3) clears?</th>
            </tr>
          </thead>
          <tbody>
            <tr className="hi"><td>pinned 1.20 GHz</td><td className="n">14.1 TFLOP/s</td><td className="n">21.0</td><td>yes — by 1.4%</td></tr>
            <tr><td>base 1.45 GHz</td><td className="n">17.1 TFLOP/s</td><td className="n">25.4</td><td>no</td></tr>
            <tr><td>boost 3.09 GHz</td><td className="n">36.4 TFLOP/s</td><td className="n">54.1</td><td>no</td></tr>
          </tbody>
        </table>
      </div>
      <div className="note">
        <p style={{ margin: 0 }}>
          I had picked the one clock that made the story work. Writing it as a threshold is what
          caught it — the rule promptly selected the <em>wrong</em> tile, because at the
          device&rsquo;s base clock 21.3 does not clear 25.4.{' '}
          <strong>A rule that flips on a 1.4% margin is numerology, not physics</strong>, and the
          intensity model ignores L2 reuse, which at these shapes is large.
        </p>
      </div>
      <p>
        What survives is the ratio between <em>machines</em>, robust where the knife-edge was not:
        the 4070&rsquo;s ridge is ~44 at base clock against this card&rsquo;s 25.4. So the rule
        shipped is empirical with the roofline as motivation — take the narrow tile unless the
        machine&rsquo;s ridge sits above the <em>wide</em> tile&rsquo;s intensity, i.e. unless it
        is bandwidth-starved enough that intensity is the limit. Both cards clear by 20–40%, it is
        evaluated at base clock so it is deterministic, and it is calibrated on two data points and
        labelled as such rather than dressed up as a law.
      </p>

      <h2>What I&rsquo;d do next</h2>
      <ol>
        <li>
          <strong><s>Raw <code>mma.sync</code></s></strong> — done, above. 8318 → 9210 GF/s on the
          ladder. The hypothesis behind it was wrong, which is the part worth keeping.
        </li>
        <li>
          <strong><s><code>cp.async</code></s></strong> — done. Worth 8.1% on the ladder, and the
          reasoning that promoted it to &ldquo;the main event&rdquo; accounted for only 2.1 of
          those 8.1 points; the asynchrony it treated as a bonus was the rest. Wiring it into the
          model&rsquo;s GEMM was tried and <em>lost</em> at every shape, and the suspected cause —
          shared footprint costing a resident block — is not the reason either. It is{' '}
          <code>BK</code>: reuse per barrier.
        </li>
        <li>
          <strong><s>Fuse attention</s></strong> — done, and then ported to tensor cores, above.
          The prescription this list used to carry — chunk the head dimension so bigger register
          tiles stop competing with more blocks per SM — <em>was</em> carried out, and it works
          structurally while buying only 0.4–3.5%. The sharpest datum from it is the config that
          frees a third resident block with nothing traded away and is not one percent faster.
          Attention is no longer computing anything on fp32 FMAs.
        </li>
        <li>
          <strong>Re-profile before choosing the next thing.</strong> The step has gone 46.8 → 40.2
          ms, and the two largest costs both moved a lot, so the old ranking cannot be trusted.
          This project is three-for-three on my intuitions losing to the profiler, and the honest
          move is to measure the new distribution rather than guess which of layernorm, the bias
          reductions or the remaining GEMM headroom is now on top.
        </li>
        <li>
          <strong><s>Multi-GPU</s></strong> — started, above. The collective is correct and cheap;
          the data-parallel driver is not yet good enough to profit from it. Two ranks need to
          genuinely overlap, which means one process per GPU or a step with no blocking calls left
          in it.
        </li>
      </ol>

      <footer>
        <p>
          Built on an RTX 4070 Laptop and continued on an RTX 5070 Ti Laptop (sm_120), CUDA 13.3,
          SM clock pinned to 1200 MHz. The two cards are not comparable and no table here mixes
          them. Every
          benchmark cell is the median of three independent sweeps, because pinning the clock fixes
          variance <em>within</em> a run and does nothing about a bad run — which cost me a
          published paragraph explaining a slowdown that three later sweeps showed did not exist.
          All numbers on this page are generated directly from <code>bench/results.csv</code>, the
          training log and the profiler output — see <code>tools/make_site_data.py</code>.
        </p>
      </footer>
    </main>
  );
}
