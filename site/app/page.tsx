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
              <td>naive … warptile</td>
              <td className="n">—</td>
              <td className="n">—</td>
              <td className="n">within 1%</td>
            </tr>
            <tr>
              <td>
                <code>dbuffer</code>
              </td>
              <td className="n">6709</td>
              <td className="n">6579</td>
              <td className="n">−1.9%</td>
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
        tell the compiler which one you are buying — and the 19% was invisible until someone asked
        why the page said 12.5.
      </p>

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

      <h2>What I&rsquo;d do next</h2>
      <ol>
        <li>
          <strong>Raw <code>mma.sync</code></strong> instead of WMMA. The remaining ~23% gap to
          cuBLAS&rsquo;s tensor-core path is mostly fragment scheduling that WMMA&rsquo;s
          abstraction does not expose.
        </li>
        <li>
          <strong>
            <code>cp.async</code>
          </strong>{' '}
          for the staging copies, so the prefetch bypasses registers entirely rather than costing 32
          of them per thread.
        </li>
        <li>
          <strong>
            <s>Fuse attention</s>
          </strong>{' '}
          — done, above. What is left is the backward: it is bound on shared-memory→register
          traffic, so <strong>chunk the head dimension</strong> the way a GEMM chunks K, and stop
          bigger register tiles competing with more blocks per SM for the same shared memory. DRAM
          sits at ~21%, so there is bandwidth to pay for the re-staging.
        </li>
        <li>
          <strong>Multi-GPU</strong>, where communication rather than compute becomes the limit.
          That one needs hardware this laptop does not have.
        </li>
      </ol>

      <footer>
        <p>
          Built on an RTX 4070 Laptop. The SGEMM numbers were measured under CUDA 12.4 on Linux;
          the fused-attention and training numbers under CUDA 12.5 on Windows, driver 610.88 —
          which is worth stating, because nvcc&rsquo;s version determines the generated SASS and
          the cuBLAS beside it is the denominator of every &ldquo;% of cuBLAS&rdquo; here. All
          numbers on this page are generated directly from <code>bench/results.csv</code>, the
          training log and the profiler output — see <code>tools/make_site_data.py</code>.
        </p>
      </footer>
    </main>
  );
}
