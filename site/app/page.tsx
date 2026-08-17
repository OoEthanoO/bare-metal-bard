import { data } from './data';

const BP = process.env.NEXT_PUBLIC_BASE_PATH || '';

// Slow -> fast. Matches the ramp used by tools/plot_results.py so the inline
// bars and the SVG charts read as one system.
const RAMP = ['#b3543f', '#c07f3c', '#c9a63c', '#9aa845', '#6a9f5c', '#3f8f74', '#2f7d8a'];

function KernelBars() {
  const max = data.cublas;
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
            <b>{k.gflops.toFixed(0)}</b> <span>{k.pct.toFixed(1)}%</span>
          </div>
        </div>
      ))}
      <div className="bar">
        <div className="name" style={{ color: 'var(--fg)' }}>
          cuBLAS
        </div>
        <div className="track">
          <div className="fill" style={{ width: '100%', background: '#8a6fb0', opacity: 0.4 }} />
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
  const best = data.kernels[data.kernels.length - 1];
  const first = data.kernels[0];
  const speedup = best.gflops / first.gflops;

  return (
    <main className="wrap">
      <header className="hero">
        <p className="eyebrow">CUDA · from scratch</p>
        <h1>Writing a CUDA matmul that reaches 90% of cuBLAS — then training a GPT on it</h1>
        <p className="lede">
          Seven rewrites of a single kernel, from the naive version everyone writes first to a
          warp-tiled one that lands within 10% of NVIDIA&rsquo;s hand-tuned library. Then a language
          model trained end to end on it, with no PyTorch and no vendor BLAS anywhere in the
          training path.
        </p>
        <div className="statgrid">
          <div className="stat">
            <span className="v">{speedup.toFixed(0)}×</span>
            <span className="k">faster than the naive kernel</span>
          </div>
          <div className="stat">
            <span className="v">{best.pct.toFixed(1)}%</span>
            <span className="k">of cuBLAS at N={data.benchSize}</span>
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
        That gap — a factor of ~170 — is the entire project. Every optimization below is the same
        move applied at a different level of the memory hierarchy:{' '}
        <em>load a value once, then spend it on as much arithmetic as possible before letting it
        go.</em>
      </p>

      <h2>The seven kernels</h2>
      <KernelBars />
      <p className="cap">
        GFLOP/s at N={data.benchSize}, fp32, SM clock pinned to 1200 MHz. cuBLAS measured in the
        same process immediately after each kernel.
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
            {data.kernels.map((k, i) => (
              <tr key={k.name} className={i === data.kernels.length - 1 ? 'hi' : undefined}>
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
        At kernel 6 I was at {data.kernels[5].pct.toFixed(1)}% and out of ideas. Nsight Compute gave
        the answer in three lines:
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
            </tr>
          </thead>
          <tbody>
            <tr className="hi">
              <td>L1/TEX throughput</td>
              <td className="n">81.4%</td>
              <td className="n">50.0%</td>
            </tr>
            <tr>
              <td>Memory throughput</td>
              <td className="n">73.8%</td>
              <td className="n">45.3%</td>
            </tr>
            <tr>
              <td>Compute (SM)</td>
              <td className="n">54.2%</td>
              <td className="n">54.8%</td>
            </tr>
            <tr>
              <td>DRAM</td>
              <td className="n">15.1%</td>
              <td className="n">15.9%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The prediction held. And the kernel is now in a genuinely different regime: neither memory
        (45%) nor compute (55%) is saturated, at ~25% occupancy. It is{' '}
        <strong>latency bound</strong> — which is what double-buffering the global-to-shared load
        would attack next.
      </p>

      <figure>
        <img src={`${BP}/sgemm_scaling.svg`} alt="Fraction of cuBLAS achieved, by matrix size" />
        <figcaption>
          Holding up across sizes. Small matrices fall off because a 128×128 block tile leaves most
          of the 36 SMs idle — at N=512 the grid is only 4×4 blocks.
        </figcaption>
      </figure>

      <h2>Then: a language model on top of it</h2>
      <p>
        A matmul is only interesting if something uses it. So the second half was a GPT — 6 layers,
        6 heads, 384 embedding, 256 context, weight-tied head, 10.8M parameters — trained on
        character-level Shakespeare. Layernorm, GELU, causal multi-head attention, softmax,
        cross-entropy and AdamW are all hand-written; every matmul routes through the kernel above.
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
          <span className="v">1.1 GB</span>
          <span className="k">activations + scratch</span>
        </div>
      </div>

      <figure>
        <img src={`${BP}/training_curve.svg`} alt="Training and validation loss" />
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

      <h2>What I&rsquo;d do next</h2>
      <ol>
        <li>
          <strong>Double-buffer</strong> the global-to-shared load, to attack the latency bound that
          kernel 7 ends on.
        </li>
        <li>
          <strong>Tensor cores.</strong> sm_89 has them and an fp32 SGEMM leaves them completely
          unused.
        </li>
        <li>
          <strong>Fuse attention</strong> FlashAttention-style. The current version materializes the
          full (B, NH, T, T) score matrix — 25 MB per layer of pure bandwidth a fused kernel would
          never spend.
        </li>
        <li>
          <strong>Multi-GPU</strong>, where communication rather than compute becomes the limit.
        </li>
      </ol>

      <footer>
        <p>
          Built on an RTX 4070 Laptop with CUDA 12.4. All numbers on this page are generated
          directly from <code>bench/results.csv</code> and the training log — see{' '}
          <code>tools/make_site_data.py</code>.
        </p>
      </footer>
    </main>
  );
}
