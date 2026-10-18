# nccl-allreduce-bench

Controlled NCCL `all_reduce_perf` experiments on a 2× Tesla T4 machine (Kaggle "GPU T4 x2"),
with no P2P access between the GPUs. The goal is to explain the busBW-vs-message-size curve from first principles: topology, transport, algorithm and protocol.
The method is to **change one variable at a time** and **prove each change took effect in the NCCL logs** before comparing numbers.

![baseline](figures/baseline_busbw.png)

## Contents

- [TL;DR](#tldr)
- [Environment](#environment)
- [Background](#background)
- [Method](#method)
- [Results](#results)
- [Open questions](#open-questions)
- [Limitations](#limitations)
- [Reading the logs](#reading-the-logs)
- [Repository layout](#repository-layout)
- [Reproduce](#reproduce)
- [中文摘要](#中文摘要)

## TL;DR

| # | Finding | Evidence |
|---|---|---|
| 1 | P2P is unavailable between the two T4s (`topo -m` = PHB, VM). NCCL falls back to **SHM** through host memory. | `intraNodeP2pSupport 0`, `via SHM/direct/direct` |
| 2 | Bandwidth plateaus at **3.97 GB/s**, ~30% of PCIe Gen3 x16 (~12–13 GB/s achievable). | 3 runs, spread < 0.5% at ≥ 4 MB |
| 3 | The curve consists of two protocols. LL (≤ 64 KB) has **α ≈ 12 µs, β ≈ 2.1 GB/s**; Simple (≥ 512 KB) has **α ≈ 29 µs, β ≈ 4.0 GB/s**. LL's β is about half of Simple's, as expected from 4-byte flags in every 8 bytes. | per-protocol α-β fit |
| 4 | With NCCL forced to Tree, large messages run **14.5% faster** than Ring (4.55 vs 3.97 GB/s). The default nonetheless picks Ring, because its model predicts 10 GB/s for Ring and only 4.6 for Tree. | `Algo RING` in diag log, TUNING table |
| 5 | The Simple-protocol thread count (Tree 512 vs Ring 256) is **not** the main cause. Ring at 512 threads gains only +1.8%, and Tree at 256 threads is unchanged. | single-variable runs + diag logs |
| 6 | Disabling SHM (falling back to NET/Socket) makes α **~4×** worse (60 µs) and cuts the plateau by **47%** (2.10 GB/s). | `via NET/Socket/0` |

## Environment

| | |
|---|---|
| GPUs | 2× Tesla T4 (16 GB, PCIe Gen3 x16), Kaggle notebook VM, 4 vCPUs |
| Topology | `nvidia-smi topo -m`: GPU0↔GPU1 = **PHB** (through the PCIe host bridge) |
| NCCL | 2.25.1 (headers and runtime library match: `nccl-headers=22501 nccl-library=22501`) |
| nccl-tests | 2.21.1, single process, `-g 2` |
| Collective | allreduce, float, sum. n = 2, so **busBW = algBW** (factor 2(n−1)/n = 1). |

## Background

The results below assume the concepts in this section.

### Topology (`nvidia-smi topo -m`)

These labels describe how far apart two devices are on the PCIe tree, from nearest to farthest:

| label | path between the two GPUs |
|---|---|
| `NV#` | NVLink (# = number of links) |
| `PIX` | at most one PCIe switch |
| `PXB` | several PCIe switches, without going through the CPU |
| `PHB` | through the CPU's PCIe host bridge (root complex) |
| `NODE` | across host bridges inside one CPU |
| `SYS` | across CPU sockets (UPI / Infinity Fabric) |

P2P is reliable at PIX/PXB. At PHB it depends on the platform, and it is usually blocked inside a VM, where the IOMMU and ACS isolate devices.

### Transport: how bytes move between two GPUs in one node

| transport | path | notes |
|---|---|---|
| P2P | GPU → GPU directly (PCIe or NVLink) | fastest; needs peer access |
| SHM | GPU → host memory → GPU | always available; crosses PCIe twice |
| NET | GPU → host → NIC or socket → host → GPU | last resort inside a node |

### Algorithm: Ring vs Tree

- **Ring:** reduce-scatter followed by all-gather. That is 2(n−1) steps, each sending S/n, so every rank sends 2(n−1)/n·S in total.
  Ring is bandwidth-optimal, but its latency grows linearly with n.
- **Tree:** reduce up to the root, then broadcast down, pipelined in chunks. Latency grows with log n.
  NCCL uses a double binary tree so that the load stays balanced.
- **With n = 2,** both algorithms reduce to the same 0⇄1 chain. Any difference between them comes from the implementation, not from the route.

### Protocol: LL vs LL128 vs Simple

A protocol is how the receiver learns that data has arrived.

| protocol | how "data ready" is signalled | efficiency | latency |
|---|---|---|---|
| LL | each 8-byte store = 4 B data + 4 B flag, written atomically | ~50% | lowest (no memory fence) |
| LL128 | each 128 B = 120 B data + 8 B flag; needs 128 B write atomicity (NVLink) | ~94% | low |
| Simple | bulk data, then a memory fence, then a flag | ~100% | highest |

NCCL normally uses LL for small messages, LL128 for medium ones (when the hardware allows it) and Simple for large ones.

### algBW vs busBW

- `algBW = S / t` is the user's view: how fast a buffer of S bytes gets reduced.
- `busBW = algBW × 2(n−1)/n` (for allreduce) is the per-link hardware view. It can be compared with link bandwidth regardless of n.
- With n = 2 the factor is 1, so the two are equal here.

### α-β model

`t ≈ α + S / β`. Here α is the fixed per-operation cost and β is the sustained bandwidth.
At the knee, `S = α·β`, the two terms are equal and busBW reaches β/2.

## Method

1. **Predict first.** From `topo -m` (PHB inside a VM), the predictions were: P2P most likely unavailable, so NCCL uses SHM, and the plateau stays below PCIe bandwidth.
2. **Diagnostic run per condition.** Each diagnostic run uses `NCCL_DEBUG=INFO` with the `INIT,ENV,TUNING,COLL` subsystems, a narrow size range and `-n 1 -w 1`.
   It confirms the transport (`via ...`), the algorithm and protocol (`AllReduce: N Bytes -> Algo X proto Y`) and thread counts (`Max NThreads`).
   Diagnostic runs are never used for timing.
3. **Timing runs.** Each timing run sweeps 8 B → 512 MB (`-f 2`) with no debug output, 3 runs per condition. Tables report the median; plots show the min–max band.
4. **Noise floor.** `NCCL_ALGO=Ring` reproduces the default (the default also picked Ring). The baseline-vs-ring difference measures run-to-run noise, which is < 0.5% at ≥ 4 MB.
5. **busBW is recomputed from the `time` column.** The `busbw` column has only 2 decimals, so small messages would otherwise read 0.00.

| condition | variable changed | diag check |
|---|---|---|
| `baseline` | none (NCCL defaults) | transport, algorithm and protocol per size |
| `ring` | `NCCL_ALGO=Ring` | identical to baseline, used as the noise floor |
| `tree` | `NCCL_ALGO=Tree` | `Algo TREE proto SIMPLE` |
| `noshm` | `NCCL_SHM_DISABLE=1` | `via NET/Socket/0` |
| `ring512` | `NCCL_ALGO=Ring NCCL_NTHREADS=512` | Ring Simple `Max NThreads` 256 → 512 |
| `tree256` | `NCCL_ALGO=Tree NCCL_NTHREADS=256` | Tree Simple `Max NThreads` 512 → 256 |

## Results

All numbers are medians of 3 timing runs. The full tables are in [`results/summary.md`](results/summary.md).

### 1. Transport: SHM, because P2P is unavailable

From `logs/diag/baseline_diag.log`:

```
Check P2P Type intraNodeP2pSupport 0 directMode 1
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
```

The GPUs appear on a flat virtual PCI bus (`0000:00:04`, `0000:00:05`). The hypervisor does not expose peer-to-peer access, so data takes this path:
GPU0 → PCIe → host memory → PCIe → GPU1.

### 2. Protocol switch and per-protocol α-β

NCCL's per-call choice (TUNING subsystem) for the default run:

| size | algorithm | protocol | channels |
|---|---|---|---|
| 1 KB, 8 KB | Ring | LL | 1 |
| 64 KB | Ring | LL | 2 |
| ≥ 512 KB | Ring | Simple | 2 |

A single α-β fit across the whole curve, with α taken from the smallest messages and β from the largest, put the knee at ~62 KB.
But the measured point where busBW reaches β/2 is ~128 KB. **The two endpoints of that fit come from different protocols.**
Fitting each protocol separately:

| protocol | fit range | α (µs) | β (GB/s) | knee α·β |
|---|---|---|---|---|
| LL | 8–64 KB | 12.4 | 2.11 | 26 KB |
| Simple | ≥ 512 KB | 29.4 | 3.97 | 117 KB |

- **LL β / Simple β = 0.53.** LL writes a 4-byte flag next to every 4 bytes of data, so it delivers about half the bandwidth.
  In exchange it needs no memory fence, so its α is about 2.4× lower.
- The two fits break even at ~76 KB. The diagnostic log brackets the actual switch between 64 KB (LL) and 512 KB (Simple).
- Simple's knee (117 KB) matches where the measured curve reaches β/2.
- LL128 is disabled on this platform (bandwidth 0.0 in the TUNING table). It relies on 128-byte write atomicity, which NVLink guarantees and PCIe does not.

NCCL's own model (TUNING table, allreduce, latency µs / bandwidth GB/s):

| | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 8.8 / 1.2 | 17.8 / 0.0 | 16.4 / 4.6 |
| Ring | 8.6 / 5.0 | 19.0 / 0.0 | 19.8 / 10.0 |

### 3. Algorithm: Tree beats Ring at large sizes

![compare](figures/compare_algo_transport.png)

| size | ring | tree |
|---|---|---|
| 16 KB | 0.83 | **0.40** |
| 1 MB | 3.56 | 3.39 |
| 64 MB | 3.97 | **4.58** |
| 512 MB | 3.98 | **4.53** |

- **Prediction:** with only 2 ranks, Ring and Tree both reduce to the same 0⇄1 chain, so they should perform about the same. **The data rejects this prediction.**
  Tree's plateau is 14.5% higher, far outside the < 0.5% noise floor.
- Tree is slower at 8 KB–1 MB. The step at 8 KB → 16 KB (21 → 41 µs) looks like an earlier LL → Simple switch.
  This fits the TUNING table, which estimates Tree LL at only 1.2 GB/s. It has not been verified with a per-size diagnostic run.
- **NCCL's model is wrong in this environment.** The TUNING table predicts Ring/Simple at 10.0 GB/s (measured 3.97) and Tree/Simple at 4.6 GB/s (measured 4.55).
  Because it overestimates Ring, the default picks the slower algorithm for large messages.

### 4. Hypothesis tested: is it the thread count?

The TUNING table shows `Max NThreads` for Simple: **Tree 512, Ring 256**. On the SHM path, data is copied by GPU threads, so more threads could mean more copies in flight.
Both directions were tested:

![nthreads](figures/nthreads_plateau.png)

| condition | verified in diag log | plateau (GB/s) | vs ring |
|---|---|---|---|
| ring (256 threads) | — | 3.97 | — |
| ring, `NCCL_NTHREADS=512` | Ring Simple = 512 | 4.04 | +1.8% |
| tree (512 threads) | — | 4.55 | +14.5% |
| tree, `NCCL_NTHREADS=256` | Tree Simple = 256 | 4.56 | +14.8% |

**Result: the hypothesis is rejected as the main cause.** More threads give Ring a small real gain (+1.8%, outside the noise floor), but that is far from the 14.5% gap.
Tree loses nothing with half the threads. One caveat: `NCCL_NTHREADS=256` also lowered the LL thread count (512 → 256), but Tree's numbers did not change at any size.

### 5. Transport: SHM vs Socket

| | SHM (default) | NET/Socket (`NCCL_SHM_DISABLE=1`) |
|---|---|---|
| small-message time (8 B–1 KB) | 14–16 µs | 60 µs |
| plateau busBW | 3.97 GB/s | 2.10 GB/s |

Socket adds a CPU proxy thread, the kernel TCP stack and system calls. This shows up mostly as fixed cost: α is about 4× higher.
Bandwidth only halves, which suggests that SHM itself (~4 GB/s) is already far from the PCIe limit.

## Open questions

1. **Why is the plateau only ~4 GB/s?** The next step is to measure single-link H2D/D2H bandwidth with a pinned `torch` copy and check the PCIe link state under load (`nvidia-smi -q -d PCI`).
   This separates "the link is slow" from "the SHM path is slow".
2. **Why is Tree 14% faster than Ring at large sizes?** Thread count is ruled out. The next single-variable candidates are `NCCL_BUFFSIZE` (pipeline depth) and `NCCL_MIN_NCHANNELS=4`.
   A definitive answer probably needs reading the SHM transport and the Ring/Tree kernels in the NCCL source.
3. **Where exactly does Tree switch protocol?** Run a narrow TUNING diagnostic with `NCCL_ALGO=Tree` over 4 KB–1 MB.

## Limitations

- **One VM, one session.** Results describe virtualized PCIe with SHM, not bare-metal P2P or NVLink.
- **Single-process `-g 2`.** Real training runs one process per GPU (`torchrun`), where P2P/SHM setup uses inter-process IPC and may behave differently.
- **Only 3 runs per condition.** The first run of a session was noticeably slower at small and medium sizes (up to ~30% at 64–128 KB), so tables use medians.
- **No single-GPU comparison by design.** With n = 1, allreduce does no communication (busBW = 0), and the earlier single-GPU data came from different hardware.

## Reading the logs

Useful greps against `logs/diag/*.log`:

```bash
grep -h "NCCL version\|nccl-library" logs/diag/*.log | sort -u    # runtime NCCL matches the build
grep -h "intraNodeP2pSupport" logs/diag/*.log | sort -u           # is P2P possible?
grep -h " via " logs/diag/*.log | sort -u                         # transport actually used
grep -h "Bytes ->" logs/diag/*.log | sort | uniq -c               # algorithm + protocol per call
grep -h "Max NThreads\|set by environment" logs/diag/*.log | sort -u
```

Columns of a timing line: `size count type redop root time algbw busbw #wrong`, first for out-of-place and then for in-place.
`time` is in µs and bandwidths are in GB/s (10⁹ B/s). `analyze.py` uses the out-of-place `time` column.

## Repository layout

```
logs/timing/   <cond>_<1..3>.log   timing runs (no debug output)
logs/diag/     <cond>_diag.log     NCCL_DEBUG=INFO runs that prove each setting took effect
scripts/run_experiments.sh         every command used, in order
scripts/analyze.py                 logs -> results/summary.md + figures/*.png
results/summary.md                 generated tables (medians, latency/plateau, α-β fits)
figures/                           generated plots
notebooks/nccl-t4x2-kaggle.ipynb   the original Kaggle notebook, with outputs
```

## Reproduce

On a 2-GPU Linux machine with CUDA (Kaggle: Accelerator = GPU T4 x2, Internet = On):

```bash
git clone https://github.com/sxwangsxwang1/nccl-allreduce-bench.git
cd nccl-allreduce-bench
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j"$(nproc)"     # add NCCL_HOME=... if nccl.h is not on the default path
bash scripts/run_experiments.sh
```

To regenerate the tables and figures from the committed logs, run the following on any machine (no GPU needed):

```bash
pip install -r requirements.txt
python scripts/analyze.py
```

On Kaggle, `apt-get install libnccl-dev` may fail because NCCL packages are held. In that case, build against the NCCL that ships with PyTorch: point `NCCL_HOME` at `python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))"`, add a `libnccl.so` symlink next to `libnccl.so.2`, and set `LD_LIBRARY_PATH` through `os.environ` in the notebook.

`run_experiments.sh` consolidates the notebook cells and uses the same debug subsystems for every diagnostic run. The committed diagnostic logs were produced cell by cell, so their subsystem flags vary slightly; see the notebook.

## 中文摘要

在 Kaggle 的 2× T4 虚拟机上，用 nccl-tests 对 allreduce 做了一组受控实验。方法有三条：一次只改一个变量；每个改动先用诊断日志确认生效；计时跑不开 DEBUG，每组跑 3 次取中位数。

- **拓扑和 transport**：两张卡之间是 PHB，又在虚拟机里，P2P 不可用（`intraNodeP2pSupport 0`），NCCL 退到经主机内存中转的 SHM。大消息带宽的平台约 3.97 GB/s，约为 PCIe Gen3 x16 的 30%。
- **协议**：按协议分段拟合 α-β。LL 的 α≈12 µs、β≈2.1 GB/s，Simple 的 α≈29 µs、β≈4.0 GB/s。LL 的带宽约为 Simple 的一半，印证了“每 8 字节里有 4 字节是 flag”。单一 α-β 模型会把拐点低估一半左右。
- **算法**：预期两个 rank 时 Ring 和 Tree 应该差不多，但实测大消息时 Tree 快 14.5%。NCCL 的模型高估了 Ring（预测 10 GB/s，实测 3.97），所以默认选了更慢的 Ring。
- **假设检验**：从正反两个方向验证了“Tree 用 512 线程、Ring 用 256 线程”这个假设，结果基本被否定：Ring 提到 512 只快了 1.8%，Tree 降到 256 没有变化。
- **关掉 SHM**：退到 Socket 后，α 约为原来的 4 倍，带宽下降 47%。
- **待解决的问题**：平台为什么只有约 4 GB/s（下一步做 H2D/D2H 对照）；Tree 为什么比 Ring 快（下一步看 `NCCL_BUFFSIZE` 和通道数）。
