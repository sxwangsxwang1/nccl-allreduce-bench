## Median busBW (GB/s), 3 runs each

| size | baseline (default) | ring | tree | no-shm (socket) | ring, NTHREADS=512 | tree, NTHREADS=256 |
|---|---|---|---|---|---|---|
| 8 | 0.000617 | 0.000643 | 0.000524 | 0.000135 | 0.000606 | 0.000604 |
| 1K | 0.0571 | 0.0801 | 0.0684 | 0.0157 | 0.0749 | 0.0626 |
| 16K | 0.83 | 0.83 | 0.396 | 0.132 | 0.803 | 0.401 |
| 128K | 2.09 | 2.03 | 1.86 | 0.519 | 1.93 | 1.86 |
| 1M | 3.55 | 3.56 | 3.39 | 1.57 | 3.58 | 3.37 |
| 4M | 3.85 | 3.85 | 4.11 | 1.98 | 3.92 | 4.1 |
| 64M | 3.96 | 3.97 | 4.58 | 2.08 | 4.04 | 4.6 |
| 512M | 3.97 | 3.98 | 4.53 | 2.13 | 4.05 | 4.54 |

## Latency (median time at 8B-1KB) and plateau (median busBW at >=64MB)

| condition | small-msg time (us) | plateau busBW (GB/s) | plateau vs ring |
|---|---|---|---|
| baseline (default) | 16.1 | 3.97 | -0.1% |
| ring | 14.2 | 3.97 | +0.0% |
| tree | 15.5 | 4.55 | +14.5% |
| no-shm (socket) | 59.9 | 2.10 | -47.2% |
| ring, NTHREADS=512 | 12.9 | 4.04 | +1.8% |
| tree, NTHREADS=256 | 14.8 | 4.56 | +14.8% |

## alpha-beta model per protocol (baseline, time = alpha + size/beta)

| protocol | fit range | alpha (us) | beta (GB/s) | knee alpha*beta (us x GB/s = KB) |
|---|---|---|---|---|
| LL | 8K-64K | 12.4 | 2.11 | 26 KB |
| Simple | >=512K | 29.4 | 3.97 | 117 KB |

LL/Simple break-even size predicted by the two fits: ~76 KB (diag log: 64K ran LL, 512K ran Simple).
LL beta / Simple beta = 0.53 (LL spends 4 of every 8 bytes on a flag).
