#!/usr/bin/env python3
"""Turn the nccl-tests logs in logs/timing/ into tables (results/) and plots (figures/).

Usage:  python scripts/analyze.py          # from the repo root

Notes
- busBW is recomputed from the `time` column instead of read from the `busbw`
  column: nccl-tests prints only 2 decimals, so small messages show up as 0.00
  and fall off a log-scale plot.
- For allreduce busBW = algBW * 2(n-1)/n; with n = 2 the factor is 1.
- Each condition has 3 runs; we report the median and plot the min-max band.
"""
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TIMING = ROOT / "logs" / "timing"
FIG = ROOT / "figures"
RES = ROOT / "results"
NRANKS = 2
BUS_FACTOR = 2 * (NRANKS - 1) / NRANKS

CONDITIONS = {  # label -> log prefix
    "baseline (default)": "baseline",
    "ring": "ring",
    "tree": "tree",
    "no-shm (socket)": "noshm",
    "ring, NTHREADS=512": "ring512",
    "tree, NTHREADS=256": "tree256",
}

# Protocol ranges, taken from logs/diag/baseline_diag.log ("Bytes -> Algo RING proto ...")
LL_FIT = (8 << 10, 64 << 10)   # 8K-64K ran with LL
SIMPLE_FIT_MIN = 512 << 10      # >=512K ran with Simple


def parse(path):
    """size (bytes) -> out-of-place time (us)."""
    d = {}
    for line in open(path):
        c = line.split()
        if len(c) >= 8 and c[0].isdigit():
            d[int(c[0])] = float(c[5])
    return d


def load(prefix):
    """size -> (median time us, median busBW, min busBW, max busBW) over the runs."""
    runs = [parse(p) for p in sorted(TIMING.glob(f"{prefix}_[0-9].log"))]
    if not runs:
        return {}
    common = set.intersection(*(set(r) for r in runs))
    out = {}
    for s in sorted(common):
        ts = [r[s] for r in runs]
        bws = [s / (t * 1e3) * BUS_FACTOR for t in ts]  # bytes/us -> GB/s (1e9)
        out[s] = (st.median(ts), st.median(bws), min(bws), max(bws))
    return out


def fit(data, lo, hi):
    """Least-squares time = alpha + size/beta over lo <= size <= hi."""
    xs = np.array([s for s in data if lo <= s <= hi], dtype=float)
    ts = np.array([data[int(s)][0] for s in xs])
    k, alpha = np.polyfit(xs, ts, 1)          # k in us/byte
    return alpha, 1e-3 / k                     # alpha in us, beta in GB/s


def fmt_size(s):
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if s >= div:
            return f"{s // div}{unit}"
    return f"{s}"


def plot(data, labels, path, title, ymin=None, linear=False, xmin=None, mark_switch=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        d = data[label]
        if not d:
            continue
        xs = [s for s in d if xmin is None or s >= xmin]
        ax.plot(xs, [d[s][1] for s in xs], marker="o", ms=3, label=label)
        ax.fill_between(xs, [d[s][2] for s in xs], [d[s][3] for s in xs], alpha=0.2)
    ax.set_xscale("log", base=2)
    if not linear:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if mark_switch:
        ax.axvspan(64 << 10, 512 << 10, color="grey", alpha=0.12)
        ax.text(2 << 10, ax.get_ylim()[0] * 3, "LL", ha="center")
        ax.text(32 << 20, ax.get_ylim()[0] * 3, "Simple", ha="center")
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("busBW (GB/s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    data = {label: load(prefix) for label, prefix in CONDITIONS.items()}
    base = data["baseline (default)"]
    lines = []

    # 1. Median busBW per condition at representative sizes
    sizes = [8, 1 << 10, 16 << 10, 128 << 10, 1 << 20, 4 << 20, 64 << 20, 512 << 20]
    labels = [l for l in CONDITIONS if data[l]]
    lines.append("## Median busBW (GB/s), 3 runs each\n")
    lines.append("| size | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for s in sizes:
        cells = [f"{data[l][s][1]:.3g}" if s in data[l] else "-" for l in labels]
        lines.append(f"| {fmt_size(s)} | " + " | ".join(cells) + " |")

    # 2. Small-message latency and plateau
    lines.append("\n## Latency (median time at 8B-1KB) and plateau (median busBW at >=64MB)\n")
    lines.append("| condition | small-msg time (us) | plateau busBW (GB/s) | plateau vs ring |")
    lines.append("|---|---|---|---|")
    ring_plateau = st.mean(v[1] for s, v in data["ring"].items() if s >= 64 << 20)
    for l in labels:
        d = data[l]
        small = st.mean(v[0] for s, v in d.items() if s <= 1 << 10)
        plat = st.mean(v[1] for s, v in d.items() if s >= 64 << 20)
        lines.append(f"| {l} | {small:.1f} | {plat:.2f} | {100 * (plat / ring_plateau - 1):+.1f}% |")

    # 3. alpha-beta per protocol (baseline = Ring)
    a_ll, b_ll = fit(base, *LL_FIT)
    a_s, b_s = fit(base, SIMPLE_FIT_MIN, max(base))
    cross = (a_s - a_ll) / (1 / b_ll - 1 / b_s) * 1e3  # bytes where LL time == Simple time
    lines.append("\n## alpha-beta model per protocol (baseline, time = alpha + size/beta)\n")
    lines.append("| protocol | fit range | alpha (us) | beta (GB/s) | knee alpha*beta (us x GB/s = KB) |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| LL | {fmt_size(LL_FIT[0])}-{fmt_size(LL_FIT[1])} | {a_ll:.1f} | {b_ll:.2f} | {a_ll * b_ll:.0f} KB |")
    lines.append(f"| Simple | >={fmt_size(SIMPLE_FIT_MIN)} | {a_s:.1f} | {b_s:.2f} | {a_s * b_s:.0f} KB |")
    lines.append(f"\nLL/Simple break-even size predicted by the two fits: ~{cross / 1e3:.0f} KB "
                 f"(diag log: 64K ran LL, 512K ran Simple).")
    lines.append(f"LL beta / Simple beta = {b_ll / b_s:.2f} (LL spends 4 of every 8 bytes on a flag).")

    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 4. Figures
    plot(data, ["baseline (default)"], FIG / "baseline_busbw.png",
         "allreduce busBW, T4x2 baseline (median of 3, band = min-max)", ymin=1e-3, mark_switch=True)
    plot(data, ["baseline (default)", "ring", "tree", "no-shm (socket)"], FIG / "compare_algo_transport.png",
         "allreduce busBW, T4x2: algorithm and transport", ymin=1e-3)
    plot(data, ["ring", "ring, NTHREADS=512", "tree", "tree, NTHREADS=256"], FIG / "nthreads_plateau.png",
         "Plateau (>=1MB): does thread count explain Tree > Ring?", linear=True, xmin=1 << 20)
    print(f"\nwrote {RES / 'summary.md'} and figures in {FIG}")


if __name__ == "__main__":
    main()
