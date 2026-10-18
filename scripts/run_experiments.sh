#!/usr/bin/env bash
# Reproduce every run in logs/ on a 2-GPU machine (tested on Kaggle "GPU T4 x2").
#
# Usage:  bash scripts/run_experiments.sh            # from the repo root
# Needs:  nccl-tests built at ./nccl-tests (see README "Reproduce").
#
# Rules followed by every experiment:
#   1. One variable changed at a time; everything else identical to the baseline.
#   2. A diagnostic run (NCCL_DEBUG=INFO, -n 1 -w 1) first, to prove the change took effect.
#   3. Timing runs never use NCCL_DEBUG, same size sweep, 3 repetitions each.
set -euo pipefail

BIN=./nccl-tests/build/all_reduce_perf
SWEEP="-b 8 -e 512M -f 2 -g 2"
OUT_T=logs/timing
OUT_D=logs/diag
mkdir -p "$OUT_T" "$OUT_D"

diag() {  # diag <name> <size args> [ENV=VAL ...]
  local name=$1 sizes=$2; shift 2
  env "$@" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING,COLL \
    $BIN $sizes -g 2 -n 1 -w 1 > "$OUT_D/${name}_diag.log" 2>&1
}

timed() {  # timed <name> [ENV=VAL ...]
  local name=$1; shift
  for i in 1 2 3; do
    env "$@" $BIN $SWEEP > "$OUT_T/${name}_${i}.log" 2>&1
  done
}

# 0. Topology (expect PHB on Kaggle T4x2)
nvidia-smi topo -m

# 1. Baseline: what does NCCL pick on its own? (transport, algorithm, protocol per size)
diag baseline "-b 1K -e 256M -f 8"
timed baseline

# 2. Experiment A: Ring vs Tree
diag  tree "-b 1M -e 1M" NCCL_ALGO=Tree   # expect "Algo TREE"
timed tree NCCL_ALGO=Tree
timed ring NCCL_ALGO=Ring                 # == baseline -> noise floor

# 3. Experiment B: transport. P2P is unavailable here (intraNodeP2pSupport 0),
#    so disable SHM and let NCCL fall back to NET/Socket.
diag  noshm "-b 1M -e 1M" NCCL_SHM_DISABLE=1   # expect "via NET/Socket"
timed noshm NCCL_SHM_DISABLE=1

# 4. Experiment C: is Tree faster because of more Simple-protocol threads (512 vs 256)?
diag  ring512 "-b 1M -e 1M" NCCL_ALGO=Ring NCCL_NTHREADS=512   # expect Ring Simple = 512
timed ring512 NCCL_ALGO=Ring NCCL_NTHREADS=512
diag  tree256 "-b 1M -e 1M" NCCL_ALGO=Tree NCCL_NTHREADS=256   # expect Tree Simple = 256
timed tree256 NCCL_ALGO=Tree NCCL_NTHREADS=256

# Quick checks
grep -h "Bytes ->" "$OUT_D"/*_diag.log | sort | uniq -c | head -20
grep -h " via "    "$OUT_D"/*_diag.log | sort -u
grep -h "Max NThreads\|set by environment" "$OUT_D"/*_diag.log | sort -u
