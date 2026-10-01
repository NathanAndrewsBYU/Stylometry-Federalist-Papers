#!/bin/bash
#SBATCH --job-name=federalist-gpt2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=m13h
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ---------------------------------------------------------------------------
# submit_job.sh
#
# SLURM batch script to run one of the Federalist Papers experiments on
# BYU's Office of Research Computing cluster.
#
# CONFIRMED WORKING CONFIGURATION (as of first successful test run,
# 2026-09-26): partition=m13h, gres=gpu:h200:1. This was reached after
# testing several alternatives:
#   - Default/no partition -> landed on an m9g node with a P100 GPU
#     (Pascal architecture, compute capability 6.0). The installed PyTorch
#     build (2.13.0) only ships kernels for compute capability >=7.5, so
#     training crashed with "CUDA error: no kernel image is available for
#     execution on the device."
#   - partition=cs (which has A100s) -> denied: "Cannot access partition cs
#     due to QOS gpu not being allowed." This partition's GPUs are
#     restricted to specific accounts/groups this account doesn't have.
#   - partition=m13h with gres=gpu:h200:1 -> WORKED. H200 is Hopper
#     architecture (compute capability 9.0), well within the installed
#     PyTorch's supported range, and this partition has no QOS restriction
#     for this account.
# If this stops working (e.g. the partition becomes unavailable or gets
# restricted), check available GPU partitions with:
#   sinfo -o "%P %N %G"
# and confirm compute capability compatibility against the installed
# PyTorch's supported list (Runtime error messages will show both). L40S
# GPUs (partition m13l, gres=gpu:l40s:1, compute capability 8.9) were also
# reachable without a QOS error and are a fallback worth trying.
#
# Per BYU ORC's AI agent policy (/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md):
#   - Every Slurm job requests CPU cores, node count, memory, and a time
#     limit (rule 4) — all present above.
#   - Check available modules before relying solely on a conda/mamba
#     environment (rule 8) — see the module load line below.
#   - Do not circumvent login-node limits; use Slurm for GPU work (rule 3) —
#     this script IS the compliant path; do not run training directly on a
#     login node or inside a plain interactive shell.
#
# Usage (from the login node, after prepare_corpus.py has been run once):
#   sbatch submit_job.sh first-try
#   sbatch submit_job.sh full
#   sbatch submit_job.sh disputed [epochs]   # epochs defaults to 30, e.g.:
#                                             #   sbatch submit_job.sh disputed 30
#                                             #   sbatch submit_job.sh disputed 80
#   sbatch submit_job.sh epoch-sweep
# ---------------------------------------------------------------------------

set -x

mkdir -p logs results

# Check for a system Python module before relying purely on mamba's bundled
# one (BYU ORC AI agent policy rule 8). Uncomment if 'module avail python'
# on the login node showed a version you want loaded inside the job too:
# module load python/3.11

# Activate the mamba environment set up per README_CLUSTER.md.
# (mamba/conda is enabled via ~/.bashrc on this system, not an Lmod module —
# see that file's "conda initialize" block.)
source ~/.bashrc
mamba activate federalist
which python
python -c "import torch; print(torch.__version__)"

# Avoid any attempt to reach the network from the compute node — GPT-2's
# tokenizer/model files were pre-downloaded and cached on the login node
# (see README_CLUSTER.md), so this forces use of that local cache instead
# of trying (and failing) to hit huggingface.co from an offline compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODE=${1:-first-try}

case "$MODE" in
  first-try)
    python run_experiment.py --mode first-try --sample-size 10 --epochs 10
    ;;
  full)
    python run_experiment.py --mode full --epochs 30
    ;;
  disputed)
    EPOCHS=${2:-30}
    python run_experiment.py --mode disputed --epochs "$EPOCHS"
    ;;
  epoch-sweep)
    python run_experiment.py --mode epoch-sweep --epoch-values 5 10 20 40 80
    ;;
  *)
    echo "Unknown mode: $MODE. Use one of: first-try, full, disputed, epoch-sweep"
    exit 1
    ;;
esac

# The H200 handled a 2-paper/2-epoch test in ~14s of actual training time
# per model. A dedicated GPU node likely has memory to spare for a larger
# --batch-size than the default of 2 (chosen originally for Colab's shared,
# memory-constrained T4s) — try --batch-size 4 or 8 on future runs and watch
# nvidia-smi (or nvtop in an salloc session) to confirm headroom before
# pushing further.