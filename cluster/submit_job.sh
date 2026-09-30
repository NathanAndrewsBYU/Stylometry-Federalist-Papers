#!/bin/bash
#SBATCH --job-name=federalist-gpt2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --partition=m13h
#SBATCH --gres=gpu:h200:1
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ---------------------------------------------------------------------------
# submit_job.sh
#
# SLURM batch script to run one of the Federalist Papers experiments on
# BYU's Office of Research Computing cluster.
#
# CONFIRMED WORKING CONFIGURATION: partition=m13h, gres=gpu:h200:1 (H200,
# Hopper architecture, compute capability 9.0 -- compatible with the
# installed PyTorch build). Avoids the default P100 nodes (Pascal, compute
# capability 6.0 -- NOT supported by this PyTorch build) and the QOS-
# restricted A100 partitions (cs, dw) this account can't access.
#
# Usage (from the login node, after prepare_corpus.py has been run once):
#   sbatch submit_job.sh first-try
#   sbatch submit_job.sh full
#   sbatch submit_job.sh disputed
#   sbatch submit_job.sh epoch-sweep
#   sbatch submit_job.sh balanced
#
# For "balanced", edit the --epoch-values and --balance-seeds lists below
# directly to control what it does: a single number in each list runs once;
# multiple numbers in --epoch-values sweeps epoch counts; multiple numbers
# in --balance-seeds sweeps which Hamilton papers get subsampled; multiple
# numbers in both runs the full grid of every combination.
# ---------------------------------------------------------------------------

mkdir -p logs results

# Activate the mamba environment. Sourced explicitly here (not just via
# ~/.bashrc) because a non-interactive batch job's shell doesn't read
# ~/.bashrc automatically.
source /apps/miniconda3/latest/etc/profile.d/conda.sh
if [ -f "/apps/miniconda3/latest/etc/profile.d/mamba.sh" ]; then
    source /apps/miniconda3/latest/etc/profile.d/mamba.sh
fi
mamba activate federalist

# Force use of GPT-2 files pre-cached on the login node, since compute
# nodes have no internet access.
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
    python run_experiment.py --mode disputed --epochs 30
    ;;
  epoch-sweep)
    python run_experiment.py --mode epoch-sweep --epoch-values 5 10 20 40 80
    ;;
  balanced)
    python run_experiment.py --mode balanced --epoch-values 80 100 --balance-seeds 1 2 3 4 5
    ;;
  *)
    echo "Unknown mode: $MODE. Use one of: first-try, full, disputed, epoch-sweep, balanced"
    exit 1
    ;;
esac
