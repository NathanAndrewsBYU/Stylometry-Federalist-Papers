#!/bin/bash
#SBATCH --job-name=federalist-gpt2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ---------------------------------------------------------------------------
# submit_job.sh
#
# SLURM batch script to run one of the Federalist Papers experiments on
# BYU's Office of Research Computing cluster (or any SLURM-managed system).
#
# IMPORTANT: the specific values above (#SBATCH lines) are a REASONABLE
# STARTING GUESS, not confirmed against BYU's actual cluster configuration.
# Ask the Office of Research Computing (403 CB) or check their documentation
# for:
#   - the correct --gres / --partition syntax for requesting a GPU on their
#     specific cluster (this varies a lot between institutions)
#   - reasonable --time and --mem limits for your account/allocation
#   - whether you need a --account or --partition flag at all (many
#     clusters require one and will reject jobs without it)
#
# Per BYU ORC's AI agent policy (/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md):
#   - Every Slurm job must request CPU cores, node count, memory, and a time
#     limit — --nodes=1 is included explicitly above for this reason.
#   - Check available modules before relying solely on a conda/mamba
#     environment — run 'module avail python' and load one if appropriate
#     (see the module load line below).
#
# Usage (from the login node, after prepare_corpus.py has been run once):
#   sbatch submit_job.sh first-try
#   sbatch submit_job.sh full
#   sbatch submit_job.sh disputed
#   sbatch submit_job.sh epoch-sweep
# ---------------------------------------------------------------------------

mkdir -p logs results

# Check for a system Python module before relying purely on mamba's bundled
# one (BYU ORC AI agent policy recommends checking Lmod modules first for
# Python work). Uncomment and adjust the version once you've checked
# 'module avail python' on the actual cluster:
# module load python/3.11

# Activate your Python environment. Adjust this to however you set yours up —
# common patterns on academic clusters:
#   module load python/3.11 cuda/12.1        # if the cluster uses environment modules
#   source ~/envs/federalist/bin/activate    # if using a venv
#   mamba activate federalist                # if using mamba/conda (matches README_CLUSTER.md)
# Ask Research Computing which pattern their system expects.
mamba activate federalist

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
  *)
    echo "Unknown mode: $MODE. Use one of: first-try, full, disputed, epoch-sweep"
    exit 1
    ;;
esac

# Once you've confirmed a dedicated cluster GPU has memory to spare (check with
# nvtop during an salloc session), try adding --batch-size 4 or --batch-size 8
# to any of the commands above for a further speedup — the default of 2 was
# chosen for Colab's shared, memory-constrained T4s and may be conservative
# for a dedicated cluster node.
