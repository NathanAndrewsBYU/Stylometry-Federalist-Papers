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

mkdir -p logs results

mamba activate federalist

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
  *)
    echo "Unknown mode: $MODE. Use one of: first-try, full, disputed, epoch-sweep"
    exit 1
    ;;
esac
