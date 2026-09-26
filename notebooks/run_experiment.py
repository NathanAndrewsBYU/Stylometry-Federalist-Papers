"""
run_experiment.py

Command-line entry point for running the Federalist Papers experiments on a
SLURM-managed cluster. This replaces manually running/editing notebook cells
with a single script controlled by arguments — the standard pattern for
scheduled batch jobs, since a job runs a script start-to-finish rather than
being clicked through interactively.

Requires corpus/federalist/ to already exist — run prepare_corpus.py once
on a login node first (see that file's docstring for why).

Usage examples:

    # Quick check: 10 random papers, 10 epochs
    python run_experiment.py --mode first-try --sample-size 10 --epochs 10

    # Full leave-one-out validation on all 65 known-authorship papers
    python run_experiment.py --mode full --epochs 30

    # Score the 12 disputed papers (only run after full validation looks good)
    python run_experiment.py --mode disputed --epochs 30

    # Epoch sweep: run full validation at several epoch counts to find the
    # point where accuracy stops improving (the overfitting check)
    python run_experiment.py --mode epoch-sweep --epoch-values 5 10 20 40 80
"""

import argparse
import os
import time

from stylometry_lib import (
    device,
    list_txt_files,
    leave_one_out_validation,
    score_disputed_papers,
    save_results_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Run Federalist Papers GPT-2 authorship experiments.")
    parser.add_argument("--mode", required=True,
                         choices=["first-try", "full", "disputed", "epoch-sweep"],
                         help="Which experiment to run.")
    parser.add_argument("--sample-size", type=int, default=None,
                         help="For --mode first-try: number of random papers to test (default: all).")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Training epochs per fine-tuned model (default: 10).")
    parser.add_argument("--epoch-values", type=int, nargs="+", default=[5, 10, 20, 40],
                         help="For --mode epoch-sweep: list of epoch counts to test.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for sample selection (reproducibility).")
    parser.add_argument("--batch-size", type=int, default=2,
                         help="Training batch size (default: 2, matching the original Colab "
                              "setup). A dedicated cluster GPU may have enough memory to handle "
                              "4 or 8 — try increasing if a job finishes with GPU memory to spare "
                              "(check via 'nvtop' in an salloc session) and watch for out-of-memory "
                              "errors if you push it too high.")
    parser.add_argument("--output-dir", default="results",
                         help="Directory to save result CSVs into.")
    parser.add_argument("--save-full-models", action="store_true",
                         help="Also save the 'full' Hamilton/Madison models trained during "
                              "leave-one-out validation to disk (off by default — they're only "
                              "needed within this run, so skipping the save avoids unnecessary "
                              "disk writes on the cluster's shared filesystem).")
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    hamilton_files = list_txt_files("hamilton")
    madison_files = list_txt_files("madison")
    print(f"Loaded {len(hamilton_files)} Hamilton papers, {len(madison_files)} Madison papers.")

    if not hamilton_files or not madison_files:
        raise RuntimeError(
            "Corpus not found. Run 'python prepare_corpus.py' on a login node first."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    if args.mode == "first-try":
        results = leave_one_out_validation(
            hamilton_files, madison_files,
            sample_size=args.sample_size or 10,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            save_full_models=args.save_full_models,
        )
        save_results_csv(results, os.path.join(args.output_dir, f"leave_one_out_first_try_n{len(results)}.csv"))

    elif args.mode == "full":
        results = leave_one_out_validation(
            hamilton_files, madison_files,
            sample_size=None,  # all 65 papers
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            save_full_models=args.save_full_models,
        )
        save_results_csv(results, os.path.join(args.output_dir, f"leave_one_out_full_epochs{args.epochs}.csv"))

    elif args.mode == "disputed":
        disputed_files = list_txt_files("disputed")
        print(f"Loaded {len(disputed_files)} disputed papers.")
        results = score_disputed_papers(hamilton_files, madison_files, disputed_files,
                                         epochs=args.epochs, batch_size=args.batch_size)
        save_results_csv(results, os.path.join(args.output_dir, f"disputed_papers_epochs{args.epochs}.csv"))

    elif args.mode == "epoch-sweep":
        # Runs full leave-one-out validation at each epoch value in turn, saving
        # a separate CSV per value. This is what actually answers "what's the
        # ideal epoch count" empirically, rather than guessing or borrowing a
        # number from a different paper's dataset.
        summary_rows = []
        for epoch_count in args.epoch_values:
            print(f"\n{'='*60}\nEpoch sweep: running full validation at epochs={epoch_count}\n{'='*60}")
            results = leave_one_out_validation(
                hamilton_files, madison_files,
                sample_size=None,
                epochs=epoch_count,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            save_results_csv(results, os.path.join(args.output_dir, f"leave_one_out_full_epochs{epoch_count}.csv"))
            accuracy = sum(r["correct"] for r in results) / len(results)
            summary_rows.append({"epochs": epoch_count, "accuracy": accuracy, "n": len(results)})

        save_results_csv(summary_rows, os.path.join(args.output_dir, "epoch_sweep_summary.csv"))
        print("\nEpoch sweep summary:")
        for row in summary_rows:
            print(f"  epochs={row['epochs']:>4}  accuracy={row['accuracy']:.1%}  (n={row['n']})")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
