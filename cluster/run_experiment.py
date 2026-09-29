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

    # Epoch sweep (UNBALANCED method): run full validation at several epoch
    # counts to find the point where accuracy stops improving
    python run_experiment.py --mode epoch-sweep --epoch-values 5 10 20 40 80

    # Balanced method (corrects for Hamilton/Madison corpus-size imbalance):
    # a single run is just one value in each list.
    python run_experiment.py --mode balanced --epoch-values 30 --balance-seeds 42

    # Balanced epoch sweep: multiple epoch values, one fixed subsample
    python run_experiment.py --mode balanced --epoch-values 10 20 30 40 --balance-seeds 42

    # Balanced seed sweep: one fixed epoch count, multiple subsamples
    python run_experiment.py --mode balanced --epoch-values 30 --balance-seeds 1 2 3 4 5

    # Full grid: multiple epoch values x multiple subsamples
    python run_experiment.py --mode balanced --epoch-values 30 40 --balance-seeds 1 2 3 4 5
"""

import argparse
import os
import time

from stylometry_lib import (
    device,
    list_txt_files,
    leave_one_out_validation,
    leave_one_out_validation_balanced,
    score_disputed_papers,
    save_results_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Run Federalist Papers GPT-2 authorship experiments.")
    parser.add_argument("--mode", required=True,
                         choices=["first-try", "full", "disputed", "epoch-sweep", "balanced"],
                         help="Which experiment to run.")
    parser.add_argument("--sample-size", type=int, default=None,
                         help="For --mode first-try: number of random papers to test (default: 10).")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Training epochs per fine-tuned model, used by --mode first-try, "
                              "full, and disputed (default: 10).")
    parser.add_argument("--epoch-values", type=int, nargs="+", default=[30],
                         help="Epoch counts to test. For --mode epoch-sweep or balanced, pass "
                              "multiple values to sweep across epoch counts; a single value "
                              "just runs once at that epoch count.")
    parser.add_argument("--balance-seeds", type=int, nargs="+", default=[42],
                         help="For --mode balanced: random seeds controlling which Hamilton "
                              "papers get sampled into the size-matched subsample. Pass "
                              "multiple values to test sensitivity to which papers get picked; "
                              "a single value just runs once with that subsample.")
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
        # UNBALANCED method (uses the full Hamilton/Madison corpora as-is).
        # Runs full leave-one-out validation at each epoch value in turn,
        # saving a separate CSV per value.
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

    elif args.mode == "balanced":
        # BALANCED method (matches Hamilton's training corpus size to
        # Madison's, to correct the corpus-size bias the unbalanced method
        # showed). Tests every combination of --epoch-values x
        # --balance-seeds. A plain single run is just one value in each list
        # (the argument defaults give you exactly that: epochs=[30], seed=[42]).
        summary_rows = []
        for epoch_count in args.epoch_values:
            for b_seed in args.balance_seeds:
                print(f"\n{'='*60}\nBalanced: epochs={epoch_count}, balance_seed={b_seed}\n{'='*60}")
                results = leave_one_out_validation_balanced(
                    hamilton_files, madison_files,
                    epochs=epoch_count,
                    seed=args.seed,
                    balance_seed=b_seed,
                    batch_size=args.batch_size,
                    save_full_models=args.save_full_models,
                )
                save_results_csv(results, os.path.join(
                    args.output_dir, f"leave_one_out_balanced_epochs{epoch_count}_seed{b_seed}.csv"))

                n_ham = sum(1 for r in results if r["true_author"] == "hamilton")
                n_mad = sum(1 for r in results if r["true_author"] == "madison")
                correct_ham = sum(1 for r in results if r["true_author"] == "hamilton" and r["correct"])
                correct_mad = sum(1 for r in results if r["true_author"] == "madison" and r["correct"])
                overall_accuracy = sum(r["correct"] for r in results) / len(results)

                summary_rows.append({
                    "epochs": epoch_count,
                    "balance_seed": b_seed,
                    "overall_accuracy": overall_accuracy,
                    "hamilton_accuracy": correct_ham / n_ham if n_ham else None,
                    "madison_accuracy": correct_mad / n_mad if n_mad else None,
                    "n": len(results),
                })

        save_results_csv(summary_rows, os.path.join(args.output_dir, "balanced_summary.csv"))
        print("\nBalanced summary:")
        for row in summary_rows:
            print(f"  epochs={row['epochs']:>4}  seed={row['balance_seed']:>2}  "
                  f"overall={row['overall_accuracy']:.1%}  hamilton={row['hamilton_accuracy']:.1%}  "
                  f"madison={row['madison_accuracy']:.1%}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()