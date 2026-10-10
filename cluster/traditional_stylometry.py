"""
traditional_stylometry.py

Classic Burrows' Delta function-word-frequency stylometric baseline, for
direct comparison against the GPT-2 perplexity method (stylometry_lib.py).
Needs no GPU or fine-tuning -- runs in seconds on a laptop or login node.
This is essentially the same family of technique Mosteller & Wallace (1963)
originally used to study Federalist Papers authorship, decades before LLMs.

Method (Burrows, 2002):
  1. Find the N most frequent words across the whole known corpus -- these
     end up being function words (the, of, to, which, ...) that authors
     use unconsciously and consistently, making them good style
     fingerprints independent of topic.
  2. For each paper, compute each of those words' frequency per 1000 words.
  3. For each candidate author, compute the MEAN and STANDARD DEVIATION of
     each word's frequency across their known papers (excluding whichever
     paper is currently held out -- same leave-one-out design as the GPT-2
     method).
  4. For the held-out paper, compute its Delta score against each author:
     the average |z-score| across all N features, using that author's
     mean/std from step 3.
  5. Attribute to whichever author gives the LOWER Delta score.

Corpus balancing:
  Hamilton has 51 known papers vs. Madison's 14. Left unbalanced, Hamilton's
  author profile (mean/std per word) is estimated from far more documents,
  making it statistically tighter/more stable. That asymmetry can pull
  ambiguous or short papers toward Hamilton's profile for reasons having
  nothing to do with actual writing style -- the same corpus-imbalance
  problem already fixed on the GPT-2 side (see
  leave_one_out_validation_balanced / score_disputed_papers in
  stylometry_lib.py). This script now subsamples Hamilton down to 14 papers
  (matching Madison's count) via a reproducible seed before building either
  author's profile or the shared vocabulary, so both profiles are built from
  equal-sized corpora. Since Delta is cheap (no GPU, seconds to run), this
  also supports sweeping multiple balance seeds to check how much the
  specific Hamilton subsample affects the result.

Disputed-included mode:
  Mirrors --mode disputed-loo on the GPT-2 side (run_experiment.py). Folds
  the 12 disputed papers into Madison's corpus (14 known + 12 disputed = 26
  total), then runs the same balanced leave-one-out procedure over that
  expanded corpus -- Hamilton is auto-subsampled down to 26 to match, same
  mechanism as the default balanced mode. Each disputed paper is held out
  one at a time and compared only against the OTHER Madison-labeled papers
  (which include other disputed papers), a style-consistency check distinct
  from a one-shot scoring against fixed models.

Usage (run directly on a login node -- no GPU needed):
    python traditional_stylometry.py                  # single seed (42)
    python traditional_stylometry.py --balance-seeds 1 2 3 4 5
    python traditional_stylometry.py --no-balance      # old unbalanced behavior
    python traditional_stylometry.py --include-disputed --balance-seeds 1 2 3 4 5
"""

import os
import re
import glob
import csv
import random
import statistics
import argparse
from collections import Counter

CORPUS_ROOT = "corpus/federalist"
N_FEATURES = 100  # most frequent words to use as features
DEFAULT_BALANCE_SEEDS = [42]


def list_txt_files(subfolder):
    return sorted(glob.glob(os.path.join(CORPUS_ROOT, subfolder, "*.txt")))


def clean_text(raw):
    text = raw
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"PUBLIUS\.?\s*$", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def load_tokens(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return tokenize(clean_text(f.read()))


def word_frequencies(tokens, vocabulary):
    total = len(tokens)
    counts = Counter(tokens)
    return {w: (counts.get(w, 0) / total) * 1000 for w in vocabulary}


def build_vocabulary(all_files, n_features):
    counts = Counter()
    for path in all_files:
        counts.update(load_tokens(path))
    return [w for w, _ in counts.most_common(n_features)]


def author_profile(files, vocabulary):
    freq_rows = [word_frequencies(load_tokens(f), vocabulary) for f in files]
    profile = {}
    for w in vocabulary:
        values = [row[w] for row in freq_rows]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        # Guard against zero variance causing a divide-by-zero below.
        stdev = stdev if stdev > 0 else 1e-6
        profile[w] = (mean, stdev)
    return profile


def burrows_delta(paper_freqs, profile, vocabulary):
    z_scores = [abs((paper_freqs[w] - profile[w][0]) / profile[w][1]) for w in vocabulary]
    return sum(z_scores) / len(z_scores)


def balance_hamilton(hamilton_files, madison_files, balance_seed):
    """Subsample Hamilton's papers down to Madison's count, reproducibly."""
    if balance_seed is None or len(hamilton_files) <= len(madison_files):
        return hamilton_files
    rng = random.Random(balance_seed)
    return sorted(rng.sample(hamilton_files, len(madison_files)))


def leave_one_out_delta(hamilton_files, madison_files, n_features=N_FEATURES,
                         balance_seed=42, verbose=True):
    balanced_hamilton = balance_hamilton(hamilton_files, madison_files, balance_seed)

    if verbose and balance_seed is not None and len(balanced_hamilton) < len(hamilton_files):
        print(f"Balancing: using {len(balanced_hamilton)}/{len(hamilton_files)} Hamilton "
              f"papers (seed={balance_seed}) vs. {len(madison_files)} Madison papers.")

    all_files = balanced_hamilton + madison_files
    vocabulary = build_vocabulary(all_files, n_features)
    if verbose:
        print(f"Using top {len(vocabulary)} most frequent words as features.")
        print(f"Most frequent 15: {vocabulary[:15]}")

    all_papers = [("hamilton", f) for f in balanced_hamilton] + [("madison", f) for f in madison_files]
    results = []

    for true_author, held_out_file in all_papers:
        remaining_hamilton = [f for f in balanced_hamilton if f != held_out_file]
        remaining_madison = [f for f in madison_files if f != held_out_file]

        ham_profile = author_profile(remaining_hamilton, vocabulary)
        mad_profile = author_profile(remaining_madison, vocabulary)

        held_out_freqs = word_frequencies(load_tokens(held_out_file), vocabulary)
        delta_h = burrows_delta(held_out_freqs, ham_profile, vocabulary)
        delta_m = burrows_delta(held_out_freqs, mad_profile, vocabulary)

        predicted_author = "hamilton" if delta_h < delta_m else "madison"
        correct = predicted_author == true_author

        results.append({
            "file": os.path.basename(held_out_file),
            "true_author": true_author,
            "predicted_author": predicted_author,
            "delta_hamilton": delta_h,
            "delta_madison": delta_m,
            "correct": correct,
            "balance_seed": balance_seed,
        })
        if verbose:
            print(f"{os.path.basename(held_out_file)} | true={true_author} pred={predicted_author} "
                  f"(Delta H:{delta_h:.3f} M:{delta_m:.3f}) {'CORRECT' if correct else 'WRONG'}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    n_ham = sum(1 for r in results if r["true_author"] == "hamilton")
    n_mad = sum(1 for r in results if r["true_author"] == "madison")
    correct_ham = sum(1 for r in results if r["true_author"] == "hamilton" and r["correct"])
    correct_mad = sum(1 for r in results if r["true_author"] == "madison" and r["correct"])

    if verbose:
        print(f"\nOverall Burrows' Delta leave-one-out accuracy (balance_seed={balance_seed}): "
              f"{accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
        print(f"  Hamilton: {correct_ham}/{n_ham} ({correct_ham/n_ham:.1%})")
        print(f"  Madison: {correct_mad}/{n_mad} ({correct_mad/n_mad:.1%})")
    return results


def save_results_csv(results, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {out_path}")


def save_summary_csv(all_seed_results, out_path):
    """One row per balance_seed, for comparing across seeds at a glance."""
    rows = []
    for seed, results in all_seed_results.items():
        n = len(results)
        n_correct = sum(r["correct"] for r in results)
        n_ham = sum(1 for r in results if r["true_author"] == "hamilton")
        n_mad = sum(1 for r in results if r["true_author"] == "madison")
        correct_ham = sum(1 for r in results if r["true_author"] == "hamilton" and r["correct"])
        correct_mad = sum(1 for r in results if r["true_author"] == "madison" and r["correct"])
        rows.append({
            "balance_seed": seed,
            "accuracy": n_correct / n,
            "n_correct": n_correct,
            "n_total": n,
            "hamilton_accuracy": correct_ham / n_ham if n_ham else float("nan"),
            "madison_accuracy": correct_mad / n_mad if n_mad else float("nan"),
        })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-seed summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Burrows' Delta baseline (balanced by default).")
    parser.add_argument("--balance-seeds", type=int, nargs="+", default=DEFAULT_BALANCE_SEEDS,
                         help="One or more seeds to subsample Hamilton's 51 papers down to "
                              "Madison's 14. Each seed is run independently and all results "
                              "are concatenated into the detailed CSV, with a per-seed summary "
                              "CSV alongside it. Default: [42].")
    parser.add_argument("--no-balance", action="store_true",
                         help="Disable balancing and use all papers from both authors "
                              "(the original, unbalanced behavior).")
    parser.add_argument("--n-features", type=int, default=N_FEATURES,
                         help="Number of most-frequent words to use as features.")
    parser.add_argument("--include-disputed", action="store_true",
                         help="Fold the 12 disputed papers into Madison's corpus (14 known + "
                              "12 disputed = 26) before running balanced LOO. Hamilton is "
                              "auto-subsampled to 26 to match. Mirrors --mode disputed-loo on "
                              "the GPT-2 side. Mutually exclusive with --no-balance.")
    args = parser.parse_args()

    hamilton_files = list_txt_files("hamilton")
    madison_files = list_txt_files("madison")
    print(f"Loaded {len(hamilton_files)} Hamilton papers, {len(madison_files)} Madison papers.")

    if args.include_disputed:
        if args.no_balance:
            raise SystemExit("--include-disputed and --no-balance are mutually exclusive.")

        disputed_files = list_txt_files("disputed")
        print(f"Loaded {len(disputed_files)} disputed papers.")
        expanded_madison = madison_files + disputed_files
        disputed_basenames = {os.path.basename(f) for f in disputed_files}
        print(f"Expanded Madison corpus: {len(madison_files)} known + "
              f"{len(disputed_files)} disputed = {len(expanded_madison)} total.")

        all_seed_results = {}
        combined_results = []
        for seed in args.balance_seeds:
            print(f"\n=== balance_seed={seed} ===")
            results = leave_one_out_delta(hamilton_files, expanded_madison,
                                           n_features=args.n_features, balance_seed=seed)
            # true_author is "madison" for every expanded-corpus row -- flag
            # which ones were originally disputed so their accuracy can be
            # reported separately from the 14 already-known Madison papers.
            for r in results:
                r["was_disputed"] = r["file"] in disputed_basenames
            all_seed_results[seed] = results
            combined_results.extend(results)

        save_results_csv(combined_results, "results/traditional_burrows_delta_disputed_included.csv")

        print("\nDisputed-included accuracy per seed:")
        summary_rows = []
        for seed, results in all_seed_results.items():
            n_disputed = sum(1 for r in results if r["was_disputed"])
            correct_disputed = sum(1 for r in results if r["was_disputed"] and r["correct"])
            overall_accuracy = sum(r["correct"] for r in results) / len(results)
            disputed_accuracy = correct_disputed / n_disputed if n_disputed else float("nan")
            print(f"  seed={seed:>2}  overall={overall_accuracy:.1%}  "
                  f"disputed-papers-correct={disputed_accuracy:.1%} (n={n_disputed})")
            summary_rows.append({
                "balance_seed": seed,
                "overall_accuracy": overall_accuracy,
                "disputed_accuracy": disputed_accuracy,
                "n_disputed": n_disputed,
                "n_total": len(results),
            })
        save_results_csv(summary_rows, "results/traditional_burrows_delta_disputed_included_summary.csv")
        return

    if args.no_balance:
        results = leave_one_out_delta(hamilton_files, madison_files,
                                       n_features=args.n_features, balance_seed=None)
        save_results_csv(results, "results/traditional_burrows_delta_unbalanced.csv")
        return

    all_seed_results = {}
    combined_results = []
    for seed in args.balance_seeds:
        print(f"\n=== balance_seed={seed} ===")
        results = leave_one_out_delta(hamilton_files, madison_files,
                                       n_features=args.n_features, balance_seed=seed)
        all_seed_results[seed] = results
        combined_results.extend(results)

    save_results_csv(combined_results, "results/traditional_burrows_delta_balanced.csv")
    if len(args.balance_seeds) > 1:
        save_summary_csv(all_seed_results, "results/traditional_burrows_delta_balanced_summary.csv")


if __name__ == "__main__":
    main()