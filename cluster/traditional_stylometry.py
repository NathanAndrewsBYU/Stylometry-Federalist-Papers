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

Usage (run directly on a login node -- no GPU needed):
    python traditional_stylometry.py
"""

import os
import re
import glob
import csv
import statistics
from collections import Counter

CORPUS_ROOT = "corpus/federalist"
N_FEATURES = 100  # most frequent words to use as features


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


def leave_one_out_delta(hamilton_files, madison_files, n_features=N_FEATURES):
    all_files = hamilton_files + madison_files
    vocabulary = build_vocabulary(all_files, n_features)
    print(f"Using top {len(vocabulary)} most frequent words as features.")
    print(f"Most frequent 15: {vocabulary[:15]}")

    all_papers = [("hamilton", f) for f in hamilton_files] + [("madison", f) for f in madison_files]
    results = []

    for true_author, held_out_file in all_papers:
        remaining_hamilton = [f for f in hamilton_files if f != held_out_file]
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
        })
        print(f"{os.path.basename(held_out_file)} | true={true_author} pred={predicted_author} "
              f"(Delta H:{delta_h:.3f} M:{delta_m:.3f}) {'CORRECT' if correct else 'WRONG'}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    n_ham = sum(1 for r in results if r["true_author"] == "hamilton")
    n_mad = sum(1 for r in results if r["true_author"] == "madison")
    correct_ham = sum(1 for r in results if r["true_author"] == "hamilton" and r["correct"])
    correct_mad = sum(1 for r in results if r["true_author"] == "madison" and r["correct"])

    print(f"\nOverall Burrows' Delta leave-one-out accuracy: "
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


def main():
    hamilton_files = list_txt_files("hamilton")
    madison_files = list_txt_files("madison")
    print(f"Loaded {len(hamilton_files)} Hamilton papers, {len(madison_files)} Madison papers.")

    results = leave_one_out_delta(hamilton_files, madison_files)
    save_results_csv(results, "results/traditional_burrows_delta.csv")


if __name__ == "__main__":
    main()