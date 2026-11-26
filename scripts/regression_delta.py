"""Regression check (plan section 5): the harness Delta arm, run in the
legacy design, must reproduce results/traditional_burrows_delta_balanced.csv
(the Colab n=65 notebook) exactly.

Legacy design reproduced here:
  - Hamilton subsampled to 14 with random.Random(seed).sample(sorted files, 14)
  - 100 most frequent words counted over the whole balanced pool, INCLUDING
    the held-out paper, with ties broken by first occurrence (Counter order)
  - per-class mean/SD profiles; asymmetric leave-one-out on the 28 pool papers
  - the other 37 Hamilton papers scored against the full-pool profiles

Usage (repo root):
  python scripts/regression_delta.py --corpus-root cluster/corpus/federalist
Exit code 0 = all rows match.
"""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from stylo.corpus import load_corpus  # noqa: E402
from stylo.evaluate import run_loo  # noqa: E402
from stylo.models.delta import DeltaArm  # noqa: E402
from stylo.representations import make_windows  # noqa: E402
from stylo.tasks import build_task  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--expected", default="results/traditional_burrows_delta_balanced.csv")
    p.add_argument("--tol", type=float, default=1e-9)
    a = p.parse_args()

    units = load_corpus(a.corpus_root)
    windows = {uid: make_windows(uid, u.cls, u.text, "raw", None) for uid, u in units.items()}

    expected: dict[tuple[int, str], dict] = {}
    with open(a.expected, newline="") as f:
        for r in csv.DictReader(f):
            expected[(int(r["balance_seed"]), r["file"])] = r

    seeds = sorted({s for s, _ in expected})
    n_checked = n_bad = 0
    max_err = 0.0
    for seed in seeds:
        task = build_task("hvm", units, seed)
        vocab_counter = Counter()
        for c in ("hamilton", "madison"):
            for uid in task.pool[c]:
                vocab_counter.update(windows[uid][0].counts)
        vocab = [w for w, _ in vocab_counter.most_common(100)]
        results = run_loo(lambda: DeltaArm("per_class", vocab=vocab, exact=True), task.pool, windows,
                          task.classes, symmetric=False, extras=task.extras)
        for r in results:
            exp = expected.get((seed, r.unit_id + ".txt"))
            if exp is None:
                print(f"seed {seed}: {r.unit_id} not in expected CSV")
                n_bad += 1
                continue
            err = max(abs(r.scores["hamilton"] - float(exp["delta_hamilton"])),
                      abs(r.scores["madison"] - float(exp["delta_madison"])))
            max_err = max(max_err, err)
            same_pred = r.predicted == exp["predicted_author"]
            if err > a.tol or not same_pred:
                n_bad += 1
                print(f"MISMATCH seed {seed} {r.unit_id}: err={err:.3e} "
                      f"pred={r.predicted} expected={exp['predicted_author']}")
            n_checked += 1
        acc = sum(r.correct for r in results) / len(results)
        print(f"seed {seed}: {len(results)} papers, accuracy {acc:.4f}")

    print(f"\nChecked {n_checked} rows (expected {len(expected)}); mismatches: {n_bad}; "
          f"max |Delta error| = {max_err:.2e}")
    ok = n_bad == 0 and n_checked == len(expected)
    print("REGRESSION CHECK PASSED" if ok else "REGRESSION CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
