"""Command-line runner for Phase A screening runs.

Examples (from the repo root; see scripts/run_screen.py for the launcher):
  python scripts/run_screen.py --corpus-root data/corpus/federalist --task hvm \
      --arm delta --rep raw --window 0 --scheme kfold --seeds 1 2 3 4 5 --n-perm 1000 --out results/phase_a/delta_raw
  python scripts/run_screen.py ... --task null_ham --seeds 1-100 --scheme kfold

Outputs in --out:
  units.csv         one row per (seed, unit): decision, per-class scores, prediction
  summary.csv       one row per seed: AUC (+ bootstrap CI), accuracies, permutation stats
  permutations.csv  one row per (seed, permutation): AUC under shuffled labels
  manifest.json     git commit, config + hash, Slurm job id, corpus/function-word hashes
Permutations can be split across Slurm array tasks with --perm-start / --n-perm;
each task then writes its own --out directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from .corpus import corpus_hash, load_corpus
from .evaluate import (run_fixed, run_kfold, run_loo, run_permutations, summarize)
from .manifest import build_manifest, write_manifest
from .metrics import permutation_summary
from .models import ARM_NAMES, make_arm_factory
from .representations import NEEDS_POS, REPRESENTATIONS, make_windows, spacy_tagger
from .tasks import TASK_NAMES, build_task
from .text import function_words_hash


def _seed_list(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--cleaner", default="federalist")
    p.add_argument("--task", required=True, choices=TASK_NAMES)
    p.add_argument("--arm", required=True, choices=ARM_NAMES)
    p.add_argument("--arm-args", default="{}", help='JSON dict of arm options, e.g. \'{"order": 4}\'')
    p.add_argument("--rep", default="raw", choices=REPRESENTATIONS)
    p.add_argument("--window", type=int, default=0, help="window size in words; 0 = whole unit")
    p.add_argument("--scheme", default="kfold", choices=["kfold", "loo_sym", "loo_legacy"],
                   help="ignored for fixed tasks (topic_shift, disputed)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seeds", nargs="+", default=["1"], help="balance or split seeds; ranges like 1-100 allowed")
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--loo-repeats", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=0)
    p.add_argument("--perm-start", type=int, default=0)
    p.add_argument("--max-train-words", type=int, default=None, help="learning curves: cap per class")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    arm_kw = json.loads(a.arm_args)
    factory = make_arm_factory(a.arm, **arm_kw)
    seeds = _seed_list(a.seeds)

    units = load_corpus(a.corpus_root, a.cleaner)
    tagger = spacy_tagger() if a.rep in NEEDS_POS else None
    win_cache: dict[str, list] = {}

    def windows_for(ids):
        for uid in ids:
            if uid not in win_cache:
                u = units[uid]
                win_cache[uid] = make_windows(uid, u.cls, u.text, a.rep, a.window or None, tagger)
        return win_cache

    unit_rows, summary_rows, perm_rows = [], [], []
    for seed in seeds:
        task = build_task(a.task, units, seed)
        all_ids = [uid for ids in list(task.pool.values()) + list(task.extras.values()) for uid in ids]
        windows = windows_for(all_ids)
        classes = task.classes

        if task.fixed:
            results = run_fixed(factory, task.pool, task.extras, windows, classes)
            scheme = "fixed"
        elif a.scheme == "kfold":
            results = run_kfold(factory, task.pool, windows, classes, k=a.k, fold_seed=a.fold_seed,
                                extras=task.extras, max_train_words=a.max_train_words, train_seed=seed)
            scheme = "kfold"
        else:
            results = run_loo(factory, task.pool, windows, classes, symmetric=(a.scheme == "loo_sym"),
                              repeats=a.loo_repeats, seed=seed, extras=task.extras)
            scheme = a.scheme

        for r in results:
            unit_rows.append({"seed": seed, **r.row(classes)})
        summ = {"seed": seed, "task": a.task, "arm": a.arm, "rep": a.rep, "window": a.window,
                "scheme": scheme, "neg": classes[0], "pos": classes[1],
                **summarize(results, classes, n_boot=a.boot, boot_seed=seed)}
        if task.fixed:
            for c in sorted(task.extras):
                rc = [r for r in results if r.true_cls == c]
                summ[f"n_{c}_pred_{classes[1]}"] = sum(r.predicted == classes[1] for r in rc)

        if a.n_perm and scheme == "kfold":
            perm_seeds = range(a.perm_start, a.perm_start + a.n_perm)
            null = run_permutations(factory, task.pool, windows, classes, a.k, a.fold_seed, perm_seeds)
            perm_rows.extend({"seed": seed, "perm_seed": ps, "auc": v} for ps, v in null)
            summ.update(permutation_summary(summ["auc_pool"], [v for _, v in null]))
        summary_rows.append(summ)
        print(f"seed={seed} {a.task} {a.arm}/{a.rep}/w{a.window} {scheme}: "
              f"AUC={summ['auc_pool']:.3f} acc_pool={summ['acc_pool']:.3f} acc_all={summ['acc_all']:.3f}"
              + (f" p_perm={summ['p_perm']:.4f} z={summ['z_perm']:.2f}" if "p_perm" in summ else ""),
              flush=True)

    _write_csv(os.path.join(a.out, "units.csv"), unit_rows)
    _write_csv(os.path.join(a.out, "summary.csv"), summary_rows)
    _write_csv(os.path.join(a.out, "permutations.csv"), perm_rows)
    config = {k: v for k, v in vars(a).items() if k != "out"}
    write_manifest(os.path.join(a.out, "manifest.json"), build_manifest(
        config,
        corpus_hash=corpus_hash(units),
        function_words_hash=function_words_hash(),
        tagger=getattr(tagger, "model_name", None),
        runtime_sec=round(time.time() - t0, 1),
    ))
    print(f"Wrote {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
