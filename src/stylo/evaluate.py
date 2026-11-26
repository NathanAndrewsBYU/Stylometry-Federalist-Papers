"""Model-agnostic evaluation: grouped cross-validation, leave-one-out with
symmetric hold-out, fixed train/test splits, and label permutations.

Arms only ever see `fit(train)` and `score(window)`; they never see labels
other than the keys of the training dict, so permutation tests are honest.

Decision convention: classes = (neg, pos). For each window,
d = score[neg] - score[pos] (scores are "lower = closer"), so d > 0 means
"looks like pos". A unit's decision is the mean d over its windows, and the
unit-level AUC treats `pos` as the positive class.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .metrics import auc, bootstrap_auc_ci
from .representations import Window
from .splits import Pool, permute_labels, stratified_group_kfold


class Arm(Protocol):
    name: str

    def fit(self, train: dict[str, list[Window]]) -> None: ...

    def score(self, window: Window) -> dict[str, float]: ...


ArmFactory = Callable[[], Arm]
WindowsByUnit = dict[str, list[Window]]


@dataclass
class UnitResult:
    unit_id: str
    true_cls: str
    in_pool: bool          # False = scored by a model trained on the full pool
    fold: int              # -1 for out-of-pool units
    decision: float
    scores: dict[str, float] = field(default_factory=dict)
    predicted: str = ""
    correct: bool = False
    n_windows: int = 0

    def row(self, classes: tuple[str, str]) -> dict:
        r = {"unit_id": self.unit_id, "true_cls": self.true_cls, "in_pool": self.in_pool,
             "fold": self.fold, "decision": self.decision, "predicted": self.predicted,
             "correct": self.correct, "n_windows": self.n_windows}
        for c in classes:
            r[f"score_{c}"] = self.scores.get(c, float("nan"))
        return r


def _train_dict(pool: Pool, exclude: set[str], windows: WindowsByUnit,
                max_words: int | None = None, rng: random.Random | None = None) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {}
    for c in sorted(pool):
        ws = [w for uid in pool[c] if uid not in exclude for w in windows[uid]]
        if max_words is not None:
            ws = list(ws)
            (rng or random.Random(0)).shuffle(ws)
            kept, total = [], 0
            for w in ws:
                if total >= max_words:
                    break
                kept.append(w)
                total += w.n_words
            ws = sorted(kept, key=lambda w: (w.unit_id, w.index))
        if not ws:
            raise ValueError(f"class {c!r} has no training windows")
        out[c] = ws
    return out


def _score_unit(arm: Arm, wins: list[Window], classes: tuple[str, str]) -> tuple[float, dict[str, float]]:
    neg, pos = classes
    ds, acc = [], {c: [] for c in classes}
    for w in wins:
        s = arm.score(w)
        ds.append(s[neg] - s[pos])
        for c in classes:
            acc[c].append(s[c])
    return statistics.fmean(ds), {c: statistics.fmean(v) for c, v in acc.items()}


def _finish(r: UnitResult, classes: tuple[str, str]) -> UnitResult:
    neg, pos = classes
    r.predicted = pos if r.decision > 0 else neg
    r.correct = r.predicted == r.true_cls
    return r


def _label_of(pool: Pool) -> dict[str, str]:
    return {uid: c for c, ids in pool.items() for uid in ids}


def _score_extras(factory: ArmFactory, pool: Pool, extras: dict[str, list[str]] | None,
                  windows: WindowsByUnit, classes, max_words, rng) -> list[UnitResult]:
    if not extras:
        return []
    arm = factory()
    arm.fit(_train_dict(pool, set(), windows, max_words, rng))
    out = []
    for c, ids in sorted(extras.items()):
        for uid in ids:
            d, s = _score_unit(arm, windows[uid], classes)
            out.append(_finish(UnitResult(uid, c, False, -1, d, s, n_windows=len(windows[uid])), classes))
    return out


def run_kfold(factory: ArmFactory, pool: Pool, windows: WindowsByUnit, classes: tuple[str, str],
              k: int = 5, fold_seed: int = 0, extras: dict[str, list[str]] | None = None,
              max_train_words: int | None = None, train_seed: int = 0) -> list[UnitResult]:
    """Stratified grouped k-fold over `pool` (symmetric hold-out when the
    classes are the same size). `extras` are scored once by a model trained
    on the whole pool (e.g. Hamilton papers outside the balanced subsample)."""
    rng = random.Random(train_seed)
    label = _label_of(pool)
    results: list[UnitResult] = []
    for fi, test in enumerate(stratified_group_kfold(pool, k, fold_seed)):
        test_ids = {uid for ids in test.values() for uid in ids}
        if not test_ids:
            continue
        arm = factory()
        arm.fit(_train_dict(pool, test_ids, windows, max_train_words, rng))
        for uid in sorted(test_ids):
            d, s = _score_unit(arm, windows[uid], classes)
            results.append(_finish(UnitResult(uid, label[uid], True, fi, d, s,
                                              n_windows=len(windows[uid])), classes))
    return results + _score_extras(factory, pool, extras, windows, classes, max_train_words, rng)


def run_loo(factory: ArmFactory, pool: Pool, windows: WindowsByUnit, classes: tuple[str, str],
            symmetric: bool = True, repeats: int = 20, seed: int = 0,
            extras: dict[str, list[str]] | None = None) -> list[UnitResult]:
    """Leave-one-unit-out. symmetric=True also drops one random unit of every
    other class from training (D-006), repeated `repeats` times with the
    decision averaged. symmetric=False is the legacy (biased) design, kept
    only for the regression check against the old CSVs."""
    rng = random.Random(seed)
    label = _label_of(pool)
    results: list[UnitResult] = []
    for i, uid in enumerate(sorted(label)):
        reps = repeats if symmetric else 1
        ds, ss = [], []
        for _ in range(reps):
            exclude = {uid}
            if symmetric:
                for c in sorted(pool):
                    if c != label[uid]:
                        exclude.add(rng.choice(sorted(pool[c])))
            arm = factory()
            arm.fit(_train_dict(pool, exclude, windows))
            d, s = _score_unit(arm, windows[uid], classes)
            ds.append(d)
            ss.append(s)
        scores = {c: statistics.fmean(s[c] for s in ss) for c in classes}
        results.append(_finish(UnitResult(uid, label[uid], True, i, statistics.fmean(ds), scores,
                                          n_windows=len(windows[uid])), classes))
    return results + _score_extras(factory, pool, extras, windows, classes, None, rng)


def run_fixed(factory: ArmFactory, train_pool: Pool, test: dict[str, list[str]],
              windows: WindowsByUnit, classes: tuple[str, str]) -> list[UnitResult]:
    """One model trained on train_pool, scoring the `test` units (class ->
    ids). Used for the topic-shift test and disputed-paper scoring."""
    arm = factory()
    arm.fit(_train_dict(train_pool, set(), windows))
    out = []
    for c, ids in sorted(test.items()):
        for uid in ids:
            d, s = _score_unit(arm, windows[uid], classes)
            out.append(_finish(UnitResult(uid, c, True, 0, d, s, n_windows=len(windows[uid])), classes))
    return out


def unit_auc(results: Iterable[UnitResult], classes: tuple[str, str], in_pool_only: bool = True):
    neg, pos = classes
    rs = [r for r in results if r.in_pool or not in_pool_only]
    return ([r.decision for r in rs if r.true_cls == pos],
            [r.decision for r in rs if r.true_cls == neg])


def summarize(results: list[UnitResult], classes: tuple[str, str], n_boot: int = 2000,
              boot_seed: int = 0) -> dict:
    neg, pos = classes
    p, n = unit_auc(results, classes)
    pool = [r for r in results if r.in_pool]
    out = {
        "n_pool": len(pool),
        "n_all": len(results),
        "auc_pool": auc(p, n),
        "acc_pool": sum(r.correct for r in pool) / len(pool) if pool else float("nan"),
        "acc_all": sum(r.correct for r in results) / len(results) if results else float("nan"),
    }
    for c in classes:
        rc = [r for r in results if r.true_cls == c]
        out[f"acc_{c}_all"] = sum(r.correct for r in rc) / len(rc) if rc else float("nan")
        out[f"n_{c}_all"] = len(rc)
    if p and n and n_boot:
        out["auc_pool_ci_lo"], out["auc_pool_ci_hi"] = bootstrap_auc_ci(p, n, n_boot, boot_seed)
    return out


def run_permutations(factory: ArmFactory, pool: Pool, windows: WindowsByUnit,
                     classes: tuple[str, str], k: int, fold_seed: int,
                     perm_seeds: Iterable[int]) -> list[tuple[int, float]]:
    """Label-permutation null for the k-fold design: relabel the pooled units
    (class sizes kept), rebuild stratified folds on the new labels, rerun
    the whole CV, record the unit-level AUC. Returns (perm_seed, auc)."""
    out = []
    for ps in perm_seeds:
        permuted = permute_labels(pool, ps)
        res = run_kfold(factory, permuted, windows, classes, k=k, fold_seed=fold_seed + ps)
        p, n = unit_auc(res, classes)
        out.append((ps, auc(p, n)))
    return out
