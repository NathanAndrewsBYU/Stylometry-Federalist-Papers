"""Federalist Papers task definitions (prereg/phase_a_plan.md section 6).

Each task turns the corpus into (pool, extras, classes, scheme_hint):
  pool    class -> unit ids evaluated by cross-validation
  extras  class -> unit ids scored by a model trained on the whole pool
  classes (neg, pos); pos is the AUC-positive class
For "fixed" tasks the pool is the training set and extras is the test set.

`seed` means the balance seed for author tasks and the split seed for the
within-author null tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Unit
from .splits import Pool, balanced_subsample, random_split

# Topic block shared by both authors (Fed 2-14); see plan section 1.
SHARED_TOPIC_BLOCK = set(range(2, 15))
EARLY_MAX, LATE_MIN = 36, 59


@dataclass
class Task:
    name: str
    pool: Pool
    extras: dict[str, list[str]]
    classes: tuple[str, str]
    fixed: bool = False        # True: train on pool, score extras only


def _ids(units: dict[str, Unit], cls: str, pred=lambda u: True) -> list[str]:
    return sorted(uid for uid, u in units.items() if u.cls == cls and pred(u))


def build_task(name: str, units: dict[str, Unit], seed: int) -> Task:
    ham, mad = _ids(units, "hamilton"), _ids(units, "madison")

    if name == "hvm":
        sub = balanced_subsample(ham, len(mad), seed)
        rest = [u for u in ham if u not in sub]
        return Task(name, {"hamilton": sub, "madison": mad}, {"hamilton": rest} if rest else {},
                    ("hamilton", "madison"))

    if name == "null_ham":
        return Task(name, random_split(ham, len(mad), seed), {}, ("group_a", "group_b"))

    if name == "null_mad":
        return Task(name, random_split(mad, len(mad) // 2, seed), {}, ("group_a", "group_b"))

    if name == "early_late":
        early = _ids(units, "hamilton", lambda u: u.number is not None and u.number <= EARLY_MAX)
        late = _ids(units, "hamilton", lambda u: u.number is not None and u.number >= LATE_MIN)
        n = min(len(mad), len(early), len(late))
        return Task(name, {"early": balanced_subsample(early, n, seed),
                           "late": balanced_subsample(late, n, seed + 1000)}, {}, ("early", "late"))

    if name in ("hvj", "mvj"):
        jay = _ids(units, "jay")
        other_cls, other = ("hamilton", ham) if name == "hvj" else ("madison", mad)
        n = len(jay)
        return Task(name, {other_cls: balanced_subsample(other, n, seed), "jay": jay}, {},
                    (other_cls, "jay"))

    if name == "topic_shift":
        in_block = lambda u: u.number in SHARED_TOPIC_BLOCK
        out_block = lambda u: u.number not in SHARED_TOPIC_BLOCK
        mad_train = _ids(units, "madison", out_block)
        ham_train = balanced_subsample(_ids(units, "hamilton", out_block), len(mad_train), seed)
        test = {"hamilton": _ids(units, "hamilton", in_block), "madison": _ids(units, "madison", in_block)}
        return Task(name, {"hamilton": ham_train, "madison": mad_train}, test,
                    ("hamilton", "madison"), fixed=True)

    if name == "disputed":
        sub = balanced_subsample(ham, len(mad), seed)
        return Task(name, {"hamilton": sub, "madison": mad}, {"disputed": _ids(units, "disputed")},
                    ("hamilton", "madison"), fixed=True)

    raise ValueError(f"unknown task {name!r}")


TASK_NAMES = ("hvm", "null_ham", "null_mad", "early_late", "hvj", "mvj", "topic_shift", "disputed")
