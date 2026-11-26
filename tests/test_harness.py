"""Unit and integration tests for src/stylo (standard library only).

Run from the repo root:
  python -m unittest discover -s tests -v
The linear arm test is skipped if scikit-learn is not installed.
"""

import importlib.util
import itertools
import math
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from stylo import metrics, splits  # noqa: E402
from stylo.corpus import load_corpus  # noqa: E402
from stylo.evaluate import run_kfold, run_loo, run_permutations, summarize, unit_auc  # noqa: E402
from stylo.models import make_arm_factory  # noqa: E402
from stylo.models.ngram import EOS, _KN  # noqa: E402
from stylo.representations import apply_representation, make_windows  # noqa: E402
from stylo.tasks import build_task  # noqa: E402
from stylo.text import load_function_words, tokenize  # noqa: E402

FW = sorted(load_function_words())
CONTENT = ("state government power union faction people law court senate army treaty tax "
           "commerce militia liberty property republic nation council election").split()


def _paper(rng, fw_pool, n_words=600):
    words = []
    for _ in range(n_words):
        if rng.random() < 0.55:
            words.append(rng.choice(fw_pool))
        else:
            words.append(rng.choice(CONTENT))
        if rng.random() < 0.08:
            words[-1] += rng.choice([",", ".", ";"])
    return " ".join(words).capitalize() + ". PUBLIUS."


def make_synthetic_corpus(root, seed=0):
    """Hamilton and Madison differ only in function-word preferences;
    content words are drawn identically, so topic carries no signal."""
    rng = random.Random(seed)
    ham_fw = FW[:60] * 3 + FW[60:120]
    mad_fw = FW[:60] + FW[60:120] * 3
    plan = {"hamilton": (list(range(1, 52)), ham_fw), "madison": (list(range(52, 66)), mad_fw),
            "jay": (list(range(66, 71)), FW[:120]), "disputed": (list(range(71, 83)), mad_fw)}
    for cls, (nums, pool) in plan.items():
        os.makedirs(os.path.join(root, cls), exist_ok=True)
        for n in nums:
            with open(os.path.join(root, cls, f"federalist_{n:02d}.txt"), "w", encoding="utf-8") as f:
                f.write(_paper(rng, pool))


class TestMetrics(unittest.TestCase):
    def test_auc_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(50):
            pos = [rng.choice([0, 1, 2, 3]) + rng.random() * rng.choice([0, 1]) for _ in range(rng.randint(1, 9))]
            neg = [rng.choice([0, 1, 2, 3]) + rng.random() * rng.choice([0, 1]) for _ in range(rng.randint(1, 9))]
            brute = sum(1.0 if p > n else 0.5 if p == n else 0.0
                        for p, n in itertools.product(pos, neg)) / (len(pos) * len(neg))
            self.assertAlmostEqual(metrics.auc(pos, neg), brute)

    def test_permutation_p(self):
        s = metrics.permutation_summary(0.9, [0.5] * 19)
        self.assertAlmostEqual(s["p_perm"], 1 / 20)


class TestSplits(unittest.TestCase):
    def test_kfold_symmetric_and_complete(self):
        pool = {"a": [f"a{i}" for i in range(14)], "b": [f"b{i}" for i in range(14)]}
        folds = splits.stratified_group_kfold(pool, 5, seed=3)
        seen = [u for f in folds for ids in f.values() for u in ids]
        self.assertEqual(sorted(seen), sorted(pool["a"] + pool["b"]))
        for f in folds:
            self.assertEqual(len(f["a"]), len(f["b"]))

    def test_permute_keeps_sizes(self):
        pool = {"a": [f"u{i}" for i in range(10)], "b": [f"u{i}" for i in range(10, 25)]}
        p = splits.permute_labels(pool, 7)
        self.assertEqual(len(p["a"]), 10)
        self.assertEqual(len(p["b"]), 15)
        self.assertEqual(sorted(p["a"] + p["b"]), sorted(pool["a"] + pool["b"]))

    def test_balanced_subsample_matches_legacy_rule(self):
        ids = [f"x{i:02d}" for i in range(51)]
        self.assertEqual(splits.balanced_subsample(ids, 14, 42),
                         sorted(random.Random(42).sample(sorted(ids), 14)))


class TestRepresentations(unittest.TestCase):
    def test_masks(self):
        toks = tokenize("The government, of the people, shall stand.")
        self.assertEqual(apply_representation("fw_x", toks),
                         ["The", "X", ",", "of", "the", "X", ",", "shall", "X", "."])
        self.assertEqual(apply_representation("content_only", toks),
                         ["X", "government", ",", "X", "X", "people", ",", "X", "stand", "."])
        self.assertEqual(apply_representation("fw_only", toks), ["the", "of", "the", "shall"])
        tags = ["DET", "NOUN", "PUNCT", "ADP", "DET", "NOUN", "PUNCT", "AUX", "VERB", "PUNCT"]
        self.assertEqual(apply_representation("fw_pos", toks, tags),
                         ["The", "NOUN", ",", "of", "the", "NOUN", ",", "shall", "VERB", "."])
        self.assertEqual(apply_representation("pos_only", toks, tags),
                         ["DET", "NOUN", ",", "ADP", "DET", "NOUN", ",", "AUX", "VERB", "."])

    def test_windows_cover_text(self):
        text = " ".join(f"w{i}" if i % 7 else "the" for i in range(1, 1201)).replace("w", "word")
        ws = make_windows("u", "c", text, "raw", 250)
        self.assertEqual([w.n_words for w in ws], [250, 250, 250, 250, 200])
        self.assertEqual(sum(w.n_words for w in ws), 1200)
        ws = make_windows("u", "c", text, "raw", 500)
        self.assertEqual([w.n_words for w in ws], [500, 500])      # 200-word tail < 250 dropped
        self.assertEqual(len(make_windows("u", "c", text, "raw", None)), 1)


class TestKneserNey(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        rng = random.Random(0)
        vocab = list("abcdefg")
        seqs = [[rng.choice(vocab) for _ in range(rng.randint(3, 15))] for _ in range(40)]
        m = _KN(3, 0.75, vocab_size=len(vocab) + 1)
        m.fit(seqs)
        for ctx in [("a", "b"), ("g", "g"), ("<s>", "<s>"), ("zz", "zz")]:
            total = sum(2 ** m.logprob2(w, ctx) for w in vocab + [EOS])
            total += 2 ** m.logprob2("<unseen>", ctx)       # the single unknown bucket
            self.assertAlmostEqual(total, 1.0, places=9)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        make_synthetic_corpus(cls.tmp)
        cls.units = load_corpus(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _windows(self, rep="raw", size=None):
        return {uid: make_windows(uid, u.cls, u.text, rep, size) for uid, u in self.units.items()}

    def _arms(self):
        names = ["delta", "delta_burrows", "cosine_delta", "ngram_kn", "lzma"]
        if importlib.util.find_spec("sklearn"):
            names.append("linear")
        return names

    def test_real_signal_is_found(self):
        task = build_task("hvm", self.units, 1)
        windows = self._windows("fw_x")
        for arm in self._arms():
            with self.subTest(arm=arm):
                res = run_kfold(make_arm_factory(arm), task.pool, windows, task.classes, k=5,
                                extras=task.extras)
                s = summarize(res, task.classes, n_boot=200)
                self.assertGreater(s["auc_pool"], 0.9, arm)
                self.assertEqual(s["n_all"], 65)

    def test_content_only_has_no_signal(self):
        # topic words are drawn identically for both classes in the synthetic corpus,
        # so content-only must be near chance (a check that masking really removes the FW signal)
        task = build_task("hvm", self.units, 1)
        res = run_kfold(make_arm_factory("delta"), task.pool, self._windows("content_only"),
                        task.classes, k=5)
        self.assertLess(summarize(res, task.classes, n_boot=0)["auc_pool"], 0.8)

    def test_null_split_is_unbiased_with_symmetric_folds(self):
        # pooled over many random Hamilton splits, k-fold AUC should centre on 0.5
        windows = self._windows("raw")
        aucs = []
        for seed in range(1, 21):
            task = build_task("null_ham", self.units, seed)
            res = run_kfold(make_arm_factory("delta"), task.pool, windows, task.classes, k=5)
            aucs.append(metrics.auc(*unit_auc(res, task.classes)))
        self.assertLess(abs(sum(aucs) / len(aucs) - 0.5), 0.1)

    def test_permutations_and_loo_run(self):
        task = build_task("hvm", self.units, 2)
        windows = self._windows("fw_only", 300)
        f = make_arm_factory("delta")
        null = run_permutations(f, task.pool, windows, task.classes, 5, 0, range(5))
        self.assertEqual(len(null), 5)
        self.assertTrue(all(0 <= v <= 1 for _, v in null))
        res = run_loo(f, task.pool, windows, task.classes, symmetric=True, repeats=3)
        self.assertEqual(len(res), 28)
        self.assertFalse(math.isnan(summarize(res, task.classes, n_boot=0)["auc_pool"]))

    def test_fixed_tasks(self):
        windows = self._windows("raw")
        for name in ("topic_shift", "disputed", "early_late", "hvj", "mvj", "null_mad"):
            with self.subTest(task=name):
                build_task(name, self.units, 1)   # builds without error on the synthetic numbering
        task = build_task("disputed", self.units, 1)
        from stylo.evaluate import run_fixed
        res = run_fixed(make_arm_factory("delta"), task.pool, task.extras, windows, task.classes)
        self.assertEqual(len(res), 12)
        self.assertGreaterEqual(sum(r.predicted == "madison" for r in res), 10)


if __name__ == "__main__":
    unittest.main()
