"""Arm C: L2-regularized logistic regression on character 2-4-grams plus
function-word relative frequencies. Needs scikit-learn (imported lazily).

Score for class c = -log P(c | window), so lower = closer, like the other arms.
"""

from __future__ import annotations

import math

from ..representations import Window
from ..text import load_function_words


class LinearArm:
    def __init__(self, C: float = 1.0, ngram_range=(2, 4), max_features: int = 20000, seed: int = 0):
        self.name = "linear"
        self.C = C
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.seed = seed

    def fit(self, train: dict[str, list[Window]]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        self.classes = sorted(train)
        texts, y = [], []
        for c in self.classes:
            for w in train[c]:
                texts.append(w.text)
                y.append(c)
        self.vec = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range, sublinear_tf=True,
                                     min_df=2, max_features=self.max_features, lowercase=True)),
            ("fw", TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z']+", lowercase=True,
                                   vocabulary=sorted(load_function_words()), use_idf=False, norm="l1")),
        ])
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed)
        self.clf.fit(X, y)

    def score(self, window: Window) -> dict[str, float]:
        proba = self.clf.predict_proba(self.vec.transform([window.text]))[0]
        return {c: -math.log(max(p, 1e-12)) for c, p in zip(self.clf.classes_, proba)}
