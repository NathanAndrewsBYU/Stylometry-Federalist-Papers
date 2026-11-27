# Phase A plan: Federalist Papers screen

**Date:** 2026-11-26
**Governing documents:** `docs/project_plan.md` (project plan), `docs/DECISION-LOG.md` (D-001 to D-007), BYU ORC agent policy (`BYU_ORC_AGENTS.md`, wins over everything here).

Phase A **screens** models. It does not select the final model (that happens in Phase B under a locked protocol). Its job is to find out which model × representation combinations detect authorial style rather than topic, series position or chance, and to drop the rest.

---

## 1. Corpus and units

**Corpus:** Project Gutenberg ebook #1404, split by `legacy/cluster/prepare_corpus.py` into 85 files. Attributions follow Adair (1944) and Mosteller & Wallace (1963):

| Class | n | Papers |
|---|---|---|
| Hamilton | 51 | 1, 6–9, 11–13, 15–17, 21–36, 59–61, 65–85 |
| Madison | 14 | 10, 14, 37–48 |
| Disputed | 12 | 49–58, 62, 63 |
| Jay | 5 | 2–5, 64 |
| Joint (H & M) | 3 | 18–20 |

**To verify before running:** word counts per paper and per author (needed for the learning curves and Isaiah-size matching).

**Units.**
- The unit of **evaluation and resampling** is the paper. All splits are grouped by paper: windows from one paper never appear in both training and test.
- Windows are non-overlapping, fixed-length word windows inside a paper: 250, 500, 1000 and 2000 words, plus the whole paper. A paper's trailing remainder shorter than half a window is dropped; record how many words that discards.
- The **primary decision** is per paper: the mean of its windows' decision values. Window-level results are secondary (used for learning curves). This avoids treating many windows from one paper as independent trials.

**Topic structure (important for interpreting everything below).**
Approximate topic blocks, from the outline in Fed 1. To be verified against the text before use.

| Papers | Topic | Hamilton | Madison | Other |
|---|---|---|---|---|
| 2–14 | Utility of the Union, faction | 6–9, 11–13 | 10, 14 | Jay 2–5 |
| 15–22 | Defects of the Confederation | 15–17, 21–22 | — | Joint 18–20 |
| 23–36 | Energetic government, taxation | 23–36 | — | — |
| 37–51 | Republican principles, powers, separation of powers | — | 37–48 | Disputed 49–51 |
| 52–61 | House of Representatives, elections | 59–61 | — | Disputed 52–58 |
| 62–66 | Senate, treaties, impeachment | 65–66 | — | Disputed 62–63; Jay 64 |
| 67–85 | Executive, judiciary, conclusion | 67–85 | — | — |

Consequences:
- 12 of Madison's 14 known papers (37–48) are in a block where Hamilton wrote nothing. **Author and topic are heavily confounded in the standard Hamilton-vs-Madison comparison.** Only block 2–14 has both authors.
- Hamilton's early papers (1–36) and late papers (59–85) differ in topic **and** date. A Hamilton early-vs-late split is therefore the most useful "same author, different topic and time" control available here.
- The disputed papers are topically mixed. 49–51 continue Madison's separation-of-powers run (47–48), so topic favors Madison there. 52–58 and 62–63 share topics with Hamilton's 59–61 and 65–66. Agreement with Mosteller–Wallace is therefore a weaker check for 49–51 than for the rest.

---

## 2. Representations

Built from the same cleaned text (`clean_text`: remove `[...]` editorial brackets and trailing `PUBLIUS`, collapse whitespace).

| Code | Representation | How |
|---|---|---|
| (a) | Raw text | Unmasked control |
| (b) | Function words + POS tags | Function words and punctuation kept verbatim; every other word replaced by its Universal POS tag (`NOUN`, `VERB`, `ADJ`, `ADV`, `PROPN`, `NUM`, …) |
| (c) | POS sequence only | Every token replaced by its POS tag; punctuation kept |
| (d) | Function-word frequency vector | Relative frequencies of the fixed function-word list (bag of words, no order) |
| (e) | Content words only | Content words kept; function words replaced by a placeholder (`mask_corpus.py --invert` logic) |

- **Function-word list:** the fixed 253-word list already in `legacy/cluster/mask_corpus.py`. It is fixed rather than "top-K frequent" so that topical words such as *state*, *government* and *power* can never count as function words. Version-pin and hash it.
- **POS tagger:** spaCy (`en_core_web_sm`, pinned version), Universal POS tags. Federalist English is 18th-century but not KJV-archaic; tagger accuracy is spot-checked on 200 hand-checked tokens and reported. (The archaic-normalization rules for the KJV/ASV are a Phase B task.)
- **Existing `X`-masked corpus** (`mask_corpus.py`): kept as a sensitivity variant of (b), since all earlier masking commands use it.
- **Ablations:** punctuation kept vs removed; case kept vs lowercased; Delta with 50 / 100 / 200 / 500 most frequent words.

---

## 3. Candidate models (arms)

Every arm implements the same interface (§5) and runs on identical units, splits and representations.

| Arm | Model | Representations | Compute |
|---|---|---|---|
| A | Burrows' Delta and Cosine Delta (most-frequent-word z-scores) | a, b, c, d, e | CPU, seconds |
| B1 | Smoothed word/tag n-gram language model (modified Kneser–Ney, n = 3–4), one per class; score = per-token cross-entropy | a, b, c, e | CPU, seconds–minutes |
| B2 | Character n-gram / PPM-style compression model, one per class | a, b, c, e | CPU, minutes |
| C | Linear classifier (L2 logistic regression, also linear SVM) on character 2–4-grams + function-word frequencies | a, b, c, d, e | CPU, seconds |
| D | Existing fine-tuned GPT-2 small (`legacy/cluster/stylometry_lib.py`), unchanged except for generalized labels and the fold logic | a, b, e (c optional) | GPU |
| E | Ensemble of the best complementary arms (stacked on out-of-fold scores) | — | CPU |
---

## 4. Validation design

### 4.1 Balanced classes
For Hamilton vs Madison, Hamilton is subsampled to 14 papers with a reproducible `balance_seed` (seeds 1–5). Every test below is repeated per subsample, and results are reported per seed and pooled.

### 4.2 Cross-validation (the same folds for every arm)
- **Stratified grouped 5-fold.** Each fold holds out about 3 papers per class, so each class trains on about 11 papers. Folds are stratified by class, so both classes always lose the same number of papers. This is the **symmetric hold-out** (D-006): it removes the leave-one-out bias that pushed the real Delta null result down to 40.7%.
- **Cheap arms (A, B, C)** also report leave-one-paper-out **with** symmetric hold-out: when a paper is held out of one class, one randomly chosen paper is also dropped from the other class's training set, with 20 repeats of that random choice. This is for comparison with the existing LOO results.
- **GPT-2 (D)** uses only the 5-fold design (D-005). Each fold retrains both class models, so 5 folds = 10 trainings per run, against 30 for balanced LOO.
- **Fold assignment** uses a fixed `fold_seed`. Arms are compared on identical folds, which allows paired comparisons.

### 4.3 Statistics
- **Primary metric:** paper-level AUC of the decision value `score[A] − score[B]`. Accuracy is also reported (threshold 0), for continuity with the earlier results.
- **Permutation null:** class labels are shuffled **at the paper level** (keeping 14/14), then the whole CV procedure is rerun. p = (b + 1) / (N + 1), where b = number of permutations with AUC ≥ observed.
  - Cheap arms: N = 1,000 per (arm × representation × balance seed).
  - GPT-2: N = 20 per (representation), balance seed 1 only. The smallest reachable p is 1/21 ≈ 0.048.
- **Confidence intervals:** bootstrap resampling **papers** (2,000 resamples), never windows.
- **Effect size against the null:** z_perm = (AUC_obs − mean of permutation AUCs) / SD of permutation AUCs. Also report AUC_obs relative to the maximum permutation AUC.

---

## 5. Harness interface (`src/stylo/`)

```python
@dataclass(frozen=True)
class Window:
    text: str                 # already in the chosen representation
    unit_id: str              # e.g. "federalist_10" (grouping key)
    cls: str                  # class label, e.g. "madison"
    start_word: int
    n_words: int
    source: str               # corpus + edition, e.g. "gutenberg-1404"

class AuthorModel(Protocol):
    name: str
    def fit(self, train: dict[str, list[Window]]) -> None: ...
    def score(self, window: Window) -> dict[str, float]: ...   # lower = closer to that class
```

- `score` returns one value per class. Perplexity, cross-entropy and Delta are all naturally per class; a classifier returns −log p(class).
- The evaluation code (splits, permutations, metrics, bootstrap) never sees model internals. Adding an arm means adding one file in `src/stylo/models/`.
- **The GPT-2 arm wraps `stylometry_lib`.** It changes only label handling (no hardcoded `hamilton`/`madison`) and the fold loop. `legacy/cluster/` stays untouched as the legacy record.
- **Every output** writes `manifest.json`: git commit, config hash, Slurm job ID (or `local`), corpus source and hash, function-word list hash, seeds, package versions.
- **Regression check before anything new:** the harness Delta arm in LOO mode (no symmetric hold-out) must reproduce `results/legacy/traditional_burrows_delta_balanced.csv` exactly.

---

## 6. Tests

1. **Hamilton vs Madison** (the real question): all arms × representations × balance seeds 1–5, 5-fold and (cheap arms) symmetric LOO.
2. **Label-permutation null:** as in §4.3.
3. **Within-author random-split null:** 28 Hamilton papers split at random into 14 vs 14, under the same CV. Cheap arms: 100 splits; GPT-2: 5 splits. Expect AUC near 0.5, judged against that test's own permutation spread. Also Madison 7 vs 7 (cheap arms only; small).
4. **Hamilton early vs late** (topic- and time-shifted, same author): papers 1–36 vs 59–85, balanced to 14 vs 14. A representation that separates these well is keying on topic or drift. Cheap arms and GPT-2.
5. **Topic-shift test:** train on all balanced papers outside block 2–14, then test on the block 2–14 papers (Hamilton 6–9, 11–13; Madison 10, 14). Madison's training set is then only constitutional-structure papers, so a model must generalize across topic to get Fed 10 and 14 right. Compare the degradation of (a) raw vs (b)/(c)/(d) masked vs (e) content-only. Small n (9 papers), so it is reported descriptively with exact per-paper results.
6. **Learning curves:** training words per class of 2.5k, 5k, 10k, 20k, and all available (verify that "all" covers this), window sizes as in §1. Cheap arms on all representations; GPT-2 on the best masked representation only, at 3 sizes. Fit a power law to AUC against training size and extrapolate to Isaiah group sizes (to be measured in Phase B).
7. **Ablations:** §2.

---

## 7. Gate 1 (thresholds approved by Nathan 2026-11-20, D-009)

A model × representation **advances to Phase B** if all of the following hold.

**Cheap arms (A, B, C, H):**
1. On a **masked** representation (b, c or d): pooled paper-level AUC ≥ 0.90 for Hamilton vs Madison (5-fold, balance seeds 1–5), with permutation p < 0.01 for every balance seed.
2. Within-author random-split null: the observed AUC lies inside the central 95% of that test's permutation distribution, and the median over the 100 splits is within 0.45–0.55.
3. Hamilton early vs late: AUC clearly below the Hamilton-vs-Madison AUC for the same arm and representation (proposed: lower by ≥ 0.15). If it is high in absolute terms (say ≥ 0.80), the arm is flagged as drift- or topic-sensitive, even if it passes the rest.
4. Topic-shift test: the masked representation loses less than raw text, and does better than content-only (e).
5. Disputed papers: at least 10 of 12 attributed to Madison, or every discrepancy explained (for example by the topic caveat).

**GPT-2 (D)** (D-005: p < 0.01 is unreachable with 20 permutations):
1. On at least one masked representation: AUC above the maximum of its 20 permutation AUCs, with z_perm ≥ 4 (proposed).
2. Within-author random-split null and Hamilton early-vs-late: as for the cheap arms (5 splits and 1 run, judged against the permutation spread).
3. **Earns its GPU cost:** on the same folds and representation, paper-level ΔAUC (GPT-2 minus the best cheap arm) has a paired-bootstrap 95% CI entirely above 0 **and** ΔAUC ≥ 0.05 (proposed). If both are at the AUC ceiling (≥ 0.98), the comparison moves to the learning curve: GPT-2 must be better at 10k training words or fewer per class.
4. Disputed papers: as for the cheap arms.

**If no arm passes, stop and report. Do not start Phase B.**

---

## 8. Compute estimates and requests (to approve before submitting)

| Work | Where | Estimate |
|---|---|---|
| Cheap arms, all tests, 1,000 permutations | Slurm CPU job array | To be measured on a 10-permutation pilot. Expected: well under 100 core-hours in total |
| GPT-2, per representation: 1 real + 20 permutations + 5 within-author splits + 1 early-vs-late = 27 runs × 10 trainings | Slurm GPU | At 30 epochs: about 27 × 10 × 0.9 min ≈ 4 GPU-hours. At 80 epochs: about 27 × 10 × 2.5 min ≈ 11 GPU-hours |
| GPT-2, 3 representations (a, b, e) | Slurm GPU | About 12 GPU-hours (30 epochs) or about 34 GPU-hours (80 epochs) |
| GPT-2 learning curve (one representation, 3 sizes, 1 seed) | Slurm GPU | 2–6 GPU-hours |

- The per-training times come from the measured 28.4 min for 30 trainings at 30 epochs, and an estimated 75 min for 30 trainings at 80 epochs, on one H200 (Isaiah handoff, §4). Fold trainings use slightly fewer papers, so these are upper-end estimates.
- Each GPT-2 batch is a separate submission that I approve. No job arrays without approval.
- Test whether `--gpus=h200:1` without a pinned partition avoids P100 nodes; if so, drop the partition pin (D-004). Consider L40S (compute capability 8.9) as fairer use for a 124M model.

---

## 9. Open problems
3. **Topic blocks** in §1: fine to verify them against Fed 1's outline and the paper headings, or is a published topic classification better?
4. **Phase B locked split** (decide before Phase B, not now): a locked set of about 20% of chapters, stratified by genre, plus nested CV on the rest or each outer CV fold used as the locked set.

## 10. Order of work after approval

1. `src/stylo/` skeleton, Window/units, representations; regression check against the existing Delta CSV (PC or login node, seconds).
2. Arms A, B1, B2, C; all tests at 10 permutations as a pilot (PC or login node, minutes). Measure runtime and use it to size the Slurm request.
3. Full cheap-arm runs (Slurm CPU, after approval).
4. GPT-2 wrapper; 2-fold, 2-epoch smoke test (Slurm GPU, minutes); then the representation runs, one approved batch at a time.
5. Gate 1 report in `reports/gate1.md`, with every number traced to a results CSV.
