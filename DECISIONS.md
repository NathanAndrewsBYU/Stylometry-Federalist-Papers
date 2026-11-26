# Decisions log

Choices that could affect how results are interpreted. Newest first. Each entry says who decided and why.

**D-011 Harness implementation choices.** (Claude, proposed, Nathan approved)
- Arm B2 uses LZMA compression cross-entropy instead of a true PPM model. A pure-Python PPM is too slow for 1,000 permutations; LZMA's 8 MB dictionary holds a whole training corpus. A PPM can be swapped in behind the same interface later.
- Permutation tests use the stratified 5-fold design only. Symmetric leave-one-out (20 repeats) is reported for the observed data but not permuted (it would need about 560,000 model fits per configuration).
- Delta's default variant is the project's per-class version (each author's own mean and SD), for continuity with the committed CSVs. Classic Burrows (pooled SD) and Cosine Delta are separate arms.
- Kneser–Ney arm: order 3, fixed discount 0.75, lowercased tokens, uniform base over the shared training vocabulary.
- Linear arm: L2 logistic regression (C = 1) on character 2–4-gram TF-IDF (top 20,000) plus function-word relative frequencies.
- Delta uses fast floating-point mean/SD by default; the exact statistics-module path is kept for the regression check only.
- The committed `traditional_burrows_delta_balanced.csv` came from the Colab notebook (scores all 65 papers), not from `cluster/traditional_stylometry.py` (scores only the 28 pool papers). The regression check targets the notebook logic.

2026-11-26
D-010 Legacy GPT-2 null test is biased; skip its 80-epoch run. (Nathan) GPT-2 null-test smoke run (Slurm job 15588902, legacy run_experiment.py --mode null-test, 5 epochs, split seed 1; cluster/results/null_smoke/null_test_summary.csv on the cluster): 5/28 = 17.9%. Exact lower-tail binomial P(X ≤ 5 | n = 28, p = 0.5) ≈ 0.0005, so this is systematic, not noise. This is probably because in the legacy balanced LOO, the held-out paper's own-class model is retrained on 13 papers while the other class's model keeps all 14. More training text lowers perplexity on any text, so the comparison is tilted toward the other class. This is the same asymmetry D-006 fixes, and it is stronger for GPT-2 than for Delta (40.7%). It appears that GPT-2 is extremely sensitive to corpus size. Consequences:

Don't run the legacy 80-epoch null test (about 3 × 75 GPU-minutes for a biased answer). The GPT-2 null test runs in the new harness with symmetric stratified 5-fold (D-005, D-006).
The existing balanced H-vs-M GPT-2 results (98.5%) had the same tilt against the correct answer for the 28 papers in the balanced pool (14 Madison + 14 sampled Hamilton). The 37 Hamilton papers outside the pool were scored against two 14-paper models with no tilt. So the bias worked against the reported accuracy, not for it. Still re-measured under the new design.
Caveat: one split at 5 epochs only. The size of the tilt at 80 epochs is unknown until the harness runs.

## 2026-11-20

**D-009 Gate 1 thresholds approved as written** in `prereg/phase_a_plan.md` §7. (Nathan)
Cheap arms: masked AUC ≥ 0.90 with permutation p < 0.01 per balance seed; within-author null inside the central 95% of its permutation distribution (median 0.45–0.55); Hamilton early-vs-late AUC at least 0.15 below Hamilton-vs-Madison (flag if ≥ 0.80); masked beats raw and content-only on the topic-shift test; ≥ 10/12 disputed papers to Madison or discrepancies explained. GPT-2: AUC above the maximum of 20 permutations with z_perm ≥ 4; same null and early-vs-late checks; paired-bootstrap ΔAUC versus the best cheap arm with 95% CI above 0 and ΔAUC ≥ 0.05 (or, at the ceiling, better at ≤ 10k training words per class); same disputed check. These must not be changed after Phase A results are seen without a logged reason.

**D-008 GPT-2 screening epochs: 80.** (Nathan)
Fixed for all Phase A GPT-2 runs; no epoch tuning. Because 80 was picked on earlier LOO scores from the same corpus, GPT-2's Phase A results carry more selection optimism than the cheap arms' results, and reports must say so. Estimated cost: about 11 GPU-hours per representation (27 runs × 10 trainings × about 2.5 min on an H200), about 34 GPU-hours for representations a, b and e.

**D-007 English text choice deferred.** (Nathan)
Which English translation(s), and whether to include the dssenglishbible.com scroll-based text, is decided later, before Phase B data setup. Candidate list and lineage groups are in the session notes and `prereg/phase_a_plan.md` §9.

**D-006 Null-test bias fix: symmetric hold-out, permutation reference.** (Nathan, on Claude's proposal)
Leave-one-out pushes true-null accuracy below 50% (the held-out paper's own group shrinks, so its profile shifts away from it; real Delta null test 2026-11-20: 57/140 = 40.7% pooled). Fixes:
- (b) Symmetric hold-out: whenever a unit is held out of one class, one unit is also removed from the other class's training set, so both classes train on equal counts. Stratified grouped k-fold gives this automatically.
- (a) P-values come from the permutation distribution of the same procedure, not from a fixed 50% / AUC 0.5 reference.

**D-005 GPT-2 evaluation budget.** (Nathan)
- Full label-permutation tests (1,000+) only for the cheap CPU arms (Delta, n-gram/PPM, linear).
- GPT-2 uses stratified grouped k-fold instead of leave-one-out (about a third of the trainings per run) and about 20 permutations.
- With 20 permutations the smallest reachable p-value is 1/21 ≈ 0.048, so GPT-2 cannot meet a p < 0.01 gate. It is judged by effect size against the permutation spread instead, and must clearly beat the cheap models to justify its GPU cost.

**D-004 Plan amendments accepted.** (Nathan, on Claude's review of `cluster/PROJECT.md`)
- GPT-2's pretraining data (WebText) very likely contains the Federalist Papers and the KJV (and possibly WEB). Treat as contamination risk in Phases A, B and D, not only D.
- 1QIsaᵃ's scribal change at about Isa 33/34 coincides with the group A (1–33) / group C (34–35) boundary; flag any Hebrew comparison crossing it.
- Isaiah groups B (36–39) and C (34–35) are too small to train on; they are only scored against models trained on larger groups.
- A strong content-only (topic-only) result is a warning to investigate, not automatic proof of a bug: some content words are genuine style markers.
- POS-tag masking needs archaic-English normalization first (thee/thou/-eth/hath); rules to be documented.
- Jeremiah 52 (parallels 2 Kings 24:18–25:30) is excluded or flagged in Phase B, as Isa 36–39 is in Phase D.
- Keep a copy of BYU's `BYU_ORC_AGENTS.md` at repo top level, referenced from `CLAUDE.md`, refreshed when older than 7 days.
- Test whether `--gpus=h200:1` alone (no pinned partition) avoids P100 nodes; drop the partition pin if so.
- Model list: drop stretch arm G (from-scratch neural model); keep H (ensemble).

**D-003 Existing code is frozen as the legacy record.** (Claude, proposed; Nathan accepted)
`cluster/` stays as-is as the record of all runs so far. New harness code goes in `src/stylo/`.

**D-002 Cluster code synced; null-test and masking code committed** (`709ed24`). Delta null test run on the real corpus: per split 53.6, 35.7, 28.6, 39.3, 46.4%; pooled 57/140 = 40.7%.

**D-001 Huang, Murakami & Grieve (2025) scope checked.**
They tested four large, modern, English, untranslated corpora (Blogs50, CCAT50, Guardian, IMDB62; 13–62 authors; Guardian smallest at about 34 texts per author), GPT-2 base fine-tuned 100 epochs, 5 random 8:2 splits. No translated, ancient, or very-small-corpus experiments. They found content words carry more authorship information than function words, and that n-gram / PPM baselines matched or beat them on the longer-text datasets (CCAT50, Guardian). So the citation supports the method in general, but not its behavior on small, translated, ancient, genre-mixed text — that is what Phases A–B must show.
