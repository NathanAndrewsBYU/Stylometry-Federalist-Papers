Project: Do Proto-, Deutero- and Trito-Isaiah differ stylistically?  
 A validated, content-masked perplexity/stylometry approach

 > Plan of record as of 2026-11-26. Supersedes `cluster/PROJECT.md`, which is kept only as history. Approved decisions in `DECISION-LOG.md` (D-001 onward) still apply; where this plan and a decision differ, the decision governs until this plan is updated.

0\. Goal, framing and phases

The goal is one peer-reviewed paper, within about two years, with Isaiah as the headline result. I will be collaborating with a BYU professor.  
Question: does Proto-Isaiah (Isa 1-39) differ stylistically from Deutero-Isaiah (40-55) and a possible Trito-Isaiah (56-66) by more than ordinary same-author variation, after neutralizing topic? A positive result doesn’t prove multiple authors and a null result doesn’t prove one author, but it would provide a valuable data point.

Prior work (for context): Radday (1970) and Kasher (1970) were early quantitative studies; Adams & Rencher (1974, BYU Studies 15.1) is the most-cited computer study, from a context with an interest in single authorship, so treat its conclusions with caution. I found no modern neural/LLM stylometry study of Isaiah specifically. For Paul there is an active but pretty contested literature, including a 2025 critique arguing stylometry on short, genre-mixed corpora yields conflicting results. That’s what this study is trying to do, so it’s worth examining. Huang, Murakami & Grieve (2025, PLOS ONE) demonstrate that the LLM approach to identify authorship of English texts can yield good results, but they didn’t analyze any truly disputed texts. A citation can justify the method; however it does not show how the method behaves on small, translated, ancient, genre-mixed corpora, which is exactly what a paper on Isaiah must do.

Contents  
0\. Overview, contents and phases  
1\. Phase A  
2\. Phase B  
3\. Texts and the English Translation Question  
4\. Phase C  
5\. Phase D  
Phases

\- Phase A: Federalist Papers screen (cheap; existing English pipeline). Build an evaluation harness that works for all models. Run null/permutation tests and content-word masking on my existing GPT-2 model and a small set of alternatives. Drop all the models that leak topic or don’t perform well.  
 Gate 1\.  
\- Phase B: Jeremiah vs. Ezekiel in English, at Isaiah-matched corpus size. One calibration pair plus within-book null splits. Compare the Phase A finalists here, select one model on development data, and confirm it once on a locked set.  
 Gate 2\.

\- Phase C: Pre-registration and review checkpoint. Decisions about the direction of the project will be made here.

1\. Phase A: Federalist Papers screen

1.1 Status of what exists

I already have a Federalist Papers model. It’s not fully described here, but a more full description is on Github. In broad strokes, two GPT-2 models are fine-tuned on the papers of Hamilton and Madison respectively and one paper is left out. The two models are then tested on the left-out paper and the one with the lower perplexity is deemed the winner.

1.2 Why the Federalist Papers are a good validation set

\- Ground truth is unusually good: of 85 papers, 51 are Hamilton's, 14 Madison's, 5 Jay's, 3 joint, and 12 are disputed between Hamilton and Madison (Mosteller & Wallace's 1963/64 analysis attributed the disputed ones to Madison, this is broadly accepted).   
\- The disputed papers share topics with undisputed papers by both candidates, so this is a natural test of whether a method keys on style rather than topic.  
\- A method that cannot recover the Mosteller-Wallace result has no business being applied to Isaiah.

1.3 Tests (with content-word masking, compared with unmasked)

Content-word masking. Replace open-class words (nouns, verbs, adjectives, most adverbs) with their part-of-speech tag (or another neutral class marker) and keep function words and punctuation verbatim. Build and compare:

\- (a) raw text (unmasked) as the control,  
\- (b) function words kept, content words replaced by POS tags,  
\- (c) POS-tag sequence only,  
\- (d) function-word frequency vector,  
\- (e) content words only (function words masked) as a deliberate "topic-only" baseline. This should separate by topic and do badly across topics; if it does not, something is wrong.

Null and control tests:

1\. Label-permutation null: shuffle author labels many times (1,000+), rerun the whole pipeline, and compare real performance to the null distribution.  
2\. Within-author split null: split one author's papers at random into two groups. The method should not separate them (AUC about 0.5 with a CI containing 0.5). Repeat many times. We will also try a chronological split of one author's papers to see how much ordinary drift looks like signal.  
3\. Topic-confound test: build splits where topic is deliberately matched or mismatched and show how much each representation depends on topic. Masked representations should degrade much less than raw text when topic shifts.  
4\. Author pairs: Hamilton vs. Madison (the real question), plus Hamilton vs. Jay and Madison vs. Jay as small-sample checks.  
5\. Window/length sensitivity and learning curves: repeat with fixed-length windows of several sizes, and plot performance vs. training-set size, down to the sizes Isaiah will give (see 2.4).  
6\. Disputed papers: apply the final method to the 12 disputed papers and compare with Mosteller-Wallace. This is just to check that the results match, as these 12 papers are now generally accepted to have been written by Madison  
7\. Ablations: remove feature classes one at a time; vary masking rules (keep/drop punctuation, capitalization).

1.4 Harness and candidate models

I will code a harness such that every candidate implements the same interface and runs on identical windows, splits, masking representations, and permutation tests. Adding a model must not require changing the evaluation code.

Core arms (required; most run in seconds to minutes on CPU):

\- A. Burrows' Delta / Cosine Delta on function-word frequencies (my existing Delta baseline).  
\- B. Smoothed n-gram language models over the masked token sequence (e.g. Kneser-Ney), and character n-gram / compression-based (PPM-style) classifiers. These are also language models, so the perplexity logic carries over, and they are language-agnostic.  
\- C. Linear classifier (logistic regression or SVM) on character n-grams and function-word frequencies.  
\- D. My existing fine-tuned GPT-2 model, exactly as it is, as the incumbent.  
\- E. An ensemble of the best models to pass gate 1\.

This part shouldn’t become too extravagant; it exists to choose the model for Isaiah, not to become a separate project.

 1.5 Gate 1

Gate 1 screens models, it does not select the final one. A model advances to Phase B if:

\- its masked representations separate Hamilton from Madison on held-out windows well above the permutation null (AUC \>= 0.90, permutation p \< 0.01). The p-value may have to be higher (\<\~.05) for GPT-2 or another language model as 100 permutations would require a lot of of compute,  
\- its within-author split null is consistent with chance,  
\- it does clearly better than the content-only baseline (e) when topic is mismatched,  
\- its disputed-paper result is consistent with the established attribution, or the discrepancy is understood.

Anything failing the within-author null is topic- or era-leaking and is rejected or flagged. If none of them pass I’ll have to figure that out but it’s interesting in itself

2\. Phase B: Jeremiah vs. Ezekiel

2.1 Purpose

The purpose is to show that the method works on translated ancient prophetic text at a small corpus size, and select the model for Isaiah from the Phase A finalists. Jeremiah and Ezekiel are different enough in genre mix and voice to be a positive control, but still similar enough to be comparable. Each is somewhat longer than Isaiah and is long enough to split for null controls.

2.2 Potential issues

\- Jeremiah is internally heterogeneous and includes poetic oracles, prose sermons, and biographical narrative. A within-Jeremiah null that mixes these may fail for legitimate reasons. I will use genre-matched and unmatched splits within the text, to compare poetry to poetry, prose to prose, but also poetry to prose. If the genre-matched split is a null result but the unmatched split is a positive result, that’s a problem for Isaiah.  
\- Masoretic Jeremiah and Septuagint Jeremiah have some significant differences, we will have to decide which to use and how to deal with them.  
\- Ezekiel is usually read as stylistically cohesive, but some scholars see editorial layers.   
\- Different translation teams translate different books, so there may be translation effects even within the same Bible translation.

2.3 Tests

1\. Jeremiah vs. Ezekiel with leave-window-out validation, masked vs. unmasked, same representations as Phase A.  
2\. Within-Jeremiah and within-Ezekiel nulls (genre-matched and unmatched).  
3\. Label-permutation null.  
4\. Learning curves.  
5\. Repeat across the translations chosen in section 3 and look at consistency.

2.4 Match the corpus size to Isaiah

Isaiah is not very big. When divided into its (alleged) constituent authors, it’s even smaller. I’ll have to downsample Jeremiah and Ezekiel to the same word counts as the Isaiah groups Proto-Isaiah,  Deutero-Isaiah, and Trito-Isaiah to see how well the models do at that level of corpus size. I will report at full length and Isaiah-matched length.

2.5  Gate 2

I haven’t fully thought through the criteria here so this is subject to change.

\- At Isaiah-matched size, Jeremiah vs. Ezekiel separates well above the permutation null (proposed: AUC \>= 0.85, p \< 0.01) in the masked representations, consistently across all chosen translations.  
\- Genre-matched within-book nulls are consistent with chance.

If none of the models make it through gate 2, I will have to stop and evaluate. It doesn’t make sense to evaluate Isaiah using a model that doesn’t get good results on texts comparable to Isaiah.

2.6 Model selection protocol

The corpus will be split into the development set and the locked set. No model will see the locked set in development.

Split by whole chapter. All windows from one chapter go on the same side, either development or locked, never both. Otherwise the model is tested on text it has partly already seen.

Make every choice on development data only. That covers the model, epochs, window size and masking, tuned with nested cross-validation inside the development set.

How the winner is picked (in order)

1. Has to pass a within-author null (random halves of one author score at chance). This is Pass/fail.  
2. Highest AUC on masked text to avoid topic leakage.  
3. Holds up best with the smallest amount of training text.

Test on the locked set once, after every choice is frozen. The two possibilities are a nested CV or one locked split (about 20% of chapters) plus repeated re-splits of the rest.

3\. Texts and the English-translation question

3.1 Different ancient textual traditions

Almost all modern translations of Isaiah follow the Masoretic text, but the vowels are a medieval addition. The Great Isaiah Scroll in the Dead Sea Scrolls is “non-pointed” and doesn’t contain these vowels. For this reason, we will prefer it as a Hebrew text. However, Nearly all English translations follow the Masoretic text.

3.2 Problems with translation distortion.

I expect distortion to be small. That is a hypothesis to test, not an assumption. It could be larger because:

\- The features that carry stylistic signal in the original are flattened or normalized by English translation  
\- Translation teams differ by book. For each translation, I will check who translated which books, and whether the books I compare were handled by different teams or periods.  
\- Archaic registers (thee/thou, \-eth in KJV and ASV) need explicit normalization rules so they do not distort function-word counts.

Phase B directly tests the risk.

3.3 Which translations, and the independence problem

\- It’s good to use several translations, but many are not independent, so agreement among them is weaker evidence. I will group by lineage and translation philosophy and report by group.  
\-Some translations are copyrighted. They shouldn’t be hard to obtain legitimately. However, it may not be possible to publish them so I will exercise caution with this.  
\- Record chapter/verse for every window; verse numbering may differ between English and Hebrew editions in places.

4\. Phase C: Pre-registration and review

At this point I will have to make a detailed write up of all work and results to this point and push it to Github. I will review the results and decide how to proceed. Assuming any model passes Phase B, we will either move on to Isaiah in English or redo phase B in Hebrew and then move on to Isaiah. Before any Isaiah analysis, I will write down and register the frozen model and settings, window size, Isaiah test groups, main comparison, exclusions and statistics. 

5\. Phase D: Isaiah

5.1 Rules to avoid contamination

Isaiah is analyzed only after pre-registration and review

The model, representation, window size and all settings are the ones frozen after Phase B. Nothing is tuned on Isaiah.

The text is English first, using several translations. I will decide in section 4 whether Hebrew follows.

5.2 Test groups

The 1–39 / 40–66 division is the hypothesis being tested, not ground truth, so Isaiah will be split into smaller groups that can be compared directly. These groups will be decided at a later date.

5.3 Method

Text is divided into fixed-length windows, sized from the Phase A and B learning curves, with every window keeping its chapter and verse.

Windows from the same chapter are never split between training and test.

For each comparison, the frozen model is trained on one group and tested on held-out windows from both groups, scored with AUC.

The permutation test (shuffled group labels) gives each comparison's p-value.

Confidence intervals are bootstrapped by chapter, not by window.

5.4 Controls

Within-group nulls: random halves of the different Isaiah groups should score at chance. This shows how much variation ordinary same-author text produces.

Calibration: Jeremiah vs Ezekiel and their within-book splits, at the same sample sizes, give the scale for "known-different" and "same-book" distances. Isaiah's between-group distances are reported against that scale.

Genre: poetry is compared with poetry wherever possible, because the sections of Isaiah differ in genre as well as in proposed author.

The same pipeline runs on every chosen translation. A result that appears in only one family of translations will be treated as a translator effect.

5.5 What the results can and cannot show

If Isaiah's groups differ much more than its within-group nulls, and as much as the calibration books, that supports a stylistic division, but does not prove separate authors.

If they differ no more than the within-group nulls, that fails to support a division, but does not prove one author.

Results are reported as distances and effect sizes with uncertainty, whatever the outcome.

An analysis in Hebrew may follow depending on the results of Phase C.

