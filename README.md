# Stylometry-Federalist-Papers
This is a project to test how effective LLMs are in stylometrically determining authorship of disputed texts, using the federalist papers as a test case. I have found that the LLM perplexity method for determining authorship is quite accurate and sometimes more accurate than traditional methods, but there are some serious caveats.

# Results
All rows use leave-one-out validation on the Federalist Papers. "Balanced" means Hamilton's training set is subsampled to match Madison's size, while every Hamilton paper is still evaluated. Hamilton and Madison columns show correct/total.

## Summary

| Test pool | GPT-2 perplexity | Burrows' Delta (100 words) |
|---|---|---|
| 65 known papers, unbalanced (51 vs 14) | 84.6% (epoch 30) | 89.2% |
| 65 known papers, balanced, mean over seeds 1–5 | 98.5% (epoch 80), 97.9% (epoch 100) | 92.3% |
| 77 papers (disputed folded into Madison), balanced, mean over seeds 1–4 | 99.7% (epoch 30) | 97.7% |

## 1. Known-authorship papers (n = 65)

| Method | Epochs | Balance seed | Accuracy | Hamilton correct | Madison correct | Misclassified papers |
|---|---|---|---|---|---|---|
| GPT-2 perplexity, unbalanced | 30 | n/a | 84.6% | 51/51 | 4/14 | 10, 14, 37, 38, 39, 42, 43, 44, 46, 47 |
| Burrows' Delta, unbalanced | n/a | n/a | 89.2% | 51/51 | 7/14 | 10, 14, 37, 38, 39, 44, 46 |
| GPT-2 perplexity, balanced | 5 | 42 | 90.8% | 45/51 | 14/14 | 9, 32, 33, 59, 68, 82 |
| GPT-2 perplexity, balanced | 10 | 42 | 89.2% | 44/51 | 14/14 | 9, 59, 60, 61, 65, 82, 84 |
| GPT-2 perplexity, balanced | 20 | 42 | 84.6% | 41/51 | 14/14 | 9, 33, 59, 66, 69, 70, 78, 82, 83, 84 |
| GPT-2 perplexity, balanced | 30 | 42 | 89.2% | 44/51 | 14/14 | 9, 59, 60, 61, 65, 66, 85 |
| GPT-2 perplexity, balanced | 40 | 42 | 93.8% | 47/51 | 14/14 | 9, 59, 60, 65 |
| GPT-2 perplexity, balanced | 50 | 42 | 95.4% | 48/51 | 14/14 | 9, 59, 69 |
| GPT-2 perplexity, balanced | 60 | 42 | 93.8% | 47/51 | 14/14 | 9, 59, 65, 66 |
| GPT-2 perplexity, balanced | 80 | 42 | 98.5% | 50/51 | 14/14 | 9 |
| GPT-2 perplexity, balanced | 100 | 42 | 98.5% | 50/51 | 14/14 | 9 |
| GPT-2 perplexity, balanced | 80 | 1 | 96.9% | 49/51 | 14/14 | 9, 60 |
| GPT-2 perplexity, balanced | 80 | 2 | 98.5% | 51/51 | 13/14 | 44 |
| GPT-2 perplexity, balanced | 80 | 3 | 98.5% | 50/51 | 14/14 | 9 |
| GPT-2 perplexity, balanced | 80 | 4 | 100% | 51/51 | 14/14 | none |
| GPT-2 perplexity, balanced | 80 | 5 | 98.5% | 50/51 | 14/14 | 65 |
| GPT-2 perplexity, balanced | 100 | 1 | 96.9% | 49/51 | 14/14 | 9, 60 |
| GPT-2 perplexity, balanced | 100 | 2 | 98.5% | 51/51 | 13/14 | 46 |
| GPT-2 perplexity, balanced | 100 | 3 | 98.5% | 50/51 | 14/14 | 9 |
| GPT-2 perplexity, balanced | 100 | 4 | 100% | 51/51 | 14/14 | none |
| GPT-2 perplexity, balanced | 100 | 5 | 95.4% | 49/51 | 13/14 | 9, 46, 60 |
| Burrows' Delta, balanced | n/a | 1 | 93.8% | 49/51 | 12/14 | 14, 23, 38, 80 |
| Burrows' Delta, balanced | n/a | 2 | 93.8% | 51/51 | 10/14 | 10, 14, 38, 44 |
| Burrows' Delta, balanced | n/a | 3 | 92.3% | 50/51 | 10/14 | 10, 14, 38, 47, 80 |
| Burrows' Delta, balanced | n/a | 4 | 90.8% | 45/51 | 14/14 | 23, 28, 33, 80, 82, 84 |
| Burrows' Delta, balanced | n/a | 5 | 90.8% | 51/51 | 8/14 | 10, 14, 38, 39, 44, 46 |

## 2. Disputed papers folded into Madison (n = 77)

The 12 disputed papers are added to Madison's class (26 total) and Hamilton is balanced to 26 for training. All 51 Hamilton and 26 Madison papers are evaluated, so both methods are scored on the same 77 papers. This measures how accurate the system is; it does not establish authorship of the 12.

| Method | Epochs | Balance seed | Accuracy | Disputed (12) correct | Misclassified papers |
|---|---|---|---|---|---|
| GPT-2 perplexity | 30 | 1 | 98.7% | 12/12 | 65 (Hamilton) |
| GPT-2 perplexity | 30 | 2 | 100% | 12/12 | none |
| GPT-2 perplexity | 30 | 3 | 100% | 12/12 | none |
| GPT-2 perplexity | 30 | 4 | 100% | 12/12 | none |
| GPT-2 perplexity | 80 | 1 | 100% | 12/12 | none |
| Burrows' Delta | n/a | 1 | 97.4% | 12/12 | 14, 38 |
| Burrows' Delta | n/a | 2 | 97.4% | 12/12 | 14, 38 |
| Burrows' Delta | n/a | 3 | 96.1% | 12/12 | 14, 38, 80 (Hamilton) |
| Burrows' Delta | n/a | 4 | 100% | 12/12 | none |
| Burrows' Delta | n/a | 5 | 97.4% | 12/12 | 14, 38 |

## 3. Scoring the 12 disputed papers (balanced, 14 vs 14)

The ratio is Hamilton perplexity divided by Madison perplexity. Higher means a stronger Madison call.

| Epochs | Papers assigned to Madison | Mean ratio | Range |
|---|---|---|---|
| 30 | 12/12 | 1.47 | 1.37–1.62 |
| 80 | 12/12 | 1.37 | 1.09–1.60 (weakest: Fed 57 at 1.09, Fed 53 at 1.13) |

full results in /results

## Insights

- **Corpus-size imbalance is the largest effect.** Without balancing, every error for both methods falls on Madison, and GPT-2 gets only 4/14 Madison papers right. Balancing brings GPT-2's Madison accuracy to 14/14 at every epoch value, though its errors move to the Hamilton side.
- **GPT-2 beats Delta once it is trained long enough.** On the 65 known papers, balanced GPT-2 at 80–100 epochs averages about 98%, versus 92.3% for Delta. At epoch 30 the single balanced run (seed 42) scored 89.2%, below Delta's average. Treat 30 epochs as uncertain until multi-seed runs exist for that setting.
- **The two methods make different mistakes.** Delta's recurring errors are Madison papers (Fed 10, 14, 38, 44). GPT-2 at 80+ epochs misses Hamilton papers instead (mostly Fed 9, 60, 65). Because their errors barely overlap, combining them could do better than either alone.
- **Fed 9 may reflect topic leakage.** GPT-2 misses it in every balanced seed-42 run, while Delta never misses it. Fed 9 deals with the same union and faction themes as Madison's Fed 10, so GPT-2 may be picking up topic as well as style, and Delta's function words are less sensitive to topic.
- **More Madison text helps both methods.** Moving from 14 to 26 Madison papers lifted Delta from 92.3% to 97.7%. Delta still misses Fed 14 and Fed 38 as near-ties. The two test pools differ, so this is suggestive, not exact.
- **Both methods put all 12 disputed papers with Madison on every seed.** This agrees with the historical consensus, but the n = 77 setup is partly circular, because the disputed papers are labeled Madison in the test. It shows the two methods agree with each other, not independent proof of authorship.
- **Accuracy depends on which Hamilton papers are sampled.** Epoch 100 ranges from 95.4% to 100% across seeds, so differences of 2–3 points between rows are within that noise.
- **The epoch sweep (5–60) used one subsample.** Its non-monotonic pattern, for example 84.6% at 20 epochs and 93.8% at 40, is not reliable. Only epochs 80 and 100 were tested across five subsamples.
- **Raw perplexity grows with epochs, so use the Hamilton/Madison ratio as the margin.** The disputed-paper margin looks smaller at 80 epochs (mean 1.37 versus 1.47), but it isn't a worse model. The ratio is the only comparison that stays meaningful across epoch settings.


# Project Structure 
   notebooks/   — Colab notebook(s)
   cluster/ code designed to run on the BYU supercomputer cluster rather than on Google Colab
   results/     — CSV output from validation runs

# Limitations
The project has some serious limitations. I have not established the ideal number of training epochs. Epochs 5–60 were each run once on a single Hamilton subsample, so the fine-grained differences between them are mostly noise. The broader trend, that 80–100 epochs clearly outperform 30 or fewer, was tested across five subsamples and is more reliable.

Accuracy appears to improve with more training text (going from 14 to 26 Madison papers raised both methods' scores), but I did not vary corpus size independently, and GPT-2's advantage over Burrows' Delta was larger with the smaller corpus, not the larger one. How small the training corpus can be before accuracy breaks down is untested.

In the n = 77 tests, it's assumed that the 12 disputed papers were written by Madison. This is a legitimate assumption based on textual and historical evidence, but those tests don't establish authorship by Madison because that would be circular.

It's possible that the LLM method is more genre-biased and prone to topic leakage than the Burrows' Delta method. The fact that it consistently misclassifies Federalist 9 seems to point to this. However, I still haven't fully tested this hypothesis.

All tests used the same GPT-2 model size. Using a different model size or a different LLM might produce different results.

The epoch count was chosen after looking at the same leave-one-out scores that are used to report accuracy. I ran the test at several epoch values (5 through 100) and focused on the best-performing range (80–100), so the reported accuracy of roughly 98% is likely somewhat optimistic. The large gap between low-epoch runs (30 or fewer) and high-epoch runs (80–100) held across five random Hamilton subsamples, so I expect that part of the result to be real, but the exact figures and the choice between neighboring epoch values may not carry over to new papers. A cleaner estimate would select the epoch count on separate data (for example, with nested leave-one-out validation) and report accuracy only on papers that played no part in that choice.


# Methodology
The methodology of this project is to train separate GPT-2 models on the Federalist Papers written by James Madison and Alexander Hamilton respectively, and then performing a leave-one-out analysis by leaving out a single paper from one of the authors and then attributing it to whichever model finds it "most predictable", measured as perplexity. I based this methodology off of the paper "Attributing authorship via the perplexity of authorial language models" by Huang, Murakami, and Grieve (2025).

Barlas and Stamatatos (2020) and Jeong and Rockova (2025) found that LLMs underperformed compared to traditional methods, but they used BERT-based cross-entropy classification and embedding methods respectively, not perplexity methods.

Hamilton wrote 51 solo papers and Madison 14. In balanced runs, a random subsample of Hamilton papers (chosen by `balance_seed`) is used for training, matching Madison's size; every Hamilton paper is still evaluated, using the model trained on the full balanced subsample when it was not in the training subsample.

Papers are downloaded from Project Gutenberg and split by author: Hamilton 51, Madison 14, disputed 12, Jay 5, joint 3.

As a baseline, I tested the same corpus under the same conditions with Burrows' Delta, which is a widely used traditional stylometry method. This will help establish a comparison between LLM methods and more traditional methods. For each paper, Burrows' Delta takes the 100 most frequent words in the combined training corpus, computes their per-1000-word frequencies, and builds an author profile (mean and standard deviation of each word's frequency). A held-out paper is assigned to the author with the smaller mean absolute z-score distance (Delta). The same leave-one-out and balancing procedure is used as for GPT-2.

# Instructions
To replicate this experiment on colab, Open notebooks/gpt2_federalist_stylometry.ipynb in Colab, set runtime to GPU, run cells in order. It will import GPT-2 and everything else necessary, and autodownloads the federalist papers from the Gutenberg Project website. Cell 8 has sample_size = 10 and epoch = 10, adjust these as necessary.

To replicate this experiment on a cluster, refer to cluster/README_CLUSTER.md

To replicate the Burrows' Delta experiment, open those notebooks on Colab, set runtime to CPU, and run the cells in order. The different notebooks are balanced n=65 and balanced n=77.

# Next Steps
The next experiment I would like to run would be to reduce the corpus size significantly to see if accuracy still holds up when models are only trained on 5-10 papers. This will be useful to see the limits of the method. 

After that, I will test the method on ancient Hebrew using the books of Jeremiah and Ezekiel, which are large Old Testament books of a similar genre. If the perplexity method can reliably distinguish them, that is a useful base case showing it is a good fit for ancient Hebrew texts and could plausibly be applied to the question of authorship within Isaiah. This comparison has limits: the two books differ in topic as well as author, so a null test within a single book and masking of content words will also be needed.
# References
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081

https://arxiv.org/pdf/2503.01869

https://link.springer.com/chapter/10.1007/978-3-030-49161-1_22
