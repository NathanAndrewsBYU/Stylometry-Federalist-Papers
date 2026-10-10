# Stylometry-Federalist-Papers
This is a project to test how effective LLMs are in stylometrically determining authorship of disputed texts, using the federalist papers as a test case.

# Results
Using a Google Colab notebook, I fine-tuned a separate GPT-2 model on federalist papers written by Alexander Hamilton and those written by James Madison. I then ran a leave one out validation, holding out one known paper at a time, retraining on the rest, and checking whether the method correctly identifies its true author. The model that found the left-out paper most predictable measured as the perplexity of the questioned document was ruled the winner. I ran this on Google Colab with a sample size of 10 papers and it performed well. I then ran it on the BYU supercomputer cluster with all 65 papers and found that it consistently identified Hamilton papers but struggled with Madison papers. This is likely because of a difference in corpus size since there are more than three times as many Hamilton papers.

   # Results

All rows use leave-one-out validation. "Balanced" means Hamilton's training set is subsampled to Madison's size, while all 51 Hamilton papers are still evaluated. Hamilton and Madison columns are correct/total.

## Known-authorship papers (n = 65)

| Method | Epochs | Balance seed | Accuracy | Hamilton correct | Madison correct | Misclassified papers |
|---|---|---|---|---|---|---|
| GPT-2 perplexity, unbalanced (51 vs 14) | 30 | n/a | 84.6% | 51/51 | 4/14 | 10, 14, 37, 38, 39, 42, 43, 44, 46, 47 |
| Burrows' Delta (100 words), unbalanced | n/a | n/a | 89.2% | 51/51 | 7/14 | 10, 14, 37, 38, 39, 44, 46 |
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
| Burrows' Delta, balanced | n/a | 1–5 | *pending* | *pending* | *pending* | *pending* |

## Disputed papers folded into Madison (n = 77)

The 12 disputed papers are added to Madison's class (26 total) and Hamilton is balanced to 26 for training. All 51 Hamilton papers and all 26 Madison papers are evaluated, so both methods are scored on the same 77 papers. This measures how accurate the system is; it does not establish authorship of the 12.

| Method | Epochs | Balance seed | Accuracy | Disputed (12) correct | Misclassified papers |
|---|---|---|---|---|---|
| GPT-2 perplexity | 30 | 1 | 98.7% | 12/12 | 65 (Hamilton) |
| GPT-2 perplexity | 30 | 2 | 100% | 12/12 | none |
| GPT-2 perplexity | 30 | 3 | 100% | 12/12 | none |
| GPT-2 perplexity | 30 | 4 | 100% | 12/12 | none |
| GPT-2 perplexity | 80 | 1 | 100% | 12/12 | none |
| Burrows' Delta (100 words) | n/a | 1 | 97.4% | 12/12 | 14, 38 |
| Burrows' Delta (100 words) | n/a | 2 | 97.4% | 12/12 | 14, 38 |
| Burrows' Delta (100 words) | n/a | 3 | 96.1% | 12/12 | 14, 38, 80 (Hamilton) |
| Burrows' Delta (100 words) | n/a | 4 | 100% | 12/12 | none |
| Burrows' Delta (100 words) | n/a | 5 | 97.4% | 12/12 | 14, 38 |

Mean accuracy on seeds 1–4: GPT-2 (epoch 30) 99.7%, Burrows' Delta 97.7%. Delta over all five seeds: 97.7%.
## Takeaways

- **Corpus-size bias.** With unbalanced training, all errors for both methods fall on the Madison side. Balancing fixed this for GPT-2 (Madison went from 4/14 to 14/14 at every epoch value tested), but the errors shifted to the Hamilton side.
- **Epoch count.** Epochs 5–60 were run on a single subsample and show no clean trend (84.6%–95.4%). Only epochs 80 and 100 were tested across five subsamples. Epoch 80 averages 98.5% and epoch 100 averages 97.9%.
- **Federalist 9.** It is misclassified in every balanced seed-42 run, so it may be a topic-leakage case (it pairs with Madison's Fed 10).
- **Delta comparison.** The 89.2% Delta figure is unbalanced and is comparable only to the 84.6% unbalanced GPT-2 row. The balanced Delta rows will give a like-for-like comparison.


# Project Structure 
   notebooks/   — Colab notebook(s)
   cluster/ code designed to run on the BYU supercomputer cluster rather than on Google Colab
   results/     — CSV output from validation runs

# Limitations
The project has some serious limitations. So far, my model has a serious bias towards the Hamilton model and incorrectly attributes many Madison papers to Hamilton. This is likely because Madison only wrote 14 papers, excluding jointly authored papers and disputed ones. Hamilton's corpus is 51, which is a significantly larger data set. I have also not established the ideal epoch number for these authors.

# Methodology
I based my methodology off of the paper "Attributing authorship via the perplexity of authorial language models" by Huang, Murakami, and Grieve (2025). Fine-tuning a language model on each candidate author's known writing, then attributing a disputed text to whichever model finds it 'most predictable,' measured as perplexity.

Barlas and Stamatatos (2020) and Jeong and Rockova (2025) found that LLMs underperformed compared to traditional methods, but they used BERT-based cross-entropy classification and embedding methods respectively, not perplexity methods.

# Instructions
To replicate this experiment on colab, Open notebooks/gpt2_federalist_stylometry.ipynb in Colab, set runtime to GPU, run cells in order. It will import GPT-2 and everything else necessary, and autodownloads the federalist papers from the Gutenberg Project website. Cell 8 has sample_size = 10 and epoch = 10, adjust these as necessary.

To replicate this experiment on a cluster, refer to cluster/README_CLUSTER.md

# Next Steps
Having now completed the full n=65 leave-one-out validation at epoch=30, my next steps are: (1) run an epoch-sweep to test whether a different epoch count reduces the Hamilton/Madison imbalance and avoids overfitting, (2) test whether balancing corpus size between authors (e.g., subsampling Hamilton's 51 papers down to Madison's ~14) removes the bias, (3) score the 12 disputed papers, and (4) compare this LLM-perplexity method against traditional function-word frequency stylometry.

# References
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081

https://arxiv.org/pdf/2503.01869

https://link.springer.com/chapter/10.1007/978-3-030-49161-1_22
