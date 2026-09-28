# Stylometry-Federalist-Papers
This is a project to test how effective LLMs are in stylometrically determining authorship of disputed texts, using the federalist papers as a test case.

# Results
Using a Google Colab notebook, I fine-tuned a separate GPT-2 model on federalist papers written by Alexander Hamilton and those written by James Madison. I then ran a leave one out validation, holding out one known paper at a time, retraining on the rest, and checking whether the method correctly identifies its true author. The model that found the left-out paper most predictable measured as the perplexity of the questioned document was ruled the winner. I ran this on Google Colab with a sample size of 10 papers and it performed well. I then ran it on the BYU supercomputer cluster with all 65 papers and found that it consistently identified Hamilton papers but struggled with Madison papers. This is likely because of a difference in corpus size since there are more than three times as many Hamilton papers.

   | Metric | Value |
   |---|---|
   | Sample size | 65 papers |
   | Accuracy | 84.6% (55/65) |
   | Hamilton-true papers | 51/51 correct |
   | Madison-true papers | 4/14 correct |

Full results: results/leave_one_out_full_epochs30.csv


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
