# Stylometry-Federalist-Papers
This is a project to test how effective LLMs are in stylometrically determining authorship of disputed texts, using the federalist papers as a test case.

# results
Using a google colab notebook, I fine-tuned a separate GPT-2 model on federalist papers written by Alexander Hamilton and those written by James Madison. I then ran a leave one out validation, holding out one known paper at a time, retraining on the rest, and checking whether the method correctly identifies its true author. The model that found the left-out paper most predictable  measured as the perplexity of the questioned document was ruled the winner. I have run this with an n=10 sample of papers and the models correctly predicted the true author every time.

   | Metric | Value |
   |---|---|
   | Sample size | 10 papers |
   | Accuracy | 100% (10/10) |
   | Hamilton-true papers | 9/9 correct |
   | Madison-true papers | 1/1 correct (narrow margin, ~9.5%) |

Full results: results/leave_one_out_n10.csv

# project structure 
   notebooks/   — Colab notebook(s)
   results/     — CSV output from validation runs

# limitations
The project has some serious limitations. Although the models were able to correctly predict the authorship of all 10 papers, only one of them was written by Madison and there was only a small margin in perplexity between the Hamilton model and the Madison model. This is likely because he only wrote 14 papers, excluding jointly authored papers and disputed ones. Hamilton's corpus is 51, which is a significantly larger data set. I have limited computing power since I have the free version of google colab, which means I only trained the models with epoch size=10, which is lower than ideal.

# methodology
I based my methodology off of the paper "Attributing authorship via the perplexity of authorial language models" by Huang, Murakami, and Grieve (2025). Fine-tuning a language model on each candidate author's known writing, then attributing a disputed text to whichever model finds it 'most predictable,' measured as perplexity.

Barlas and Stamatatos (2020) and Jeong and Rockova (2025) found that LLMs underperformed compared to traditional methods, but they used BERT-based cross-entropy classification and embedding methods respectively, not perplexity methods.

# instructions
To replicate this experiment, Open notebooks/gpt2_federalist_stylometry.ipynb in Colab, set runtime to GPU, run cells in order. It will import GPT-2 and everything else necessary, and autodownloads the federalist papers from the Gutenberg Project website. Cell 8 has sample_size = 10 and epoch = 10, adjust these as necessary.

# next steps
I plan to re-run the leave one out test with n=65 all papers reliably attributed to Madison and Hamilton, not just 10. I also plan to test the models on the 12 papers with disputed authorship (although statistical models and historians generally agree that they were written by Madison). After I've run a general test with epoch = 10, I plan to rerun the experiment with varied epoch levels in order to see what the ideal epoch range is to adequately train the LLM while avoiding overfitting. I also eventually plan to directly compare the LLM perplexity methodology with more traditional stylometric methods such as function-word frequency analysis to see the relative strengths of each approach.

# references
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081
https://arxiv.org/pdf/2503.01869
https://link.springer.com/chapter/10.1007/978-3-030-49161-1_22
