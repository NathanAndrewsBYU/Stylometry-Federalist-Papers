# Stylometry-Federalist-Papers
This is a project to test how effective LLMs are in stylometrically determining authorship of disputed texts, using the federalist papers as a test case.

# results now 
Using a google colab notebook, I fine-tuned a separate GPT-2 model on federalist papers written by Alexander Hamilton and those written by James Madison. I then ran a leave one out validation, holding out one known paper at a time, retraining on the rest, and checking whether the method correctly identifies its true author. The model that found the left-out paper most predictable  measured as the perplexity of the questioned document was ruled the winner. I have run this with an n=10 sample of papers and the models correctly predicted the true author every time.

# limitations
The project has some serious limitations. Although the models were able to correctly predict the authorship of all 10 papers, only one of them was written by Madison and there was only a small margin in perplexity between the Hamilton model and the Madison model. This is likely because he only wrote 14 papers, excluding jointly authored papers and disputed ones. Hamilton's corpus is 51, which is a significantly larger data set. I have limited computing power since I have the free version of google colab, which means I only trained the models with epoch size=10, which is lower than ideal.

# methodology
I based my methodology off of the paper "Attributing authorship via the perplexity of authorial language models" by Huang, Murakami, et al. published July of 2025. Fine-tuning a language model on each candidate author's known writing, then attributing a disputed text to whichever model finds it 'most predictable,' measured as perplexity

# instructions
To replicate this experiment, Open notebooks/gpt2_federalist_stylometry.ipynb in Colab, set runtime to GPU, run cells in order.

# next steps
I plan to re-run the leave one out test with n=65 all papers reliably attributed to Madison and Hamilton, not just 10. I also plan to test the models on the 12 papers with disputed authorship (although almost everyone now agrees they were written by Madison)

# references
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081
