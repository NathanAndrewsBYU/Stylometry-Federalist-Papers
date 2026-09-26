"""
stylometry_lib.py

Shared functions for the Federalist Papers GPT-2 authorship attribution project.
Designed for BYU's Office of Research Computing SLURM cluster (or any similar
SLURM-managed HPC system) — plain Python/PyTorch, no notebook or Colab-specific
code, so it runs identically via `python script.py` or inside a batch job.

EFFICIENCY CHANGES IN THIS VERSION (vs. the original Colab notebook):

1. Mixed precision (fp16=True) in TrainingArguments — real speedup on GPUs with
   Tensor Cores (T4, V100, A100), since it lets matrix multiplications run in
   half precision instead of full FP32.

2. Redundant-retraining fix in leave_one_out_validation: only the model
   belonging to the held-out paper's true author actually needs retraining
   each iteration, since the OTHER author's full corpus hasn't changed. This
   takes leave-one-out validation from 130 trainings (65 papers x 2 models
   each) down to 67 (2 "full" models trained once + 65 "minus-one" retrains).

3. NEW — no disk round-trip for throwaway models: the original notebook wrote
   every fine-tuned model to disk with trainer.save_model(), then immediately
   reloaded it with GPT2LMHeadModel.from_pretrained() before scoring it. For
   the 65 "minus-one" models in leave-one-out validation, this model is only
   ever needed once, immediately, in the same process — so this version keeps
   it in memory and skips the save+reload round trip entirely. GPT-2 small's
   checkpoint is a few hundred MB; avoiding writing and re-reading that from
   disk 65 times is one of the largest savings here, especially on a shared
   cluster filesystem where I/O can be a real bottleneck. Models you actually
   want to keep after the run (the two "full" models, and score_disputed_papers'
   final models) still save to disk via save_to_disk=True.

4. NEW — no temp file round-trip for training text: the original notebook wrote
   cleaned text to a "*_train.txt" file, then had SimpleTextDataset re-read
   that file from disk. This version tokenizes the cleaned text directly in
   memory, skipping that intermediate file write+read entirely.

5. NEW — cleaned file contents are cached in memory (load_and_clean_cached):
   the original re-read and re-regex-cleaned every file from disk on every
   call. Since the same 65 files get read repeatedly across leave-one-out
   iterations, caching the cleaned text after the first read avoids redundant
   disk I/O and redundant regex processing on every subsequent iteration.

6. NEW — torch.backends.cudnn.benchmark = True: lets cuDNN auto-tune its
   algorithm selection for your specific GPU and input sizes, which are
   constant across this workload (fixed block_size) — a small but free win.
"""

import os
import re
import glob
import csv
import random
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt2"  # options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
CORPUS_ROOT = "corpus/federalist"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()  # only meaningful with a GPU present

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # auto-tune conv/matmul algorithms for fixed input sizes

# Author attribution lists (Adair 1944; consensus reflected in Mosteller & Wallace 1963)
HAMILTON_SOLO = set([1] + list(range(6, 10)) + list(range(11, 14)) + list(range(15, 18)) +
                     list(range(21, 37)) + list(range(59, 62)) + list(range(65, 86)))
MADISON_SOLO = set([10, 14] + list(range(37, 49)))
JAY = set([2, 3, 4, 5, 64])
JOINT = set([18, 19, 20])
DISPUTED = set(list(range(49, 59)) + [62, 63])

assert len(HAMILTON_SOLO) == 51, f"Hamilton count wrong: {len(HAMILTON_SOLO)}"
assert len(MADISON_SOLO) == 14, f"Madison count wrong: {len(MADISON_SOLO)}"
assert len(JAY) == 5
assert len(JOINT) == 3
assert len(DISPUTED) == 12


def get_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


# Module-level tokenizer, loaded once and reused everywhere
tokenizer = get_tokenizer()


# ---------------------------------------------------------------------------
# Corpus loading / cleaning
# ---------------------------------------------------------------------------

def list_txt_files(subfolder):
    return sorted(glob.glob(os.path.join(CORPUS_ROOT, subfolder, "*.txt")))


def clean_text(raw):
    text = raw
    text = re.sub(r"\[.*?\]", "", text)                     # editorial brackets
    text = re.sub(r"PUBLIUS\.?\s*$", "", text.strip())        # trailing signature
    text = re.sub(r"\s+", " ", text)                          # collapse whitespace
    return text.strip()


_CLEAN_TEXT_CACHE = {}  # path -> cleaned text, populated lazily


def load_and_clean_cached(file_list):
    """Like load_and_clean, but caches cleaned text per file path in memory so
    repeated calls across leave-one-out iterations don't re-read or re-regex
    the same files from disk over and over."""
    chunks = []
    for path in file_list:
        if path not in _CLEAN_TEXT_CACHE:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                _CLEAN_TEXT_CACHE[path] = clean_text(f.read())
        chunks.append(_CLEAN_TEXT_CACHE[path])
    return chunks


# Kept as an alias for clarity when caching isn't the point being made
def load_and_clean(file_list):
    return load_and_clean_cached(file_list)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tok, text, stride=512, max_length=1024):
    """Sliding-window perplexity for texts longer than the model's context window."""
    encodings = tok(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end)
    return ppl.item()


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

class InMemoryTextDataset(Dataset):
    """Tokenizes a text STRING (not a file path) and chunks it into fixed-length
    blocks. Replaces transformers' removed TextDataset class, and skips the
    temp-file write+read round trip that the file-based version needed."""

    def __init__(self, tok, text, block_size=256):
        token_ids = tok.encode(text)
        self.examples = [
            token_ids[i:i + block_size]
            for i in range(0, len(token_ids) - block_size + 1, block_size)
        ]
        if not self.examples:
            self.examples = [token_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def fine_tune_on_files(file_list, output_dir=None, epochs=10, block_size=256,
                        batch_size=2, save_to_disk=False):
    """Fine-tunes a fresh GPT-2 model on the given files, returning the model
    object directly (in memory).

    epochs defaults to 10 (not the ALM paper's 100) as a starting point that
    fits within typical cluster job time limits and reduces overfitting risk
    on small corpora (e.g. Madison's 14 papers) — see README for the planned
    epoch-sweep experiment to find the right value empirically.

    save_to_disk: set True only for models you want to keep after this run
    (e.g. the "full" Hamilton/Madison models, or score_disputed_papers' final
    models). Leave False for one-off "minus-one" models used and discarded
    within the same leave-one-out iteration — this skips a full checkpoint
    write+read that would otherwise happen 65 times during a full validation
    run for no benefit, since that model is never needed again afterward.
    """
    texts = load_and_clean_cached(file_list)
    combined_text = "\n\n".join(texts)

    dataset = InMemoryTextDataset(tok=tokenizer, text=combined_text, block_size=block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir or "models/_tmp_training",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        fp16=USE_FP16,
    )

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)
    trainer.train()

    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    return model  # returned directly — caller uses it in-memory, no reload needed


def load_finetuned_model(model_dir):
    """Only needed for models that were actually saved to disk (save_to_disk=True)."""
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    return model, tok


# ---------------------------------------------------------------------------
# Validation experiments
# ---------------------------------------------------------------------------

def leave_one_out_validation(hamilton_files, madison_files, sample_size=None, epochs=10, seed=None,
                              save_full_models=False, full_models_dir="models", batch_size=2):
    """
    Returns a list of dicts: {file, true_author, predicted_author, ppl_hamilton, ppl_madison, correct}
    Set sample_size to an integer to test on a random subset first (faster first try).

    Trains each author's FULL model once (2 trainings total), reused whenever the
    held-out paper belongs to the OTHER author. Only retrains a "minus-one" model
    for whichever author the held-out paper actually belongs to (up to 65 more
    trainings) — 67 total instead of 130, with identical results. "Minus-one"
    models are kept in memory only and never written to disk (see
    fine_tune_on_files' save_to_disk parameter) since they're used once and discarded.
    """
    results = []
    if seed is not None:
        random.seed(seed)

    all_papers = [("hamilton", f) for f in hamilton_files] + [("madison", f) for f in madison_files]
    if sample_size:
        all_papers = random.sample(all_papers, min(sample_size, len(all_papers)))

    print("Training full Hamilton and Madison models (reused across iterations)...", flush=True)
    full_ham_model = fine_tune_on_files(
        hamilton_files, output_dir=os.path.join(full_models_dir, "full_hamilton"),
        epochs=epochs, batch_size=batch_size, save_to_disk=save_full_models,
    )
    full_mad_model = fine_tune_on_files(
        madison_files, output_dir=os.path.join(full_models_dir, "full_madison"),
        epochs=epochs, batch_size=batch_size, save_to_disk=save_full_models,
    )

    for true_author, held_out_file in all_papers:
        if true_author == "hamilton":
            remaining_hamilton = [f for f in hamilton_files if f != held_out_file]
            ham_model = fine_tune_on_files(remaining_hamilton, epochs=epochs, batch_size=batch_size, save_to_disk=False)
            mad_model = full_mad_model  # unchanged, reuse in-memory model directly
        else:
            remaining_madison = [f for f in madison_files if f != held_out_file]
            mad_model = fine_tune_on_files(remaining_madison, epochs=epochs, batch_size=batch_size, save_to_disk=False)
            ham_model = full_ham_model  # unchanged, reuse in-memory model directly

        held_out_text = load_and_clean_cached([held_out_file])[0]
        ppl_hamilton = compute_perplexity(ham_model, tokenizer, held_out_text)
        ppl_madison = compute_perplexity(mad_model, tokenizer, held_out_text)

        predicted_author = "hamilton" if ppl_hamilton < ppl_madison else "madison"
        correct = predicted_author == true_author

        results.append({
            "file": os.path.basename(held_out_file),
            "true_author": true_author,
            "predicted_author": predicted_author,
            "ppl_hamilton": ppl_hamilton,
            "ppl_madison": ppl_madison,
            "correct": correct,
        })
        print(f"{os.path.basename(held_out_file)} | true={true_author} pred={predicted_author} "
              f"(H:{ppl_hamilton:.1f} M:{ppl_madison:.1f}) {'CORRECT' if correct else 'WRONG'}",
              flush=True)

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nOverall leave-one-out accuracy: {accuracy:.1%} "
          f"({sum(r['correct'] for r in results)}/{len(results)})", flush=True)
    return results


def score_disputed_papers(hamilton_files, madison_files, disputed_files, epochs=10,
                           models_dir="models", batch_size=2):
    """Trains final models on ALL solely-attributed papers (no held-out), then
    scores each disputed paper under both. These models ARE saved to disk since
    they represent your final, reusable result."""
    ham_model = fine_tune_on_files(
        hamilton_files, output_dir=os.path.join(models_dir, "final_hamilton"),
        epochs=epochs, batch_size=batch_size, save_to_disk=True,
    )
    mad_model = fine_tune_on_files(
        madison_files, output_dir=os.path.join(models_dir, "final_madison"),
        epochs=epochs, batch_size=batch_size, save_to_disk=True,
    )

    results = []
    for path in disputed_files:
        text = load_and_clean_cached([path])[0]
        ppl_h = compute_perplexity(ham_model, tokenizer, text)
        ppl_m = compute_perplexity(mad_model, tokenizer, text)
        verdict = "Hamilton" if ppl_h < ppl_m else "Madison"
        results.append({"file": os.path.basename(path), "ppl_hamilton": ppl_h,
                         "ppl_madison": ppl_m, "verdict": verdict})
        print(f"{os.path.basename(path)}: H={ppl_h:.1f}  M={ppl_m:.1f}  -> {verdict}", flush=True)

    madison_count = sum(1 for r in results if r["verdict"] == "Madison")
    print(f"\n{madison_count}/{len(results)} disputed papers attributed to Madison "
          f"(scholarly consensus attributes all 12 to Madison).", flush=True)
    return results


def save_results_csv(results, out_path):
    if not results:
        print(f"No results to save for {out_path}")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {out_path}")
