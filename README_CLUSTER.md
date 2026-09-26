# Running on BYU's Office of Research Computing cluster

This is a script-based port of the original Colab notebook (`gpt2_federalist_stylometry.ipynb`),
restructured to run as scheduled SLURM jobs instead of interactive notebook cells. All the
actual model logic (fine-tuning, perplexity calculation, leave-one-out validation) is identical
to the notebook, only the delivery mechanism has changed.

## Files

| File | Purpose |
|---|---|
| `stylometry_lib.py` | All shared functions — cleaning, perplexity, fine-tuning, validation. Imported by everything else. |
| `prepare_corpus.py` | One-time download + split of the Federalist Papers. Run on a **login node**, not in a job. |
| `run_experiment.py` | Main CLI entry point. Run via `sbatch`, not directly. |
| `submit_job.sh` | SLURM batch script — this is what you actually submit with `sbatch`. |
| `requirements.txt` | Pinned dependency versions (avoids the `TextDataset`/`TrainingArguments` breaking changes hit during Colab development). |

## First-time setup

1. **Log in** to the cluster (ask Research Computing, 403 CB, for account setup if you don't have one yet).
2. **Set up a Python environment** (ask them which pattern their system expects — venv, conda, or environment modules):
   ```bash
   python -m venv ~/envs/federalist
   source ~/envs/federalist/bin/activate
   pip install -r requirements.txt
   ```
3. **Clone this repo** onto the cluster:
   ```bash
   git clone https://github.com/NathanAndrewsBYU/Stylometry-Federalist-Papers.git
   cd Stylometry-Federalist-Papers
   ```
4. **Download and split the corpus** (needs internet access — do this on the login node, not inside a job):
   ```bash
   python prepare_corpus.py
   ```
   This creates `corpus/federalist/{hamilton,madison,jay,joint,disputed}/` with 85 individual paper files.

## Running experiments

Everything after setup runs through `sbatch`, not directly:

```bash
sbatch submit_job.sh sanity        # 10-paper sanity check, ~few minutes to an hour depending on GPU
sbatch submit_job.sh full          # full 65-paper leave-one-out validation
sbatch submit_job.sh disputed      # score the 12 disputed papers (run after 'full' looks good)
sbatch submit_job.sh epoch-sweep   # runs full validation at epochs = 5, 10, 20, 40, 80 to find the best epoch count
```

Check job status with `squeue -u $USER`. Output and errors land in `logs/`, and results CSVs land in `results/`.

## What's different from Colab

- **No `!pip install`** — dependencies are installed once into a persistent environment, not reinstalled every session.
- **No browser upload dialogs** — the corpus lives on disk from `prepare_corpus.py`, and results are just files in `results/` you can inspect directly or `scp` to your own machine.
- **No live progress bars** — since jobs run detached from any browser session, check `logs/*.out` for progress instead of watching a cell execute in real time. This also means **closing your laptop doesn't kill anything** — once `sbatch` submits the job, it runs independently of your connection.
- **Reproducible seeds** — `run_experiment.py` takes a `--seed` argument so sample selection for the sanity check is repeatable, not silently different every run (the original notebook's `random.sample` had no fixed seed).

## Known unknowns — confirm these with Research Computing before your first real run

- The exact `#SBATCH --gres=gpu:1` syntax in `submit_job.sh` is a common pattern but may need adjusting for BYU's specific cluster (some systems require `--partition=gpu` or an `--account=` flag as well).
- Whether compute nodes have any internet access at all (this determines whether `prepare_corpus.py` could theoretically run inside a job — recommended default here is still to run it on the login node regardless).
- Reasonable `--time` and `--mem` limits for a student allocation.
