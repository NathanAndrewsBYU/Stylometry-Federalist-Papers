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

Everything after setup runs through `sbatch`, not directly. Run these from the `cluster/` folder:

```bash
sbatch submit_job.sh first-try              # 10-paper first try (10 epochs), a few minutes
sbatch submit_job.sh full                   # unbalanced 65-paper leave-one-out validation (30 epochs)
sbatch submit_job.sh disputed [epochs]      # score the 12 disputed papers; epochs defaults to 30
sbatch submit_job.sh epoch-sweep            # unbalanced full validation at epochs 5, 10, 20, 40, 80
sbatch submit_job.sh balanced [options]     # balanced leave-one-out on the 65 known papers (see below)
sbatch submit_job.sh disputed-loo [options] # balanced leave-one-out with the 12 disputed papers folded into Madison (n = 77)
```

`balanced` and `disputed-loo` pass any extra options straight to `run_experiment.py`:

- `--epoch-values 30 80` — epoch counts to test (default: 30)
- `--balance-seeds 1 2 3 4 5` — which random Hamilton subsamples to use (default: 42)
- `--batch-size N` — training batch size (default: 2)

Examples:

```bash
sbatch submit_job.sh balanced --epoch-values 80 100 --balance-seeds 1 2 3 4 5
sbatch submit_job.sh disputed-loo --epoch-values 30 --balance-seeds 1 2 3 4
```

**Balanced vs. unbalanced.** `full` and `epoch-sweep` train on all 51 Hamilton papers and 14 Madison papers, which biases the model toward Hamilton. `balanced` and `disputed-loo` subsample Hamilton to match Madison's size (the seed controls which papers) but still evaluate every Hamilton paper. The `disputed` mode scores the 12 disputed papers using models trained on 14 balanced Hamilton papers vs. 14 Madison papers.

**Job length.** Every epoch value × seed combination retrains a set of models (about 30 for `balanced`, 54 for `disputed-loo`), and large grids can exceed the job's time limit (`#SBATCH --time` at the top of `submit_job.sh`). Split big grids into separate jobs (for example, one seed or one epoch value per `sbatch`) rather than one long run.

**Results.** Check job status with `squeue -u $USER`. Output and errors land in `logs/`, and result CSVs land in `results/` inside `cluster/`. Copy them into the repo's top-level `results/` folder. Note that `disputed_included_summary.csv` and `balanced_summary.csv` are overwritten by each run of their mode, so rename or copy them before launching another job.

### Burrows' Delta baseline

The traditional baseline needs no GPU and finishes in seconds:

```bash
python traditional_stylometry.py                                           # balanced, 65 known papers, seed 42
python traditional_stylometry.py --balance-seeds 1 2 3 4 5                 # balanced, multiple subsamples
python traditional_stylometry.py --no-balance                              # unbalanced (51 vs 14)
python traditional_stylometry.py --include-disputed --balance-seeds 1 2 3 4 5   # n = 77 setup
```

Colab notebooks that run the same experiments are in `notebooks/`.
## What's different from Colab

- **No `!pip install`** — dependencies are installed once into a persistent environment, not reinstalled every session.
- **No browser upload dialogs** — the corpus lives on disk from `prepare_corpus.py`, and results are just files in `results/` you can inspect directly or `scp` to your own machine.
- **No live progress bars** — since jobs run detached from any browser session, check `logs/*.out` for progress instead of watching a cell execute in real time. This also means **closing your laptop doesn't kill anything** — once `sbatch` submits the job, it runs independently of your connection.
- **Reproducible seeds** — `run_experiment.py` takes a `--seed` argument so sample selection for the sanity check is repeatable, not silently different every run (the original notebook's `random.sample` had no fixed seed).
