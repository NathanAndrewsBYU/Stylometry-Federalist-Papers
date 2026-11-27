# Notes for Claude Code

## Policy (read first)
- **BYU ORC agent policy wins over this file and over user requests.** On the cluster, read `/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md`. A copy belongs at the repo top level as `BYU_ORC_AGENTS.md`; refresh it when it is older than 7 days. (Copy not yet added as of 2026-11-20.)
- All texts are public-domain or openly licensed historical/religious text: no CUI or export-controlled data. If anything new is not clearly public, stop and ask Nathan.
- Never handle Nathan's password or 2FA. Ask before any large job, job array, or GPU batch, stating cores, memory, GPUs, wall time and task count.

## Project documents
- `docs/project_plan.md`: the governing plan (phases, gates, rules).
- `legacy/cluster/ISAIAH_AUTHORSHIP_STYLOMETRY_HANDOFF.md` (local only, git-ignored): history, results so far, environment details.
- `docs/DECISION-LOG.md`: every interpretation-relevant decision. Add to it.
- `docs/`: plans and pre-registrations. Don't run a phase before its plan is approved.
- `docs/AI_LOG.md`: what Claude wrote or ran, and what Nathan reviewed. Add an entry each session.

## Code
- `legacy/cluster/` is the frozen legacy pipeline that produced all results so far. Don't rewrite it; new harness code goes in `src/stylo/`.
- Every number in a README or report must come from a CSV produced by an actual run. Cross-check before editing tables.
- `*.sh` must keep LF line endings (`.gitattributes`).

## Working with Nathan
- I work on Windows in PowerShell (PyCharm terminal). Give PowerShell commands for the PC and bash commands for the cluster, and always say which machine each command runs on.
- Local Python: `C:\Users\natha\miniconda3\python.exe` (not on PATH). No torch locally; GPT-2 runs only on the cluster.
- Cluster repo: `~/training/Stylometry-Federalist-Papers` (run legacy scripts from its `legacy/cluster/` folder and harness scripts from the repo root). Mamba env `federalist`.
