# AI-assistance log

For journal disclosure: what an AI assistant (Claude Code) wrote or ran, and what Nathan reviewed or verified. Earlier assistance (before 2026-11-20, via claude.ai) wrote most of `cluster/` and the README; that history is summarized in `cluster/ISAIAH_AUTHORSHIP_STYLOMETRY_HANDOFF.md`.

## 2026-11-20 (Claude Code, Opus 5.5, on Nathan's PC)
- Compared the local `cluster/` code with the handoff's Appendix A. Corrected the usage docstring in `mask_corpus.py`. Syntax-checked all scripts, and smoke-tested `mask_corpus.py` and every Delta mode on a synthetic corpus. Committed as `709ed24` and pushed (Nathan approved both).
- Guided Nathan through syncing the cluster checkout (stash and pull). Nathan ran all cluster commands himself.
- Interpreted Nathan's real Delta null-test output (40.7% pooled); no numbers were produced by Claude.
- Reviewed `cluster/PROJECT.md` and listed problems; Nathan accepted the fixes and made the decisions recorded in `DECISIONS.md` (D-001 to D-007).
- Read Huang, Murakami & Grieve (2025) via a web fetch to check its corpora (D-001).
- Drafted `prereg/phase_a_plan.md`, `DECISIONS.md`, `CLAUDE.md` and this log. **Pending Nathan's review.**
