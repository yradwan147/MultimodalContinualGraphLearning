# MCGL Project Phase Plans

This directory contains the implementation plan for the **Multimodal Continual Graph Learning (MCGL)** project, broken into phases to enable incremental execution across multiple Claude sessions.

## Phase Index

| Phase | File | Status | Description |
|-------|------|--------|-------------|
| 0 | (this session) | DONE | Git init, project scaffolding, CLAUDE.md, worklog |
| 1 | [phase1-setup-and-literature.md](phase1-setup-and-literature.md) | PENDING | Environment setup, literature review, PrimeKG exploration |
| 2 | [phase2-benchmark-construction.md](phase2-benchmark-construction.md) | PENDING | Temporal benchmark from PrimeKG snapshots |
| 3 | [phase3-baseline-implementation.md](phase3-baseline-implementation.md) | PENDING | 6 baseline methods implementation |
| 4 | [phase4-cmkl-development.md](phase4-cmkl-development.md) | PENDING | Novel CMKL method development |
| 5 | [phase5-experiments-and-ablations.md](phase5-experiments-and-ablations.md) | PENDING | Full experiment matrix + ablation studies |
| 6 | [phase6-paper-a-benchmark.md](phase6-paper-a-benchmark.md) | PENDING | Paper A: Benchmark paper writing |
| 7 | [phase7-paper-b-method.md](phase7-paper-b-method.md) | PENDING | Paper B: Method paper writing |

## How to Use

1. At the start of each Claude session, read `CLAUDE.md` (loaded automatically) and `worklog.md` to recover context
2. Open the relevant phase plan file for the current work
3. Follow the step-by-step instructions in the phase plan
4. Update `worklog.md` after every significant change
5. Mark phase status as DONE in this README when completed

## Source Document
All plans derive from: `MCGL Full Project Guide.pdf` (50 pages) in the project root.
