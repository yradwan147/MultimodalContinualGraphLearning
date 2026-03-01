# Experiment Results Tracking

*Updated after each experiment run. All results use 5 random seeds, reporting mean +/- std.*

## Baseline Results

| Method | AP | AF | BWT | FWT | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------|----|----|-----|-----|-----|--------|--------|---------|
| Naive Sequential | | | | | | | | |
| Joint Training | | | | | | | | |
| EWC | | | | | | | | |
| Experience Replay | | | | | | | | |
| LKGE | | | | | | | | |
| RAG Agent | | | | | | | | |

## CMKL Results

| Variant | AP | AF | BWT | FWT | MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|----|----|-----|-----|-----|--------|--------|---------|
| CMKL (full) | | | | | | | | |
| CMKL + RAG | | | | | | | | |

## Ablation Results

| Ablation | AP | AF | BWT | FWT |
|----------|----|----|-----|-----|
| Struct only | | | | |
| Text only | | | | |
| Concat fusion | | | | |
| Global EWC | | | | |
| Random replay | | | | |
