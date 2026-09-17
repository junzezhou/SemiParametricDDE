# SemiDDE CSP sample-size scaling study

## Design

- Sample sizes: `500, 1000, 2000, 4000, 8000`
- Replicates: 100 per scenario and sample size (1000 fits total)
- Seed: replicate `r` uses `20260916 + r`
- Primary metrics: exact width recovery rate and mean L1 width error
- All 1000 fits completed successfully.
- No recursive structural cap was applied to a fitted overspecified hierarchy.

### D2

- True widths: `(9, 3)`; maximum initialized widths: `(18, 6)`
- `t_spike=(0.02, 0.04)`, `temp=0.8`
- 3 outer iterations, 20 inner iterations, optimizer maximum 30

| N | Exact rate | Mean L1 | Median seconds | Fitted widths |
|---:|---:|---:|---:|:---|
| 500 | 0.98 | 0.02 | 0.36 | `(9,3)` 98; `(9,4)` 2 |
| 1000 | 0.99 | 0.01 | 0.51 | `(9,3)` 99; `(9,4)` 1 |
| 2000 | 0.99 | 0.01 | 0.60 | `(9,3)` 99; `(9,4)` 1 |
| 4000 | 0.82 | 0.19 | 1.02 | `(9,3)` 82; `(9,4)` 17; `(9,5)` 1 |
| 8000 | 0.83 | 0.18 | 1.87 | `(9,3)` 83; `(9,4)` 16; `(9,5)` 1 |

D2 is highly reliable through N=2000. At N=4000 and N=8000 it develops
mild second-layer under-shrinkage, almost always retaining one extra node.

### D3

- True widths: `(18, 6, 2)`; maximum initialized widths: `(36, 12, 4)`
- `t_spike=(0.01, 0.03, 0.0075)`, `temp=0.85`
- 1 outer iteration, 50 inner iterations, optimizer maximum 50

| N | Exact rate | Mean L1 | Median seconds | Dominant fitted widths |
|---:|---:|---:|---:|:---|
| 500 | 0.33 | 0.74 | 1.10 | `(18,6,1)` 39; `(18,6,2)` 33; `(18,6,3)` 19 |
| 1000 | 0.38 | 0.68 | 1.66 | `(18,6,1)` 44; `(18,6,2)` 38; `(18,6,3)` 13 |
| 2000 | 0.12 | 0.98 | 2.65 | `(18,6,1)` 77; `(18,6,2)` 12; `(18,7,1)` 9 |
| 4000 | 0.00 | 1.30 | 4.63 | `(18,6,1)` 70; `(18,7,1)` 30 |
| 8000 | 0.00 | 1.32 | 8.55 | `(18,6,1)` 70; `(18,7,1)` 28; `(18,8,1)` 2 |

D3 is not sample-size stable under fixed CSP hyperparameters. Its best point
is N=1000. From N=2000 onward, the deepest layer is increasingly
over-shrunk from two nodes to one, while the middle layer begins to retain
extra nodes at N=4000 and N=8000.

## Interpretation

The observed deterioration is systematic rather than Monte Carlo noise. The
likelihood contribution grows with N, while the shared Copula/Normal CSP
column-scale update is driven mainly by coefficient magnitudes and the number
of active child rows rather than an explicit sample-size normalization. This
can make small middle-layer coefficients harder to eliminate at large N. At
the same time, the deepest layer has only a few active rows and its ordered
CSP allocation continues to absorb one true node into the spike component.

The next useful experiment is therefore sample-size-dependent layerwise spike
scaling, especially the second D2 scale and the middle/deep D3 scales. Adding
a recursive width cap would hide this calibration problem and is not
recommended.
