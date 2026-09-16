# Poisson CSP dimension-recovery test

| Scenario | N | OK | Shallow exact | Shallow L1 | All exact | All L1 | K1 exact | K2 exact | Under | Over | Iter. | Limit | Conv. | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D4_MATLAB_sparse_overspecified_J192 | 500 | 100/100 | 0.000 | 5.470 | 0.000 | 6.090 | 0.610 | 0.000 | 1.000 | 0.430 | 50.0 | 1.000 | 0.000 | 28.38 |
| D4_MATLAB_sparse_overspecified_J192 | 1000 | 100/100 | 0.020 | 3.120 | 0.010 | 3.860 | 0.350 | 0.040 | 0.960 | 0.870 | 50.0 | 1.000 | 0.000 | 56.83 |
| D4_MATLAB_sparse_overspecified_J192 | 2000 | 100/100 | 0.040 | 2.340 | 0.010 | 3.220 | 0.190 | 0.340 | 0.630 | 0.980 | 50.0 | 1.000 | 0.000 | 130.52 |
| D4_MATLAB_sparse_overspecified_J192 | 4000 | 100/100 | 0.060 | 2.500 | 0.000 | 3.410 | 0.200 | 0.350 | 0.360 | 0.970 | 50.0 | 1.000 | 0.000 | 327.67 |
| D4_MATLAB_sparse_overspecified_J192 | 8000 | 100/100 | 0.050 | 2.920 | 0.000 | 3.880 | 0.110 | 0.330 | 0.230 | 1.000 | 50.0 | 1.000 | 0.000 | 635.36 |

The exact-shrinkage metric equals one only when every fitted layer width equals its generating width. Dimension L1 is the sum of the absolute layer-wise width errors. Shallow metrics use only the first two latent layers.

## Settings

```json
{
  "backend": "mcem",
  "C": 1,
  "max_iter": 50,
  "optimizer_maxiter": 100,
  "optimizer_tol": 1e-06,
  "tol": 0.005,
  "convergence_window": 7,
  "convergence_mean_tol": 0.01,
  "burn": 5,
  "tau": 0.3,
  "penalty_strength": 1.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260916,
  "settings_id": "3903f7882040b5a8"
}
```
