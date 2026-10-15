# Poisson CSP dimension-recovery test

| Scenario | N | OK | Shallow exact | Shallow L1 | All exact | All L1 | K1 exact | K2 exact | Under | Over | Iter. | Limit | Conv. | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_MATLAB_sparse_overspecified_J27 | 4000 | 100/100 | 0.710 | 0.390 | 0.710 | 0.390 | 0.890 | 0.720 | 0.000 | 0.290 | 30.0 | 1.000 | 0.000 | 44.24 |
| D2_MATLAB_sparse_overspecified_J27 | 8000 | 100/100 | 0.590 | 0.420 | 0.590 | 0.420 | 0.970 | 0.610 | 0.000 | 0.410 | 30.0 | 1.000 | 0.000 | 89.35 |
| D3_MATLAB_sparse_overspecified_J64 | 4000 | 100/100 | 0.540 | 0.600 | 0.540 | 0.600 | 0.750 | 0.680 | 0.050 | 0.420 | 30.0 | 1.000 | 0.000 | 99.35 |
| D3_MATLAB_sparse_overspecified_J64 | 8000 | 100/100 | 0.620 | 0.480 | 0.620 | 0.480 | 0.900 | 0.660 | 0.080 | 0.310 | 30.0 | 1.000 | 0.000 | 52.46 |

The exact-shrinkage metric equals one only when every fitted layer width equals its generating width. Dimension L1 is the sum of the absolute layer-wise width errors. Shallow metrics use only the first two latent layers.

## Settings

```json
{
  "backend": "mcem",
  "C": 1,
  "max_iter": 30,
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
  "settings_id": "37f97fe97f52e510"
}
```
