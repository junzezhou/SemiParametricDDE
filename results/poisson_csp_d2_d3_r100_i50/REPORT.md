# Poisson CSP dimension-recovery test

| Scenario | N | OK | Shallow exact | Shallow L1 | All exact | All L1 | K1 exact | K2 exact | Under | Over | Iter. | Limit | Conv. | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_MATLAB_sparse_overspecified_J27 | 500 | 100/100 | 0.650 | 0.370 | 0.650 | 0.370 | 0.960 | 0.670 | 0.000 | 0.350 | 50.0 | 1.000 | 0.000 | 7.26 |
| D2_MATLAB_sparse_overspecified_J27 | 1000 | 100/100 | 0.750 | 0.290 | 0.750 | 0.290 | 0.930 | 0.780 | 0.000 | 0.250 | 50.0 | 1.000 | 0.000 | 11.50 |
| D2_MATLAB_sparse_overspecified_J27 | 2000 | 100/100 | 0.700 | 0.350 | 0.700 | 0.350 | 0.950 | 0.700 | 0.000 | 0.300 | 50.0 | 1.000 | 0.000 | 20.47 |
| D3_MATLAB_sparse_overspecified_J64 | 500 | 100/100 | 0.570 | 0.580 | 0.570 | 0.580 | 0.730 | 0.790 | 0.050 | 0.400 | 50.0 | 1.000 | 0.000 | 23.40 |
| D3_MATLAB_sparse_overspecified_J64 | 1000 | 100/100 | 0.550 | 0.600 | 0.550 | 0.600 | 0.720 | 0.740 | 0.020 | 0.440 | 50.0 | 1.000 | 0.000 | 46.91 |
| D3_MATLAB_sparse_overspecified_J64 | 2000 | 100/100 | 0.580 | 0.530 | 0.580 | 0.530 | 0.820 | 0.690 | 0.070 | 0.360 | 50.0 | 1.000 | 0.000 | 96.43 |

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
