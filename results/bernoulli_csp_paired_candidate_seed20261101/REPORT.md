# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 30/30 | 0.000 | 2.333 | 1.000 | 0.914 | 0.753 | 50.0 | 1.000 | 4.02 | 200.8 |
| D3_Bs_overspecified_J64 | 1000 | 30/30 | 0.267 | 1.033 | 0.633 | 0.924 | 0.775 | 50.0 | 1.000 | 7.17 | 219.2 |
| D3_Bs_overspecified_J64 | 2000 | 30/30 | 0.267 | 1.033 | 0.733 | 0.958 | 0.767 | 50.0 | 1.000 | 12.06 | 245.6 |
| D3_Bs_overspecified_J64 | 4000 | 30/30 | 0.167 | 1.167 | 0.800 | 0.966 | 0.787 | 50.0 | 1.000 | 22.35 | 288.2 |
| D3_Bs_overspecified_J64 | 8000 | 30/30 | 0.267 | 1.067 | 0.633 | 0.962 | 0.783 | 50.0 | 1.000 | 43.16 | 350.9 |

## Automatic findings

- Numerical completion: 150/150 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.267 at N=8000; time 4.02s to 43.16s; final conditional-log-loss gap to oracle +0.0058.

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
  "penalty_n_power": 0.6666666666666666,
  "penalty_reference_n": 1000.0,
  "penalty_min_strength": 1.0,
  "shrinkage_burn": 10,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261101,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
