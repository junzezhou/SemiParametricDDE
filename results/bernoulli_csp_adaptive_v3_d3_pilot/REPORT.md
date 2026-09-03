# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 30/30 | 0.033 | 2.600 | 0.967 | 0.913 | 0.751 | 50.0 | 1.000 | 4.23 | 201.6 |
| D3_Bs_overspecified_J64 | 1000 | 30/30 | 0.433 | 0.933 | 0.567 | 0.926 | 0.775 | 50.0 | 1.000 | 7.65 | 217.7 |
| D3_Bs_overspecified_J64 | 2000 | 30/30 | 0.500 | 0.733 | 0.467 | 0.960 | 0.787 | 50.0 | 1.000 | 13.87 | 241.9 |
| D3_Bs_overspecified_J64 | 4000 | 30/30 | 0.167 | 1.400 | 0.767 | 0.964 | 0.783 | 50.0 | 1.000 | 26.06 | 276.7 |
| D3_Bs_overspecified_J64 | 8000 | 30/30 | 0.400 | 1.067 | 0.600 | 0.965 | 0.791 | 50.0 | 1.000 | 44.66 | 361.2 |

## Automatic findings

- Numerical completion: 150/150 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.033 at N=500 to 0.400 at N=8000; time 4.23s to 44.66s; final conditional-log-loss gap to oracle +0.0058.

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
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
