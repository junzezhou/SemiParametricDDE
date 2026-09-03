# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.200 | 2.000 | 0.800 | 0.916 | 0.759 | 50.0 | 1.000 | 3.85 | 201.4 |
| D3_Bs_overspecified_J64 | 1000 | 5/5 | 0.400 | 0.800 | 0.600 | 0.927 | 0.789 | 50.0 | 1.000 | 7.43 | 212.8 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.600 | 0.400 | 0.400 | 0.953 | 0.784 | 50.0 | 1.000 | 13.59 | 234.3 |
| D3_Bs_overspecified_J64 | 4000 | 5/5 | 0.200 | 1.200 | 0.600 | 0.968 | 0.797 | 50.0 | 1.000 | 26.01 | 266.2 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.200 | 1.000 | 0.400 | 0.968 | 0.808 | 50.0 | 1.000 | 47.70 | 341.8 |

## Automatic findings

- Numerical completion: 25/25 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.200 at N=500 to 0.200 at N=8000; time 3.85s to 47.70s; final conditional-log-loss gap to oracle +0.0044.

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
  "penalty_n_power": 0.5,
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
