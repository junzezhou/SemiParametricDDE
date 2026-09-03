# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 30/30 | 0.000 | 3.100 | 1.000 | 0.912 | 0.751 | 50.0 | 1.000 | 3.60 | 193.8 |
| D3_Bs_overspecified_J64 | 1000 | 30/30 | 0.067 | 2.067 | 0.933 | 0.919 | 0.762 | 50.0 | 1.000 | 6.51 | 205.1 |
| D3_Bs_overspecified_J64 | 2000 | 30/30 | 0.033 | 2.133 | 0.933 | 0.936 | 0.763 | 50.0 | 1.000 | 10.93 | 223.1 |
| D3_Bs_overspecified_J64 | 4000 | 30/30 | 0.067 | 2.033 | 0.933 | 0.954 | 0.776 | 50.0 | 1.000 | 20.33 | 240.4 |
| D3_Bs_overspecified_J64 | 8000 | 30/30 | 0.267 | 1.333 | 0.733 | 0.950 | 0.768 | 50.0 | 1.000 | 38.94 | 280.7 |

## Automatic findings

- Numerical completion: 150/150 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.267 at N=8000; time 3.60s to 38.94s; final conditional-log-loss gap to oracle +0.0084.

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
  "penalty_n_power": 0.0,
  "penalty_reference_n": 500.0,
  "penalty_min_strength": 1.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261101,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
