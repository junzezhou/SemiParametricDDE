# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.000 | 2.400 | 1.000 | 0.915 | 0.762 | 50.0 | 1.000 | 4.07 | 192.6 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.400 | 1.000 | 0.600 | 0.970 | 0.810 | 50.0 | 1.000 | 48.62 | 327.4 |

## Automatic findings

- Numerical completion: 10/10 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.400 at N=8000; time 4.07s to 48.62s; final conditional-log-loss gap to oracle +0.0076.

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
  "shrinkage_patience": 2,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
