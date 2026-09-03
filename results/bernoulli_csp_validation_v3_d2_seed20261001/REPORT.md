# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 10/10 | 0.700 | 0.300 | 0.200 | 0.832 | 0.861 | 50.0 | 1.000 | 1.16 | 191.7 |
| D2_Bs_overspecified_J27 | 1000 | 10/10 | 0.800 | 0.300 | 0.000 | 0.883 | 0.867 | 50.0 | 1.000 | 2.14 | 196.6 |
| D2_Bs_overspecified_J27 | 2000 | 10/10 | 1.000 | 0.000 | 0.000 | 0.976 | 0.883 | 50.0 | 1.000 | 4.31 | 205.3 |
| D2_Bs_overspecified_J27 | 4000 | 10/10 | 0.800 | 0.200 | 0.100 | 0.980 | 0.876 | 50.0 | 1.000 | 8.15 | 222.3 |
| D2_Bs_overspecified_J27 | 8000 | 10/10 | 0.500 | 0.500 | 0.400 | 0.949 | 0.856 | 50.0 | 1.000 | 15.28 | 228.4 |

## Automatic findings

- Numerical completion: 50/50 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.700 at N=500 to 0.500 at N=8000; time 1.16s to 15.28s; final conditional-log-loss gap to oracle +0.0086.

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
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261001,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
