# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 1.000 | 0.000 | 0.000 | 0.848 | 0.879 | 50.0 | 1.000 | 1.12 | 191.8 |
| D2_Bs_overspecified_J27 | 1000 | 5/5 | 0.800 | 0.200 | 0.200 | 0.883 | 0.858 | 50.0 | 1.000 | 2.01 | 195.8 |
| D2_Bs_overspecified_J27 | 2000 | 5/5 | 1.000 | 0.000 | 0.000 | 0.967 | 0.880 | 50.0 | 1.000 | 4.58 | 202.2 |
| D2_Bs_overspecified_J27 | 4000 | 5/5 | 0.800 | 0.200 | 0.200 | 0.968 | 0.865 | 50.0 | 1.000 | 7.94 | 219.7 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.800 | 0.200 | 0.000 | 0.976 | 0.874 | 50.0 | 1.000 | 13.15 | 244.1 |

## Automatic findings

- Numerical completion: 25/25 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 1.000 at N=500 to 0.800 at N=8000; time 1.12s to 13.15s; final conditional-log-loss gap to oracle +0.0014.

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
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
