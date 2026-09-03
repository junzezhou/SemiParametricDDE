# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 1.000 | 0.000 | 0.000 | 0.848 | 0.879 | 50.0 | 1.000 | 1.16 | 186.7 |
| D2_Bs_overspecified_J27 | 1000 | 5/5 | 0.600 | 0.600 | 0.400 | 0.914 | 0.853 | 50.0 | 1.000 | 2.14 | 193.3 |
| D2_Bs_overspecified_J27 | 2000 | 5/5 | 1.000 | 0.000 | 0.000 | 0.974 | 0.883 | 50.0 | 1.000 | 4.95 | 194.8 |
| D2_Bs_overspecified_J27 | 4000 | 5/5 | 0.800 | 0.200 | 0.200 | 0.969 | 0.866 | 50.0 | 1.000 | 8.44 | 208.6 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.800 | 0.200 | 0.000 | 0.976 | 0.874 | 50.0 | 1.000 | 14.44 | 235.6 |

## Automatic findings

- Numerical completion: 25/25 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 1.000 at N=500 to 0.800 at N=8000; time 1.16s to 14.44s; final conditional-log-loss gap to oracle +0.0014.

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
  "penalty_reference_n": 500.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
