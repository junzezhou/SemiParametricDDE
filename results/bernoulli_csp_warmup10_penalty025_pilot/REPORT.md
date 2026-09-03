# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 0.800 | 0.200 | 0.000 | 0.856 | 0.881 | 50.0 | 1.000 | 1.37 | 186.5 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.600 | 0.600 | 0.000 | 0.971 | 0.874 | 50.0 | 1.000 | 15.02 | 245.4 |
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.200 | 2.000 | 0.800 | 0.916 | 0.759 | 50.0 | 1.000 | 3.79 | 203.2 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.600 | 0.400 | 0.400 | 0.953 | 0.784 | 50.0 | 1.000 | 13.09 | 249.7 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.200 | 1.200 | 0.000 | 0.966 | 0.813 | 50.0 | 1.000 | 48.07 | 346.0 |

## Automatic findings

- Numerical completion: 25/25 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.800 at N=500 to 0.600 at N=8000; time 1.37s to 15.02s; final conditional-log-loss gap to oracle +0.0006.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.200 at N=500 to 0.200 at N=8000; time 3.79s to 48.07s; final conditional-log-loss gap to oracle +0.0017.

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
  "penalty_n_power": 0.25,
  "penalty_reference_n": 500.0,
  "shrinkage_burn": 10,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
