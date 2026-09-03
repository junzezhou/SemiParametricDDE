# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 4000 | 10/10 | 0.800 | 0.200 | 0.100 | 0.971 | 0.871 | 50.0 | 1.000 | 7.96 | 208.7 |
| D2_Bs_overspecified_J27 | 8000 | 10/10 | 0.500 | 0.500 | 0.200 | 0.963 | 0.864 | 50.0 | 1.000 | 15.53 | 232.4 |

## Automatic findings

- Numerical completion: 20/20 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.800 at N=4000 to 0.500 at N=8000; time 7.96s to 15.53s; final conditional-log-loss gap to oracle +0.0045.

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
