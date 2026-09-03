# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 10/10 | 0.000 | 3.100 | 1.000 | 0.908 | 0.743 | 50.0 | 1.000 | 3.98 | 199.4 |
| D3_Bs_overspecified_J64 | 1000 | 10/10 | 0.400 | 0.800 | 0.600 | 0.927 | 0.771 | 50.0 | 1.000 | 8.23 | 216.2 |
| D3_Bs_overspecified_J64 | 2000 | 10/10 | 0.200 | 1.000 | 0.800 | 0.957 | 0.788 | 50.0 | 1.000 | 13.77 | 245.7 |
| D3_Bs_overspecified_J64 | 4000 | 10/10 | 0.500 | 0.600 | 0.500 | 0.967 | 0.794 | 50.0 | 1.000 | 25.13 | 278.0 |
| D3_Bs_overspecified_J64 | 8000 | 10/10 | 0.200 | 1.000 | 0.800 | 0.965 | 0.788 | 50.0 | 1.000 | 47.87 | 346.1 |

## Automatic findings

- Numerical completion: 50/50 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.200 at N=8000; time 3.98s to 47.87s; final conditional-log-loss gap to oracle +0.0044.

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
  "base_seed": 20261001,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
