# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 4000 | 10/10 | 0.500 | 0.700 | 0.500 | 0.966 | 0.792 | 50.0 | 1.000 | 25.84 | 265.4 |
| D3_Bs_overspecified_J64 | 8000 | 10/10 | 0.000 | 1.200 | 0.700 | 0.963 | 0.787 | 50.0 | 1.000 | 51.37 | 336.4 |

## Automatic findings

- Numerical completion: 20/20 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.500 at N=4000 to 0.000 at N=8000; time 25.84s to 51.37s; final conditional-log-loss gap to oracle +0.0028.

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
  "base_seed": 20261001,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
