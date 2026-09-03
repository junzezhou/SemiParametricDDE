# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 100/100 | 0.000 | 4.490 | 1.000 | 0.899 | 0.712 | 50.0 | 1.000 | 2.38 | 198.2 |

## Automatic findings

- Numerical completion: 100/100 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.000 at N=500; time 2.38s to 2.38s; final conditional-log-loss gap to oracle +0.0175.

## Settings

```json
{
  "backend": "mcem_reinitialize",
  "C": 1,
  "max_iter": 50,
  "optimizer_maxiter": 50,
  "optimizer_tol": 1e-06,
  "tol": 0.0,
  "convergence_window": 7,
  "convergence_mean_tol": 0.01,
  "burn": 3,
  "tau": 0.3,
  "penalty_n_power": 0.0,
  "penalty_reference_n": 500.0,
  "penalty_min_strength": 1.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261301,
  "keep_trace": true
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
