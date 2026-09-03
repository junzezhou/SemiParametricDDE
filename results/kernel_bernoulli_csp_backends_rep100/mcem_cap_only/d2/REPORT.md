# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 100/100 | 0.640 | 0.580 | 0.360 | 0.890 | 0.816 | 40.0 | 1.000 | 0.43 | 191.5 |

## Automatic findings

- Numerical completion: 100/100 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.640 at N=500 to 0.640 at N=500; time 0.43s to 0.43s; final conditional-log-loss gap to oracle +0.0070.

## Settings

```json
{
  "backend": "mcem_cap_only",
  "C": 1,
  "max_iter": 40,
  "optimizer_maxiter": 50,
  "optimizer_tol": 1e-06,
  "tol": 0.0,
  "convergence_window": 7,
  "convergence_mean_tol": 0.01,
  "burn": 3,
  "tau": 0.3,
  "penalty_n_power": 0.0,
  "penalty_reference_n": 500.0,
  "penalty_min_strength": 2.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261301,
  "keep_trace": true
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
