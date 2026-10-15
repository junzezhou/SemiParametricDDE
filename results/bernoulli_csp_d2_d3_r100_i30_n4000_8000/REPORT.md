# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 4000 | 100/100 | 0.690 | 0.460 | 0.060 | 0.956 | 0.868 | 30.0 | 1.000 | 8.45 | 224.5 |
| D2_Bs_overspecified_J27 | 8000 | 100/100 | 0.530 | 0.630 | 0.070 | 0.949 | 0.861 | 30.0 | 1.000 | 23.56 | 256.1 |
| D3_Bs_overspecified_J64 | 4000 | 100/100 | 0.090 | 1.670 | 0.790 | 0.944 | 0.761 | 30.0 | 1.000 | 44.24 | 268.4 |
| D3_Bs_overspecified_J64 | 8000 | 100/100 | 0.140 | 1.580 | 0.590 | 0.943 | 0.755 | 30.0 | 1.000 | 89.62 | 334.4 |

## Automatic findings

- Numerical completion: 400/400 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.690 at N=4000 to 0.530 at N=8000; time 8.45s to 23.56s; final conditional-log-loss gap to oracle +0.0020.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.090 at N=4000 to 0.140 at N=8000; time 44.24s to 89.62s; final conditional-log-loss gap to oracle +0.0074.

## Settings

```json
{
  "backend": "mcem",
  "C": 1,
  "max_iter": 30,
  "optimizer_maxiter": 100,
  "optimizer_tol": 1e-06,
  "tol": 0.005,
  "convergence_window": 7,
  "convergence_mean_tol": 0.01,
  "burn": 5,
  "tau": 0.3,
  "penalty_n_power": 0.0,
  "penalty_reference_n": 500.0,
  "penalty_min_strength": 1.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
