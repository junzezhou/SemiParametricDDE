# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 1/1 | 1.000 | 0.000 | 0.000 | 0.828 | 0.878 | 150.0 | 1.000 | 2.63 | 191.7 |
| D3_Bs_overspecified_J64 | 500 | 1/1 | 0.000 | 4.000 | 1.000 | 0.907 | 0.736 | 150.0 | 1.000 | 6.78 | 197.3 |

## Automatic findings

- Numerical completion: 2/2 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 1.000 at N=500 to 1.000 at N=500; time 2.63s to 2.63s; final conditional-log-loss gap to oracle -0.0009.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.000 at N=500; time 6.78s to 6.78s; final conditional-log-loss gap to oracle +0.0114.

## Settings

```json
{
  "backend": "mcem",
  "C": 1,
  "max_iter": 150,
  "optimizer_maxiter": 100,
  "optimizer_tol": 1e-06,
  "tol": 0.005,
  "convergence_window": 7,
  "convergence_mean_tol": 0.01,
  "burn": 5,
  "tau": 0.3,
  "base_seed": 20260902,
  "keep_trace": true
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
