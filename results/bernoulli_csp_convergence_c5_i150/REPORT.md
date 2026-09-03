# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 1/1 | 0.000 | 1.000 | 0.000 | 0.600 | 0.663 | 150.0 | 1.000 | 8.91 | 198.1 |
| D3_Bs_overspecified_J64 | 500 | 1/1 | 0.000 | 11.000 | 1.000 | 0.803 | 0.602 | 150.0 | 1.000 | 24.25 | 216.4 |

## Automatic findings

- Numerical completion: 2/2 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.000 at N=500 to 0.000 at N=500; time 8.91s to 8.91s; final conditional-log-loss gap to oracle +0.0356.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.000 at N=500; time 24.25s to 24.25s; final conditional-log-loss gap to oracle +0.0873.

## Settings

```json
{
  "backend": "mcem",
  "C": 5,
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
