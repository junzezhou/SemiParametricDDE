# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Bs_overspecified_J64 | 500 | 1/1 | 0.000 | 4.000 | 1.000 | 0.911 | 0.721 | 50.0 | 1.000 | 2.76 | 184.4 |

## Automatic findings

- Numerical completion: 1/1 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.000 at N=500; time 2.76s to 2.76s; final conditional-log-loss gap to oracle +0.0143.

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
  "t_spike_multiplier": 2.0,
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
