# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 0.800 | 0.200 | 0.000 | 0.856 | 0.881 | 50.0 | 1.000 | 1.30 | 190.8 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.600 | 1.000 | 0.000 | 0.964 | 0.874 | 50.0 | 1.000 | 15.37 | 244.4 |
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.200 | 2.000 | 0.800 | 0.916 | 0.759 | 50.0 | 1.000 | 3.82 | 204.2 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.200 | 0.800 | 0.200 | 0.942 | 0.776 | 50.0 | 1.000 | 13.91 | 238.7 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.200 | 1.800 | 0.200 | 0.961 | 0.811 | 50.0 | 1.000 | 49.28 | 333.3 |

## Automatic findings

- Numerical completion: 25/25 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.800 at N=500 to 0.600 at N=8000; time 1.30s to 15.37s; final conditional-log-loss gap to oracle +0.0007.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.200 at N=500 to 0.200 at N=8000; time 3.82s to 49.28s; final conditional-log-loss gap to oracle +0.0005.

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
  "penalty_n_power": 0.0,
  "penalty_reference_n": 500.0,
  "shrinkage_burn": 10,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
