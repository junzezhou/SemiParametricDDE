# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 0.800 | 0.200 | 0.000 | 0.856 | 0.881 | 50.0 | 1.000 | 1.28 | 189.9 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.600 | 0.600 | 0.000 | 0.976 | 0.875 | 50.0 | 1.000 | 14.94 | 245.6 |
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.200 | 2.000 | 0.800 | 0.916 | 0.759 | 50.0 | 1.000 | 3.88 | 198.0 |
| D3_Bs_overspecified_J64 | 1000 | 5/5 | 0.000 | 1.600 | 1.000 | 0.943 | 0.772 | 50.0 | 1.000 | 6.73 | 209.4 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.400 | 0.600 | 0.600 | 0.964 | 0.786 | 50.0 | 1.000 | 12.91 | 235.7 |
| D3_Bs_overspecified_J64 | 4000 | 5/5 | 0.200 | 1.400 | 0.600 | 0.969 | 0.806 | 50.0 | 1.000 | 23.73 | 273.4 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.600 | 0.800 | 0.400 | 0.968 | 0.809 | 50.0 | 1.000 | 46.63 | 320.0 |

## Automatic findings

- Numerical completion: 35/35 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.800 at N=500 to 0.600 at N=8000; time 1.28s to 14.94s; final conditional-log-loss gap to oracle +0.0009.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.200 at N=500 to 0.600 at N=8000; time 3.88s to 46.63s; final conditional-log-loss gap to oracle +0.0076.

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
  "penalty_reference_n": 500.0,
  "shrinkage_burn": 10,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
