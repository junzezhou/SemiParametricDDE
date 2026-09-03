# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 1.000 | 0.000 | 0.000 | 0.848 | 0.879 | 50.0 | 1.000 | 1.22 | 181.0 |
| D2_Bs_overspecified_J27 | 1000 | 5/5 | 0.600 | 0.400 | 0.400 | 0.940 | 0.856 | 50.0 | 1.000 | 2.30 | 192.7 |
| D2_Bs_overspecified_J27 | 2000 | 5/5 | 0.800 | 0.200 | 0.200 | 0.974 | 0.873 | 50.0 | 1.000 | 4.65 | 198.3 |
| D2_Bs_overspecified_J27 | 4000 | 5/5 | 0.600 | 0.400 | 0.400 | 0.968 | 0.864 | 50.0 | 1.000 | 8.59 | 213.0 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.600 | 0.400 | 0.400 | 0.972 | 0.862 | 50.0 | 1.000 | 15.54 | 229.8 |
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.000 | 3.400 | 1.000 | 0.911 | 0.740 | 50.0 | 1.000 | 4.03 | 197.0 |
| D3_Bs_overspecified_J64 | 1000 | 5/5 | 0.000 | 2.800 | 1.000 | 0.952 | 0.768 | 50.0 | 1.000 | 7.03 | 204.1 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.000 | 4.200 | 1.000 | 0.958 | 0.755 | 50.0 | 1.000 | 12.09 | 217.4 |
| D3_Bs_overspecified_J64 | 4000 | 5/5 | 0.000 | 3.600 | 1.000 | 0.963 | 0.774 | 50.0 | 1.000 | 22.41 | 244.9 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.000 | 3.600 | 1.000 | 0.965 | 0.772 | 50.0 | 1.000 | 40.49 | 269.2 |

## Automatic findings

- Numerical completion: 50/50 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 1.000 at N=500 to 0.600 at N=8000; time 1.22s to 15.54s; final conditional-log-loss gap to oracle +0.0107.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.000 at N=8000; time 4.03s to 40.49s; final conditional-log-loss gap to oracle +0.0226.

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
  "penalty_n_power": 1.0,
  "penalty_reference_n": 500.0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
