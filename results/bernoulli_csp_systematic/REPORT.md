# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 30/30 | 0.767 | 0.267 | 0.233 | 0.828 | 0.856 | 50.0 | 1.000 | 1.15 | 191.2 |
| D2_Bs_overspecified_J27 | 1000 | 30/30 | 0.800 | 0.200 | 0.100 | 0.884 | 0.869 | 50.0 | 1.000 | 2.14 | 193.8 |
| D2_Bs_overspecified_J27 | 2000 | 30/30 | 0.800 | 0.233 | 0.067 | 0.942 | 0.874 | 50.0 | 1.000 | 4.81 | 197.7 |
| D2_Bs_overspecified_J27 | 4000 | 30/30 | 0.733 | 0.400 | 0.033 | 0.968 | 0.872 | 50.0 | 1.000 | 8.36 | 214.8 |
| D2_Bs_overspecified_J27 | 8000 | 30/30 | 0.500 | 0.600 | 0.100 | 0.956 | 0.863 | 50.0 | 1.000 | 14.58 | 246.7 |
| D3_Bs_overspecified_J64 | 500 | 30/30 | 0.000 | 3.367 | 1.000 | 0.912 | 0.742 | 50.0 | 1.000 | 4.32 | 197.6 |
| D3_Bs_overspecified_J64 | 1000 | 30/30 | 0.067 | 2.233 | 0.933 | 0.922 | 0.761 | 50.0 | 1.000 | 6.93 | 206.4 |
| D3_Bs_overspecified_J64 | 2000 | 30/30 | 0.200 | 1.567 | 0.767 | 0.940 | 0.777 | 50.0 | 1.000 | 11.77 | 223.7 |
| D3_Bs_overspecified_J64 | 4000 | 30/30 | 0.067 | 2.100 | 0.933 | 0.951 | 0.776 | 50.0 | 1.000 | 21.19 | 252.4 |
| D3_Bs_overspecified_J64 | 8000 | 30/30 | 0.133 | 1.667 | 0.767 | 0.954 | 0.776 | 50.0 | 1.000 | 40.07 | 286.4 |

## Automatic findings

- Numerical completion: 300/300 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.767 at N=500 to 0.500 at N=8000; time 1.15s to 14.58s; final conditional-log-loss gap to oracle +0.0025.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.133 at N=8000; time 4.32s to 40.07s; final conditional-log-loss gap to oracle +0.0079.

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
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20260902,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
