# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 5/5 | 1.000 | 0.000 | 0.000 | 0.818 | 0.856 | 50.0 | 1.000 | 1.08 | 185.9 |
| D2_Bs_overspecified_J27 | 1000 | 5/5 | 0.600 | 0.800 | 0.400 | 0.852 | 0.815 | 50.0 | 1.000 | 2.02 | 196.8 |
| D2_Bs_overspecified_J27 | 2000 | 5/5 | 1.000 | 0.000 | 0.000 | 0.883 | 0.809 | 50.0 | 1.000 | 4.32 | 204.2 |
| D2_Bs_overspecified_J27 | 4000 | 5/5 | 0.600 | 0.400 | 0.200 | 0.917 | 0.817 | 50.0 | 1.000 | 8.24 | 219.9 |
| D2_Bs_overspecified_J27 | 8000 | 5/5 | 0.600 | 0.400 | 0.000 | 0.943 | 0.861 | 50.0 | 1.000 | 15.34 | 246.7 |
| D3_Bs_overspecified_J64 | 500 | 5/5 | 0.000 | 4.800 | 1.000 | 0.901 | 0.707 | 50.0 | 1.000 | 3.51 | 196.6 |
| D3_Bs_overspecified_J64 | 1000 | 5/5 | 0.000 | 4.000 | 1.000 | 0.901 | 0.717 | 50.0 | 1.000 | 6.58 | 199.0 |
| D3_Bs_overspecified_J64 | 2000 | 5/5 | 0.200 | 2.200 | 0.800 | 0.912 | 0.727 | 50.0 | 1.000 | 11.44 | 224.2 |
| D3_Bs_overspecified_J64 | 4000 | 5/5 | 0.200 | 1.800 | 0.600 | 0.929 | 0.747 | 50.0 | 1.000 | 21.48 | 242.6 |
| D3_Bs_overspecified_J64 | 8000 | 5/5 | 0.200 | 1.800 | 0.600 | 0.929 | 0.740 | 50.0 | 1.000 | 39.66 | 274.0 |

## Automatic findings

- Numerical completion: 50/50 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 1.000 at N=500 to 0.600 at N=8000; time 1.08s to 15.34s; final conditional-log-loss gap to oracle +0.0004.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.200 at N=8000; time 3.51s to 39.66s; final conditional-log-loss gap to oracle +0.0082.

## Settings

```json
{
  "backend": "mcem_cap_only",
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
