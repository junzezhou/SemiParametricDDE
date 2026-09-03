# Bernoulli CSP systematic test

Initial overspecified recovery and performance matrix.

| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_Bs_overspecified_J27 | 500 | 10/10 | 0.700 | 0.300 | 0.200 | 0.832 | 0.861 | 50.0 | 1.000 | 1.15 | 190.4 |
| D2_Bs_overspecified_J27 | 1000 | 10/10 | 0.800 | 0.300 | 0.000 | 0.883 | 0.867 | 50.0 | 1.000 | 2.16 | 196.9 |
| D2_Bs_overspecified_J27 | 2000 | 10/10 | 1.000 | 0.000 | 0.000 | 0.961 | 0.884 | 50.0 | 1.000 | 4.54 | 202.0 |
| D2_Bs_overspecified_J27 | 4000 | 10/10 | 0.800 | 0.200 | 0.000 | 0.965 | 0.876 | 50.0 | 1.000 | 8.44 | 223.1 |
| D2_Bs_overspecified_J27 | 8000 | 10/10 | 0.500 | 0.500 | 0.200 | 0.960 | 0.864 | 50.0 | 1.000 | 15.62 | 241.8 |
| D3_Bs_overspecified_J64 | 500 | 10/10 | 0.000 | 3.500 | 1.000 | 0.904 | 0.737 | 50.0 | 1.000 | 4.01 | 197.6 |
| D3_Bs_overspecified_J64 | 1000 | 10/10 | 0.000 | 2.300 | 1.000 | 0.921 | 0.769 | 50.0 | 1.000 | 7.22 | 205.9 |
| D3_Bs_overspecified_J64 | 2000 | 10/10 | 0.100 | 1.700 | 0.900 | 0.938 | 0.784 | 50.0 | 1.000 | 12.21 | 224.1 |
| D3_Bs_overspecified_J64 | 4000 | 10/10 | 0.200 | 1.300 | 0.800 | 0.954 | 0.786 | 50.0 | 1.000 | 22.74 | 253.7 |
| D3_Bs_overspecified_J64 | 8000 | 10/10 | 0.200 | 2.100 | 0.800 | 0.953 | 0.773 | 50.0 | 1.000 | 42.97 | 282.5 |

## Automatic findings

- Numerical completion: 100/100 fits succeeded.
- Mean grouped max-iteration rate: 1.000.
- D2_Bs_overspecified_J27: exact-width rate changes from 0.700 at N=500 to 0.500 at N=8000; time 1.15s to 15.62s; final conditional-log-loss gap to oracle +0.0044.
- D3_Bs_overspecified_J64: exact-width rate changes from 0.000 at N=500 to 0.200 at N=8000; time 4.01s to 42.97s; final conditional-log-loss gap to oracle +0.0102.

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
  "penalty_min_strength": 1.0,
  "shrinkage_burn": 0,
  "t_spike_multiplier": 1.0,
  "initialization": "data",
  "base_seed": 20261001,
  "keep_trace": false
}
```

Peak RSS is the operating-system process high-water mark. Each worker handles one fit, so rows are comparable across configurations.
