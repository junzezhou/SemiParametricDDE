# Bernoulli fixed-DDE paper-aligned recovery report

## Experiment

- Family and path: Bernoulli, fixed-width DDE, data initialization.
- Replicates: 100 per scenario/sample-size combination; all 1,200 fits succeeded.
- Sample sizes: 500, 1,000, 2,000, 4,000, 8,000, and 16,000.
- D=2: paper model `B_s`, `(J,K1,K2)=(18,6,2)`.
- D=3: recursive `B_s` extension, `(J,K1,K2,K3)=(54,18,6,2)`.
- Hyperparameters: `lambda=N**0.25` at every layer,
  `tau=max(3*N**-0.3,0.3)`, `C=1`, `tol=K_D/2`, `max_iter=50`.
- Evaluation: complete coefficient matrices were thresholded as in the MATLAB
  reporting script, followed by bottom-up squared-L2 Hungarian alignment.
- Paper-style RMSE is `sqrt(mean(MSE))` after MATLAB-default-style median/MAD
  outlier removal. The unfiltered RMSE remains available in `summary.csv`.

The paper reports Bernoulli results only for D=2 and N up to 8,000. D=3 and
N=16,000 are new extension results and have no paper reference value.

## D=2 comparison with paper

| N | Python Acc(G) | Paper Acc(G) | Delta | Python RMSE | Paper RMSE | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.9243 | 0.899 | +0.0253 | 0.5172 | 0.498 | +0.0192 |
| 1,000 | 0.9417 | 0.931 | +0.0107 | 0.4479 | 0.431 | +0.0169 |
| 2,000 | 0.9357 | 0.956 | -0.0203 | 0.4152 | 0.386 | +0.0292 |
| 4,000 | 0.9477 | 0.974 | -0.0263 | 0.3983 | 0.359 | +0.0393 |
| 8,000 | 0.9545 | 0.985 | -0.0305 | 0.3923 | 0.341 | +0.0513 |
| 16,000 | 0.9566 | not reported | — | 0.3884 | not reported | — |

D=2 is numerically stable and ordinarily stops in 3–5 iterations. Only one
N=16,000 replicate reached the 50-iteration limit. The corrected experiment
therefore resolves the severe failures caused by the earlier simulation
protocol, but it does not reproduce the paper's large-N recovery rate. At
N=8,000, the remaining differences are -0.0305 in graph accuracy and +0.0513
in RMSE. Sensitivity rises from 0.898 at N=500 to approximately 0.985 at large
N, while specificity plateaus near 0.946. Residual false positives are the
main reason graph accuracy does not approach the paper's 0.985.

## D=3 extension

| N | Overall Acc(G) | Layer Acc(G), B1/B2/B3 | RMSE | Mean iterations | At 50-iteration limit |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.8896 | 0.9049 / 0.7719 / 0.7075 | 0.5496 | 6.71 | 0% |
| 1,000 | 0.9170 | 0.9357 / 0.7700 / 0.7258 | 0.5050 | 10.80 | 3% |
| 2,000 | 0.9332 | 0.9525 / 0.7799 / 0.7458 | 0.5143 | 41.13 | 70% |
| 4,000 | 0.9423 | 0.9636 / 0.7778 / 0.6933 | 0.5295 | 50.00 | 100% |
| 8,000 | 0.9415 | 0.9620 / 0.7865 / 0.6742 | 0.5419 | 50.00 | 100% |
| 16,000 | 0.9383 | 0.9604 / 0.7698 / 0.6683 | 0.5469 | 50.00 | 100% |

The first layer improves with N, but the second and third layers remain much
less accurate. From N=4,000 onward every fit reaches the iteration cap, while
RMSE increases rather than decreases. Consequently, the large-N D=3 rows are
valid records of the current implementation under the requested settings but
must not be interpreted as converged statistical estimates. They identify a
D=3 SAEM convergence/deep-layer problem that requires kernel-level diagnosis.

## Main conclusion

Correcting the simulator and evaluation protocol explains most of the earlier
D=2 failure, especially at small N. It does not fully explain the discrepancy:
the Python D=2 path still underperforms the paper increasingly at large N.
Because D=2 converges quickly, its residual gap points primarily to numerical
implementation differences such as the penalized M-step optimizer and
SVD/varimax initialization. D=3 exhibits a separate and stronger convergence
problem and should be investigated using per-iteration traces before changing
its statistical acceptance criteria.
