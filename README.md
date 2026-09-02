# dde

`dde` is a Python package for Deep Discrete Encoder models, including
parametric Normal and Bernoulli DDE estimation,
semi-parametric DDE estimation for continuous observations and copula/rank
likelihood DDE estimation for mixed-scale observations.

Install in editable mode from this directory:

```bash
pip install -e .
```

Then import from any script or notebook:

```python
from dde import fit
```

A fixed-width Normal DDE supplies its latent widths and layer penalties.  A
CSP fit omits `K_list`; the package initializes the recursive maximal widths
`floor(J / 3), floor(floor(J / 3) / 3), ...` and returns the CSP-active widths
in `result.diagnostics["K_list"]`:

```python
result = fit(
    Y,
    method="DDE",
    family="Normal",
    D=3,
    dimension_method="csp",
    t_spike=[0.01, 0.03, 0.005],  # Laplace spike scales, not rates
    temp=0.9,
    tau=0.3,
    C=1,
    tol=0.005,
    max_it=50,
    csp_backend="mcem",
    random_state=123,
)
```

Parametric CSP provides three MCEM backends.  ``"mcem"`` retains the original
simultaneous MCEM updates, ``"mcem_reinitialize"`` activates CSP layer by
layer and rebuilds each waiting parent, and ``"mcem_cap_only"`` uses the same
staged schedule while preserving eligible parent states and only enforcing the
child-implied structural cap.

Bernoulli DDE uses the same fixed and CSP interfaces, requires a binary 0/1
matrix, and has no `gamma` dispersion parameter. For fixed-width parametric
DDEs, `tol` is the RMSE of the postprocessed parameter update and defaults to
`0.005`:

```python
result = fit(
    Y_binary,
    method="DDE",
    family="Bernoulli",
    D=2,
    dimension_method="fixed",
    K_list=[6, 2],
    pen_list=[0.1, 0.1],
    tau=0.3,
    C=1,
    tol=0.005,
)
```
