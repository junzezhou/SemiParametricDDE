# dde

`dde` is a Python package for Deep Discrete Encoder models, including
parametric Normal DDE estimation,
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
    random_state=123,
)
```
