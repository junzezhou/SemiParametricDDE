# dde

`dde` is a Python package for Deep Discrete Encoder models, including
parametric Normal, Bernoulli, and Poisson DDE estimation,
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
child-implied structural cap. Bernoulli CSP selects ``"mcem"`` by default when
``csp_backend`` is omitted; the staged backends remain explicit options.

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

For a lightweight fixed-width convergence trace, pass `keep_trace=True`.
The returned `result.diagnostics["trace"]` records per-sweep update RMSEs,
E/M-step timings, and accumulator sizes. It does not retain coefficient
matrices or latent draws, so tracing does not scale with parameter-history
storage.

Bernoulli CSP uses the simultaneous MCEM path with a persistent systematic
coordinate-Gibbs transition: its
`C` draws are consecutive sweeps, and the last draw becomes the next MCEM
state.  The transition respects CSP stage caps and the current annealing
temperature.  For simultaneous MCEM with three or more layers, its automatic
defaults delay irreversible CSP shrinkage for 10 iterations when at least 20
outer iterations are requested, and scale the CSP penalty by
`max(1, (N / 1000) ** (2 / 3))`.  Supplying
`mcem_shrinkage_burn` or `mcem_penalty_strength` overrides the corresponding
automatic value.  Two-layer Bernoulli CSP and Normal CSP retain their
established settings; Normal also retains its synchronous approximate-MCEM
transition for numerical backward compatibility.

Poisson DDE uses a canonical log link, requires finite nonnegative integer
counts, and has no `gamma` dispersion parameter. It currently supports only
fixed-width estimation, with any positive number of latent layers:

```python
result = fit(
    Y_counts,
    method="DDE",
    family="Poisson",
    D=3,
    dimension_method="fixed",
    K_list=[9, 3, 1],
    pen_list=[0.1, 0.1, 0.1],
    tau=0.3,
    C=1,
    tol=0.005,
)
```
