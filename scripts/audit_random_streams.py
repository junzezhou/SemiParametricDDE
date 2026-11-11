"""Which seed reproduces each fit path?  Run twice per mode in fresh processes.

    PYTHONPATH=src python scripts/audit_random_streams.py MODE OUT.json
MODE: "global" (np.random.seed only), "state" (random_state only), "both".
Prints/saves a hash of every estimate (or the error); compare two runs.
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit

warnings.simplefilter("ignore")
mode, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dde import fit, select_hyperparameters  # noqa: E402
from dde.identifiability import build_true_params  # noqa: E402
from dde.simulation import generate_X_normal  # noqa: E402


def bs_data(family, N=400, seed=1):
    rng = np.random.default_rng(seed)
    K = 6
    slopes = np.zeros((18, 6))
    for k in range(6):
        slopes[k, k] = slopes[k + 6, k] = 4.0
    for k in range(3):
        slopes[k + 12, k] = 4.0; slopes[k + 15, k] = -4 / 3; slopes[k + 12, k + 3] = -4 / 3; slopes[k + 15, k + 3] = 4.0
    B1 = np.column_stack([np.r_[np.full(6, -2.0), np.full(6, -4.0), np.full(6, -2.0)], slopes])
    A2 = rng.binomial(1, 0.5, size=(N, 2))
    B2 = np.array([[-2, 4, 0], [-2, 0, 4], [-4, 4, 0], [-4, 0, 4], [-2, 4, -4 / 3], [-2, -4 / 3, 4]], float)
    A1 = rng.binomial(1, expit(B2[:, 0] + A2 @ B2[:, 1:].T))
    eta = B1[:, 0] + A1 @ B1[:, 1:].T
    G = [(B1[:, 1:] != 0).astype(int), (B2[:, 1:] != 0).astype(int)]
    if family == "bernoulli":
        return rng.binomial(1, expit(eta)), G
    if family == "poisson":
        return rng.poisson(np.exp(eta)), G
    return eta + rng.normal(size=eta.shape), G


def copdde_data(N=400):
    from copdde_paper_metrics import load_truth
    from dde.simulation.generate_copula_D import generate_X_Cop_marg_emp_D

    root = Path(__file__).resolve().parents[2]
    truth = load_truth(root / "DDE-Copula-m/Results/copdde_baseline/settings/d2_j100.mat")
    return generate_X_Cop_marg_emp_D(N, truth["prop_true"], truth["lambdas"], truth["gen_gamma"],
                                     truth["gen_B"], random_state=3)["X"]


Xn, G = bs_data("normal")
Xb, _ = bs_data("bernoulli")
Xp, _ = bs_data("poisson")
Xc = copdde_data()
fixed = dict(dimension_method="fixed", K_list=[6, 2], pen_list=[1.0, 1.0], tau=0.3, C=1, max_it=8)
csp = dict(dimension_method="csp", D=2, t_spike=[0.01, 0.03], temp=0.9, tau=0.3, C=1, max_it=8)
cases = {
    "DDE Normal fixed": lambda **s: fit(Xn, model="DDE", family="Normal", D=2, **fixed, **s),
    "DDE Bernoulli fixed": lambda **s: fit(Xb, model="DDE", family="Bernoulli", D=2, **fixed, **s),
    "DDE Poisson fixed": lambda **s: fit(Xp, model="DDE", family="Poisson", D=2, **fixed, **s),
    "DDE Normal fixed + G_list": lambda **s: fit(Xn, model="DDE", family="Normal", G_list=G, dimension_method="fixed",
                                                 pen_list=[1.0, 1.0], tau=0.3, max_it=8, **s),
    "DDE Normal CSP": lambda **s: fit(Xn, model="DDE", family="Normal", **csp, **s),
    "DDE Bernoulli CSP": lambda **s: fit(Xb, model="DDE", family="Bernoulli", **csp, **s),
    "DDE Poisson CSP": lambda **s: fit(Xp, model="DDE", family="Poisson", **csp, **s),
    "SemiDDE fixed": lambda **s: fit(np.exp(Xn), model="SemiDDE", D=2, **fixed, **s),
    "SemiDDE CSP": lambda **s: fit(np.exp(Xn), model="SemiDDE", dimension_method="csp", D=2, t_spike=[0.04, 0.08],
                                   temp=0.8, tau=0.3, C=1, max_it=8, **s),
    "CopDDE fixed": lambda **s: fit(Xc, model="CopDDE", dimension_method="fixed", K_list=[10, 3], pen_list=[1.0, 1.0],
                                    tau=0.3, C=1, tol=0.01, max_it=4, **s),
    "CopDDE CSP": lambda **s: fit(Xc, model="CopDDE", dimension_method="csp", K_list=[12, 4], t_spike=[0.05, 0.1],
                                  temp=0.7, tau=0.3, C=1, max_it=4, **s),
    "select_hyperparameters DDE fixed": lambda **s: select_hyperparameters(
        Xn, model="DDE", family="Normal", D=2, dimension_method="fixed", K_list=[6, 2], criterion="ebic",
        candidates=[{"pen_list": [1.0, 1.0], "tau": 0.3}, {"pen_list": [2.0, 2.0], "tau": 0.3}], max_it=4, **s),
}
results = {}
for name, run in cases.items():
    if mode == "global":
        np.random.seed(11); kwargs = {}
    elif mode == "state":
        kwargs = {"random_state": 11}
    else:
        np.random.seed(11); kwargs = {"random_state": 11}
    try:
        res = run(verbose=False, **kwargs)
        h = hashlib.md5()
        for B in res.B_list:
            h.update(np.ascontiguousarray(B, dtype=float).tobytes())
        results[name] = h.hexdigest()[:12]
    except Exception as exc:  # noqa: BLE001
        results[name] = f"error: {type(exc).__name__}: {str(exc)[:90]}"
Path(out).write_text(json.dumps(results, indent=1))
