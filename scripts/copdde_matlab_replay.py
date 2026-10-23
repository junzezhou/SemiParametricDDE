"""Replay MATLAB CopDDE CSP snapshots against the Python port, step by step.

The MATLAB baseline (DDE-Copula-m/Scripts/Sim_CopDDE_baseline.m) stores, for a
few replicates, the full state of selected SAEM iterations of the D-layer
routine (get_SAEM_RL_CSP_D_trace.m).  Starting from each stored input state,
this script recomputes every deterministic sub-step with the Python port
(dde.algorithms.get_SAEM_RL_CSP_D) and compares it with MATLAB's output:

  estep     Bernoulli conditional probabilities (ber) for every layer
  cusp      CUSP allocation z (argmax) and the penalty matrix theta
  mstep1    Gaussian layer-1 objective at MATLAB's optimum under both F_1
            scalings (MATLAB code: gamma_j/temp; F_1's declared order: temp/gamma_j),
            and the Python L-BFGS-B optimum from MATLAB's start
  mstepd    logistic layers d >= 2: objective at MATLAB's optimum and Python's
  gamma     gamma update from MATLAB's layer-1 optimum
  gating    thresholding/gating of MATLAB's pre-threshold B under the MATLAB
            rule (layers 1..D-1) and the Python rule (first two layers)
  temp      temperature schedule

Usage: python scripts/copdde_matlab_replay.py --root <copdde_baseline dir> --out <report dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from dde.algorithms.csp import update_csp_layer
from dde.algorithms.get_SAEM_RL_CSP_D import (
    _bounded_minimize_value_grad,
    _gaussian_objective_value_grad_scaled,
    _logistic_objective_value_grad,
    _make_bounds,
    _threshold,
    _update_bernoulli_probabilities_D,
)


def load_mat(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def as_list(cell):
    """MATLAB cell array (squeezed) -> list of float arrays."""
    if isinstance(cell, np.ndarray) and cell.dtype == object:
        return [np.asarray(v, dtype=float) for v in cell]
    return [np.asarray(cell, dtype=float)]


def as_2d(x, n_rows=None):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None] if n_rows is not None and x.size == n_rows else x[None, :]
    return x


def snapshot_state(s, N):
    D = len(as_list(s.B))
    st = {
        "D": D,
        "A_old": [as_2d(a, N) for a in as_list(s.A_old)],
        "B": [as_2d(b) for b in as_list(s.B)],
        "Z": np.asarray(s.Z, dtype=float),
        "gamma": np.atleast_1d(np.asarray(s.gamma, dtype=float)),
        "prop": np.atleast_1d(np.asarray(s.prop, dtype=float)),
        "temp": float(s.temp),
        "omega": [np.atleast_1d(o) for o in as_list(s.omega)],
        "theta_slab": [np.atleast_1d(t) for t in as_list(s.theta_slab)],
        "num_act_in": np.atleast_1d(np.asarray(s.num_act, dtype=float)).astype(int),
        "ber": [as_2d(b, N) for b in as_list(s.ber)],
        "A_sample": [as_2d(a, N) for a in as_list(s.A_sample)],
        "Z_after": np.asarray(s.Z_after, dtype=float),
        "prob_mat": [as_2d(p) for p in as_list(s.prob_mat)],
        "z_cusp": [np.atleast_1d(z).astype(int) for z in as_list(s.z_cusp)],
        "theta_slab_out": [np.atleast_1d(t) for t in as_list(s.theta_slab_out)],
        "theta_out": [as_2d(t) for t in as_list(s.theta_out)],
        "prop_update": np.atleast_1d(np.asarray(s.prop_update, dtype=float)),
        "B_pre": [as_2d(b) for b in as_list(s.B_pre_threshold)],
        "fval": [np.atleast_1d(f) for f in as_list(s.fval)],
        "exitflag": [np.atleast_1d(f) for f in as_list(s.exitflag)],
        "gamma_update": np.atleast_1d(np.asarray(s.gamma_update, dtype=float)),
        "B_post": [as_2d(b) for b in as_list(s.B_post)],
        "num_act": np.atleast_1d(np.asarray(s.num_act, dtype=float)).astype(int),
        "temp_out": float(s.temp_out),
        "it": int(s.it),
    }
    # The snapshot stores num_act_in under 'num_act' at the start and the updated
    # value under the same name at the end; the end value wins in loadmat, so
    # recover the input widths from the CUSP inputs instead (number of columns
    # active after the previous iteration = columns of B with any nonzero entry
    # is not reliable); use the trace record of the previous iteration instead.
    return st


def cusp_probs(B_d, omega, spike, num_rows):
    """Allocation probabilities exactly as in update_csp_layer (deterministic part)."""
    from scipy.special import gammaln

    abs_sum = np.sum(np.abs(B_d[:, 1:]), axis=0)
    Kd = B_d.shape[1] - 1
    log_spike = -num_rows * np.log(2.0 * spike) - abs_sum / spike
    log_slab = np.log(5.0) - gammaln(1.0) - num_rows * np.log(2.0) + gammaln(1.0 + num_rows) \
        - (1.0 + num_rows) * np.log(5.0 + abs_sum)
    log_omega = np.log(np.maximum(omega, np.finfo(float).tiny))
    P = np.empty((Kd, Kd + 1))
    for h in range(Kd):
        lp = log_omega.copy()
        lp[: h + 1] += log_spike[h]
        lp[h + 1:] += log_slab[h]
        p = np.exp(lp - lp.max())
        P[h] = p / p.sum()
    return P


def replay_snapshot(st, meta, prev_num_act, maxiter=15):
    D = st["D"]
    N, J = st["Z"].shape
    K = [a.shape[1] for a in st["A_old"]]
    t_spike = np.atleast_1d(np.asarray(meta.t_spike, dtype=float))
    tau = float(meta.tau)
    temp = st["temp"]
    out = {"it": st["it"], "N": N, "D": D}

    # E-step
    ber_py = _update_bernoulli_probabilities_D(st["Z"], st["A_old"], st["B"], st["gamma"], st["prop"], temp)
    out["estep_max_abs"] = [float(np.max(np.abs(ber_py[d] - st["ber"][d]))) for d in range(D)]

    # CUSP (deterministic part: probabilities and argmax allocation)
    num_rows = [J] + [int(prev_num_act[d - 1]) for d in range(1, D)]
    out["cusp_prob_max_abs"] = []
    out["cusp_z_mismatch"] = []
    for d in range(D):
        P = cusp_probs(st["B"][d], st["omega"][d], t_spike[d], num_rows[d])
        out["cusp_prob_max_abs"].append(float(np.max(np.abs(P - st["prob_mat"][d]))))
        z_py = np.argmax(P, axis=1) + 1
        out["cusp_z_mismatch"].append(int(np.sum(z_py != st["z_cusp"][d])))
        # package update_csp_layer: same allocation (random draws excluded)
        z_pkg, _, _, _ = update_csp_layer(st["B"][d], st["omega"][d], t_spike[d], st["theta_slab"][d], num_rows[d])
        out.setdefault("cusp_pkg_z_mismatch", []).append(int(np.sum(np.asarray(z_pkg) != st["z_cusp"][d])))
    theta_chk = []
    for d in range(D):
        active = st["z_cusp"][d] > np.arange(1, K[d] + 1)
        col = np.where(active, st["theta_slab_out"][d], t_spike[d])
        theta_chk.append(float(np.max(np.abs(np.broadcast_to(col, st["theta_out"][d].shape) - st["theta_out"][d]))))
    out["theta_max_abs"] = theta_chk

    # M-step layer 1
    lb, ub = _make_bounds(K, D)
    Xd = np.column_stack([np.ones(N), st["A_sample"][0]])
    rows = []
    for j in range(J):
        y = st["Z_after"][:, j]
        w = np.concatenate(([0.0], 1.0 / st["theta_out"][0][j]))
        g = st["gamma"][j]
        scale_mat = g / temp      # what get_SAEM_RL_CSP_D.m computes (argument swap)
        scale_dec = temp / g      # F_1's declared order (Python port)
        x_m = st["B_pre"][0][j]
        f_mat_at_m, _ = _gaussian_objective_value_grad_scaled(x_m, Xd, y, w, scale_mat)
        f_dec_at_m, _ = _gaussian_objective_value_grad_scaled(x_m, Xd, y, w, scale_dec)
        x0 = st["B"][0][j]
        x_py15 = _bounded_minimize_value_grad(
            lambda x: _gaussian_objective_value_grad_scaled(x, Xd, y, w, scale_mat), x0, lb[0], ub[0], maxiter=maxiter)
        x_pyc = _bounded_minimize_value_grad(
            lambda x: _gaussian_objective_value_grad_scaled(x, Xd, y, w, scale_mat), x0, lb[0], ub[0], maxiter=2000)
        f_py15 = _gaussian_objective_value_grad_scaled(x_py15, Xd, y, w, scale_mat)[0]
        f_pyc = _gaussian_objective_value_grad_scaled(x_pyc, Xd, y, w, scale_mat)[0]
        x_dec = _bounded_minimize_value_grad(
            lambda x: _gaussian_objective_value_grad_scaled(x, Xd, y, w, scale_dec), x0, lb[0], ub[0], maxiter=maxiter)
        rows.append((st["fval"][0][j], f_mat_at_m, f_dec_at_m, f_py15, f_pyc,
                     np.max(np.abs(x_py15 - x_m)), np.max(np.abs(x_pyc - x_m)), np.max(np.abs(x_dec - x_m)),
                     np.sum(np.abs(x_m[1:]) >= tau), np.sum(np.abs(x_py15[1:]) >= tau),
                     np.sum(np.abs(x_pyc[1:]) >= tau), np.sum(np.abs(x_dec[1:]) >= tau)))
    r = np.array(rows, dtype=float)
    fval = np.maximum(np.abs(r[:, 0]), 1e-12)
    out["mstep1"] = {
        "fval_rel_err_matlab_scaling": float(np.max(np.abs(r[:, 1] - r[:, 0]) / fval)),
        "fval_rel_err_declared_scaling": float(np.max(np.abs(r[:, 2] - r[:, 0]) / fval)),
        "py15_minus_matlab_obj_mean": float(np.mean(r[:, 3] - r[:, 0])),
        "pyconv_minus_matlab_obj_mean": float(np.mean(r[:, 4] - r[:, 0])),
        "matlab_obj_gap_to_converged_median_rel": float(np.median((r[:, 0] - r[:, 4]) / fval)),
        "coef_max_abs_py15": float(np.max(r[:, 5])),
        "coef_max_abs_pyconv": float(np.max(r[:, 6])),
        "coef_max_abs_declared_scaling": float(np.max(r[:, 7])),
        "support_matlab": int(r[:, 8].sum()),
        "support_py15": int(r[:, 9].sum()),
        "support_pyconv": int(r[:, 10].sum()),
        "support_declared_scaling": int(r[:, 11].sum()),
        "matlab_exitflag0_rows": int(np.sum(st["exitflag"][0] == 0)),
    }

    # M-step logistic layers
    out["mstepd"] = []
    for d in range(1, D):
        Xd = np.column_stack([np.ones(N), st["A_sample"][d]])
        rows = []
        for k in range(K[d - 1]):
            y = st["A_sample"][d - 1][:, k]
            w = np.concatenate(([0.0], 1.0 / st["theta_out"][d][k]))
            x_m = st["B_pre"][d][k]
            f_at_m = _logistic_objective_value_grad(x_m, Xd, y, w, temp)[0]
            x_py15 = _bounded_minimize_value_grad(
                lambda x: _logistic_objective_value_grad(x, Xd, y, w, temp), st["B"][d][k], lb[d], ub[d], maxiter=maxiter)
            x_pyc = _bounded_minimize_value_grad(
                lambda x: _logistic_objective_value_grad(x, Xd, y, w, temp), st["B"][d][k], lb[d], ub[d], maxiter=2000)
            rows.append((st["fval"][d][k], f_at_m,
                         _logistic_objective_value_grad(x_py15, Xd, y, w, temp)[0],
                         _logistic_objective_value_grad(x_pyc, Xd, y, w, temp)[0],
                         np.max(np.abs(x_py15 - x_m)), np.max(np.abs(x_pyc - x_m))))
        r = np.array(rows, dtype=float)
        fval = np.maximum(np.abs(r[:, 0]), 1e-12)
        out["mstepd"].append({
            "layer": d + 1,
            "fval_rel_err": float(np.max(np.abs(r[:, 1] - r[:, 0]) / fval)),
            "py15_minus_matlab_obj_mean": float(np.mean(r[:, 2] - r[:, 0])),
            "pyconv_minus_matlab_obj_mean": float(np.mean(r[:, 3] - r[:, 0])),
            "matlab_obj_gap_to_converged_median_rel": float(np.median((r[:, 0] - r[:, 3]) / fval)),
            "coef_max_abs_py15": float(np.max(r[:, 4])),
            "coef_max_abs_pyconv": float(np.max(r[:, 5])),
            "matlab_exitflag0_rows": int(np.sum(st["exitflag"][d] == 0)),
        })

    # gamma update from MATLAB's pre-threshold layer-1 optimum
    Xd = np.column_stack([np.ones(N), st["A_sample"][0]])
    sse = np.sum((st["Z_after"] - Xd @ st["B_pre"][0].T) ** 2, axis=0)
    gamma_py = (1.0 + 0.5 * sse) / (1.0 + 0.5 * N + 1.0)
    out["gamma_max_rel"] = float(np.max(np.abs(gamma_py - st["gamma_update"]) / st["gamma_update"]))

    # thresholding / gating
    inactive = [np.where(st["z_cusp"][d] <= np.arange(1, K[d] + 1))[0] for d in range(D)]
    for rule, layers in (("matlab_rule", range(D - 1)), ("python_rule", range(min(2, D)))):
        Bt = [b.copy() for b in st["B_pre"]]
        for d in layers:
            Bt[d][:, 1:] = _threshold(Bt[d][:, 1:], tau)
        for d in range(D - 1):
            Bt[d + 1][inactive[d], :] = 0.0
        out[f"gating_{rule}_max_abs"] = [float(np.max(np.abs(Bt[d] - st["B_post"][d]))) for d in range(D)]

    # temperature schedule
    it = st["it"]
    t_next = min(1.0, temp + 0.01 * ((it - 5) - 1)) if it > 5 else temp
    out["temp_abs"] = abs(t_next - st["temp_out"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(args.root / "fits" / "*" / "csp_D" / "N*" / "rep*.mat")))
    results = []
    for path in files:
        scenario = Path(path).parts[-4]
        if args.scenarios and scenario not in args.scenarios:
            continue
        m = load_mat(path)
        fit = m["fit"]
        if "snapshots" not in fit._fieldnames:
            continue
        meta = fit.meta
        iters = np.atleast_1d(fit.trace.iter)
        num_act_by_it = {int(r.it): np.atleast_1d(np.asarray(r.num_act, dtype=float)).astype(int) for r in iters}
        K_max = np.atleast_1d(np.asarray(meta.K_max, dtype=float)).astype(int)
        N = int(meta.N)
        for s in np.atleast_1d(fit.snapshots):
            st = snapshot_state(s, N)
            prev = num_act_by_it.get(st["it"] - 1, K_max)
            res = replay_snapshot(st, meta, prev)
            res.update(scenario=scenario, rep=int(meta.rep), file=os.path.relpath(path, args.root))
            results.append(res)
            print(json.dumps({k: res[k] for k in ("scenario", "N", "rep", "it", "estep_max_abs", "cusp_z_mismatch",
                                                    "theta_max_abs", "gamma_max_rel", "temp_abs")}))
            print("   mstep1:", json.dumps(res["mstep1"]))
            for md in res["mstepd"]:
                print("   mstepd:", json.dumps(md))
            print("   gating matlab/python:", res["gating_matlab_rule_max_abs"], res["gating_python_rule_max_abs"])
        if args.max_files and len({r["file"] for r in results}) >= args.max_files:
            break
    (args.out / "replay_results.json").write_text(json.dumps(results, indent=1))
    print("wrote", args.out / "replay_results.json", len(results), "snapshots")


if __name__ == "__main__":
    main()
