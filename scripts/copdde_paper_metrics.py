"""Paper-style CopDDE evaluation, ported from the MATLAB simulation scripts.

``paper_graph_accuracy`` reproduces the graph-accuracy computation of
Sim_Overspecified.m (D=2) and Deeper_Sims.m (D=3): fitted columns are aligned to
the true ones by Hungarian assignment on the summed fitted coefficients over each
true column's positive support (format_estimates.m), layer by layer, carrying the
layer-d permutation into the rows of B_{d+1}; the support (intercept column
included) is then compared with the truth padded to the maximal widths.  Graph
accuracy depends only on alignment and support, so the rescaling step of
format_estimates.m is not needed.

``load_truth`` reads the truth saved by Sim_CopDDE_baseline.m (settings/<scenario>.mat).
"""

from __future__ import annotations

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment


def load_truth(settings_path):
    scn = sio.loadmat(settings_path, squeeze_me=True, struct_as_record=False)["scn"]
    D = int(scn.D)
    J = int(scn.J)
    K_true = np.atleast_1d(np.asarray(scn.K_true, dtype=float)).astype(int)
    K_max = np.atleast_1d(np.asarray(scn.K_max, dtype=float)).astype(int)
    truth = {
        "id": str(scn.id), "D": D, "J": J, "K_true": K_true, "K_max": K_max,
        "t_spike": np.atleast_1d(np.asarray(scn.t_spike, dtype=float)), "temp0": float(scn.temp0),
        "lambdas": np.atleast_1d(np.asarray(scn.lambdas, dtype=float)),
        "prop_true": np.atleast_1d(np.asarray(scn.prop_true, dtype=float)),
    }
    if D == 2:
        truth["gen_B"] = [np.asarray(scn.B1_true_scale, dtype=float), np.asarray(scn.B2_true, dtype=float)]
        truth["gen_gamma"] = np.atleast_1d(np.asarray(scn.gamma_scale, dtype=float))
        align = [np.asarray(scn.B1_true_aug_unscale, dtype=float), np.asarray(scn.B2_true_aug, dtype=float)]
        support = [np.asarray(scn.B1_true_aug_scale, dtype=float) != 0, np.asarray(scn.B2_true_aug, dtype=float) != 0]
    else:
        cells = [np.asarray(b, dtype=float) for b in scn.B_cell_true]
        truth["gen_B"] = cells
        truth["gen_gamma"] = np.atleast_1d(np.asarray(scn.gamma_true, dtype=float))
        B1_aug = np.column_stack([cells[0], np.zeros((J, K_max[0] - K_true[0]))])
        align = [B1_aug, np.asarray(scn.B2_true_aug, dtype=float), np.asarray(scn.B3_true_aug, dtype=float)]
        support = [b != 0 for b in align]
    truth["align"] = align
    truth["support"] = support
    paper = scn.paper
    truth["paper"] = {"N": np.atleast_1d(paper.N).astype(int), "K": np.atleast_2d(paper.K), "G": np.atleast_2d(paper.G)}
    return truth


def _assign(B_true_slopes, B_fit_slopes, n_true=None):
    K = B_fit_slopes.shape[1]
    cost = np.zeros((K, K))
    rows = B_true_slopes.shape[1] if n_true is None else n_true
    for k in range(rows):
        cost[k] = -np.sum(B_fit_slopes[B_true_slopes[:, k] > 0], axis=0)
    _, cols = linear_sum_assignment(cost)
    return cols


def paper_graph_accuracy(truth, B_fit):
    """Per-layer graph accuracy exactly as the MATLAB simulation scripts compute it."""
    B_fit = [np.asarray(b, dtype=float) for b in B_fit]
    align = truth["align"]
    est = []
    a1 = _assign(align[0][:, 1:], B_fit[0][:, 1:])
    est.append(np.column_stack([B_fit[0][:, 0], B_fit[0][:, 1:][:, a1]]))
    B2_rows = B_fit[1][a1, :]
    a2 = _assign(align[1][:, 1:], B2_rows[:, 1:])
    est.append(np.column_stack([B2_rows[:, 0], B2_rows[:, 1:][:, a2]]))
    if truth["D"] == 3:
        B3_rows = B_fit[2][a2, :]
        a3 = _assign(align[2][:, 1:], B3_rows[:, 1:], n_true=int(truth["K_true"][2]))
        est.append(np.column_stack([B3_rows[:, 0], B3_rows[:, 1:][:, a3]]))
    return [float(np.mean(truth["support"][d] == (est[d] != 0))) for d in range(truth["D"])]
