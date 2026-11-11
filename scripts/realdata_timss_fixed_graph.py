"""TIMSS response times with the provisional Q-matrix fixed (paper Table S.19).

The DDE paper fits the TIMSS response times by a two-layer lognormal DDE with
``K_list = [7, 1]``, the graph ``G1`` fixed to the provisional Q-matrix and
``G2`` to a column of ones (the confirmatory analysis of Supplement S.5.1;
``TIMSS_analysis.m`` with ``get_EM_multimodal_confirm.m``).  Table S.19
averages the estimated latent variables (each layer's most probable state)
within the categories of the held-out survey item "Mathematics is one of my
favorite subjects".  ``TIMSS.csv`` stores the log response times, which the
paper's code models as Normal.

This script fits the same model with ``dde.fit(..., G_list=[Q, ones])``
(Normal DDE, exact EM on the fixed graph, ``--n-init`` starts) and reports
the table next to the paper's values, plus a SemiDDE fit of the same graph.
The paper's code additionally bounds the response-time slopes and the
layer-2 slopes below by zero; the package fit is unconstrained, and the
signs of its slopes are reported.  The likelihood is multimodal (nine local
optima in 50 starts), so the table depends on the start: when
``matlab_rt_confirm/timss_rt_matlab.mat`` (the paper's confirmatory EM with
the accuracy terms removed, its ``rng(5)`` start; ``run_timss_rt.m``) is
present, its solution is also evaluated and continued by the package EM.

    python scripts/realdata_timss_fixed_graph.py

Outputs under ``Results/realdata_timss_fixed_graph``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Deep-Discrete-Encoders" / "real data" / "TIMSS" / "dataset"
DEFAULT_OUT = REPO_ROOT / "Results" / "realdata_timss_fixed_graph"
SKILLS = ["Number", "Algebra", "Geometry", "DataProb", "Knowing", "Applying", "Reasoning"]
CATEGORIES = {1: "Agree a lot", 2: "Agree a little", 3: "Disagree a little", 4: "Disagree a lot"}
# Table S.19: A2_1, then A1 for Number, Algebra, Geometry, Data and Probability.
PAPER_TABLE_S19 = {
    1: [0.83, 0.72, 0.88, 0.86, 0.87],
    2: [0.80, 0.72, 0.85, 0.84, 0.85],
    3: [0.76, 0.74, 0.85, 0.84, 0.84],
    4: [0.69, 0.67, 0.76, 0.75, 0.78],
}


def load_timss():
    raw = pd.read_csv(DATA_DIR / "TIMSS.csv", header=None).to_numpy(float)
    q_raw = pd.read_csv(DATA_DIR / "Q.csv", header=None, encoding="utf-8-sig").to_numpy(float)
    survey = pd.read_csv(DATA_DIR / "survey_response.csv", header=None).to_numpy(float).ravel()
    return raw[:, 29:58], q_raw[:, 1:], survey


def survey_table(A1, A2, survey):
    rows = []
    for category, label in CATEGORIES.items():
        index = survey == category
        row = {"category": label, "n": int(index.sum()), "A2_1": float(A2[index, 0].mean())}
        for k, skill in enumerate(SKILLS):
            row[f"A1_{skill}"] = float(A1[index, k].mean())
        paper = PAPER_TABLE_S19[category]
        row["paper_A2_1"] = paper[0]
        for k, skill in enumerate(SKILLS[:4]):
            row[f"paper_A1_{skill}"] = paper[k + 1]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from dde import fit
    from dde.algorithms.dde import fit_fixed_graph_reference

    Y, Q, survey = load_timss()
    G_list = [Q, np.ones((7, 1))]
    summary = {"N": int(Y.shape[0]), "J": int(Y.shape[1]), "K_list": [7, 1]}
    tables = {}

    # The confirmatory fit is the unpenalized exact EM on the graph, the
    # package's reference implementation (not part of fit()).
    normal = fit_fixed_graph_reference(
        Y, family="normal", G_list=G_list, n_init=args.n_init,
        random_state=args.random_state, verbose=False,
    )
    tables["normal_dde_em"] = survey_table(normal.A_list[0], normal.A_list[1], survey)
    B1, B2 = normal.B_list
    summary["normal_dde_em"] = {
        "graph_algorithm": normal.diagnostics["graph_algorithm"],
        "iterations": normal.diagnostics["iterations"],
        "converged": normal.diagnostics["converged"],
        "loglik": normal.diagnostics["loglik"],
        "start_logliks": normal.diagnostics["start_logliks"],
        "best_start": normal.diagnostics["best_start"],
        "prop": normal.prop.tolist(),
        "negative_time_slopes": int(np.sum(B1[:, 1:][Q > 0] < 0)),
        "time_slopes_on_graph": int(np.sum(Q > 0)),
        "layer2_slopes": B2[:, 1].tolist(),
        "layer2_intercepts": B2[:, 0].tolist(),
        "skill_prevalence": normal.A_list[0].mean(axis=0).tolist(),
    }
    np.save(args.out / "normal_dde_B1.npy", B1)
    np.save(args.out / "normal_dde_B2.npy", B2)

    matlab_path = args.out / "matlab_rt_confirm" / "timss_rt_matlab.mat"
    if matlab_path.exists():
        import scipy.io

        from dde.algorithms.fixed_graph import fit_fixed_graph_em

        m = scipy.io.loadmat(matlab_path, squeeze_me=True)
        support = [Q > 0, np.ones((7, 1), dtype=bool)]
        B_m = [np.asarray(m["B1_time"], float), np.asarray(m["B2"], float).reshape(7, 2)]
        start = (np.atleast_1d(m["prop"]).astype(float), B_m, np.asarray(m["gamma"], float))
        at_matlab = fit_fixed_graph_em(Y, *start, support, family="normal", max_iter=0)
        continued = fit_fixed_graph_em(Y, *start, support, family="normal")
        tables["matlab_rng5"] = survey_table(at_matlab[3][0], at_matlab[3][1], survey)
        tables["matlab_rng5_continued"] = survey_table(continued[3][0], continued[3][1], survey)
        summary["matlab_rng5"] = {
            "matlab_iterations": int(m["itera"]),
            "loglik": at_matlab[5]["loglik"],
            "loglik_after_package_em": continued[5]["loglik"],
            "prop": float(m["prop"]),
            "layer2_slopes": B_m[1][:, 1].tolist(),
        }

    # Confirmatory (unpenalized) fit on the graph: no penalty and a negligible threshold.
    semi = fit(Y, model="SemiDDE", dimension_method="fixed", G_list=G_list,
               pen_list=[0.0, 0.0], tau=1e-8, C=1, tol=0.005, verbose=False)
    tables["semidde_saem"] = survey_table(semi.A_list[0], semi.A_list[1], survey)
    B1s, B2s = semi.B_list
    summary["semidde_saem"] = {
        "iterations": semi.diagnostics["iterations"],
        "g_iterations": semi.diagnostics.get("g_iterations"),
        "prop": np.asarray(semi.prop).tolist(),
        "negative_time_slopes": int(np.sum(B1s[:, 1:][Q > 0] < 0)),
        "layer2_slopes": B2s[:, 1].tolist(),
        "skill_prevalence": np.asarray(semi.A_list[0]).mean(axis=0).tolist(),
    }

    for name, table in tables.items():
        table.to_csv(args.out / f"table_s19_{name}.csv", index=False)
        print(f"\n{name}")
        print(table.round(2).to_string(index=False))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
