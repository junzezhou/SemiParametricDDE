"""Score the MATLAB fixed-graph reference fits in the CopDDE paper's format.

Reads ``DDE-Copula-m/Results/copdde_fixed_graph_reference`` (written by
``Scripts/Sim_CopDDE_fixed_graph_reference.m``: the authors' ``Normal_init_G``
and ``get_SAEM_RL_G`` with the true graph, on the data of the MATLAB CSP
baseline), applies the alignment and metric of
``evaluate_copdde_fixed_graph.py`` (B^(1) standardized by the final ``Z``'s
empirical moments, padded to K_max) and pairs each fit with the baseline's
graph-estimated RL CSP fit of the same replicate.

    python scripts/score_copdde_fixed_graph_reference.py

Writes ``matlab_reference_rows.csv`` and ``matlab_reference_summary.csv``
under ``Results/copdde_fixed_graph_v1``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from copdde_paper_metrics import load_truth  # noqa: E402
from evaluate_copdde_fixed_graph import (  # noqa: E402
    BASELINE,
    BASELINE_ALGO,
    DEFAULT_OUT,
    ROOT,
    SETTINGS,
    align_to_truth,
    padded,
)

REFERENCE = ROOT / "DDE-Copula-m" / "Results" / "copdde_fixed_graph_reference" / "fits"


def baseline_rows(scenario):
    with open(BASELINE / f"summary_{scenario}_{BASELINE_ALGO[scenario]}.csv", newline="") as handle:
        return {
            (int(float(r["N"])), int(float(r["rep"]))): r
            for r in csv.DictReader(handle) if r["status"] == "ok"
        }


def main():
    rows = []
    for scenario_dir in sorted(REFERENCE.glob("d2_*")):
        scenario = scenario_dir.name
        truth = load_truth(SETTINGS / f"{scenario}.mat")
        J = truth["J"]
        K_max = truth["K_max"]
        T_list = [truth["gen_B"][0], truth["gen_B"][1]]
        G_list = [B[:, 1:] != 0 for B in T_list]
        shapes = [(J, K_max[0] + 1), (K_max[0], K_max[1] + 1)]
        base = baseline_rows(scenario)
        for path in sorted(scenario_dir.glob("N*/rep*.mat")):
            fit = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["fit"]
            N, rep = int(fit.N), int(fit.rep)
            row = {"scenario": scenario, "N": N, "rep": rep, "status": str(fit.status),
                   "time_sec": float(fit.time_sec)}
            if row["status"] == "ok":
                B_aligned, _, maps = align_to_truth(
                    [np.atleast_2d(fit.B1), np.atleast_2d(fit.B2)], np.atleast_1d(fit.prop), T_list, G_list
                )
                mu, sd = np.atleast_1d(fit.mu_Z), np.atleast_1d(fit.sd_Z)
                B_aligned[0][:, 0] = (B_aligned[0][:, 0] - mu) / sd
                B_aligned[0][:, 1:] = B_aligned[0][:, 1:] / sd[:, None]
                for d in range(2):
                    row[f"mse_B{d + 1}"] = float(np.mean(
                        (padded(B_aligned[d], shapes[d]) - padded(T_list[d], shapes[d])) ** 2
                    ))
                    row[f"active_mse_B{d + 1}"] = float(np.mean((B_aligned[d] - T_list[d]) ** 2))
                row["it"] = int(fit.it)
                row["flipped"] = sum(len(m[1]) for m in maps)
                paired = base.get((N, rep))
                if paired is not None:
                    row["csp_mse_B1"] = float(paired["mseB_1"])
                    row["csp_mse_B2"] = float(paired["mseB_2"])
            rows.append(row)
    out = DEFAULT_OUT
    fields = ["scenario", "N", "rep", "status", "mse_B1", "mse_B2", "active_mse_B1", "active_mse_B2",
              "csp_mse_B1", "csp_mse_B2", "it", "flipped", "time_sec"]
    with open(out / "matlab_reference_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for N in sorted({r["N"] for r in rows if r["scenario"] == scenario}):
            cell = [r for r in rows if r["scenario"] == scenario and r["N"] == N and r["status"] == "ok"]
            paired = [r for r in cell if "csp_mse_B1" in r]
            entry = {"scenario": scenario, "N": N, "replicates": len(cell),
                     "failed": sum(r["status"] != "ok" for r in rows if r["scenario"] == scenario and r["N"] == N),
                     "median_it": float(np.median([r["it"] for r in cell])) if cell else "",
                     "median_seconds": float(np.median([r["time_sec"] for r in cell])) if cell else ""}
            for key in ("mse_B1", "mse_B2", "active_mse_B1", "active_mse_B2"):
                entry[key] = float(np.mean([r[key] for r in cell])) if cell else ""
            for d in (1, 2):
                entry[f"csp_mse_B{d}"] = float(np.mean([r[f"csp_mse_B{d}"] for r in paired])) if paired else ""
                entry[f"fixed_below_csp_B{d}"] = (
                    float(np.mean([r[f"mse_B{d}"] < r[f"csp_mse_B{d}"] for r in paired])) if paired else ""
                )
            summary.append(entry)
    with open(out / "matlab_reference_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for entry in summary:
        print(entry)


if __name__ == "__main__":
    main()
