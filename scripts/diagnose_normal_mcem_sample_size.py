"""Targeted, paired diagnostics for the Normal-DDE MCEM CSP path.

This script deliberately holds the simulated data, spectral initialization,
and MCEM random seed fixed within each sample size.  Its configurations isolate
four mechanisms that can otherwise be confounded in a sample-size comparison:
the entrywise hard threshold, early stopping, annealing, and the ordered CSP
coordinate convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from dde import fit
from dde.algorithms.normal_csp import _canonicalize_csp_layer_order

from evaluate_normal_csp_recovery import (
    _initialization,
    _matched_binary_accuracy,
    _scenario_d3,
    _simulate,
    _tau_default,
)


def _width_changes(trace):
    changes = []
    previous = None
    for entry in trace:
        width = tuple(int(value) for value in entry["num_active"])
        if width != previous:
            changes.append((int(entry["iteration"]), width))
            previous = width
    return changes


def _run_configuration(
    *,
    scenario,
    X,
    A_true,
    initialization,
    fit_seed,
    name,
    tau,
    temperature,
    convergence_tol,
    canonicalize,
    max_iter,
    optimizer_maxiter,
):
    prop0, B0, gamma0, A0 = initialization
    if canonicalize:
        prop0, B0, A0, permutations = _canonicalize_csp_layer_order(
            prop0, B0, A0
        )
    else:
        prop0 = np.asarray(prop0).copy()
        B0 = [np.asarray(B).copy() for B in B0]
        A0 = [np.asarray(A).copy() for A in A0]
        permutations = None

    started = time.perf_counter()
    result = fit(
        X,
        method="DDE",
        D=scenario["D"],
        family="Normal",
        dimension_method="csp",
        t_spike=scenario["base_t_spike"],
        temp=temperature,
        tau=tau,
        C=1,
        tol=convergence_tol,
        max_it=max_iter,
        prop_in=prop0,
        B_list_in=B0,
        gamma_in=gamma0,
        A_list_in=A0,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_tol=1e-6,
        random_state=fit_seed,
        csp_backend="mcem",
        keep_trace=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    trace = result.diagnostics["trace"]
    final_accuracy = [
        _matched_binary_accuracy(A_est, A_layer_true)
        for A_est, A_layer_true in zip(result.A_list, A_true)
    ]
    final_l1 = [
        np.sum(np.abs(B[:, 1:]), axis=0).tolist() for B in result.B_list
    ]
    return {
        "name": name,
        "N": int(X.shape[0]),
        "tau": float(tau),
        "temperature": float(temperature),
        "convergence_tol": float(convergence_tol),
        "canonicalize": bool(canonicalize),
        "activation_permutations": (
            None
            if permutations is None
            else [permutation.tolist() for permutation in permutations]
        ),
        "K_fit": list(map(int, result.K_list)),
        "iterations": int(result.diagnostics["iterations"]),
        "dimension_l1_error": int(
            sum(
                abs(fitted - truth)
                for fitted, truth in zip(result.K_list, scenario["K_true"])
            )
        ),
        "final_A_accuracy": final_accuracy,
        "final_column_l1": final_l1,
        "width_changes": _width_changes(trace),
        "relative_tail_mean": [
            float(np.mean(history[-7:]))
            for history in result.diagnostics["relative_history"]
        ],
        "relative_tail_std": [
            float(np.std(history[-7:]))
            for history in result.diagnostics["relative_history"]
        ],
        "fit_seconds": float(elapsed),
    }


def _write_summary_csv(path, records):
    fields = [
        "name",
        "N",
        "tau",
        "temperature",
        "convergence_tol",
        "canonicalize",
        "K_fit",
        "iterations",
        "dimension_l1_error",
        "final_A_accuracy",
        "relative_tail_mean",
        "relative_tail_std",
        "fit_seconds",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: (
                        json.dumps(record[field])
                        if isinstance(record[field], (list, tuple))
                        else record[field]
                    )
                    for field in fields
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Results/normal_dde_csp_mcem_investigation/ablations"),
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _scenario_d3()
    seed_base = 20260828
    records = []
    configurations = {
        500: [
            ("paper", _tau_default(500), scenario["base_temp"], 0.005, False),
            ("low_tau_0.3", 0.3, scenario["base_temp"], 0.005, False),
        ],
        8000: [
            ("paper", _tau_default(8000), scenario["base_temp"], 0.005, False),
            ("fixed_tau_n500", _tau_default(500), scenario["base_temp"], 0.005, False),
            ("force_100", _tau_default(8000), scenario["base_temp"], 0.0, False),
            ("temperature_1", _tau_default(8000), 1.0, 0.005, False),
            ("canonical_order", _tau_default(8000), scenario["base_temp"], 0.005, True),
        ],
    }

    for N, variants in configurations.items():
        data_seed = seed_base + 100_000 * scenario["D"] + 1_000 * N
        init_seed = data_seed + 101
        fit_seed = data_seed + 202
        X, A_true = _simulate(scenario, N, data_seed)
        initialization = _initialization(
            scenario, X, A_true, "data", init_seed
        )
        for name, tau, temperature, convergence_tol, canonicalize in variants:
            record = _run_configuration(
                scenario=scenario,
                X=X,
                A_true=A_true,
                initialization=initialization,
                fit_seed=fit_seed,
                name=name,
                tau=tau,
                temperature=temperature,
                convergence_tol=convergence_tol,
                canonicalize=canonicalize,
                max_iter=args.max_iter,
                optimizer_maxiter=args.optimizer_maxiter,
            )
            records.append(record)
            print(
                f"N={N} {name}: K={record['K_fit']} "
                f"iterations={record['iterations']}",
                flush=True,
            )

    with (args.output_dir / "diagnostics.json").open("w") as stream:
        json.dump(records, stream, indent=2)
    _write_summary_csv(args.output_dir / "summary.csv", records)


if __name__ == "__main__":
    main()
