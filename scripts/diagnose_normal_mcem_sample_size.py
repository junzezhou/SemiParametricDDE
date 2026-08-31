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
    replicate,
    name,
    t_spike,
    tau,
    temperature,
    convergence_tol,
    canonicalize,
    stagewise,
    stage_min_iter,
    stage_width_window,
    stage_freeze_deeper,
    penalty_strength,
    mc_samples,
    stage_reinitialize_next,
    stage_spectral_gap_threshold,
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
        t_spike=t_spike,
        temp=temperature,
        tau=tau,
        C=mc_samples,
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
        mcem_stagewise=stagewise,
        mcem_stage_min_iter=stage_min_iter,
        mcem_stage_width_window=stage_width_window,
        mcem_stage_freeze_deeper=stage_freeze_deeper,
        mcem_penalty_strength=penalty_strength,
        mcem_stage_reinitialize_next=stage_reinitialize_next,
        mcem_stage_spectral_gap_threshold=stage_spectral_gap_threshold,
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
        "t_spike": list(map(float, t_spike)),
        "N": int(X.shape[0]),
        "replicate": int(replicate),
        "tau": float(tau),
        "temperature": float(temperature),
        "convergence_tol": float(convergence_tol),
        "canonicalize": bool(canonicalize),
        "stagewise": bool(stagewise),
        "stage_min_iter": int(stage_min_iter),
        "stage_width_window": int(stage_width_window),
        "stage_freeze_deeper": bool(stage_freeze_deeper),
        "penalty_strength": list(map(float, penalty_strength)),
        "mc_samples": int(mc_samples),
        "stage_reinitialize_next": bool(stage_reinitialize_next),
        "stage_spectral_gap_threshold": stage_spectral_gap_threshold,
        "stage_caps": list(map(int, result.diagnostics["stage_caps"])),
        "stage_transitions": result.diagnostics["stage_transitions"],
        "stage_reinitializations": result.diagnostics[
            "stage_reinitializations"
        ],
        "final_stage": int(result.diagnostics["final_stage"]),
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
        "t_spike",
        "N",
        "replicate",
        "tau",
        "temperature",
        "convergence_tol",
        "canonicalize",
        "stagewise",
        "stage_min_iter",
        "stage_width_window",
        "stage_freeze_deeper",
        "penalty_strength",
        "mc_samples",
        "stage_reinitialize_next",
        "stage_spectral_gap_threshold",
        "stage_caps",
        "stage_transitions",
        "stage_reinitializations",
        "final_stage",
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
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 8000]
    )
    parser.add_argument(
        "--initialization-mode", choices=("data", "truth"), default="data"
    )
    parser.add_argument(
        "--t-spike-multiplier",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--stage-reinitialize-next", action="store_true")
    parser.add_argument(
        "--stage-spectral-gap-threshold", type=float, default=None
    )
    parser.add_argument("--stage-min-iter", type=int, default=20)
    parser.add_argument("--stage-width-window", type=int, default=10)
    parser.add_argument("--stage-convergence-tol", type=float, default=0.0)
    parser.add_argument(
        "--no-stage-freeze-deeper", action="store_true"
    )
    parser.add_argument(
        "--penalty-strength",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--penalty-power",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help=(
            "Multiply each layer's penalty strength by "
            "(N / 500) ** power; use 0 0.5 0 for a sqrt-N B2 pilot."
        ),
    )
    parser.add_argument(
        "--staged-pilot",
        action="store_true",
        help="Run only the paired Normal-MCEM staged-CSP pilot.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _scenario_d3()
    seed_base = 20260828
    records = []
    if args.staged_pilot:
        configurations = {
            N: [
                (
                    "stage_min20_w10",
                    _tau_default(N),
                    scenario["base_temp"],
                    args.stage_convergence_tol,
                    False,
                    True,
                    args.stage_min_iter,
                    args.stage_width_window,
                    not args.no_stage_freeze_deeper,
                    [
                        base_strength * (N / 500.0) ** power
                        for base_strength, power in zip(
                            args.penalty_strength, args.penalty_power
                        )
                    ],
                )
            ]
            for N in args.sample_sizes
        }
    else:
        configurations = {
            500: [
                ("paper", _tau_default(500), scenario["base_temp"], 0.005, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
                ("low_tau_0.3", 0.3, scenario["base_temp"], 0.005, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
            ],
            8000: [
                ("paper", _tau_default(8000), scenario["base_temp"], 0.005, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
                ("fixed_tau_n500", _tau_default(500), scenario["base_temp"], 0.005, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
                ("force_100", _tau_default(8000), scenario["base_temp"], 0.0, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
                ("temperature_1", _tau_default(8000), 1.0, 0.005, False, False, 20, 10, True, [1.0, 1.0, 1.0]),
                ("canonical_order", _tau_default(8000), scenario["base_temp"], 0.005, True, False, 20, 10, True, [1.0, 1.0, 1.0]),
            ],
        }

    for N, variants in configurations.items():
        for replicate in range(args.replicates):
            data_seed = (
                seed_base
                + 100_000 * scenario["D"]
                + 1_000 * N
                + replicate
            )
            init_seed = data_seed + 101
            fit_seed = data_seed + 202
            X, A_true = _simulate(scenario, N, data_seed)
            initialization = _initialization(
                scenario, X, A_true, args.initialization_mode, init_seed
            )
            for (
                name,
                tau,
                temperature,
                convergence_tol,
                canonicalize,
                stagewise,
                stage_min_iter,
                stage_width_window,
                stage_freeze_deeper,
                penalty_strength,
            ) in variants:
                record = _run_configuration(
                    scenario=scenario,
                    X=X,
                    A_true=A_true,
                    initialization=initialization,
                    fit_seed=fit_seed,
                    replicate=replicate,
                    name=name,
                    t_spike=(
                        scenario["base_t_spike"]
                        * np.asarray(args.t_spike_multiplier, dtype=float)
                    ),
                    tau=tau,
                    temperature=temperature,
                    convergence_tol=convergence_tol,
                    canonicalize=canonicalize,
                    stagewise=stagewise,
                    stage_min_iter=stage_min_iter,
                    stage_width_window=stage_width_window,
                    stage_freeze_deeper=stage_freeze_deeper,
                    penalty_strength=penalty_strength,
                    mc_samples=args.mc_samples,
                    stage_reinitialize_next=args.stage_reinitialize_next,
                    stage_spectral_gap_threshold=(
                        args.stage_spectral_gap_threshold
                    ),
                    max_iter=args.max_iter,
                    optimizer_maxiter=args.optimizer_maxiter,
                )
                records.append(record)
                print(
                    f"N={N} rep={replicate} {name}: K={record['K_fit']} "
                    f"iterations={record['iterations']}",
                    flush=True,
                )

    with (args.output_dir / "diagnostics.json").open("w") as stream:
        json.dump(records, stream, indent=2)
    _write_summary_csv(args.output_dir / "summary.csv", records)


if __name__ == "__main__":
    main()
