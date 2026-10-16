"""Fixed-true-width controls for the Normal-DDE CSP development benchmark."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dde import fit
from dde.algorithms.normal_init_semi import Normal_init_semi

from normal_csp_scenarios import (
    _matched_binary_accuracy,
    _max_abs_column_correlation,
    _read_csv,
    _scenario_d2,
    _scenario_d3,
    _simulate,
    _tau_default,
    _write_csv,
)


def _run_one(scenario, N, replicate, args):
    data_seed = args.seed_base + 100_000 * scenario["D"] + 1_000 * N + replicate
    init_seed = data_seed + 101
    fit_seed = data_seed + 202
    X, A_true = _simulate(scenario, N, data_seed)

    np.random.seed(init_seed)
    prop0, B0, gamma0, A0 = Normal_init_semi(
        X,
        scenario["K_true"][0],
        scenario["K_true"],
        mode="data",
    )[:4]
    initial_A_accuracy = tuple(
        _matched_binary_accuracy(A_est, A_layer_true)
        for A_est, A_layer_true in zip(A0, A_true)
    )

    np.random.seed(fit_seed)
    started = time.perf_counter()
    result = fit(
        X,
        method="DDE",
        D=scenario["D"],
        family="Normal",
        dimension_method="fixed",
        K_list=scenario["K_true"],
        pen_list=[float(N) ** (2.0 / 8.0)] * scenario["D"],
        tau=_tau_default(N),
        C=args.mc_samples,
        tol=args.tol,
        max_it=args.max_iter,
        prop_in=prop0,
        B_list_in=B0,
        gamma_in=gamma0,
        A_list_in=A0,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    final_A_accuracy = tuple(
        _matched_binary_accuracy(A_est, A_layer_true)
        for A_est, A_layer_true in zip(result.A_list, A_true)
    )

    record = {
        "scenario": scenario["name"],
        "estimator": "fixed_true_width",
        "N": int(N),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "fit_seed": int(fit_seed),
        "K_true": json.dumps(list(scenario["K_true"])),
        "K_fit": json.dumps(list(result.K_list)),
        "penalty": float(N) ** (2.0 / 8.0),
        "tau": _tau_default(N),
        "mc_samples": args.mc_samples,
        "iterations": result.diagnostics["iterations"],
        "hit_max_iter": int(result.diagnostics["iterations"] >= args.max_iter),
        "fit_seconds": elapsed,
    }
    for d in range(scenario["D"]):
        record[f"A{d + 1}_initial_accuracy"] = initial_A_accuracy[d]
        record[f"A{d + 1}_final_accuracy"] = final_A_accuracy[d]
        record[f"A{d + 1}_max_abs_correlation"] = _max_abs_column_correlation(
            result.A_list[d]
        )
    return record


def _summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], int(float(record["N"])))].append(record)
    summary = []
    for (scenario, N), rows in sorted(grouped.items()):
        D = len(json.loads(rows[0]["K_true"]))
        item = {
            "scenario": scenario,
            "N": N,
            "fits": len(rows),
            "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
            "hit_max_iter_rate": float(
                np.mean([row["hit_max_iter"] for row in rows])
            ),
            "mean_fit_seconds": float(
                np.mean([row["fit_seconds"] for row in rows])
            ),
        }
        for d in range(D):
            item[f"mean_A{d + 1}_initial_accuracy"] = float(
                np.mean([row[f"A{d + 1}_initial_accuracy"] for row in rows])
            )
            item[f"mean_A{d + 1}_final_accuracy"] = float(
                np.mean([row[f"A{d + 1}_final_accuracy"] for row in rows])
            )
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument("--seed-base", type=int, default=20260828)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("d2_j50", "d3_j108"),
        default=["d2_j50", "d3_j108"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Results/normal_dde_csp_development/fixed_true_width"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        scenario
        for scenario in (_scenario_d2(), _scenario_d3())
        if scenario["name"] in args.scenarios
    ]
    fits_path = args.output_dir / "normal_fixed_control_fits.csv"
    records = _read_csv(fits_path) if args.resume and fits_path.exists() else []
    completed = {
        (
            record["scenario"],
            int(float(record["N"])),
            int(float(record["replicate"])),
        )
        for record in records
    }
    targets = [
        (scenario, N, replicate)
        for scenario in scenarios
        for N in args.sample_sizes
        for replicate in range(
            args.replicate_start, args.replicate_start + args.replicates
        )
    ]

    for index, (scenario, N, replicate) in enumerate(targets, start=1):
        key = (scenario["name"], int(N), int(replicate))
        if key in completed:
            continue
        record = _run_one(scenario, int(N), int(replicate), args)
        records.append(record)
        completed.add(key)
        _write_csv(fits_path, records)
        _write_csv(
            args.output_dir / "normal_fixed_control_summary.csv", _summary(records)
        )
        print(
            f"[{index}/{len(targets)}] {record['scenario']} N={record['N']} "
            f"rep={record['replicate']} iterations={record['iterations']}",
            flush=True,
        )

    metadata = {
        "replicates": args.replicates,
        "replicate_start": args.replicate_start,
        "sample_sizes": args.sample_sizes,
        "max_iter": args.max_iter,
        "mc_samples": args.mc_samples,
        "tol": args.tol,
        "seed_base": args.seed_base,
        "scenarios_selected": args.scenarios,
        "fit_count": len(records),
    }
    with (args.output_dir / "normal_fixed_control_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)


if __name__ == "__main__":
    main()
