"""Evaluate SemiDDE-CSP latent-width recovery on nonlinear margins.

The primary endpoints are

* exact dimension-shrinkage recovery rate; and
* mean L1 distance between fitted and true width vectors.

Each replicate is appended to ``fits.csv`` immediately so interrupted runs can
be resumed with ``--resume``.

Fits use the SemiDDE CSP defaults of ``fit`` (the copula kernel with a ``g``
step after every committed iteration); ``--max-inner-iter``, ``--tol`` and
``--optimizer-maxiter`` override the kernel's budget when given.  The MCEM
results of earlier runs are reproducible from the git tag ``csp-mcem-last``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from dde import fit
from dde.algorithms.csp import maximal_identifiable_widths
from dde.utils import dimension_recovery_metrics


def _block_matrix(n_child, n_parent, strength):
    slopes = np.zeros((n_child, n_parent), dtype=float)
    edges = np.linspace(0, n_child, n_parent + 1, dtype=int)
    for parent in range(n_parent):
        slopes[edges[parent] : edges[parent + 1], parent] = strength
    return slopes


def _scenario(name):
    scenarios = {
        "d2": {
            "D": 2,
            "J": 54,
            "K_true": (9, 3),
            "t_spike": (0.02, 0.04),
            "temp": 0.8,
        },
        "d3": {
            "D": 3,
            "J": 108,
            "K_true": (18, 6, 2),
            "t_spike": (0.01, 0.03, 0.0075),
            "temp": 0.85,
        },
    }
    try:
        scenario = dict(scenarios[name])
    except KeyError as exc:
        raise ValueError(f"Unknown scenario {name!r}.") from exc

    K_true = scenario["K_true"]
    J = scenario["J"]
    B_list = [
        np.column_stack(
            [np.full(J, -1.5), _block_matrix(J, K_true[0], 3.0)]
        )
    ]
    for layer in range(1, scenario["D"]):
        n_child = K_true[layer - 1]
        n_parent = K_true[layer]
        B_list.append(
            np.column_stack(
                [
                    np.full(n_child, -1.5),
                    _block_matrix(n_child, n_parent, 3.0),
                ]
            )
        )
    scenario["B_list"] = B_list
    scenario["prop"] = np.full(K_true[-1], 0.5)
    scenario["gamma"] = np.ones(J)
    scenario["K_max"] = maximal_identifiable_widths(J, scenario["D"])
    return scenario


def _simulate_semidde(scenario, N, seed):
    rng = np.random.default_rng(seed)
    B_list = scenario["B_list"]
    D = scenario["D"]
    A_list = [None] * D
    A_list[-1] = rng.binomial(
        1, scenario["prop"], size=(N, scenario["K_true"][-1])
    ).astype(np.uint8)
    for layer in range(D - 2, -1, -1):
        design = np.column_stack([np.ones(N), A_list[layer + 1]])
        probability = expit(design @ B_list[layer + 1].T)
        A_list[layer] = rng.binomial(1, probability).astype(np.uint8)

    mean = np.column_stack([np.ones(N), A_list[0]]) @ B_list[0].T
    X = mean + rng.normal(
        scale=np.sqrt(scenario["gamma"]), size=mean.shape
    )

    # Strictly increasing, heterogeneous margins preserve the copula while
    # making the observed layer genuinely semiparametric.
    powers = 1.0 + (np.arange(scenario["J"]) % 3) / 2.0
    Y = np.sign(X) * np.abs(X) ** powers[None, :]
    return Y


def _fit_one(scenario, *, N, replicate, seed, args):
    replicate_seed = int(seed + replicate)
    Y = _simulate_semidde(scenario, N, replicate_seed)
    started = time.perf_counter()
    try:
        result = fit(
            Y,
            model="SemiDDE",
            D=scenario["D"],
            dimension_method="csp",
            t_spike=scenario["t_spike"],
            temp=scenario["temp"],
            tau=args.tau,
            C=args.mc_draws,
            max_enum_bits=args.max_enum_bits,
            random_state=replicate_seed,
            verbose=False,
            **{
                name: value
                for name, value in (
                    ("tol", args.tol),
                    ("max_it", args.max_inner_iter),
                    ("optimizer_maxiter", args.optimizer_maxiter),
                )
                if value is not None
            },
        )
        widths = tuple(int(value) for value in result.K_list)
        metrics = dimension_recovery_metrics(widths, scenario["K_true"])
        return {
            "replicate": replicate,
            "seed": replicate_seed,
            "status": "ok",
            "true_widths": json.dumps(scenario["K_true"]),
            "max_widths": json.dumps(scenario["K_max"]),
            "fitted_widths": json.dumps(widths),
            **metrics,
            "g_updates": len(result.diagnostics["g_relative_history"]),
            "inner_iterations": result.diagnostics["iterations"],
            "elapsed_seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # keep systematic runs checkpointable
        return {
            "replicate": replicate,
            "seed": replicate_seed,
            "status": "error",
            "true_widths": json.dumps(scenario["K_true"]),
            "max_widths": json.dumps(scenario["K_max"]),
            "fitted_widths": "",
            "exact_dimension_recovery": "",
            "dimension_l1_error": "",
            "g_updates": "",
            "inner_iterations": "",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_completed(path):
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["replicate"]) for row in csv.DictReader(handle)}


def _append_row(path, row):
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _summarize(fits_path, summary_path, report_path, scenario):
    with fits_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if row["status"] == "ok"]
    exact = [float(row["exact_dimension_recovery"]) for row in successful]
    l1 = [float(row["dimension_l1_error"]) for row in successful]
    summary = {
        "scenario": f"D={scenario['D']}, J={scenario['J']}",
        "true_widths": json.dumps(scenario["K_true"]),
        "max_widths": json.dumps(scenario["K_max"]),
        "n_requested": len(rows),
        "n_successful": len(successful),
        "dimension_shrinkage_accuracy": float(np.mean(exact)) if exact else np.nan,
        "mean_dimension_l1_error": float(np.mean(l1)) if l1 else np.nan,
    }
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    report_path.write_text(
        "# SemiDDE CSP dimension recovery\n\n"
        f"- True widths: `{scenario['K_true']}`\n"
        f"- Maximum initialized widths: `{scenario['K_max']}`\n"
        f"- Successful fits: {len(successful)}/{len(rows)}\n"
        f"- Exact dimension-shrinkage recovery rate: "
        f"{summary['dimension_shrinkage_accuracy']:.4f}\n"
        f"- Mean L1 width error: {summary['mean_dimension_l1_error']:.4f}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("d2", "d3"), default="d2")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-inner-iter", type=int, default=None, help="kernel default when omitted (50)")
    parser.add_argument("--optimizer-maxiter", type=int, default=None, help="kernel default when omitted (15)")
    parser.add_argument("--mc-draws", type=int, default=1)
    parser.add_argument("--max-enum-bits", type=int, default=20)
    parser.add_argument(
        "--t-spike",
        type=float,
        nargs="+",
        default=None,
        help="Override the scenario spike scales with one value or D values.",
    )
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--tol", type=float, default=None, help="kernel default when omitted (0.005)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/semidde_csp_recovery"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.replicates < 1 or args.sample_size < 1:
        parser.error("--replicates and --sample-size must be positive")

    scenario = _scenario(args.scenario)
    if args.t_spike is not None:
        if len(args.t_spike) not in {1, scenario["D"]}:
            parser.error("--t-spike requires one value or exactly D values")
        scenario["t_spike"] = tuple(args.t_spike)
    if args.temp is not None:
        scenario["temp"] = float(args.temp)
    args.output.mkdir(parents=True, exist_ok=True)
    fits_path = args.output / "fits.csv"
    if fits_path.exists() and not args.resume:
        parser.error(f"{fits_path} exists; use --resume or a different --output")
    completed = _read_completed(fits_path) if args.resume else set()

    metadata = {
        "scenario": args.scenario,
        "D": scenario["D"],
        "J": scenario["J"],
        "K_true": scenario["K_true"],
        "K_max": scenario["K_max"],
        "replicates": args.replicates,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "t_spike": scenario["t_spike"],
        "temperature": scenario["temp"],
        "backend": "copula",
        "max_inner_iter": args.max_inner_iter,
        "optimizer_maxiter": args.optimizer_maxiter,
        "mc_draws": args.mc_draws,
        "max_enum_bits": args.max_enum_bits,
        "tau": args.tau,
        "tol": args.tol,
        "primary_metrics": [
            "dimension_shrinkage_accuracy",
            "mean_dimension_l1_error",
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    for replicate in range(1, args.replicates + 1):
        if replicate in completed:
            continue
        row = _fit_one(
            scenario,
            N=args.sample_size,
            replicate=replicate,
            seed=args.seed,
            args=args,
        )
        _append_row(fits_path, row)
        print(
            f"replicate={replicate} status={row['status']} "
            f"widths={row['fitted_widths']} l1={row['dimension_l1_error']}"
        )

    _summarize(
        fits_path,
        args.output / "summary.csv",
        args.output / "REPORT.md",
        scenario,
    )


if __name__ == "__main__":
    main()
