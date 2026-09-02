"""Diagnose D=3 Bernoulli SAEM sampling bias and convergence behavior.

This script does not change the production kernel. It runs the same M-step
with either the current synchronous/parallel latent update or a systematic
coordinate Gibbs sweep that preserves the joint posterior. Per-iteration
parameter, graph, convergence-error, and timing metrics are saved as CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from dde.algorithms.bernoulli_init import Bernoulli_init
from dde.algorithms.parametric_families import (
    BERNOULLI_FAMILY,
    _tlp_penalty_value_grad,
)
from dde.algorithms.semi_saem import (
    _compute_bernoulli_probabilities_parametric,
    _make_bernoulli_layer_accumulator,
)


FIELDS = [
    "N", "replicate", "start", "sampler", "iteration", "graph_accuracy",
    "parameter_rmse", "layer_graph_accuracy", "latent_accuracy",
    "raw_error", "postprocessed_error", "support_normalized_error",
    "sample_seconds", "observed_mstep_seconds", "latent_mstep_seconds",
    "iteration_seconds",
]


def _tiny_posterior(depth):
    states = np.asarray(
        list(itertools.product((0.0, 1.0), repeat=int(depth)))
    )
    coefficients = [np.array([-1.2, 2.4])] + [
        np.array([-1.0, 2.0]) for _ in range(depth - 1)
    ]
    weights = []
    for state in states:
        probability = 0.45 if state[-1] else 0.55
        for layer in range(depth - 2, -1, -1):
            success = expit(
                coefficients[layer + 1][0]
                + coefficients[layer + 1][1] * state[layer + 1]
            )
            probability *= success if state[layer] else 1.0 - success
        success = expit(coefficients[0][0] + coefficients[0][1] * state[0])
        probability *= success  # Condition on the observed bit X=1.
        weights.append(probability)
    posterior = np.asarray(weights, dtype=float)
    posterior /= np.sum(posterior)
    return states, posterior


def _tiny_conditional(states, posterior, state, coordinate):
    mask = np.ones(states.shape[0], dtype=bool)
    for index in range(states.shape[1]):
        if index != coordinate:
            mask &= states[:, index] == state[index]
    selected = np.flatnonzero(mask)
    numerator = np.sum(
        posterior[selected[states[selected, coordinate] == 1.0]]
    )
    return float(numerator / np.sum(posterior[selected]))


def exact_stationarity_diagnostic(depth, sampler):
    states, posterior = _tiny_posterior(depth)
    transition = np.zeros((states.shape[0], states.shape[0]), dtype=float)
    for source_index, source in enumerate(states):
        if sampler == "parallel":
            probabilities = [
                _tiny_conditional(states, posterior, source, coordinate)
                for coordinate in range(depth)
            ]
            for target_index, target in enumerate(states):
                transition[source_index, target_index] = np.prod([
                    probabilities[coordinate]
                    if target[coordinate]
                    else 1.0 - probabilities[coordinate]
                    for coordinate in range(depth)
                ])
        elif sampler == "sequential":
            paths = {tuple(source): 1.0}
            for coordinate in range(depth):
                updated_paths = {}
                for intermediate, path_probability in paths.items():
                    intermediate = np.asarray(intermediate, dtype=float)
                    probability = _tiny_conditional(
                        states, posterior, intermediate, coordinate
                    )
                    for value, draw_probability in (
                        (0.0, 1.0 - probability), (1.0, probability)
                    ):
                        target = intermediate.copy()
                        target[coordinate] = value
                        key = tuple(target)
                        updated_paths[key] = (
                            updated_paths.get(key, 0.0)
                            + path_probability * draw_probability
                        )
                paths = updated_paths
            for target, probability in paths.items():
                target_index = np.flatnonzero(
                    np.all(states == np.asarray(target), axis=1)
                )[0]
                transition[source_index, target_index] = probability
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary = np.real(
        eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))]
    )
    stationary = np.abs(stationary)
    stationary /= np.sum(stationary)
    return {
        "depth": depth,
        "sampler": sampler,
        "target_one_step_max_error": float(
            np.max(np.abs(posterior @ transition - posterior))
        ),
        "stationary_total_variation": float(
            0.5 * np.sum(np.abs(stationary - posterior))
        ),
        "posterior": posterior.tolist(),
        "stationary": stationary.tolist(),
    }


def _load_recovery_module():
    path = Path(__file__).with_name("evaluate_bernoulli_fixed_recovery.py")
    spec = importlib.util.spec_from_file_location("bernoulli_recovery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parallel_sample(X, prop, B_list, A_list, rng):
    probabilities = _compute_bernoulli_probabilities_parametric(
        X, prop, B_list, None, A_list, family=BERNOULLI_FAMILY
    )
    sampled = [
        (rng.random(probability.shape) < probability).astype(np.uint8)
        for probability in probabilities
    ]
    return sampled, probabilities[-1]


def _sequential_sample(X, prop, B_list, A_list, rng):
    """Perform one bottom-up systematic coordinate Gibbs sweep."""
    X = np.asarray(X, dtype=float)
    prop = np.clip(np.asarray(prop, dtype=float), 1e-15, 1.0 - 1e-15)
    state = [np.asarray(A, dtype=np.uint8).copy() for A in A_list]
    D = len(state)
    top_probabilities = np.empty_like(state[-1], dtype=float)

    for layer in range(D):
        coefficient = np.asarray(B_list[layer], dtype=float)
        natural = (
            coefficient[:, 0][None, :]
            + state[layer] @ coefficient[:, 1:].T
        )
        children = X if layer == 0 else state[layer - 1]

        for coordinate in range(state[layer].shape[1]):
            slope = coefficient[:, coordinate + 1]
            natural_zero = natural - (
                state[layer][:, coordinate, None] * slope[None, :]
            )
            natural_one = natural_zero + slope[None, :]
            likelihood_logit = children @ slope
            likelihood_logit -= np.sum(
                np.logaddexp(0.0, natural_one)
                - np.logaddexp(0.0, natural_zero),
                axis=1,
            )

            if layer == D - 1:
                prior_logit = np.log(
                    prop[coordinate] / (1.0 - prop[coordinate])
                )
            else:
                above = np.asarray(B_list[layer + 1], dtype=float)
                prior_logit = (
                    above[coordinate, 0]
                    + state[layer + 1] @ above[coordinate, 1:]
                )
            probability = expit(prior_logit + likelihood_logit)
            updated = (rng.random(X.shape[0]) < probability).astype(np.uint8)
            state[layer][:, coordinate] = updated
            natural = natural_zero + updated[:, None] * slope[None, :]
            if layer == D - 1:
                top_probabilities[:, coordinate] = probability

    return state, top_probabilities


def _augment(A_list):
    augmented = []
    for state in A_list:
        sample = np.empty((state.shape[0], state.shape[1] + 1, 1), dtype=np.uint8)
        sample[:, 0, 0] = 1
        sample[:, 1:, 0] = state
        augmented.append(sample)
    return augmented


def _latent_mstep(
    lower_sample,
    upper_sample,
    coefficient_start,
    accumulator,
    iteration,
    penalty,
    tau,
):
    accumulator.update(lower_sample, upper_sample, iteration)
    update = np.zeros_like(coefficient_start)
    bounds = BERNOULLI_FAMILY.fixed_latent_bounds(update.shape[1], 10.0)
    for row in range(update.shape[0]):

        def objective(x, __row=row):
            value, gradient = accumulator.value_and_grad(__row, x)
            penalty_value, penalty_gradient = _tlp_penalty_value_grad(
                x[1:], penalty, tau
            )
            gradient[1:] += penalty_gradient
            return value + penalty_value, gradient

        result = minimize(
            objective,
            coefficient_start[row],
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 10},
        )
        update[row] = result.x
    return update


def run_variant(N, replicate, start, sampler, iterations, seed_base):
    with threadpool_limits(limits=1):
        return _run_variant_single_thread(
            N, replicate, start, sampler, iterations, seed_base
        )


def _run_variant_single_thread(
    N, replicate, start, sampler, iterations, seed_base
):
    recovery = _load_recovery_module()
    scenario = recovery.build_scenario("d3")
    data_seed = seed_base + 1000 * int(N) + int(replicate)
    fit_seed = data_seed + 500_000_000
    X, A_true = recovery.simulate_bernoulli_dde(scenario, N, data_seed)

    if start == "truth":
        prop = scenario["prop"].copy()
        B_list = [B.copy() for B in scenario["B_list"]]
        A_list = [A.copy() for A in A_true]
    elif start == "data":
        prop, B_list, _, A_list = Bernoulli_init(
            X, scenario["K_list"], mode="data"
        )[:4]
    else:
        raise ValueError(f"Unknown start: {start}")

    rng = np.random.default_rng(fit_seed)
    penalty = float(N) ** 0.25
    tau = max(3.0 * float(N) ** -0.3, 0.3)
    p_top = np.zeros_like(A_list[-1], dtype=float)
    observed_accumulator = BERNOULLI_FAMILY.make_fixed_accumulator(
        scenario["J"], scenario["K_list"][0] + 1
    )
    latent_accumulators = [
        _make_bernoulli_layer_accumulator(
            scenario["K_list"][layer], B_list[layer].shape[0]
        )
        for layer in range(1, scenario["D"])
    ]
    n_parameters = prop.size + sum(B.size for B in B_list)
    records = []

    for iteration in range(1, iterations + 1):
        iteration_started = time.perf_counter()
        sample_started = time.perf_counter()
        if sampler == "parallel":
            A_new, top_probability = _parallel_sample(
                X, prop, B_list, A_list, rng
            )
        elif sampler == "sequential":
            A_new, top_probability = _sequential_sample(
                X, prop, B_list, A_list, rng
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        sample_seconds = time.perf_counter() - sample_started
        samples = _augment(A_new)

        p_top = (
            (1.0 - 1.0 / iteration) * p_top
            + (1.0 / iteration) * top_probability
        )
        prop_update = np.mean(p_top, axis=0)

        observed_started = time.perf_counter()
        B1_update, _ = BERNOULLI_FAMILY.fixed_mstep(
            X,
            samples[0],
            B_list[0],
            None,
            observed_accumulator,
            iteration=iteration,
            depth=3,
            penalty=penalty,
            tau=tau,
            bound_val=10.0,
            response_cache=None,
        )
        observed_seconds = time.perf_counter() - observed_started

        latent_started = time.perf_counter()
        raw_updates = [B1_update]
        for layer in range(1, scenario["D"]):
            raw_updates.append(_latent_mstep(
                samples[layer - 1],
                samples[layer],
                B_list[layer],
                latent_accumulators[layer - 1],
                iteration,
                penalty,
                tau,
            ))
        latent_seconds = time.perf_counter() - latent_started

        raw_error = np.sqrt(
            np.sum((prop - prop_update) ** 2)
            + sum(
                np.sum((old - new) ** 2)
                for old, new in zip(B_list, raw_updates)
            )
        )
        processed = BERNOULLI_FAMILY.postprocess_fixed_coefficients(
            [value.copy() for value in raw_updates], tau=tau, depth=3
        )
        postprocessed_error = np.sqrt(
            np.sum((prop - prop_update) ** 2)
            + sum(
                np.sum((old - new) ** 2)
                for old, new in zip(B_list, processed)
            )
        )
        support_error_sq = np.sum((prop - prop_update) ** 2)
        for old, new in zip(B_list, processed):
            support_error_sq += np.sum((new * (old != 0.0) - old) ** 2)
        support_normalized_error = np.sqrt(support_error_sq / n_parameters)

        prop = prop_update
        B_list = processed
        A_list = A_new
        metrics = recovery.calculate_metrics(
            prop, B_list, A_list, scenario, A_true, tau
        )
        records.append({
            "N": N,
            "replicate": replicate,
            "start": start,
            "sampler": sampler,
            "iteration": iteration,
            "graph_accuracy": metrics["graph_accuracy"],
            "parameter_rmse": metrics["parameter_rmse"],
            "layer_graph_accuracy": metrics["layer_graph_accuracy"],
            "latent_accuracy": metrics["latent_accuracy"],
            "raw_error": raw_error,
            "postprocessed_error": postprocessed_error,
            "support_normalized_error": support_normalized_error,
            "sample_seconds": sample_seconds,
            "observed_mstep_seconds": observed_seconds,
            "latent_mstep_seconds": latent_seconds,
            "iteration_seconds": time.perf_counter() - iteration_started,
        })
    return records


def _write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[500, 2000])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--starts", nargs="+", choices=("data", "truth"), default=["data", "truth"])
    parser.add_argument("--samplers", nargs="+", choices=("parallel", "sequential"), default=["parallel", "sequential"])
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed-base", type=int, default=20260902)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/bernoulli_d3_diagnostics"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_results = [
        exact_stationarity_diagnostic(depth, sampler)
        for depth in (2, 3)
        for sampler in ("parallel", "sequential")
    ]
    (output_dir / "exact_gibbs_stationarity.json").write_text(
        json.dumps(exact_results, indent=2), encoding="utf-8"
    )
    if args.exact_only:
        return
    tasks = [
        (N, replicate, start, sampler, args.iterations, args.seed_base)
        for N in args.sample_sizes
        for replicate in range(1, args.replicates + 1)
        for start in args.starts
        for sampler in args.samplers
    ]
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(run_variant, *task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            rows.sort(key=lambda row: (
                int(row["N"]), int(row["replicate"]), row["start"],
                row["sampler"], int(row["iteration"]),
            ))
            _write_csv(output_dir / "saem_trajectories.csv", rows)
            print(f"completed {index}/{len(tasks)} variants", flush=True)

    metadata = {
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "iterations": args.iterations,
        "starts": args.starts,
        "samplers": args.samplers,
        "n_jobs": args.n_jobs,
        "seed_base": args.seed_base,
        "wall_seconds": time.perf_counter() - started,
        "note": "Diagnostic implementation only; production code is unchanged.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
