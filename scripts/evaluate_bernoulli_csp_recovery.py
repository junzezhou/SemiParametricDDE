"""Systematic recovery and performance tests for Bernoulli-DDE CSP.

The scenarios retain the paper's sparse Bernoulli ``B_s`` signal while adding
uninformative observed variables so CSP starts from a genuinely overspecified
identifiable width:

* D=2: true (6, 2), maximal (9, 3), J=27;
* D=3: true (18, 6, 2), maximal (21, 7, 2), J=64.

The default experiment covers N=(500, 1000, 2000, 4000, 8000), checkpoints
every completed fit, and records dimension recovery, graph support recovery,
latent matching, reconstruction diagnostics, convergence, time, and worker
peak RSS. Graph support is reported pooled (``graph_accuracy``) and per layer
(``layer_graph_accuracy``, ``layer_sensitivity``, ``layer_specificity``,
``layer_matched_columns``; see ``dde.utils.layer_structure_recovery_metrics``),
both with the support thresholded at ``tau``. Re-running the command resumes
completed configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from threadpoolctl import threadpool_limits

from dde import fit
from dde.algorithms.csp import maximal_identifiable_widths
from dde.algorithms.dde import DEFAULT_PARAMETRIC_CONVERGENCE_TOL
from dde.utils import layer_structure_recovery_metrics


SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000)
STRUCTURE_FIELDS = (
    "layer_graph_accuracy", "layer_sensitivity", "layer_specificity",
    "layer_matched_columns", "layer_graph_counts", "layer_column_pairs",
)
SCENARIO_SPECS = {
    "d2": {
        "name": "D2_Bs_overspecified_J27",
        "D": 2,
        "J": 27,
        "K_true": (6, 2),
        "extra_observed": 9,
        "t_spike": (0.03, 0.05),
        "temperature": 0.9,
    },
    "d3": {
        "name": "D3_Bs_overspecified_J64",
        "D": 3,
        "J": 64,
        "K_true": (18, 6, 2),
        "extra_observed": 10,
        "t_spike": (0.01, 0.03, 0.005),
        "temperature": 0.9,
    },
}


def _paper_sparse_coefficients(n_parent: int) -> np.ndarray:
    """Return the sparse 3K-by-K Bernoulli coefficients from equation S.36."""
    n_parent = int(n_parent)
    if n_parent < 1 or n_parent % 2:
        raise ValueError("The B_s construction requires a positive even width.")
    slopes = np.zeros((3 * n_parent, n_parent), dtype=float)
    slopes[:n_parent] = 4.0 * np.eye(n_parent)
    slopes[n_parent : 2 * n_parent] = 4.0 * np.eye(n_parent)
    third = 4.0 * np.eye(n_parent)
    half = n_parent // 2
    for row in range(n_parent):
        third[row, (row + half) % n_parent] = 4.0 / 3.0
    slopes[2 * n_parent :] = third
    intercept = np.concatenate(
        [
            np.full(n_parent, -2.0),
            np.full(n_parent, -4.0),
            np.full(n_parent, -2.0),
        ]
    )
    return np.column_stack([intercept, slopes])


def build_scenario(key: str) -> dict:
    spec = SCENARIO_SPECS[key]
    widths = spec["K_true"]
    B_true = [_paper_sparse_coefficients(width) for width in widths]
    extra = int(spec["extra_observed"])
    if extra:
        noise_rows = np.zeros((extra, widths[0] + 1), dtype=float)
        B_true[0] = np.row_stack([B_true[0], noise_rows])
    if B_true[0].shape[0] != spec["J"]:
        raise AssertionError("Scenario observed width is inconsistent.")
    K_max = maximal_identifiable_widths(spec["J"], spec["D"])
    if any(true >= maximum for true, maximum in zip(widths, K_max)):
        # The D=3 top layer intentionally reaches its structural maximum.
        if not (
            key == "d3"
            and widths[-1] == K_max[-1]
            and all(t < m for t, m in zip(widths[:-1], K_max[:-1]))
        ):
            raise AssertionError("Scenario must be overspecified by CSP.")
    return {
        **spec,
        "K_max": tuple(int(value) for value in K_max),
        "B_true": B_true,
        "G_true": [B[:, 1:] != 0.0 for B in B_true],
        "prop": np.full(widths[-1], 0.5, dtype=float),
    }


def simulate(scenario: dict, N: int, seed: int):
    rng = np.random.default_rng(int(seed))
    A_true = [None] * scenario["D"]
    A_true[-1] = rng.binomial(
        1, scenario["prop"], size=(N, scenario["K_true"][-1])
    ).astype(np.uint8)
    for layer in range(scenario["D"] - 2, -1, -1):
        B = scenario["B_true"][layer + 1]
        eta = B[:, 0][None, :] + A_true[layer + 1] @ B[:, 1:].T
        A_true[layer] = rng.binomial(1, expit(eta)).astype(np.uint8)
    B1 = scenario["B_true"][0]
    eta = B1[:, 0][None, :] + A_true[0] @ B1[:, 1:].T
    X = rng.binomial(1, expit(eta)).astype(np.uint8)
    return X, A_true


def _padded_truth_initialization(scenario, A_true, seed):
    """Embed generating parameters and states in the maximal CSP widths."""
    rng = np.random.default_rng(int(seed))
    K_true = scenario["K_true"]
    K_max = scenario["K_max"]
    B_true = scenario["B_true"]
    B_start = [np.zeros((scenario["J"], K_max[0] + 1), dtype=float)]
    B_start[0][:, 0] = B_true[0][:, 0]
    B_start[0][:, 1 : K_true[0] + 1] = B_true[0][:, 1:]
    for layer in range(1, scenario["D"]):
        coefficient = np.zeros(
            (K_max[layer - 1], K_max[layer] + 1), dtype=float
        )
        coefficient[: K_true[layer - 1], 0] = B_true[layer][:, 0]
        coefficient[
            : K_true[layer - 1], 1 : K_true[layer] + 1
        ] = B_true[layer][:, 1:]
        B_start.append(coefficient)

    A_start = []
    for truth, true_width, maximum_width in zip(A_true, K_true, K_max):
        state = rng.integers(
            0, 2, size=(truth.shape[0], maximum_width), dtype=np.uint8
        )
        state[:, :true_width] = truth
        A_start.append(state)
    prop_start = np.full(K_max[-1], 0.5, dtype=float)
    prop_start[: K_true[-1]] = scenario["prop"]
    return prop_start, B_start, A_start


def _matched_latent_layer(estimated, truth):
    estimated = np.asarray(estimated, dtype=np.uint8)
    truth = np.asarray(truth, dtype=np.uint8)
    score = np.empty((truth.shape[1], estimated.shape[1]), dtype=float)
    for true_index in range(truth.shape[1]):
        agreement = np.mean(
            estimated == truth[:, true_index : true_index + 1], axis=0
        )
        score[true_index] = np.maximum(agreement, 1.0 - agreement)
    true_indices, estimated_indices = linear_sum_assignment(-score)
    pairs = [
        (int(true_index), int(estimated_index))
        for true_index, estimated_index in zip(true_indices, estimated_indices)
    ]
    matched_score = float(np.mean(score[true_indices, estimated_indices]))
    coverage_adjusted = float(
        (
            np.sum(score[true_indices, estimated_indices])
            + 0.5 * (truth.shape[1] - len(pairs))
        )
        / truth.shape[1]
    )
    return pairs, matched_score, coverage_adjusted


def _align_support(estimated, truth, row_pairs, column_pairs, threshold):
    estimated = np.abs(np.asarray(estimated, dtype=float)[:, 1:]) >= threshold
    truth = np.asarray(truth, dtype=bool)
    n_true_rows, n_true_columns = truth.shape
    n_estimated_rows, n_estimated_columns = estimated.shape

    if row_pairs is None:
        row_map = {index: index for index in range(n_estimated_rows)}
        aligned_rows = n_true_rows
    else:
        row_map = {estimated_index: true_index for true_index, estimated_index in row_pairs}
        next_row = n_true_rows
        for estimated_index in range(n_estimated_rows):
            if estimated_index not in row_map:
                row_map[estimated_index] = next_row
                next_row += 1
        aligned_rows = next_row

    column_map = {
        estimated_index: true_index for true_index, estimated_index in column_pairs
    }
    next_column = n_true_columns
    for estimated_index in range(n_estimated_columns):
        if estimated_index not in column_map:
            column_map[estimated_index] = next_column
            next_column += 1

    estimated_aligned = np.zeros((aligned_rows, next_column), dtype=bool)
    for estimated_row in range(n_estimated_rows):
        for estimated_column in range(n_estimated_columns):
            estimated_aligned[
                row_map[estimated_row], column_map[estimated_column]
            ] = estimated[estimated_row, estimated_column]
    truth_aligned = np.zeros_like(estimated_aligned)
    truth_aligned[:n_true_rows, :n_true_columns] = truth
    return estimated_aligned, truth_aligned


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def calculate_metrics(result, X, A_true, scenario, threshold):
    latent_pairs = []
    latent_match = []
    latent_adjusted = []
    for estimated, truth in zip(result.A_list, A_true):
        pairs, matched, adjusted = _matched_latent_layer(estimated, truth)
        latent_pairs.append(pairs)
        latent_match.append(matched)
        latent_adjusted.append(adjusted)

    estimated_support = []
    true_support = []
    for layer, (estimated_B, truth_G) in enumerate(
        zip(result.B_list, scenario["G_true"])
    ):
        row_pairs = None if layer == 0 else latent_pairs[layer - 1]
        estimated_layer, true_layer = _align_support(
            estimated_B,
            truth_G,
            row_pairs,
            latent_pairs[layer],
            threshold,
        )
        estimated_support.append(estimated_layer.ravel())
        true_support.append(true_layer.ravel())
    estimated_support = np.concatenate(estimated_support)
    true_support = np.concatenate(true_support)
    structure = layer_structure_recovery_metrics(
        result.B_list, scenario["B_true"], result.A_list, A_true,
        threshold=threshold,
    )
    tp = int(np.sum(estimated_support & true_support))
    tn = int(np.sum(~estimated_support & ~true_support))
    fp = int(np.sum(estimated_support & ~true_support))
    fn = int(np.sum(~estimated_support & true_support))

    B1 = np.asarray(result.B_list[0], dtype=float)
    eta = B1[:, 0][None, :] + result.A_list[0] @ B1[:, 1:].T
    probabilities = expit(eta)
    log_loss = float(np.mean(np.logaddexp(0.0, eta) - X * eta))
    brier = float(np.mean((X - probabilities) ** 2))
    prevalence = np.clip(np.mean(X, axis=0), 1e-12, 1.0 - 1e-12)
    prevalence_logit = np.log(prevalence / (1.0 - prevalence))
    prevalence_log_loss = float(
        np.mean(np.logaddexp(0.0, prevalence_logit) - X * prevalence_logit)
    )
    true_B1 = scenario["B_true"][0]
    true_eta = (
        true_B1[:, 0][None, :] + A_true[0] @ true_B1[:, 1:].T
    )
    oracle_log_loss = float(
        np.mean(np.logaddexp(0.0, true_eta) - X * true_eta)
    )

    all_coefficients = np.concatenate([B.ravel() for B in result.B_list])
    all_slopes = np.concatenate([B[:, 1:].ravel() for B in result.B_list])
    K_fit = tuple(int(value) for value in result.K_list)
    return {
        "K_fit": json.dumps(K_fit),
        "exact_width": int(K_fit == scenario["K_true"]),
        "dimension_l1_error": int(
            sum(abs(fitted - truth) for fitted, truth in zip(K_fit, scenario["K_true"]))
        ),
        "underselected": int(any(f < t for f, t in zip(K_fit, scenario["K_true"]))),
        "overselected": int(any(f > t for f, t in zip(K_fit, scenario["K_true"]))),
        "latent_accuracy": float(np.mean(latent_match)),
        "latent_coverage_adjusted_accuracy": float(np.mean(latent_adjusted)),
        "layer_latent_accuracy": json.dumps(latent_match),
        "layer_latent_coverage_adjusted_accuracy": json.dumps(latent_adjusted),
        "latent_alignment": json.dumps(latent_pairs),
        "graph_accuracy": float(np.mean(estimated_support == true_support)),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "false_negative_rate": _safe_ratio(fn, fn + tp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "support_threshold": float(threshold),
        "structure_alignment": structure["alignment"],
        **{key: json.dumps(structure[key]) for key in STRUCTURE_FIELDS},
        "conditional_log_loss": log_loss,
        "prevalence_log_loss": prevalence_log_loss,
        "oracle_conditional_log_loss": oracle_log_loss,
        "brier_score": brier,
        "probability_saturation_fraction": float(
            np.mean((probabilities < 1e-6) | (probabilities > 1.0 - 1e-6))
        ),
        "coefficient_max_abs": float(np.max(np.abs(all_coefficients))),
        "slope_zero_fraction": float(np.mean(all_slopes == 0.0)),
        "slope_lower_bound_fraction": float(np.mean(np.isclose(all_slopes, -2.5))),
        "coefficient_upper_bound_fraction": float(
            np.mean(np.isclose(all_coefficients, 5.0))
        ),
    }


def _peak_rss_mib():
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage / (1024.0**2)
    return usage / 1024.0


def _seed_pair(base_seed, scenario_key, N, replicate):
    depth_offset = 20_000_000 if scenario_key == "d2" else 30_000_000
    data_seed = int(base_seed + depth_offset + 1000 * N + replicate)
    return data_seed, data_seed + 500_000_000


def run_one(task):
    scenario_key, N, replicate, settings = task
    scenario = build_scenario(scenario_key)
    data_seed, fit_seed = _seed_pair(
        settings["base_seed"], scenario_key, N, replicate
    )
    penalty_strength = max(
        settings["penalty_min_strength"],
        (N / settings["penalty_reference_n"]) ** settings[
            "penalty_n_power"
        ],
    )
    common = {
        "scenario": scenario["name"],
        "D": scenario["D"],
        "J": scenario["J"],
        "K_true": json.dumps(scenario["K_true"]),
        "K_max": json.dumps(scenario["K_max"]),
        "N": int(N),
        "replicate": int(replicate),
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "backend": settings["backend"],
        "t_spike": json.dumps(
            (
                np.asarray(scenario["t_spike"], dtype=float)
                * settings["t_spike_multiplier"]
            ).tolist()
        ),
        "t_spike_multiplier": settings["t_spike_multiplier"],
        "temperature": scenario["temperature"],
        "tau": settings["tau"],
        "penalty_strength": penalty_strength,
        "penalty_n_power": settings["penalty_n_power"],
        "penalty_reference_n": settings["penalty_reference_n"],
        "penalty_min_strength": settings["penalty_min_strength"],
        "shrinkage_burn": settings["shrinkage_burn"],
        "C": settings["C"],
        "tol": settings["tol"],
        "max_iter": settings["max_iter"],
        "optimizer_maxiter": settings["optimizer_maxiter"],
        "initialization": settings["initialization"],
    }
    started = time.perf_counter()
    try:
        X, A_true = simulate(scenario, N, data_seed)
        t_spike = (
            np.asarray(scenario["t_spike"], dtype=float)
            * settings["t_spike_multiplier"]
        )
        initialization_arguments = {}
        if settings["initialization"] == "truth":
            prop_start, B_start, A_start = _padded_truth_initialization(
                scenario, A_true, data_seed + 101
            )
            initialization_arguments = {
                "prop_in": prop_start,
                "B_list_in": B_start,
                "A_list_in": A_start,
            }
        with threadpool_limits(limits=1):
            result = fit(
                X,
                method="DDE",
                family="Bernoulli",
                D=scenario["D"],
                dimension_method="csp",
                t_spike=t_spike,
                temp=scenario["temperature"],
                tau=settings["tau"],
                C=settings["C"],
                tol=settings["tol"],
                max_it=settings["max_iter"],
                optimizer_maxiter=settings["optimizer_maxiter"],
                optimizer_tol=settings["optimizer_tol"],
                convergence_window=settings["convergence_window"],
                convergence_mean_tol=settings["convergence_mean_tol"],
                burn=settings["burn"],
                mcem_penalty_strength=penalty_strength,
                mcem_shrinkage_burn=settings["shrinkage_burn"],
                random_state=fit_seed,
                csp_backend=settings["backend"],
                keep_trace=settings["keep_trace"],
                verbose=False,
                **initialization_arguments,
            )
        metrics = calculate_metrics(result, X, A_true, scenario, settings["tau"])
        finite = bool(
            np.all(np.isfinite(result.prop))
            and all(np.all(np.isfinite(B)) for B in result.B_list)
            and np.isfinite(metrics["conditional_log_loss"])
        )
        trace = result.diagnostics.get("trace") or []
        return {
            **common,
            "status": "ok" if finite else "nonfinite",
            "error": "",
            "finite": finite,
            "iterations": int(result.diagnostics["iterations"]),
            "hit_iteration_limit": int(
                result.diagnostics["iterations"] >= settings["max_iter"]
            ),
            "stopped_by_convergence": int(
                result.diagnostics.get("stopped_by_convergence", False)
            ),
            "seconds": time.perf_counter() - started,
            "worker_peak_rss_mib": _peak_rss_mib(),
            "num_stage_transitions": len(
                result.diagnostics.get("stage_transitions", ())
            ),
            "K_trace": json.dumps(
                [entry["num_active"].astype(int).tolist() for entry in trace]
            ),
            "relative_change_trace": json.dumps(
                [list(entry["relative_change"]) for entry in trace]
            ),
            "final_max_relative_change": (
                max(trace[-1]["relative_change"]) if trace else ""
            ),
            "last_window_mean_max_relative_change": (
                float(
                    np.mean(
                        [
                            max(entry["relative_change"])
                            for entry in trace[-settings["convergence_window"] :]
                        ]
                    )
                )
                if trace
                else ""
            ),
            **metrics,
        }
    except Exception as exc:
        return {
            **common,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finite": False,
            "seconds": time.perf_counter() - started,
            "worker_peak_rss_mib": _peak_rss_mib(),
        }


def _write_csv(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else float("nan")


def _layer_mean(rows, field):
    values = [json.loads(row[field]) for row in rows if row.get(field, "") != ""]
    if not values:
        return ""
    return json.dumps(np.nanmean(np.asarray(values, dtype=float), axis=0).tolist())


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], int(row["N"]))].append(row)
    summary = []
    for (scenario, N), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "ok"]
        record = {
            "scenario": scenario,
            "D": int(group[0]["D"]),
            "N": N,
            "fits": len(group),
            "successful_fits": len(successful),
            "failed_fits": len(group) - len(successful),
        }
        for field in (
            "exact_width",
            "dimension_l1_error",
            "underselected",
            "overselected",
            "latent_accuracy",
            "latent_coverage_adjusted_accuracy",
            "graph_accuracy",
            "sensitivity",
            "specificity",
            "conditional_log_loss",
            "prevalence_log_loss",
            "oracle_conditional_log_loss",
            "brier_score",
            "probability_saturation_fraction",
            "slope_zero_fraction",
            "coefficient_max_abs",
            "coefficient_upper_bound_fraction",
            "iterations",
            "hit_iteration_limit",
            "stopped_by_convergence",
            "seconds",
            "worker_peak_rss_mib",
        ):
            record[f"{field}_mean"] = _mean(successful, field)
        for field in ("layer_latent_accuracy",) + STRUCTURE_FIELDS[:4]:
            record[f"{field}_mean"] = _layer_mean(successful, field)
        width_counts = defaultdict(int)
        for row in successful:
            width_counts[row["K_fit"]] += 1
        record["K_fit_counts"] = json.dumps(width_counts, sort_keys=True)
        summary.append(record)
    return summary


def write_report(path: Path, summary: list[dict], settings: dict):
    lines = [
        "# Bernoulli CSP systematic test",
        "",
        "Initial overspecified recovery and performance matrix.",
        "",
        "| Scenario | N | OK | Exact K | Mean K error | Under | Graph acc. | Latent acc. | Iter. | Max-it | Seconds | Peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {scenario} | {N} | {successful_fits}/{fits} | "
            "{exact_width_mean:.3f} | {dimension_l1_error_mean:.3f} | "
            "{underselected_mean:.3f} | {graph_accuracy_mean:.3f} | "
            "{latent_coverage_adjusted_accuracy_mean:.3f} | "
            "{iterations_mean:.1f} | {hit_iteration_limit_mean:.3f} | "
            "{seconds_mean:.2f} | {worker_peak_rss_mib_mean:.1f} |".format(**row)
        )
    total_fits = sum(int(row["fits"]) for row in summary)
    successful_fits = sum(int(row["successful_fits"]) for row in summary)
    overall_max_iter = np.mean(
        [float(row["hit_iteration_limit_mean"]) for row in summary]
    )
    lines.extend(
        [
            "",
            "## Automatic findings",
            "",
            f"- Numerical completion: {successful_fits}/{total_fits} fits succeeded.",
            f"- Mean grouped max-iteration rate: {overall_max_iter:.3f}.",
        ]
    )
    by_scenario = defaultdict(list)
    for row in summary:
        by_scenario[row["scenario"]].append(row)
    for scenario, rows in sorted(by_scenario.items()):
        rows.sort(key=lambda value: int(value["N"]))
        first, last = rows[0], rows[-1]
        log_loss_gap = (
            float(last["conditional_log_loss_mean"])
            - float(last["oracle_conditional_log_loss_mean"])
        )
        lines.append(
            f"- {scenario}: exact-width rate changes from "
            f"{float(first['exact_width_mean']):.3f} at N={first['N']} to "
            f"{float(last['exact_width_mean']):.3f} at N={last['N']}; "
            f"time {float(first['seconds_mean']):.2f}s to "
            f"{float(last['seconds_mean']):.2f}s; final conditional-log-loss "
            f"gap to oracle {log_loss_gap:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Settings",
            "",
            "```json",
            json.dumps(settings, indent=2),
            "```",
            "",
            "Peak RSS is the operating-system process high-water mark. Each worker "
            "handles one fit, so rows are comparable across configurations.",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_SPECS, default=["d2", "d3"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=list(SAMPLE_SIZES))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--backend", choices=("mcem", "mcem_reinitialize", "mcem_cap_only"), default="mcem")
    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--optimizer-tol", type=float, default=1e-6)
    parser.add_argument("--tol", type=float, default=DEFAULT_PARAMETRIC_CONVERGENCE_TOL)
    parser.add_argument("--convergence-window", type=int, default=7)
    parser.add_argument("--convergence-mean-tol", type=float, default=0.01)
    parser.add_argument("--burn", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--penalty-n-power", type=float, default=0.0)
    parser.add_argument("--penalty-reference-n", type=float, default=500.0)
    parser.add_argument("--penalty-min-strength", type=float, default=1.0)
    parser.add_argument("--shrinkage-burn", type=int, default=0)
    parser.add_argument("--t-spike-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--initialization", choices=("data", "truth"), default="data"
    )
    parser.add_argument("--base-seed", type=int, default=20260902)
    parser.add_argument("--keep-trace", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bernoulli_csp_systematic"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replicates < 1 or args.workers < 1 or args.C < 1:
        raise ValueError("replicates, workers, and C must be positive.")
    if args.max_iter < 1 or args.optimizer_maxiter < 1:
        raise ValueError("iteration budgets must be positive.")
    if args.shrinkage_burn < 0 or args.shrinkage_burn >= args.max_iter:
        raise ValueError("shrinkage_burn must lie in [0, max_iter).")
    if any(N < 1 for N in args.sample_sizes):
        raise ValueError("sample sizes must be positive.")
    if (
        not np.isfinite(args.t_spike_multiplier)
        or args.t_spike_multiplier <= 0
    ):
        raise ValueError("t_spike_multiplier must be finite and positive.")
    if not np.isfinite(args.penalty_n_power):
        raise ValueError("penalty_n_power must be finite.")
    if (
        not np.isfinite(args.penalty_reference_n)
        or args.penalty_reference_n <= 0
    ):
        raise ValueError("penalty_reference_n must be finite and positive.")
    if (
        not np.isfinite(args.penalty_min_strength)
        or args.penalty_min_strength <= 0
    ):
        raise ValueError("penalty_min_strength must be finite and positive.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / "fits.csv"
    rows = _read_csv(fits_path) if args.resume else []

    settings = {
        "backend": args.backend,
        "C": args.C,
        "max_iter": args.max_iter,
        "optimizer_maxiter": args.optimizer_maxiter,
        "optimizer_tol": args.optimizer_tol,
        "tol": args.tol,
        "convergence_window": args.convergence_window,
        "convergence_mean_tol": args.convergence_mean_tol,
        "burn": args.burn,
        "tau": args.tau,
        "penalty_n_power": args.penalty_n_power,
        "penalty_reference_n": args.penalty_reference_n,
        "penalty_min_strength": args.penalty_min_strength,
        "shrinkage_burn": args.shrinkage_burn,
        "t_spike_multiplier": args.t_spike_multiplier,
        "initialization": args.initialization,
        "base_seed": args.base_seed,
        "keep_trace": args.keep_trace,
    }
    completed = {
        (row["scenario"], int(row["N"]), int(row["replicate"])) for row in rows
    }
    tasks = []
    for key in args.scenarios:
        scenario = build_scenario(key)
        for N in args.sample_sizes:
            for replicate in range(1, args.replicates + 1):
                identity = (scenario["name"], int(N), replicate)
                if identity not in completed:
                    tasks.append((key, int(N), replicate, settings))

    total = len(tasks) + len(completed)
    print(
        f"Bernoulli CSP systematic test: {len(tasks)} pending / {total} total; "
        f"workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers, max_tasks_per_child=1
        ) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                rows.sort(
                    key=lambda value: (
                        value["scenario"], int(value["N"]), int(value["replicate"])
                    )
                )
                _write_csv(fits_path, rows)
                print(
                    f"[{index}/{len(tasks)}] {row['scenario']} N={row['N']} "
                    f"rep={row['replicate']} status={row['status']} "
                    f"K={row.get('K_fit', '')} sec={float(row['seconds']):.2f}",
                    flush=True,
                )

    summary = summarize(rows)
    _write_csv(output_dir / "summary.csv", summary)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sample_sizes": args.sample_sizes,
        "replicates": args.replicates,
        "workers": args.workers,
        "settings": settings,
        "scenarios": {
            key: {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in build_scenario(key).items()
                if name not in {"B_true", "G_true", "prop"}
            }
            for key in args.scenarios
        },
        "completed_rows": len(rows),
        "successful_rows": sum(row["status"] == "ok" for row in rows),
        "failed_rows": sum(row["status"] != "ok" for row in rows),
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    _write_json(output_dir / "metadata.json", metadata)
    write_report(output_dir / "REPORT.md", summary, settings)
    print(f"results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
