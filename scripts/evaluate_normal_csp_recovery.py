"""Systematic recovery diagnostics for Normal-DDE CSP estimation.

The scenarios mirror the overspecified D=2 and D=3 structures used by the
CopDDE reference simulations.  The script writes per-fit and grouped summary
CSVs so initialization, CSP allocation, thresholding, and temperature effects
can be inspected separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

from dde import fit
from dde.algorithms.csp import csp_active_mask, update_csp_layer
from dde.algorithms.normal_init_semi import Normal_init_semi


def _block_loadings(
    J,
    K,
    block_size,
    positive,
    negative,
    seed,
    *,
    negative_columns=None,
):
    rng = np.random.default_rng(seed)
    out = np.zeros((J, K), dtype=float)
    n_blocks = int(np.ceil(J / block_size))
    if K < n_blocks:
        raise ValueError("K must be at least the number of loading blocks.")
    for block in range(n_blocks):
        start = block * block_size
        stop = min((block + 1) * block_size, J)
        positive_column = block
        if negative_columns is None:
            choices = np.delete(np.arange(K), positive_column)
            negative_column = int(rng.choice(choices))
        else:
            negative_column = int(negative_columns[block])
        out[start:stop, positive_column] += positive
        out[start:stop, negative_column] -= negative
    return out


def _deep_loading(n_child, n_parent, maximum):
    out = np.zeros((n_child, n_parent), dtype=float)
    indices = np.floor(np.linspace(1, n_child, n_parent + 1)).astype(int)
    for i in range(n_parent):
        start = indices[i] - 1
        stop = indices[i + 1]
        out[start:stop, i] = maximum
        if i % 2 == 0 and i < n_parent - 2:
            out[start:stop, i + 1] = -maximum / 3.0
        elif i % 2 == 1:
            out[start:stop, i - 1] = -maximum / 3.0
    return out


def _scenario_d2():
    J = 50
    K_true = (10, 3)
    K_max = (16, 5)
    B1_slope = _block_loadings(J, K_true[0], 5, 10.0, 5.0, seed=1)
    thirds = J // 3
    B1_intercept = np.concatenate(
        [
            np.full(thirds, -2.0),
            np.full(thirds, -4.0),
            np.full(J - 2 * thirds, -2.0),
        ]
    )
    B2_slope = _deep_loading(K_true[0], K_true[1], 4.0)
    child_thirds = K_true[0] // 3
    B2_intercept = np.concatenate(
        [
            np.full(child_thirds, -2.0),
            np.full(child_thirds, -1.0),
            np.full(K_true[0] - 2 * child_thirds, -2.0),
        ]
    )
    return {
        "name": "d2_j50",
        "D": 2,
        "J": J,
        "K_true": K_true,
        "K_max": K_max,
        "B_true": [
            np.column_stack([B1_intercept, B1_slope]),
            np.column_stack([B2_intercept, B2_slope]),
        ],
        "gamma": np.ones(J),
        "prop": np.full(K_true[-1], 0.5),
        "base_t_spike": np.array([0.02, 0.04]),
        "base_temp": 0.7,
    }


def _scenario_d3():
    J = 108
    K_true = (18, 6, 2)
    K_max = (36, 12, 4)
    B1_negative_columns = np.array(
        [4, 14, 10, 13, 2, 16, 14, 9, 4, 17, 1, 17, 4, 1, 8, 4, 4, 5]
    ) - 1
    B2_negative_columns = np.array([2, 5, 4, 5, 1, 5]) - 1
    B1_slope = _block_loadings(
        J,
        K_true[0],
        6,
        10.0,
        5.0,
        seed=1,
        negative_columns=B1_negative_columns,
    )
    B2_slope = _block_loadings(
        K_true[0],
        K_true[1],
        3,
        4.0,
        4.0 / 3.0,
        seed=1,
        negative_columns=B2_negative_columns,
    )
    B3_slope = _deep_loading(K_true[1], K_true[2], 4.0)
    return {
        "name": "d3_j108",
        "D": 3,
        "J": J,
        "K_true": K_true,
        "K_max": K_max,
        "B_true": [
            np.column_stack([np.full(J, -2.0), B1_slope]),
            np.column_stack([np.full(K_true[0], -2.0), B2_slope]),
            np.column_stack([np.full(K_true[1], -2.0), B3_slope]),
        ],
        "gamma": np.ones(J),
        "prop": np.full(K_true[-1], 0.5),
        "base_t_spike": np.array([0.01, 0.03, 0.005]),
        "base_temp": 0.9,
    }


def _simulate(scenario, N, seed):
    rng = np.random.default_rng(seed)
    B_true = scenario["B_true"]
    K_true = scenario["K_true"]
    D = scenario["D"]
    A_true = [None] * D
    A_true[-1] = rng.binomial(
        1, scenario["prop"], size=(N, K_true[-1])
    ).astype(np.uint8)
    for d in range(D - 2, -1, -1):
        design = np.column_stack([np.ones(N), A_true[d + 1]])
        probability = expit(design @ B_true[d + 1].T)
        A_true[d] = rng.binomial(1, probability).astype(np.uint8)
    mean = np.column_stack([np.ones(N), A_true[0]]) @ B_true[0].T
    X = mean + rng.normal(
        scale=np.sqrt(scenario["gamma"]), size=mean.shape
    )
    return X, A_true


def _padded_truth(scenario, A_true, seed):
    rng = np.random.default_rng(seed)
    K_true = scenario["K_true"]
    K_max = scenario["K_max"]
    B_true = scenario["B_true"]
    J = scenario["J"]
    D = scenario["D"]

    B_max = [np.zeros((J, K_max[0] + 1), dtype=float)]
    B_max[0][:, 0] = B_true[0][:, 0]
    B_max[0][:, 1 : K_true[0] + 1] = B_true[0][:, 1:]
    for d in range(1, D):
        B_d = np.zeros((K_max[d - 1], K_max[d] + 1), dtype=float)
        B_d[: K_true[d - 1], 0] = B_true[d][:, 0]
        B_d[: K_true[d - 1], 1 : K_true[d] + 1] = B_true[d][:, 1:]
        B_max.append(B_d)

    A_max = []
    for true_A, true_width, max_width in zip(A_true, K_true, K_max):
        augmented = rng.integers(
            0, 2, size=(true_A.shape[0], max_width), dtype=np.uint8
        )
        augmented[:, :true_width] = true_A
        A_max.append(augmented)
    prop_max = np.full(K_max[-1], 0.5)
    return prop_max, B_max, scenario["gamma"].copy(), A_max


def _initialization(scenario, X, A_true, mode, seed):
    truth = _padded_truth(scenario, A_true, seed)
    if mode == "truth":
        return truth
    _, B_max, _, _ = truth
    G_max = [(B[:, 1:] != 0) for B in B_max]
    np.random.seed(seed)
    if mode == "data":
        return Normal_init_semi(
            X,
            scenario["K_max"][0],
            scenario["K_max"],
            mode="data",
        )[:4]
    if mode == "oracle":
        return Normal_init_semi(
            X,
            scenario["K_max"][0],
            scenario["K_max"],
            B_max,
            G_max,
            mode="oracle",
        )[:4]
    raise ValueError(f"Unknown initialization mode: {mode}")


def _matched_binary_accuracy(A_est, A_true):
    A_est = np.asarray(A_est, dtype=np.uint8)
    A_true = np.asarray(A_true, dtype=np.uint8)
    if A_true.shape[1] == 0 or A_est.shape[1] == 0:
        return float("nan")
    score = np.empty((A_true.shape[1], A_est.shape[1]), dtype=float)
    for j in range(A_true.shape[1]):
        agreement = np.mean(A_est == A_true[:, j : j + 1], axis=0)
        score[j] = np.maximum(agreement, 1.0 - agreement)
    rows, columns = linear_sum_assignment(-score)
    return float(np.mean(score[rows, columns]))


def _max_abs_column_correlation(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] < 2:
        return float("nan")
    centered = A - np.mean(A, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    np.fill_diagonal(correlation, 0.0)
    return float(np.max(np.abs(correlation)))


def _allocation_counts(B_list, t_spike, K_max, seed):
    counts = []
    masks = []
    for d, B in enumerate(B_list):
        omega = np.ones(K_max[d] + 1) / (K_max[d] + 1)
        num_rows = B.shape[0]
        allocation, _, _, _ = update_csp_layer(
            B,
            omega,
            t_spike[d],
            np.ones(K_max[d]),
            num_rows,
            rng=np.random.default_rng(seed + d),
        )
        mask = csp_active_mask(allocation)
        counts.append(int(np.sum(mask)))
        masks.append(mask)
    return tuple(counts), masks


def _tau_default(N):
    return max(0.3, 3.0 * N ** -0.3)


def _configuration_grid(scenario, sample_sizes, replicates, replicate_start=0):
    configurations = set()
    replicate_indices = range(replicate_start, replicate_start + replicates)
    for N in sample_sizes:
        for multiplier in (0.5, 1.0, 2.0):
            for replicate in replicate_indices:
                configurations.add(
                    ("spike_grid", N, replicate, "data", multiplier, scenario["base_temp"], "paper")
                )

    diagnostic_N = sample_sizes[0]
    for replicate in replicate_indices:
        for mode in ("oracle", "truth"):
            configurations.add(
                ("initialization", diagnostic_N, replicate, mode, 1.0, scenario["base_temp"], "paper")
            )
        for temperature in (scenario["base_temp"], 1.0):
            for threshold_name in ("paper", "near_zero"):
                if temperature == scenario["base_temp"] and threshold_name == "paper":
                    continue
                configurations.add(
                    ("anneal_threshold", diagnostic_N, replicate, "data", 1.0, temperature, threshold_name)
                )
    return sorted(configurations)


def _run_one(
    scenario,
    configuration,
    max_iter,
    optimizer_maxiter,
    mc_samples,
    seed_base,
    dataset_cache,
    initialization_cache,
    keep_trace=False,
    csp_backend="saem",
    csp_min_iter=30,
):
    group, N, replicate, init_mode, multiplier, temperature, threshold_name = configuration
    data_seed = seed_base + 100_000 * scenario["D"] + 1_000 * N + replicate
    init_seed = data_seed + 101
    fit_seed = data_seed + 202
    dataset_key = (scenario["name"], N, replicate)
    if dataset_key not in dataset_cache:
        dataset_cache[dataset_key] = _simulate(scenario, N, data_seed)
    X, A_true = dataset_cache[dataset_key]
    initialization_key = (*dataset_key, init_mode)
    if initialization_key not in initialization_cache:
        initialization_cache[initialization_key] = _initialization(
            scenario, X, A_true, init_mode, init_seed
        )
    prop0, B0, gamma0, A0 = initialization_cache[initialization_key]
    t_spike = scenario["base_t_spike"] * multiplier
    initial_counts, _ = _allocation_counts(
        B0, t_spike, scenario["K_max"], init_seed + 500
    )
    true_padded = _padded_truth(scenario, A_true, init_seed + 700)
    truth_counts, _ = _allocation_counts(
        true_padded[1], t_spike, scenario["K_max"], init_seed + 900
    )
    initial_A_accuracy = tuple(
        _matched_binary_accuracy(A_est, A_layer_true)
        for A_est, A_layer_true in zip(A0, A_true)
    )

    tau = _tau_default(N) if threshold_name == "paper" else 1e-8
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
        tol=0.005,
        max_it=max_iter,
        prop_in=prop0,
        B_list_in=B0,
        gamma_in=gamma0,
        A_list_in=A0,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_tol=1e-6,
        random_state=fit_seed,
        keep_trace=keep_trace,
        csp_backend=csp_backend,
        csp_min_iter=csp_min_iter,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    K_fit = tuple(int(value) for value in result.K_list)
    final_A_accuracy = tuple(
        _matched_binary_accuracy(A_est, A_layer_true)
        for A_est, A_layer_true in zip(result.A_list, A_true)
    )
    trace = result.diagnostics.get("trace", [])
    first_csp_entry = next(
        (entry for entry in trace if entry.get("phase") == "saem_csp"),
        trace[0] if trace else None,
    )
    first_active = (
        tuple(first_csp_entry["num_active"].tolist())
        if first_csp_entry is not None
        else None
    )

    record = {
        "scenario": scenario["name"],
        "group": group,
        "N": N,
        "replicate": replicate,
        "init_mode": init_mode,
        "spike_multiplier": multiplier,
        "t_spike": json.dumps(t_spike.tolist()),
        "temperature": temperature,
        "threshold_name": threshold_name,
        "tau": tau,
        "mc_samples": mc_samples,
        "csp_backend": csp_backend,
        "csp_min_iter": csp_min_iter,
        "K_true": json.dumps(list(scenario["K_true"])),
        "K_max": json.dumps(list(scenario["K_max"])),
        "K_initial_allocation": json.dumps(list(initial_counts)),
        "K_truth_allocation": json.dumps(list(truth_counts)),
        "K_first_iteration": (
            json.dumps(list(first_active)) if first_active is not None else ""
        ),
        "K_fit": json.dumps(list(K_fit)),
        "K_trace": json.dumps(
            [entry["num_active"].astype(int).tolist() for entry in trace]
        ),
        "allocation_trace": json.dumps(
            [
                [values.astype(int).tolist() for values in entry["allocation"]]
                for entry in trace
            ]
        ),
        "omega_trace": json.dumps(
            [
                [values.tolist() for values in entry["omega"]]
                for entry in trace
            ]
        ),
        "allocation_input_column_l1_trace": json.dumps(
            [
                [values.tolist() for values in entry["allocation_input_column_l1"]]
                for entry in trace
            ]
        ),
        "updated_column_l1_trace": json.dumps(
            [
                [values.tolist() for values in entry["updated_column_l1"]]
                for entry in trace
            ]
        ),
        "exact_width": int(K_fit == scenario["K_true"]),
        "collapsed_layer": int(any(value == 0 for value in K_fit)),
        "iterations": result.diagnostics["iterations"],
        "sa_iterations": result.diagnostics.get("sa_iterations", 0),
        "csp_start_iteration": result.diagnostics.get(
            "csp_start_iteration", np.nan
        ),
        "stopped_by_convergence": int(
            result.diagnostics.get("stopped_by_convergence", False)
        ),
        "hit_max_iter": int(result.diagnostics["iterations"] >= max_iter),
        "dimension_l1_error": int(
            sum(
                abs(fitted - truth)
                for fitted, truth in zip(K_fit, scenario["K_true"])
            )
        ),
        "fit_seconds": elapsed,
    }
    for d in range(scenario["D"]):
        record[f"K{d + 1}_true"] = scenario["K_true"][d]
        record[f"K{d + 1}_initial"] = initial_counts[d]
        record[f"K{d + 1}_truth_allocation"] = truth_counts[d]
        record[f"K{d + 1}_first"] = (
            first_active[d] if first_active is not None else np.nan
        )
        record[f"K{d + 1}_fit"] = K_fit[d]
        record[f"K{d + 1}_error"] = K_fit[d] - scenario["K_true"][d]
        record[f"A{d + 1}_initial_accuracy"] = initial_A_accuracy[d]
        record[f"A{d + 1}_final_accuracy"] = final_A_accuracy[d]
        record[f"A{d + 1}_max_abs_correlation"] = _max_abs_column_correlation(
            result.A_list[d]
        )
        record[f"B{d + 1}_final_column_l1"] = json.dumps(
            np.sum(np.abs(result.B_list[d][:, 1:]), axis=0).tolist()
        )
    return record


def _write_csv(path, records):
    fields = sorted({key for record in records for key in record})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _read_csv(path):
    categorical = {
        "scenario",
        "group",
        "init_mode",
        "threshold_name",
        "csp_backend",
        "t_spike",
        "K_true",
        "K_max",
        "K_initial_allocation",
        "K_truth_allocation",
        "K_first_iteration",
        "K_fit",
        "K_trace",
        "allocation_trace",
        "omega_trace",
        "allocation_input_column_l1_trace",
        "updated_column_l1_trace",
    }
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for key, value in tuple(record.items()):
            if key in categorical or key.startswith("B") or value == "":
                continue
            try:
                record[key] = float(value)
            except ValueError:
                pass
    return records


def _record_configuration_key(record):
    return (
        record["scenario"],
        record["group"],
        int(float(record["N"])),
        int(float(record["replicate"])),
        record["init_mode"],
        float(record["spike_multiplier"]),
        float(record["temperature"]),
        record["threshold_name"],
        int(float(record["mc_samples"])),
        record.get("csp_backend", "mcem"),
        int(float(record.get("csp_min_iter", 0))),
    )


def _target_configuration_key(
    scenario,
    configuration,
    mc_samples,
    csp_backend,
    csp_min_iter,
):
    group, N, replicate, init_mode, multiplier, temperature, threshold_name = configuration
    return (
        scenario["name"],
        group,
        int(N),
        int(replicate),
        init_mode,
        float(multiplier),
        float(temperature),
        threshold_name,
        int(mc_samples),
        str(csp_backend),
        int(csp_min_iter),
    )


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if np.any(finite) else float("nan")


def _summarize(records):
    groups = defaultdict(list)
    keys = (
        "scenario",
        "group",
        "N",
        "init_mode",
        "spike_multiplier",
        "temperature",
        "threshold_name",
        "mc_samples",
        "csp_backend",
    )
    for record in records:
        groups[
            tuple(
                record.get(key, "mcem" if key == "csp_backend" else None)
                for key in keys
            )
        ].append(record)

    summary = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(keys, key))
        item["fits"] = len(rows)
        item["exact_width_rate"] = float(np.mean([row["exact_width"] for row in rows]))
        item["collapse_rate"] = float(np.mean([row["collapsed_layer"] for row in rows]))
        item["hit_max_iter_rate"] = float(np.mean([row["hit_max_iter"] for row in rows]))
        item["mean_dimension_l1_error"] = float(
            np.mean([row["dimension_l1_error"] for row in rows])
        )
        item["mean_fit_seconds"] = float(np.mean([row["fit_seconds"] for row in rows]))
        D = len(json.loads(rows[0]["K_true"]))
        for d in range(D):
            for suffix in ("initial", "truth_allocation", "first", "fit", "error"):
                item[f"mean_K{d + 1}_{suffix}"] = float(
                    np.mean([row[f"K{d + 1}_{suffix}"] for row in rows])
                )
            item[f"mean_A{d + 1}_initial_accuracy"] = _nanmean_or_nan(
                [row[f"A{d + 1}_initial_accuracy"] for row in rows]
            )
            item[f"mean_A{d + 1}_final_accuracy"] = _nanmean_or_nan(
                [row[f"A{d + 1}_final_accuracy"] for row in rows]
            )
            item[f"mean_A{d + 1}_max_abs_correlation"] = _nanmean_or_nan(
                [row[f"A{d + 1}_max_abs_correlation"] for row in rows]
            )
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument(
        "--csp-backend", choices=("saem", "mcem"), default="saem"
    )
    parser.add_argument("--csp-min-iter", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=20260828)
    parser.add_argument(
        "--keep-trace",
        action="store_true",
        help="Store full per-iteration CSP histories in the fit CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configurations already checkpointed in the output fit CSV.",
    )
    parser.add_argument(
        "--spike-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--initialization-modes",
        nargs="+",
        choices=("oracle", "truth"),
        default=["oracle", "truth"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("d2_j50", "d3_j108"),
        default=["d2_j50", "d3_j108"],
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("spike_grid", "initialization", "anneal_threshold"),
        default=["spike_grid", "initialization", "anneal_threshold"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Results/normal_dde_csp_diagnostics"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        scenario
        for scenario in (_scenario_d2(), _scenario_d3())
        if scenario["name"] in args.scenarios
    ]
    allowed_groups = set(args.groups)
    configuration_by_scenario = {
        scenario["name"]: [
            configuration
            for configuration in _configuration_grid(
                scenario,
                args.sample_sizes,
                args.replicates,
                args.replicate_start,
            )
            if configuration[0] in allowed_groups
            and (
                configuration[0] != "spike_grid"
                or configuration[4] in args.spike_multipliers
            )
            and (
                configuration[0] != "initialization"
                or configuration[3] in args.initialization_modes
            )
        ]
        for scenario in scenarios
    }
    fits_path = args.output_dir / "normal_csp_recovery_fits.csv"
    records = _read_csv(fits_path) if args.resume and fits_path.exists() else []
    completed_keys = {_record_configuration_key(record) for record in records}
    dataset_cache = {}
    initialization_cache = {}
    total = sum(
        len(configuration_by_scenario[scenario["name"]])
        for scenario in scenarios
    )
    completed = sum(
        _target_configuration_key(
            scenario,
            configuration,
            args.mc_samples,
            args.csp_backend,
            args.csp_min_iter,
        )
        in completed_keys
        for scenario in scenarios
        for configuration in configuration_by_scenario[scenario["name"]]
    )
    for scenario in scenarios:
        for configuration in configuration_by_scenario[scenario["name"]]:
            configuration_key = _target_configuration_key(
                scenario,
                configuration,
                args.mc_samples,
                args.csp_backend,
                args.csp_min_iter,
            )
            if configuration_key in completed_keys:
                continue
            record = _run_one(
                scenario,
                configuration,
                args.max_iter,
                args.optimizer_maxiter,
                args.mc_samples,
                args.seed_base,
                dataset_cache,
                initialization_cache,
                keep_trace=args.keep_trace,
                csp_backend=args.csp_backend,
                csp_min_iter=args.csp_min_iter,
            )
            records.append(record)
            completed_keys.add(configuration_key)
            _write_csv(fits_path, records)
            completed += 1
            print(
                f"[{completed}/{total}] {record['scenario']} {record['group']} "
                f"N={record['N']} init={record['init_mode']} "
                f"mult={record['spike_multiplier']} K={record['K_fit']}",
                flush=True,
            )

    summary = _summarize(records)
    _write_csv(fits_path, records)
    _write_csv(args.output_dir / "normal_csp_recovery_summary.csv", summary)
    metadata = {
        "replicates": args.replicates,
        "replicate_start": args.replicate_start,
        "sample_sizes": args.sample_sizes,
        "max_iter": args.max_iter,
        "optimizer_maxiter": args.optimizer_maxiter,
        "mc_samples": args.mc_samples,
        "csp_backend": args.csp_backend,
        "csp_min_iter": args.csp_min_iter,
        "seed_base": args.seed_base,
        "keep_trace": args.keep_trace,
        "scenarios_selected": args.scenarios,
        "groups_selected": args.groups,
        "spike_multipliers_selected": args.spike_multipliers,
        "initialization_modes_selected": args.initialization_modes,
        "fit_count": len(records),
        "scenarios": [
            {
                "name": scenario["name"],
                "K_true": scenario["K_true"],
                "K_max": scenario["K_max"],
                "base_t_spike": scenario["base_t_spike"].tolist(),
                "base_temp": scenario["base_temp"],
            }
            for scenario in scenarios
        ],
    }
    with (args.output_dir / "normal_csp_recovery_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)


if __name__ == "__main__":
    main()
