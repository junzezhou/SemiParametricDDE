"""Shared Normal-DDE CSP simulation scenarios and helpers.

The D=2 and D=3 scenarios mirror the overspecified structures used by the
CopDDE reference simulations (Gaussian observations, no marginal transform).


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


def _scenario_d2_sparse_deep():
    """D=2 falsification case whose top width is below its structural cap."""
    scenario = _scenario_d2()
    K_true = (scenario["K_true"][0], 2)
    B2_slope = _deep_loading(K_true[0], K_true[1], 4.0)
    scenario.update(
        name="d2_j50_sparse_deep",
        K_true=K_true,
        B_true=[
            scenario["B_true"][0],
            np.column_stack([np.full(K_true[0], -2.0), B2_slope]),
        ],
        prop=np.full(K_true[-1], 0.5),
    )
    return scenario


def _scenario_d3():
    J = 108
    K_true = (18, 6, 2)
    K_max = (36, 12, 4)
    B1_negative_columns = (
        np.array([4, 14, 10, 13, 2, 16, 14, 9, 4, 17, 1, 17, 4, 1, 8, 4, 4, 5]) - 1
    )
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
    A_true[-1] = rng.binomial(1, scenario["prop"], size=(N, K_true[-1])).astype(
        np.uint8
    )
    for d in range(D - 2, -1, -1):
        design = np.column_stack([np.ones(N), A_true[d + 1]])
        probability = expit(design @ B_true[d + 1].T)
        A_true[d] = rng.binomial(1, probability).astype(np.uint8)
    mean = np.column_stack([np.ones(N), A_true[0]]) @ B_true[0].T
    X = mean + rng.normal(scale=np.sqrt(scenario["gamma"]), size=mean.shape)
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
    return max(0.3, 3.0 * N**-0.3)


def _write_csv(path, records):
    fields = sorted({key for record in records for key in record})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _write_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _append_jsonl(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")))
        stream.write("\n")


def _read_jsonl(path):
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if any(remaining.strip() for remaining in stream):
                    raise ValueError(f"Invalid JSONL checkpoint at line {line_number}.")
                break
    return records


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
        "mcem_penalty_strength",
        "stage_transitions",
        "stage_caps",
        "stage_transition_events",
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


