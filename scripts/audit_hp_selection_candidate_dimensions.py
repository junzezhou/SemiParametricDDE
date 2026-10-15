"""Inspect fitted widths for every CSP candidate in selected validation cells."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dde import auto_grid_copula_csp, select_hyperparameters
from evaluate_hp_selection_validation import DEFAULT_OUTPUT, _data, _fit_kwargs


FIELDS = ("kernel", "D", "N", "seed", "candidate_index", "candidate",
          "score", "full_grid_score", "score_difference", "true_K",
          "fitted_K", "dimension_exact", "status", "error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_OUTPUT / "candidate_dimension_audit.csv")
    parser.add_argument("--kernel", choices=("dde_bernoulli_csp", "dde_normal_csp",
                                              "semidde_csp"), required=True)
    parser.add_argument("--depth", type=int, choices=(2, 3), required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=20261014)
    parser.add_argument("--dde-iterations", type=int, default=20)
    parser.add_argument("--semi-outer-iterations", type=int, default=2)
    parser.add_argument("--semi-inner-iterations", type=int, default=20)
    parser.add_argument("--optimizer-iterations", type=int, default=30)
    parser.add_argument("--score-mc-samples", type=int, default=4096)
    args = parser.parse_args()
    kernel, depth, N = args.kernel, args.depth, args.sample_size
    seed = int(args.seed_base + 10_000_000 * depth + 10_000 * N)
    X, truth = _data(kernel, depth, N, seed)
    candidates = auto_grid_copula_csp(N, depth, tau_values=[0.3])
    method = "DDE" if kernel.startswith("dde_") else "SemiDDE"
    family = kernel.split("_")[1].title() if method == "DDE" else None
    reference = DEFAULT_OUTPUT / "cells" / f"{kernel}_d{depth}_n{N}_r0.json"
    full_scores = [row["mean_score"] for row in json.loads(reference.read_text())["cv_table"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.exists():
        with args.output.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    completed = {(row["kernel"], int(row["D"]), int(row["N"]),
                  int(row["candidate_index"])) for row in existing}
    for index, candidate in enumerate(candidates):
        key = (kernel, depth, N, index)
        if key in completed:
            continue
        row = {"kernel": kernel, "D": depth, "N": N, "seed": seed,
               "candidate_index": index, "candidate": json.dumps(candidate),
               "full_grid_score": full_scores[index],
               "true_K": json.dumps(truth["K_true"]), "status": "error", "error": ""}
        try:
            np.random.seed(seed)
            with threadpool_limits(limits=1):
                result = select_hyperparameters(
                    X, D=depth, method=method, family=family,
                    dimension_method="csp", candidates=[candidate],
                    criterion="ebic", base_fit_kwargs=_fit_kwargs(kernel, args),
                    random_state=seed, refit=True,
                )
            fitted = tuple(map(int, result.K_list))
            score = float(result.diagnostics["cv_table"][0]["mean_score"])
            row.update(status="ok", score=score,
                       score_difference=score-float(full_scores[index]),
                       fitted_K=json.dumps(fitted),
                       dimension_exact=int(fitted == tuple(truth["K_true"])))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        existing.append(row)
        temporary = args.output.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(temporary, args.output)
        print(f"{kernel} D={depth} N={N} candidate {index+1}/{len(candidates)}: "
              f"{row['status']} {row.get('fitted_K','')}", flush=True)


if __name__ == "__main__":
    main()
