"""Refit leading Copula CSP candidates to expose their active layer widths."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dde import fit
from evaluate_hp_selection_validation import DEFAULT_OUTPUT, _data


def audit(source: Path, output: Path, depth: int, N: int, top_k: int):
    fits = list(csv.DictReader((source / "fits.csv").open(newline="")))
    match = [row for row in fits if row["kernel"] == "copula_csp"
             and int(row["D"]) == depth and int(row["N"]) == N]
    assert len(match) == 1
    seed = int(match[0]["seed"])
    X, truth = _data("copula_csp", depth, N, seed)
    diagnostic = json.loads((source / "cells" / f"copula_csp_d{depth}_n{N}_r0.json").read_text())
    candidates = sorted(diagnostic["cv_table"], key=lambda row: row["mean_pmse"])[:top_k]
    existing = list(csv.DictReader(output.open(newline=""))) if output.exists() else []
    completed = {int(row["candidate_index"]) for row in existing}
    fields = ("D", "N", "seed", "candidate_index", "mean_pmse", "t_spike",
              "fitted_K", "true_K", "dimension_exact", "seconds", "status", "error")
    output.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        index = int(candidate["candidate_index"])
        if index in completed:
            continue
        row = {"D": depth, "N": N, "seed": seed, "candidate_index": index,
               "mean_pmse": candidate["mean_pmse"],
               "t_spike": json.dumps(candidate["t_spike"]),
               "true_K": json.dumps(truth["K_true"]), "status": "error", "error": ""}
        started = time.perf_counter()
        try:
            np.random.seed(seed)
            with threadpool_limits(limits=1):
                fitted = fit(
                    X, method="CopDDE", D=depth, dimension_method="csp",
                    K_list=candidate["K_list"], C=candidate["C"],
                    max_iter=candidate["max_iter"],
                    optimizer_maxiter=candidate["optimizer_maxiter"],
                    temp=candidate["temp"], tau=candidate["tau"],
                    t_spike=candidate["t_spike"],
                    initialize=candidate["initialize"],
                    verbose=False, random_state=seed,
                )
            widths = tuple(map(int, fitted.K_list))
            row.update(status="ok", fitted_K=json.dumps(widths),
                       dimension_exact=int(widths == tuple(truth["K_true"])))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - started
        existing.append(row)
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(temporary, output)
        print(f"Candidate {index}: {row['status']} {row.get('fitted_K', '')}", flush=True)
    if int(match[0]["best_index"]) in {int(row["candidate_index"]) for row in existing}:
        best = next(row for row in existing if int(row["candidate_index"]) == int(match[0]["best_index"]))
        assert best["fitted_K"] == match[0]["fitted_K"], "Refit disagrees with selected result"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT / "copula_d3_default_pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "copula_candidate_width_audit.csv")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    audit(args.source, args.output, args.depth, args.sample_size, args.top_k)
