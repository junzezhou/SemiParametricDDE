"""Replace superseded pilot cells with seed-controlled, paper-matched runs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn

from evaluate_hp_selection_validation import DEFAULT_OUTPUT, KERNELS, N_VALUES, _write
from summarize_hp_selection_validation import summarize
from repair_hp_selection_npz import repair
from check_hp_selection_validation import check


def _read(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _key(row):
    return row["kernel"], int(row["D"]), int(row["N"]), int(row["replicate"])


def _check_semi_mc_equivalence(output: Path):
    basename = "semidde_csp_d3_n500_r0.json"
    reduced = json.loads((output / "noncop_csp_default_full" / "cells" / basename).read_text())
    full = json.loads((output / "semidde_full_mc_pilot" / "cells" / basename).read_text())
    reduced_scores = np.asarray([row["mean_score"] for row in reduced["cv_table"]], dtype=float)
    full_scores = np.asarray([row["mean_score"] for row in full["cv_table"]], dtype=float)
    assert reduced_scores.shape == full_scores.shape == (27,)
    maximum = float(np.max(np.abs(reduced_scores - full_scores)))
    assert maximum < 1e-10, f"SemiDDE MC-budget score discrepancy: {maximum}"
    reduced_fit = _read(output / "noncop_csp_default_full" / "fits.csv")
    full_fit = _read(output / "semidde_full_mc_pilot" / "fits.csv")
    reduced_row = next(row for row in reduced_fit if _key(row) == ("semidde_csp", 3, 500, 0))
    assert len(full_fit) == 1 and full_fit[0]["fitted_K"] == reduced_row["fitted_K"]
    details = {"scenario": "semidde_csp D=3 N=500", "candidates": 27,
               "mc_samples_compared": [4096, 50000],
               "max_absolute_candidate_score_difference": maximum,
               "same_selected_widths": True,
               "interpretation": "This hierarchy uses exact conditional enumeration; the MC sample setting is inactive."}
    (output / "semidde_mc_equivalence.json").write_text(json.dumps(details, indent=2))


def _save_provenance(output: Path):
    repo = Path(__file__).resolve().parents[1]
    source_names = (
        repo / "scripts" / "evaluate_hp_selection_validation.py",
        repo / "scripts" / "summarize_hp_selection_validation.py",
        repo / "scripts" / "finalize_hp_selection_validation.py",
        repo / "scripts" / "audit_hp_selection_candidate_dimensions.py",
        repo / "scripts" / "audit_copula_candidate_widths.py",
        repo / "scripts" / "check_hp_selection_validation.py",
        repo / "scripts" / "repair_hp_selection_npz.py",
        repo / "tests" / "test_hp_selection_validation_metrics.py",
    )
    hashes = {}
    for source in source_names:
        hashes[str(source.relative_to(repo))] = hashlib.sha256(source.read_bytes()).hexdigest()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              text=True, capture_output=True, check=True).stdout.strip()
    patch = subprocess.run(["git", "diff", "--binary"], cwd=repo,
                           capture_output=True, check=True).stdout
    (output / "package_source_diff.patch").write_bytes(patch)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "h5py": h5py.__version__,
        "dde_git_head": git_head,
        "dde_uncommitted_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "runner_sha256": hashes,
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))


def finalize(output: Path = DEFAULT_OUTPUT):
    main_path = output / "fits.csv"
    main = _read(main_path)
    expected = {(kernel, depth, N, 0) for kernel in KERNELS
                for depth in (2, 3) for N in N_VALUES}
    paper_kernels = {"dde_normal_fixed", "dde_bernoulli_fixed",
                     "dde_poisson_fixed", "semidde_fixed"}
    expected_repeats = {(kernel, 2, N, rep) for kernel in paper_kernels
                        for N in N_VALUES for rep in range(10)}
    current_keys = {_key(row) for row in main}
    if current_keys == expected | expected_repeats:
        _check_semi_mc_equivalence(output)
        repair(output)
        check(output, expected_unique=100, expected_rows=280)
        _save_provenance(output)
        summarize(output)
        return
    assert current_keys == expected and len(main) == len(expected), "Main run has missing or duplicate cells"
    assert all(row["status"] == "ok" for row in main), "Main run has failed cells"
    for name, unique, total in (
        ("paper_fixed_default_replicates", 20, 200),
        ("noncop_fixed_d3_default_full", 20, 20),
        ("noncop_csp_default_full", 40, 40),
        ("copula_default_budget_pilot", 2, 2),
        ("copula_d3_default_pilot", 1, 1),
        ("semidde_full_mc_pilot", 1, 1),
    ):
        check(output / name, expected_unique=unique, expected_rows=total)
    _check_semi_mc_equivalence(output)

    source_specs = [
        (kernel, {3}, output / "noncop_fixed_d3_default_full")
        for kernel in paper_kernels
    ] + [
        (kernel, {2, 3}, output / "noncop_csp_default_full")
        for kernel in KERNELS if kernel.endswith("csp") and not kernel.startswith("copula")
    ]
    corrections = {}
    for kernel, depths, source in source_specs:
        source_rows = [row for row in _read(source / "fits.csv") if row["kernel"] == kernel]
        expected_subset = {key for key in expected if key[0] == kernel and key[1] in depths}
        assert {_key(row) for row in source_rows} == expected_subset
        assert all(row["status"] == "ok" for row in source_rows)
        corrections.update({_key(row): (row, source) for row in source_rows})

    replicate_source = output / "paper_fixed_default_replicates"
    replicate_rows = _read(replicate_source / "fits.csv")
    assert {_key(row) for row in replicate_rows} == expected_repeats
    assert all(row["status"] == "ok" for row in replicate_rows)
    repeat_map = {_key(row): row for row in replicate_rows}

    archive = output / "superseded_initial"
    archive.mkdir(exist_ok=True)
    if not (archive / "fits_before_corrections.csv").exists():
        shutil.copy2(main_path, archive / "fits_before_corrections.csv")
    merged = []
    for row in main:
        key = _key(row)
        if key in repeat_map:
            corrected, source = repeat_map[key], replicate_source
        elif key in corrections:
            corrected, source = corrections[key]
        else:
            merged.append(row)
            continue
        basename = f"{key[0]}_d{key[1]}_n{key[2]}_r{key[3]}"
        for suffix in ("json", "npz"):
            filename = f"{basename}.{suffix}"
            old = output / "cells" / filename
            replacement = source / "cells" / filename
            if not (archive / filename).exists():
                shutil.copy2(old, archive / filename)
            shutil.copy2(replacement, old)
        merged.append(corrected)
    for key in sorted(expected_repeats):
        if key[3] == 0:
            continue
        row = repeat_map[key]
        basename = f"{key[0]}_d{key[1]}_n{key[2]}_r{key[3]}"
        for suffix in ("json", "npz"):
            filename = f"{basename}.{suffix}"
            shutil.copy2(replicate_source / "cells" / filename,
                         output / "cells" / filename)
        merged.append(row)
    merged.sort(key=_key)
    _write(main_path, merged)
    manifest = {
        "target_cells": len(expected),
        "corrected_or_replaced_base_cells": len(corrections) + 20,
        "additional_replicate_fits": len(expected_repeats) - 20,
        "d2_fixed_budget": "Default DDE 50 iterations, SemiDDE 10 outer x 50 inner, optimizer maxiter 100; ten seeded replicates per cell.",
        "other_noncop_budget": "Default DDE 50 iterations, SemiDDE 10 outer x 50 inner, optimizer maxiter 100; SemiDDE mc_samples is configured as 4096 but does not affect these D=3 Mode-B calculations (verified against 50000). One seeded run per cell.",
        "copula_budget": "Screening budget: 10 kernel iterations, optimizer maxiter 30, pMSE 2 repeats x 3 folds x 30 trees; one seeded run per cell.",
        "corrections": {
            "dde_fixed_d2": "Ten default-budget, paper-matched, seed-controlled replicates for paper RMSE comparison.",
            "semidde_fixed_d2": "Ten default-budget, data-initialized replicates for paper RMSE comparison.",
            "noncop_fixed_d3": "Default-budget, seed-controlled replacements for all four methods.",
            "noncop_csp_d2_d3": "Default-budget replacements for all four non-Copula CSP methods.",
        },
        "copula_d2": "Current manuscript J=100 MATLAB stored setting; full Cartesian grid.",
        "copula_d3": "CopDDE paper J=108 MATLAB stored setting; full Cartesian grid.",
        "copula_default_pilots": "N=500 full-grid reruns for D=2 fixed/CSP and D=3 CSP using default kernel and pMSE budgets.",
        "semidde_full_mc_pilot": "D=3 N=500 full-grid CSP rerun with 50000 Monte Carlo samples; all 27 scores and selected widths exactly match mc_samples=4096.",
        "paper_sources": [
            "Papers/References/DDE - 2501.01414v2.txt (Eq. S.36, Tables S.9-S.11)",
            "Results/paper_results/simulation_semidde_summary.csv",
            "Results/paper_results/simulation_copula_summary.csv",
            "Papers/References/CopDDE.txt (Table S.3)",
        ],
        "superseded_raw": str(archive.relative_to(output)),
    }
    (output / "validation_manifest.json").write_text(json.dumps(manifest, indent=2))
    repair(output)
    check(output, expected_unique=100, expected_rows=280)
    _save_provenance(output)
    summarize(output)


if __name__ == "__main__":
    finalize()
