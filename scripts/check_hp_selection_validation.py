"""Check checkpoint completeness and internal consistency of validation output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_hp_selection_validation import DEFAULT_OUTPUT


def check(output: Path, expected_unique: int | None = None,
          expected_rows: int | None = None) -> dict:
    with (output / "fits.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "No validation rows"
    keys = [(r["kernel"], int(r["D"]), int(r["N"]), int(r["replicate"])) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate fit rows"
    assert all(r["status"] == "ok" for r in rows), "Failed fit row"
    unique = {(k, d, n) for k, d, n, _ in keys}
    if expected_unique is not None:
        assert len(unique) == expected_unique, f"Expected {expected_unique} unique settings, found {len(unique)}"
    if expected_rows is not None:
        assert len(rows) == expected_rows, f"Expected {expected_rows} rows, found {len(rows)}"
    counts = {}
    for row, (kernel, depth, N, replicate) in zip(rows, keys):
        basename = f"{kernel}_d{depth}_n{N}_r{replicate}"
        diagnostic = json.loads((output / "cells" / f"{basename}.json").read_text())
        table = diagnostic["cv_table"]
        timings = diagnostic["timing_table"]
        count = int(row["candidate_count"])
        assert len(table) == len(timings) == count, basename
        assert [int(r["candidate_index"]) for r in table] == list(range(count)), basename
        scores = np.asarray([r["mean_score"] for r in table], dtype=float)
        assert np.isfinite(scores).all(), basename
        assert int(np.argmax(scores)) == int(row["best_index"]) == int(diagnostic["best_index"]), basename
        assert np.isclose(scores[int(row["best_index"])], float(row["best_score"])), basename
        assert np.isclose(float(row["selected_tau"]),
                          float(table[int(row["best_index"])]["tau"])), basename
        assert all(np.isfinite(float(t["total_seconds"])) and float(t["total_seconds"]) >= 0 for t in timings), basename
        widths = tuple(json.loads(row["fitted_K"]))
        truth = tuple(json.loads(row["true_K"]))
        assert int(row["dimension_exact"]) == int(widths == truth), basename
        assert int(row["dimension_l1"]) == sum(abs(a - b) for a, b in zip(widths, truth)), basename
        if kernel.endswith("fixed"):
            assert all(np.isfinite(float(row[field])) and float(row[field]) >= 0 for field in
                       ("parameter_mse_raw", "parameter_rmse_raw",
                        "parameter_mse_thresholded", "parameter_rmse_thresholded")), basename
        with np.load(output / "cells" / f"{basename}.npz") as fitted:
            assert all(np.isfinite(fitted[name]).all() for name in fitted.files), basename
            assert fitted["prop"].shape == (widths[-1],), basename
            if "gamma" in fitted:
                assert fitted["gamma"].shape == (int(row["J"]),), basename
            previous = int(row["J"])
            for layer, width in enumerate(widths, start=1):
                assert fitted[f"B{layer}"].shape == (previous, width + 1), basename
                previous = width
        expected = (18 * 3 ** (depth - 2) if kernel == "copula_fixed" else
                    3 ** depth if kernel.endswith("csp") else 3 * 2 ** depth)
        assert count == expected, basename
        counts[kernel] = counts.get(kernel, 0) + 1
    report = {"rows": len(rows), "unique_settings": len(unique),
              "all_status_ok": True, "all_candidate_grids_complete": True,
              "all_selected_parameters_finite_and_shape_consistent": True,
              "rows_by_kernel": counts}
    (output / "integrity_check.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-unique", type=int)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    print(json.dumps(check(args.output, args.expected_unique, args.expected_rows), indent=2))
