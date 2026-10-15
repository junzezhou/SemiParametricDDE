"""Independent file-level integrity check for selected-CSP rep100 output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from simulate_selected_csp_hp_rep100 import DEFAULT_OUTPUT, DEPTHS, KERNELS, SAMPLE_SIZES


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output: Path) -> dict:
    config = json.loads((output / "config.json").read_text())
    environment = json.loads((output / "environment.json").read_text())
    selected = json.loads((output / "selected_hp.json").read_text())
    frozen_hp = {(row["kernel"], row["D"], row["N"]): row["hp"] for row in selected}
    expected_settings = {(k, d, n) for k in KERNELS for d in DEPTHS for n in SAMPLE_SIZES}
    assert set(frozen_hp) == expected_settings
    assert _sha(Path(config["source"])) == config["source_sha256"]
    source_rows = [row for row in _csv(Path(config["source"])) if row["kernel"] in KERNELS]
    assert len(source_rows) == len(expected_settings)
    for row in source_rows:
        setting = row["kernel"], int(row["D"]), int(row["N"])
        assert row["status"] == "ok" and json.loads(row["selected_hyperparameters"]) == frozen_hp[setting]

    rows = _csv(output / "fits.csv")
    expected_count = len(expected_settings) * config["replicates"]
    assert len(rows) == expected_count
    keys, seeds = set(), set()
    by_setting: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = row["kernel"], int(row["D"]), int(row["N"]), int(row["replicate"])
        setting = key[:3]
        assert setting in expected_settings and 0 <= key[3] < config["replicates"]
        assert key not in keys
        keys.add(key)
        seed = int(row["seed"])
        assert seed not in seeds
        seeds.add(seed)
        assert row["status"] == "ok"
        assert json.loads(row["hp"]) == frozen_hp[setting]
        filename = f"{key[0]}_d{key[1]}_n{key[2]}_r{key[3]:03d}.json"
        cell = json.loads((output / "cells" / filename).read_text())
        assert cell["seed"] == seed and cell["status"] == "ok"
        assert cell["hp"] == frozen_hp[setting]
        true_K, fitted_K = json.loads(row["true_K"]), json.loads(row["fitted_K"])
        assert true_K == cell["true_K"] and fitted_K == cell["fitted_K"]
        differences = [f - t for f, t in zip(fitted_K, true_K)]
        assert len(differences) == key[1]
        assert int(row["exact"]) == cell["exact"] == int(fitted_K == true_K)
        assert int(row["dimension_l1"]) == cell["dimension_l1"] == sum(abs(v) for v in differences)
        assert int(row["under_width"]) == cell["under_width"] == sum(max(-v, 0) for v in differences)
        assert int(row["over_width"]) == cell["over_width"] == sum(max(v, 0) for v in differences)
        assert math.isclose(float(row["seconds"]), cell["seconds"], rel_tol=1e-12)
        by_setting.setdefault(setting, []).append(row)
    assert set(by_setting) == expected_settings
    assert all(len(group) == config["replicates"] for group in by_setting.values())
    assert len(list((output / "cells").glob("*.json"))) == expected_count

    summary = _csv(output / "summary.csv")
    assert len(summary) == len(expected_settings)
    summary_keys = set()
    for row in summary:
        setting = row["kernel"], int(row["D"]), int(row["N"])
        assert setting not in summary_keys
        summary_keys.add(setting)
        group = by_setting[setting]
        exact = sum(int(item["exact"]) for item in group)
        assert int(row["completed"]) == int(row["successful"]) == config["replicates"]
        assert int(row["failed"]) == 0 and int(row["exact_count"]) == exact
        assert math.isclose(float(row["exact_rate_all"]), exact / config["replicates"], abs_tol=1e-12)
        assert math.isclose(float(row["mean_dimension_l1"]),
                            sum(int(item["dimension_l1"]) for item in group) / config["replicates"],
                            abs_tol=1e-12)
    assert summary_keys == expected_settings

    figures = ("exact_rate_all_vs_n.png", "mean_dimension_l1_vs_n.png", "median_seconds_vs_n.png")
    for name in figures:
        with Image.open(output / name) as figure:
            figure.verify()
    for filename, digest in (("simulate_selected_csp_hp_rep100.py", "runner_sha256"),
                             ("summarize_selected_csp_hp_rep100.py", "summarizer_sha256"),
                             ("evaluate_hp_selection_validation.py", "data_generator_sha256")):
        assert _sha(output / "code_snapshot" / filename) == environment[digest]
    result = {"settings": len(expected_settings), "replicates_per_setting": config["replicates"],
              "successful_fits": len(rows), "failed_fits": 0,
              "unique_seeds": len(seeds), "exact_fits": sum(int(row["exact"]) for row in rows),
              "csv_matches_atomic_checkpoints": True,
              "all_metrics_recomputed": True,
              "all_selected_hyperparameters_frozen": True,
              "figures_verified": list(figures),
              "code_snapshot_hashes_verified": True}
    (output / "integrity_check.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(check(args.output.resolve()), indent=2))
