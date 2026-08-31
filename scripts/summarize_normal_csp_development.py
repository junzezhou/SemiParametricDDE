"""Summarize the paper-locked Normal-DDE CSP development benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


COPDDE_MEAN_WIDTHS = {
    "d2_j50": {
        500: (10.02, 3.55),
        1000: (10.06, 3.13),
        2000: (10.09, 3.03),
        4000: (10.02, 3.01),
        8000: (10.00, 3.01),
    },
    "d3_j108": {
        500: (17.66, 9.82, 3.27),
        1000: (17.89, 8.93, 3.42),
        2000: (17.99, 7.75, 3.26),
        4000: (18.08, 6.66, 2.74),
        8000: (18.04, 6.22, 2.26),
    },
}


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path, records):
    fields = sorted({key for record in records for key in record})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _f(record, key):
    return float(record[key])


def _i(record, key):
    return int(float(record[key]))


def _main_summary(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario"], _i(record, "N"))].append(record)

    summary = []
    frequencies = []
    for (scenario, N), rows in sorted(groups.items()):
        truth = tuple(json.loads(rows[0]["K_true"]))
        fitted = [tuple(json.loads(row["K_fit"])) for row in rows]
        paper = COPDDE_MEAN_WIDTHS[scenario][N]
        mean_width = tuple(np.mean(fitted, axis=0))
        item = {
            "scenario": scenario,
            "D": len(truth),
            "N": N,
            "fits": len(rows),
            "K_true": json.dumps(truth),
            "normal_mean_width": json.dumps([round(value, 6) for value in mean_width]),
            "copdde_published_mean_width": json.dumps(paper),
            "full_exact_rate": float(np.mean([value == truth for value in fitted])),
            "mean_dimension_l1_error": float(
                np.mean(
                    [
                        sum(abs(value[d] - truth[d]) for d in range(len(truth)))
                        for value in fitted
                    ]
                )
            ),
            "hit_max_iter_rate": float(
                np.mean([_i(row, "hit_max_iter") for row in rows])
            ),
            "mean_iterations": float(np.mean([_f(row, "iterations") for row in rows])),
            "mean_fit_seconds": float(np.mean([_f(row, "fit_seconds") for row in rows])),
        }
        normal_bias_total = 0.0
        paper_bias_total = 0.0
        for d, true_width in enumerate(truth):
            values = np.asarray([value[d] for value in fitted], dtype=float)
            normal_bias = abs(mean_width[d] - true_width)
            paper_bias = abs(paper[d] - true_width)
            normal_bias_total += normal_bias
            paper_bias_total += paper_bias
            item[f"K{d + 1}_mean"] = float(mean_width[d])
            item[f"K{d + 1}_mean_bias"] = float(mean_width[d] - true_width)
            item[f"K{d + 1}_abs_mean_bias"] = float(normal_bias)
            item[f"K{d + 1}_mae"] = float(np.mean(np.abs(values - true_width)))
            item[f"K{d + 1}_rmse"] = float(
                np.sqrt(np.mean((values - true_width) ** 2))
            )
            item[f"K{d + 1}_exact_rate"] = float(np.mean(values == true_width))
            item[f"K{d + 1}_over_rate"] = float(np.mean(values > true_width))
            item[f"K{d + 1}_under_rate"] = float(np.mean(values < true_width))
            item[f"K{d + 1}_copdde_abs_mean_bias"] = float(paper_bias)
            item[f"A{d + 1}_initial_accuracy"] = float(
                np.mean([_f(row, f"A{d + 1}_initial_accuracy") for row in rows])
            )
            item[f"A{d + 1}_final_accuracy"] = float(
                np.mean([_f(row, f"A{d + 1}_final_accuracy") for row in rows])
            )
        item["normal_total_abs_mean_bias"] = normal_bias_total
        item["copdde_total_abs_mean_bias"] = paper_bias_total
        item["total_abs_mean_bias_delta_vs_copdde"] = (
            normal_bias_total - paper_bias_total
        )
        summary.append(item)

        for widths, count in sorted(Counter(fitted).items()):
            selected = [
                row for row, value in zip(rows, fitted) if value == widths
            ]
            frequencies.append(
                {
                    "scenario": scenario,
                    "N": N,
                    "K_fit": json.dumps(widths),
                    "count": count,
                    "rate": count / len(rows),
                    "mean_iterations": float(
                        np.mean([_f(row, "iterations") for row in selected])
                    ),
                }
            )
    return summary, frequencies


def _control_summary(truth_rows, fixed_rows):
    output = []
    for estimator, rows in (
        ("csp_padded_truth", truth_rows),
        ("fixed_true_width", fixed_rows),
    ):
        by_scenario = defaultdict(list)
        for row in rows:
            by_scenario[row["scenario"]].append(row)
        for scenario, group in sorted(by_scenario.items()):
            truth = tuple(json.loads(group[0]["K_true"]))
            fitted = [tuple(json.loads(row["K_fit"])) for row in group]
            item = {
                "estimator": estimator,
                "scenario": scenario,
                "N": _i(group[0], "N"),
                "fits": len(group),
                "K_true": json.dumps(truth),
                "mean_K_fit": json.dumps(
                    [round(value, 6) for value in np.mean(fitted, axis=0)]
                ),
                "full_exact_rate": float(
                    np.mean([value == truth for value in fitted])
                ),
                "hit_max_iter_rate": float(
                    np.mean([_i(row, "hit_max_iter") for row in group])
                ),
                "mean_iterations": float(
                    np.mean([_f(row, "iterations") for row in group])
                ),
                "mean_fit_seconds": float(
                    np.mean([_f(row, "fit_seconds") for row in group])
                ),
            }
            for d in range(len(truth)):
                item[f"A{d + 1}_initial_accuracy"] = float(
                    np.mean([_f(row, f"A{d + 1}_initial_accuracy") for row in group])
                )
                item[f"A{d + 1}_final_accuracy"] = float(
                    np.mean([_f(row, f"A{d + 1}_final_accuracy") for row in group])
                )
            output.append(item)
    return output


def _widths(item):
    return "(" + ", ".join(f"{value:.2f}" for value in json.loads(item["normal_mean_width"])) + ")"


def _paper_widths(item):
    return "(" + ", ".join(f"{value:.2f}" for value in json.loads(item["copdde_published_mean_width"])) + ")"


def _report(summary, controls, total_seconds):
    lines = [
        "# Normal DDE CSP paper-locked development benchmark",
        "",
        "Date: 2026-08-30",
        "",
        "## Design",
        "",
        "The main benchmark uses D=2/J=50/K_true=(10,3)/K_max=(16,5) and "
        "D=3/J=108/K_true=(18,6,2)/K_max=(36,12,4), with N in "
        "{500,1000,2000,4000,8000}, 10 development replicates, C=1, "
        "paper-locked spike scales/temperature/threshold, data initialization, "
        "100 outer iterations, and 100 optimizer iterations. Full traces were disabled.",
        "",
        "Published CopDDE values are descriptive external benchmarks based on 100 "
        "replicates; this 10-replicate development run is not an inferential comparison.",
        "",
        "## Main dimension recovery",
        "",
        "| Scenario | N | Normal mean K | CopDDE published mean K | Exact | Mean L1 error | Hit max | Mean sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['scenario']} | {item['N']} | {_widths(item)} | "
            f"{_paper_widths(item)} | {item['full_exact_rate']:.0%} | "
            f"{item['mean_dimension_l1_error']:.2f} | "
            f"{item['hit_max_iter_rate']:.0%} | {item['mean_fit_seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic controls at N=500",
            "",
            "| Estimator | Scenario | Mean K | Exact | Hit max | Initial A accuracy | Final A accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in controls:
        D = len(json.loads(item["K_true"]))
        initial = "(" + ", ".join(
            f"{item[f'A{d + 1}_initial_accuracy']:.3f}" for d in range(D)
        ) + ")"
        final = "(" + ", ".join(
            f"{item[f'A{d + 1}_final_accuracy']:.3f}" for d in range(D)
        ) + ")"
        mean_k = "(" + ", ".join(
            f"{value:.2f}" for value in json.loads(item["mean_K_fit"])
        ) + ")"
        lines.append(
            f"| {item['estimator']} | {item['scenario']} | {mean_k} | "
            f"{item['full_exact_rate']:.0%} | {item['hit_max_iter_rate']:.0%} | "
            f"{initial} | {final} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. D=2 is broadly functional but not monotone in N: exact recovery is "
            "90%, 80%, 100%, 90%, and 80%. The N=8000 misses are both (11,4), "
            "so large-N false positives remain.",
            "2. D=3 beats the published CopDDE mean-width bias at N<=2000 but this "
            "partly reflects cancellation between over- and under-selection. Full exact "
            "rates are 50%, 0%, 20%, 0%, and 0%.",
            "3. D=3 deteriorates at N=4000 and N=8000, driven by second-layer "
            "over-selection and deepest-layer under-selection. This rejects monotone "
            "dimension consistency for the current implementation.",
            "4. Padded-truth initialization gives D=2 exact recovery in 10/10 fits, "
            "but D=3 only 7/10; all three failures shrink K3 from 2 to 1. The failure "
            "therefore is not solely caused by spectral initialization.",
            "5. Fixed-width SAEM preserves D=3 width and reaches mean final latent "
            "accuracies near (1.000, 0.893, 0.776), whereas padded-truth CSP reaches "
            "approximately (1.000, 0.856, 0.685). The current CSP MCEM update damages "
            "the deep latent representation relative to the SAEM control.",
            "6. The convergence rule is unreliable: some poor N=8000 D=3 fits stop "
            "after 17-23 iterations, while most smaller-N fits exhaust all 100. Stable "
            "variability of relative changes is not sufficient evidence of a good fixed point.",
            "",
            f"Total measured fit time across the main and diagnostic runs was "
            f"{total_seconds / 60.0:.1f} minutes.",
            "",
            "## Decision",
            "",
            "Do not launch the 100-replicate frozen evaluation yet. The development "
            "test has identified a reproducible D=3 algorithm defect. The next change "
            "should prioritize SAEM-style sufficient-statistic accumulation in the CSP "
            "M-step and a convergence rule requiring both stable widths and small/stable "
            "parameter changes. Re-run the same frozen development seeds after each change.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("../Results/normal_dde_csp_development"),
    )
    args = parser.parse_args()
    main_rows = _read(
        args.results_root
        / "paper_locked_main_r10"
        / "normal_csp_recovery_fits.csv"
    )
    truth_rows = _read(
        args.results_root
        / "padded_truth_n500_r10"
        / "normal_csp_recovery_fits.csv"
    )
    fixed_rows = _read(
        args.results_root
        / "fixed_true_width_n500_r10"
        / "normal_fixed_control_fits.csv"
    )

    summary, frequencies = _main_summary(main_rows)
    controls = _control_summary(truth_rows, fixed_rows)
    _write(args.results_root / "paper_comparison_summary.csv", summary)
    _write(args.results_root / "width_frequency.csv", frequencies)
    _write(args.results_root / "diagnostic_controls_summary.csv", controls)
    total_seconds = sum(
        _f(row, "fit_seconds") for row in main_rows + truth_rows + fixed_rows
    )
    report = _report(summary, controls, total_seconds)
    temporary = args.results_root / "REPORT.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.results_root / "REPORT.md")


if __name__ == "__main__":
    main()
