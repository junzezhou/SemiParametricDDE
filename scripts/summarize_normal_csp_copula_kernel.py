"""Render markdown tables from ``evaluate_normal_csp_copula_kernel.py`` output.

The copula paper's mean CSP widths (MATLAB kernel, 100 replicates, mixed
marginals) are printed alongside as the reference shrinkage pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PAPER_COPULA_MEANS = {
    "d2_j50": {500: (9.65, 2.34), 1000: (10.06, 2.38), 2000: (10.13, 2.47),
               4000: (10.19, 2.52), 8000: (10.25, 2.69)},
    "d3_j108": {500: (17.66, 9.82, 3.27), 1000: (17.89, 8.93, 3.42),
                2000: (17.99, 7.75, 3.26), 4000: (18.08, 6.66, 2.74),
                8000: (18.04, 6.22, 2.26)},
}


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _fmt_widths(values):
    return "(" + ", ".join(f"{float(v):.2f}" for v in values) + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    summary = _read(args.out_dir / "summary.csv")
    lines = []
    scenarios = sorted({row["scenario"] for row in summary})
    backends = sorted({row["backend"] for row in summary})
    for scenario in scenarios:
        rows = [row for row in summary if row["scenario"] == scenario]
        K_true = json.loads(rows[0]["K_true"])
        D = len(K_true)
        lines.append(f"### {scenario}: truth {tuple(K_true)}\n")
        header = ["N"] + [f"{b} mean widths" for b in backends] + ["paper CopDDE (MATLAB) means"] \
            + [f"{b} exact" for b in backends] + [f"{b} L1" for b in backends] + [f"{b} iters" for b in backends]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        by_key = {(row["backend"], int(row["N"])): row for row in rows}
        for N in sorted({int(row["N"]) for row in rows}):
            cells = [str(N)]
            for b in backends:
                row = by_key.get((b, N))
                cells.append(
                    _fmt_widths([row[f"K{d + 1}_mean"] for d in range(D)]) if row else "—"
                )
            paper = PAPER_COPULA_MEANS.get(scenario, {}).get(N)
            cells.append(_fmt_widths(paper) if paper else "—")
            for b in backends:
                row = by_key.get((b, N))
                cells.append(f"{float(row['exact_rate']):.2f}" if row else "—")
            for b in backends:
                row = by_key.get((b, N))
                cells.append(f"{float(row['mean_dimension_l1_error']):.2f}" if row else "—")
            for b in backends:
                row = by_key.get((b, N))
                cells.append(f"{float(row['mean_iterations']):.1f}" if row else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        fits = int(rows[0]["fits"])
        lines.append(f"Replicates per cell: {fits}.  Mean fit seconds: "
                     + ", ".join(f"{b} {float(by_key[(b, N)]['mean_fit_seconds']):.0f}s@N={N}"
                                 for b in backends for N in (min(int(r['N']) for r in rows), max(int(r['N']) for r in rows))
                                 if (b, N) in by_key) + ".\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
