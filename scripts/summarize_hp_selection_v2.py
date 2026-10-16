"""Summarize an HP-selection validation sweep: trends in N per kernel and depth.

Fixed paths: Hungarian-aligned all-parameter RMSE (thresholded at the selected
tau) by N, with the DDE paper SAEM / PEM references at D=2.  CSP paths: mean
fitted widths, all-layer exact rate and L1 width error by N, plus a
per-cell audit of whether some candidate in the grid reached the true widths
(selection problem) or none did (kernel problem).  Writes markdown tables to
stdout and ``summary_v2.csv`` / ``csp_candidate_audit.csv`` into the sweep
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DDE_PAPER_RMSE_D2 = {
    "dde_normal_fixed": (0.189, 0.156, 0.147, 0.137, 0.133),
    "dde_bernoulli_fixed": (0.498, 0.431, 0.386, 0.359, 0.341),
    "dde_poisson_fixed": (0.281, 0.258, 0.245, 0.243, 0.242),
}
DDE_PAPER_PEM_RMSE_D2 = {
    "dde_normal_fixed": (0.170, 0.128, 0.080, 0.063, 0.054),
    "dde_bernoulli_fixed": (0.404, 0.304, 0.231, 0.203, 0.184),
    "dde_poisson_fixed": (0.219, 0.158, 0.108, 0.080, 0.061),
}
N_VALUES = (500, 1000, 2000, 4000, 8000)


def _fmt(values, digits=3):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "—"
    if values.size == 1:
        return f"{values[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def _trend(means):
    """Spearman correlation of the metric with log N (negative = shrinking)."""
    means = np.asarray(means, dtype=float)
    if means.size < 3 or np.all(np.isnan(means)):
        return float("nan")
    x = np.argsort(np.argsort(np.arange(means.size)))
    y = np.argsort(np.argsort(means))
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--previous", type=Path, default=None,
                        help="fits.csv of an earlier sweep for side-by-side columns")
    args = parser.parse_args()
    rows = [r for r in csv.DictReader((args.sweep / "fits.csv").open()) if r["status"] == "ok"]
    errors = [r for r in csv.DictReader((args.sweep / "fits.csv").open()) if r["status"] != "ok"]
    previous = {}
    if args.previous and args.previous.exists():
        for r in csv.DictReader(args.previous.open()):
            if r["status"] == "ok":
                previous.setdefault((r["kernel"], int(r["D"]), int(r["N"])), []).append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kernel"], int(r["D"]), int(r["N"]))].append(r)
    kernels = sorted({k for k, _, _ in groups})
    summary = []
    lines = [f"Sweep `{args.sweep.name}`: {len(rows)} successful cells, {len(errors)} errors.\n"]

    # ---- fixed paths
    lines.append("## Fixed paths: thresholded all-parameter RMSE by N (mean ± sd over replicates)\n")
    for depth in (2, 3):
        fixed = [k for k in kernels if k.endswith("fixed") and any((k, depth, N) in groups for N in N_VALUES)]
        if not fixed:
            continue
        lines.append(f"### D={depth}\n")
        lines.append("| kernel | " + " | ".join(f"N={N}" for N in N_VALUES) + " | reps | trend (Spearman vs N) | previous sweep (N=500 … 8000) |")
        lines.append("|---|" + "---|" * (len(N_VALUES) + 3))
        for k in fixed:
            cells, means, reps = [], [], []
            for N in N_VALUES:
                vals = [float(r["parameter_rmse_thresholded"]) for r in groups.get((k, depth, N), [])]
                cells.append(_fmt(vals)); means.append(np.mean(vals) if vals else np.nan); reps.append(len(vals))
                summary.append({"kernel": k, "D": depth, "N": N, "metric": "rmse_thresholded",
                                "mean": np.mean(vals) if vals else np.nan, "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan, "n": len(vals)})
            prev = " / ".join(f"{np.mean([float(r['parameter_rmse_thresholded']) for r in previous[(k, depth, N)]]):.3f}" if (k, depth, N) in previous else "—" for N in N_VALUES)
            lines.append(f"| {k} | " + " | ".join(cells) + f" | {'/'.join(map(str, reps))} | {_trend(means):+.2f} | {prev} |")
            if depth == 2 and k in DDE_PAPER_RMSE_D2:
                lines.append(f"| {k} paper SAEM / PEM | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in zip(DDE_PAPER_RMSE_D2[k], DDE_PAPER_PEM_RMSE_D2[k])) + " | 100 | | |")
        lines.append("")

    # ---- csp paths
    lines.append("## CSP paths: mean fitted widths (exact rate, mean L1 width error) by N\n")
    audit_rows = []
    for depth in (2, 3):
        csp = [k for k in kernels if k.endswith("csp") and any((k, depth, N) in groups for N in N_VALUES)]
        if not csp:
            continue
        lines.append(f"### D={depth}\n")
        lines.append("| kernel | truth | " + " | ".join(f"N={N}" for N in N_VALUES) + " | reps | L1 trend | previous sweep L1 |")
        lines.append("|---|---|" + "---|" * (len(N_VALUES) + 3))
        for k in csp:
            cells, l1_means, reps, truth = [], [], [], None
            for N in N_VALUES:
                rs = groups.get((k, depth, N), [])
                if not rs:
                    cells.append("—"); l1_means.append(np.nan); reps.append(0); continue
                truth = rs[0]["true_K"]
                K = np.array([json.loads(r["fitted_K"]) for r in rs], dtype=float)
                exact = np.mean([float(r["dimension_exact"]) for r in rs])
                l1 = np.mean([float(r["dimension_l1"]) for r in rs])
                cells.append("(" + ", ".join(f"{v:.1f}" for v in K.mean(0)) + f") ({exact:.2f}, {l1:.2f})")
                l1_means.append(l1); reps.append(len(rs))
                summary.append({"kernel": k, "D": depth, "N": N, "metric": "dimension_l1", "mean": l1, "sd": np.nan, "n": len(rs)})
                summary.append({"kernel": k, "D": depth, "N": N, "metric": "dimension_exact", "mean": exact, "sd": np.nan, "n": len(rs)})
                # candidate audit from the cell diagnostics
                for r in rs:
                    cell = args.sweep / "cells" / f"{k}_d{depth}_n{N}_r{r['replicate']}.json"
                    if not cell.exists():
                        continue
                    d = json.load(cell.open())
                    true_K = tuple(d["true_K"])
                    table = d["cv_table"]
                    width_key = "fitted_K" if "fitted_K" in table[0] else None
                    exact_candidates = (
                        [c for c in table if tuple(c[width_key]) == true_K] if width_key else []
                    )
                    best = table[d["best_index"]]
                    audit_rows.append({
                        "kernel": k, "D": depth, "N": N, "replicate": r["replicate"],
                        "true_K": json.dumps(list(true_K)), "selected_K": json.dumps(best.get("K_list")),
                        "selected_t_spike": json.dumps(best.get("t_spike")), "selected_temp": best.get("temp"), "selected_tau": best.get("tau"),
                        "candidates": len(table), "candidates_exact": len(exact_candidates),
                        "best_score": best["mean_score"],
                        "best_exact_score": max(c["mean_score"] for c in exact_candidates) if exact_candidates else "",
                        "verdict": ("unknown (candidate widths not recorded)" if not width_key else
                                    "selected exact" if tuple(best[width_key]) == true_K else
                                    "selection problem (an exact candidate exists)" if exact_candidates else
                                    "kernel problem (no candidate exact)"),
                    })
            prev = " / ".join(f"{np.mean([float(r['dimension_l1']) for r in previous[(k, depth, N)]]):.2f}" if (k, depth, N) in previous else "—" for N in N_VALUES)
            lines.append(f"| {k} | {truth} | " + " | ".join(cells) + f" | {'/'.join(map(str, reps))} | {_trend(l1_means):+.2f} | {prev} |")
        lines.append("")

    if audit_rows:
        lines.append("## CSP candidate audit (per cell)\n")
        counts = defaultdict(lambda: defaultdict(int))
        for a in audit_rows:
            counts[(a["kernel"], a["D"])][a["verdict"]] += 1
        lines.append("| kernel | D | selected exact | selection problem | kernel problem | unknown |")
        lines.append("|---|---|---|---|---|---|")
        for (k, depth), c in sorted(counts.items()):
            lines.append(f"| {k} | {depth} | {c['selected exact']} | {c['selection problem (an exact candidate exists)']} | {c['kernel problem (no candidate exact)']} | {c['unknown (candidate widths not recorded)']} |")
        lines.append("")
        with (args.sweep / "csp_candidate_audit.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader(); writer.writerows(audit_rows)
    if errors:
        lines.append("## Errors\n")
        for r in errors:
            lines.append(f"- {r['kernel']} D={r['D']} N={r['N']} r{r['replicate']}: {r.get('error', '')[:160]}")
    with (args.sweep / "summary_v2.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kernel", "D", "N", "metric", "mean", "sd", "n"])
        writer.writeheader(); writer.writerows(summary)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
