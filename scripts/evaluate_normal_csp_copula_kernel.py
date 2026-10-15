"""Dimension-shrinkage simulation for the Normal DDE CSP backends.

The scenarios are the CopDDE paper designs reused by
``evaluate_normal_csp_recovery.py``: D=2 with J=50 and true widths (10, 3)
fitted from the maximal widths (16, 5), and D=3 with J=108 and true widths
(18, 6, 2) fitted from (36, 12, 4).  Observations are Gaussian, so the data
are the latent ``Z`` of the copula design without any marginal transform.

For every scenario, sample size, replicate, and backend the script fits the
model once from a data initialization and records the fitted active widths.
The headline summary is the mean fitted width per layer, which is the
quantity the CopDDE paper tabulates for its CSP path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_normal_csp_recovery as base  # noqa: E402

from dde import fit  # noqa: E402


SCENARIOS = {
    "d2_j50": base._scenario_d2,
    "d3_j108": base._scenario_d3,
}
BACKEND_DEFAULT_OPTIMIZER_MAXITER = {"copula": 15, "mcem": 100}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="d2_j50,d3_j108")
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000]
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--backends", default="copula,mcem")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument(
        "--initializer", default="auto", choices=["auto", "copula", "family"]
    )
    parser.add_argument(
        "--threshold-rule", default="first_two",
        choices=["first_two", "all_but_last", "none"],
    )
    parser.add_argument(
        "--mstep-solver", default="hybrid", choices=["lbfgsb", "fista", "hybrid"],
        help="copula backend M-step; 'lbfgsb' reproduces the main_r20 baseline",
    )
    parser.add_argument("--fista-maxiter", type=int, default=None,
                        help="copula backend: FISTA budget (default: package default, 300)")
    parser.add_argument("--slab-shape", type=float, default=1.0,
                        help="copula backend: Gamma shape of the slab-rate hyperprior (CopDDE 1)")
    parser.add_argument("--slab-rate", type=float, default=5.0,
                        help="copula backend: Gamma rate of the slab-rate hyperprior (CopDDE 5)")
    parser.add_argument("--penalty-power", type=float, nargs="+", default=[0.0],
                        help="copula backend: penalty multiplier (N / reference) ** power; "
                             "one value or one per layer")
    parser.add_argument("--penalty-reference", type=float, default=500.0)
    parser.add_argument("--penalty-layers", default="all", choices=["all", "deep"])
    parser.add_argument("--penalty-allocation", action="store_true",
                        help="copula backend: CUSP allocation sees the scaled coefficients")
    parser.add_argument(
        "--mstep-latent", default="draw", choices=["draw", "mean_field"],
        help="copula backend: M-step design from Bernoulli draws or mean-field probabilities",
    )
    parser.add_argument(
        "--observed-noise-scale", type=float, default=0.0,
        help="copula backend: sd multiplier on sqrt(gamma/temp) noise added to X in the M-step",
    )
    parser.add_argument(
        "--tau-fixed", type=float, default=None,
        help="use this constant tau instead of max(0.3, 3 N^-0.3)",
    )
    parser.add_argument(
        "--t-spike-multipliers", default=None,
        help="comma-separated per-layer multipliers applied to the scenario t_spike",
    )
    parser.add_argument("--standardize", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--seed-base", type=int, default=20261015)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-trace", action="store_true")
    return parser.parse_args()


def _tau_default(N):
    return max(0.3, 3.0 * N ** -0.3)


def _tau_for(N, settings):
    if settings.get("tau_fixed") is not None:
        return float(settings["tau_fixed"])
    return _tau_default(N)


def _t_spike_for(scenario, settings):
    t_spike = np.asarray(scenario["base_t_spike"], dtype=float)
    multipliers = settings.get("t_spike_multipliers")
    if multipliers:
        mult = np.asarray([float(v) for v in multipliers.split(",")], dtype=float)
        if mult.size != t_spike.size:
            raise ValueError("t_spike multipliers must have one entry per layer.")
        t_spike = t_spike * mult
    return t_spike


def _run_one(task):
    scenario_name, N, replicate, backend, settings = task
    scenario = SCENARIOS[scenario_name]()
    data_seed = settings["seed_base"] + 100_000 * scenario["D"] + 1_000 * N + replicate
    fit_seed = data_seed + 202
    X, A_true = base._simulate(scenario, N, data_seed)
    kwargs = dict(
        method="DDE",
        family="Normal",
        D=scenario["D"],
        dimension_method="csp",
        t_spike=_t_spike_for(scenario, settings),
        temp=float(scenario["base_temp"]),
        tau=_tau_for(N, settings),
        C=settings["mc_samples"],
        tol=0.005,
        max_it=settings["max_iter"],
        csp_backend=backend,
        random_state=fit_seed,
        keep_trace=settings["keep_trace"],
        verbose=False,
    )
    if settings["initializer"] != "auto":
        kwargs["csp_initializer"] = settings["initializer"]
    if settings["standardize"] != "auto":
        kwargs["standardize"] = settings["standardize"] == "true"
    optimizer_maxiter = settings["optimizer_maxiter"]
    if optimizer_maxiter is None:
        optimizer_maxiter = BACKEND_DEFAULT_OPTIMIZER_MAXITER[backend]
    kwargs["optimizer_maxiter"] = optimizer_maxiter
    if backend == "copula":
        kwargs["csp_threshold_rule"] = settings["threshold_rule"]
        kwargs["csp_mstep_solver"] = settings["mstep_solver"]
        kwargs["csp_mstep_latent"] = settings.get("mstep_latent", "draw")
        kwargs["csp_observed_noise_scale"] = settings.get("observed_noise_scale", 0.0)
        kwargs["csp_fista_maxiter"] = settings.get("fista_maxiter")
        kwargs["csp_slab_shape"] = settings.get("slab_shape", 1.0)
        kwargs["csp_slab_rate"] = settings.get("slab_rate", 5.0)
        power = settings.get("penalty_power", 0.0)
        kwargs["csp_penalty_scaling_power"] = (
            power[0] if isinstance(power, (list, tuple)) and len(power) == 1 else power
        )
        kwargs["csp_penalty_scaling_reference"] = settings.get("penalty_reference", 500.0)
        kwargs["csp_penalty_scaling_layers"] = settings.get("penalty_layers", "all")
        kwargs["csp_penalty_scaling_allocation"] = settings.get("penalty_allocation", False)

    started = time.perf_counter()
    error = ""
    try:
        with threadpool_limits(limits=1):
            result = fit(X, **kwargs)
    except Exception as exc:  # noqa: BLE001 - record and continue
        result = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started

    record = {
        "scenario": scenario_name,
        "N": int(N),
        "replicate": int(replicate),
        "backend": backend,
        "initializer": settings["initializer"],
        "standardize": settings["standardize"],
        "optimizer_maxiter": int(optimizer_maxiter),
        "max_iter": int(settings["max_iter"]),
        "mc_samples": int(settings["mc_samples"]),
        "tau": float(_tau_for(N, settings)),
        "t_spike": json.dumps(_t_spike_for(scenario, settings).tolist()),
        "mstep_latent": settings.get("mstep_latent", "draw"),
        "fista_maxiter": settings.get("fista_maxiter"),
        "observed_noise_scale": float(settings.get("observed_noise_scale", 0.0)),
        "temp": float(scenario["base_temp"]),
        "K_true": json.dumps(list(scenario["K_true"])),
        "K_max": json.dumps(list(scenario["K_max"])),
        "fit_seconds": float(elapsed),
        "error": error,
    }
    if result is None:
        record.update(K_fit="", exact_width=np.nan, dimension_l1_error=np.nan,
                      iterations=np.nan, stopped_by_convergence=np.nan)
        for d in range(scenario["D"]):
            record[f"K{d + 1}_true"] = scenario["K_true"][d]
            record[f"K{d + 1}_fit"] = np.nan
            record[f"A{d + 1}_final_accuracy"] = np.nan
        return record

    K_fit = tuple(int(value) for value in result.K_list)
    record.update(
        K_fit=json.dumps(list(K_fit)),
        exact_width=int(K_fit == tuple(scenario["K_true"])),
        dimension_l1_error=int(
            sum(abs(f - t) for f, t in zip(K_fit, scenario["K_true"]))
        ),
        iterations=int(result.diagnostics["iterations"]),
        stopped_by_convergence=int(
            bool(result.diagnostics.get("stopped_by_convergence", False))
        ),
        initialization=result.diagnostics.get("initialization", ""),
        standardized=int(bool(result.diagnostics.get("standardized", False))),
    )
    for d in range(scenario["D"]):
        record[f"K{d + 1}_true"] = scenario["K_true"][d]
        record[f"K{d + 1}_fit"] = K_fit[d]
        record[f"A{d + 1}_final_accuracy"] = base._matched_binary_accuracy(
            result.A_list[d], A_true[d]
        )
        dup, matched, weak = _column_diagnostics(result.A_list[d], A_true[d])
        record[f"A{d + 1}_duplicate_pairs"] = dup
        record[f"A{d + 1}_matched_truth"] = matched
        record[f"A{d + 1}_weak_columns"] = weak
    if settings["keep_trace"]:
        trace = result.diagnostics.get("trace", [])
        record["K_trace"] = json.dumps(
            [np.asarray(entry["num_active"]).astype(int).tolist() for entry in trace]
        )
    return record


def _column_diagnostics(A_fit, A_true):
    """Count near-duplicate fitted columns and columns unrelated to any truth."""
    A_fit = np.asarray(A_fit, dtype=float)
    A_true = np.asarray(A_true, dtype=float)
    if A_fit.shape[1] == 0:
        return 0, 0, 0
    with np.errstate(invalid="ignore", divide="ignore"):
        within = np.nan_to_num(np.corrcoef(A_fit.T)) if A_fit.shape[1] > 1 else np.zeros((1, 1))
        cross = np.nan_to_num(
            np.corrcoef(A_fit.T, A_true.T)[: A_fit.shape[1], A_fit.shape[1]:]
        )
    np.fill_diagonal(within, 0.0)
    duplicate_pairs = int(np.sum(np.triu(np.abs(within) > 0.95)))
    best = np.max(np.abs(cross), axis=1)
    matched_truth = len(set(np.argmax(np.abs(cross), axis=1)[best > 0.5]))
    weak_columns = int(np.sum(best < 0.5))
    return duplicate_pairs, matched_truth, weak_columns


def _record_key(record):
    return (record["scenario"], int(record["N"]), int(record["replicate"]), record["backend"])


def _summarize(records, out_dir):
    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario"], record["backend"], int(record["N"]))].append(record)

    summary_rows = []
    layer_rows = []
    for (scenario, backend, N), rows in sorted(groups.items()):
        ok = [r for r in rows if not r["error"]]
        D = len(json.loads(rows[0]["K_true"]))
        K_true = json.loads(rows[0]["K_true"])
        row = {
            "scenario": scenario,
            "backend": backend,
            "N": N,
            "K_true": rows[0]["K_true"],
            "fits": len(rows),
            "successful_fits": len(ok),
            "exact_rate": float(np.mean([r["exact_width"] for r in ok])) if ok else np.nan,
            "mean_dimension_l1_error": float(np.mean([r["dimension_l1_error"] for r in ok])) if ok else np.nan,
            "mean_iterations": float(np.mean([r["iterations"] for r in ok])) if ok else np.nan,
            "convergence_rate": float(np.mean([r["stopped_by_convergence"] for r in ok])) if ok else np.nan,
            "mean_fit_seconds": float(np.mean([r["fit_seconds"] for r in rows])),
        }
        mean_widths = []
        for d in range(D):
            values = np.asarray([r[f"K{d + 1}_fit"] for r in ok], dtype=float)
            mean_widths.append(float(np.mean(values)) if ok else np.nan)
            row[f"K{d + 1}_mean"] = mean_widths[-1]
            row[f"K{d + 1}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"A{d + 1}_accuracy_mean"] = (
                float(np.nanmean([r[f"A{d + 1}_final_accuracy"] for r in ok])) if ok else np.nan
            )
            layer_rows.append(
                {
                    "scenario": scenario,
                    "backend": backend,
                    "N": N,
                    "layer": d + 1,
                    "K_true": K_true[d],
                    "mean_K_fit": mean_widths[-1],
                    "sd_K_fit": row[f"K{d + 1}_sd"],
                    "mean_bias": mean_widths[-1] - K_true[d] if ok else np.nan,
                    "exact_rate": float(np.mean(values == K_true[d])) if ok else np.nan,
                    "under_rate": float(np.mean(values < K_true[d])) if ok else np.nan,
                    "over_rate": float(np.mean(values > K_true[d])) if ok else np.nan,
                }
            )
        row["mean_widths"] = json.dumps([round(v, 2) for v in mean_widths])
        summary_rows.append(row)

    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "layer_summary.csv", layer_rows)
    return summary_rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "seed_base": args.seed_base,
        "max_iter": args.max_iter,
        "optimizer_maxiter": args.optimizer_maxiter,
        "mc_samples": args.mc_samples,
        "initializer": args.initializer,
        "threshold_rule": args.threshold_rule,
        "mstep_solver": args.mstep_solver,
        "mstep_latent": args.mstep_latent,
        "fista_maxiter": args.fista_maxiter,
        "slab_shape": args.slab_shape,
        "slab_rate": args.slab_rate,
        "penalty_power": args.penalty_power,
        "penalty_reference": args.penalty_reference,
        "penalty_layers": args.penalty_layers,
        "penalty_allocation": args.penalty_allocation,
        "observed_noise_scale": args.observed_noise_scale,
        "tau_fixed": args.tau_fixed,
        "t_spike_multipliers": args.t_spike_multipliers,
        "standardize": args.standardize,
        "keep_trace": args.keep_trace,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "scenarios": args.scenarios,
                "sample_sizes": args.sample_sizes,
                "replicates": args.replicates,
                "replicate_start": args.replicate_start,
                "backends": args.backends,
                "workers": args.workers,
            },
            indent=2,
        )
    )

    checkpoint = out_dir / "fits.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[_record_key(record)] = record

    tasks = []
    for scenario_name in args.scenarios.split(","):
        for N in args.sample_sizes:
            for replicate in range(args.replicate_start, args.replicate_start + args.replicates):
                for backend in args.backends.split(","):
                    task = (scenario_name, N, replicate, backend, settings)
                    if (scenario_name, N, replicate, backend) not in done:
                        tasks.append(task)
    # Largest problems first so the pool stays busy at the end.
    tasks.sort(key=lambda t: (-t[1], t[0]))
    print(f"{len(done)} cached fits, {len(tasks)} fits to run", flush=True)

    records = list(done.values())

    def _store(record):
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(
            f"{record['scenario']} N={record['N']} rep={record['replicate']} "
            f"{record['backend']}: K_fit={record['K_fit']} iters={record['iterations']} "
            f"{record['fit_seconds']:.1f}s {record['error']}",
            flush=True,
        )

    if args.workers <= 1:
        for task in tasks:
            _store(_run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, task) for task in tasks]
            for future in as_completed(futures):
                _store(future.result())

    base._write_csv(out_dir / "fits.csv", records)
    summary = _summarize(records, out_dir)
    print("\nMean fitted widths (truth in K_true):")
    for row in summary:
        print(
            f"  {row['scenario']:8s} {row['backend']:7s} N={row['N']:5d} "
            f"mean={row['mean_widths']} exact={row['exact_rate']:.2f} "
            f"L1={row['mean_dimension_l1_error']:.2f} iters={row['mean_iterations']:.1f} "
            f"({row['successful_fits']}/{row['fits']} ok, {row['mean_fit_seconds']:.1f}s)"
        )


if __name__ == "__main__":
    main()
