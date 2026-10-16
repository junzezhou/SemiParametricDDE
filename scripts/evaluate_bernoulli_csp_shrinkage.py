"""Dimension-shrinkage comparison of the Bernoulli DDE CSP backends.

Scenarios, data generation, seeds, and metrics come from
``evaluate_bernoulli_csp_recovery.py`` (D=2: J=27, truth (6, 2), maximal
(9, 3); D=3: J=64, truth (18, 6, 2), maximal (21, 7, 2)), so the ``mcem``
arm reproduces the stored MCEM baselines for the same replicates.  The
``copula`` arm runs the CopDDE-derived kernel with a logistic observed layer.
The headline summary is the mean fitted width per layer by N.
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
import evaluate_bernoulli_csp_recovery as base  # noqa: E402

from dde import fit  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="d2,d3")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 2000, 8000])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--backends", default="copula,mcem")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--t-spike-multiplier", type=float, default=1.0)
    parser.add_argument("--optimizer-maxiter", type=int, default=None,
                        help="copula backend only (default 15); mcem keeps 100")
    parser.add_argument("--mstep-solver", default="fista_lower")
    parser.add_argument("--fista-maxiter", type=int, default=None)
    parser.add_argument("--penalty-power", type=float, nargs="+", default=[0.0])
    parser.add_argument("--penalty-reference", type=float, default=500.0)
    parser.add_argument("--penalty-layers", default="lower", choices=["all", "deep", "lower"])
    parser.add_argument("--slab-shape", type=float, default=1.0)
    parser.add_argument("--slab-rate", type=float, default=5.0)
    parser.add_argument("--latent-transition", default="systematic_coordinate_gibbs")
    parser.add_argument("--merge-cosine", type=float, default=0.0)
    parser.add_argument("--shrinkage-burn", type=int, default=0,
                        help="copula backend: slab-start burn-in iterations (default 0 here; the package default is 10 for D>=3)")
    parser.add_argument("--no-reorder-after-burn", action="store_true",
                        help="copula backend: skip the strength ordering at the end of the burn-in")
    parser.add_argument("--seed-base", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-trace", action="store_true")
    return parser.parse_args()


def _run_one(task):
    scenario_key, N, replicate, backend, settings = task
    scenario = base.build_scenario(scenario_key)
    data_seed, fit_seed = base._seed_pair(settings["seed_base"], scenario_key, N, replicate)
    X, A_true = base.simulate(scenario, N, data_seed)
    t_spike = np.asarray(scenario["t_spike"], dtype=float) * settings["t_spike_multiplier"]
    kwargs = dict(
        method="DDE", family="Bernoulli", D=scenario["D"], dimension_method="csp",
        t_spike=t_spike, temp=float(scenario["temperature"]), tau=settings["tau"],
        C=settings["mc_samples"], tol=0.005, max_it=settings["max_iter"],
        csp_backend=backend, random_state=fit_seed, keep_trace=settings["keep_trace"],
        verbose=False,
    )
    if backend == "copula":
        kwargs.update(
            optimizer_maxiter=settings["optimizer_maxiter"] or 15,
            csp_mstep_solver=(settings["mstep_solver"].split(",") if "," in settings["mstep_solver"] else settings["mstep_solver"]),
            csp_fista_maxiter=settings["fista_maxiter"],
            csp_penalty_scaling_power=(settings["penalty_power"][0] if len(settings["penalty_power"]) == 1 else settings["penalty_power"]),
            csp_penalty_scaling_reference=settings["penalty_reference"],
            csp_penalty_scaling_layers=settings["penalty_layers"],
            csp_slab_shape=settings["slab_shape"],
            csp_slab_rate=settings["slab_rate"],
            csp_latent_transition=settings["latent_transition"],
            csp_merge_cosine=settings["merge_cosine"],
            csp_shrinkage_burn=settings["shrinkage_burn"],
            csp_reorder_after_shrinkage_burn=not settings["no_reorder_after_burn"],
        )
    else:
        kwargs.update(optimizer_maxiter=100)

    started = time.perf_counter()
    error = ""
    try:
        with threadpool_limits(limits=1):
            result = fit(X, **kwargs)
    except Exception as exc:  # noqa: BLE001
        result = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    record = {
        "scenario": scenario["name"], "scenario_key": scenario_key, "N": int(N),
        "replicate": int(replicate), "backend": backend, "D": scenario["D"],
        "K_true": json.dumps(list(scenario["K_true"])), "K_max": json.dumps(list(scenario["K_max"])),
        "t_spike": json.dumps(t_spike.tolist()), "tau": settings["tau"],
        "fit_seconds": float(elapsed), "error": error,
    }
    if result is None:
        record.update(K_fit="", exact_width=np.nan, dimension_l1_error=np.nan, iterations=np.nan)
        for d in range(scenario["D"]):
            record[f"K{d + 1}_fit"] = np.nan
        return record
    metrics = base.calculate_metrics(result, X, A_true, scenario, settings["tau"])
    record.update(metrics)
    record["iterations"] = int(result.diagnostics["iterations"])
    record["stopped_by_convergence"] = int(bool(result.diagnostics.get("stopped_by_convergence", False)))
    K_fit = json.loads(metrics["K_fit"])
    for d in range(scenario["D"]):
        record[f"K{d + 1}_fit"] = K_fit[d]
    if settings["keep_trace"]:
        record["K_trace"] = json.dumps(
            [np.asarray(e["num_active"]).astype(int).tolist() for e in result.diagnostics.get("trace", [])]
        )
    return record


def _summarize(records, out_dir):
    groups = defaultdict(list)
    for r in records:
        groups[(r["scenario_key"], r["backend"], int(r["N"]))].append(r)
    rows = []
    for (key, backend, N), rs in sorted(groups.items()):
        ok = [r for r in rs if not r["error"]]
        D = int(rs[0]["D"])
        row = {"scenario": key, "backend": backend, "N": N, "K_true": rs[0]["K_true"],
               "fits": len(rs), "successful_fits": len(ok)}
        widths = []
        for d in range(D):
            vals = np.asarray([r[f"K{d + 1}_fit"] for r in ok], dtype=float)
            widths.append(float(vals.mean()) if ok else np.nan)
            row[f"K{d + 1}_mean"] = widths[-1]
        row["mean_widths"] = json.dumps([round(w, 2) for w in widths])
        for key_name in ("exact_width", "dimension_l1_error", "iterations", "latent_accuracy", "graph_accuracy", "conditional_log_loss", "fit_seconds"):
            row[f"{key_name}_mean"] = float(np.mean([r[key_name] for r in ok])) if ok else np.nan
        rows.append(row)
    if rows:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (out_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, restval="")
            writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    args = _parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in (
        "max_iter", "mc_samples", "tau", "t_spike_multiplier", "optimizer_maxiter",
        "mstep_solver", "fista_maxiter", "penalty_power", "penalty_reference",
        "penalty_layers", "slab_shape", "slab_rate", "latent_transition",
        "merge_cosine", "shrinkage_burn", "no_reorder_after_burn", "seed_base",
        "keep_trace")}
    (out_dir / "metadata.json").write_text(json.dumps({"settings": settings, "args": vars(args)}, indent=2, default=str))
    checkpoint = out_dir / "fits.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["scenario_key"], r["N"], r["replicate"], r["backend"])] = r
    tasks = [
        (key, N, rep, backend, settings)
        for key in args.scenarios.split(",")
        for N in args.sample_sizes
        for rep in range(args.replicate_start, args.replicate_start + args.replicates)
        for backend in args.backends.split(",")
        if (key, N, rep, backend) not in done
    ]
    tasks.sort(key=lambda t: (-t[1], t[0]))
    print(f"{len(done)} cached fits, {len(tasks)} fits to run", flush=True)
    records = list(done.values())

    def store(record):
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), default=float) + "\n")
        print(f"{record['scenario_key']} N={record['N']} rep={record['replicate']} {record['backend']}: "
              f"K_fit={record['K_fit']} iters={record['iterations']} {record['fit_seconds']:.1f}s {record['error']}", flush=True)

    if args.workers <= 1:
        for task in tasks:
            store(_run_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, task) for task in tasks]
            for future in as_completed(futures):
                store(future.result())
    fields = sorted({k for r in records for k in r})
    with (out_dir / "fits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    print("\nMean fitted widths (truth in K_true):")
    for row in _summarize(records, out_dir):
        print(f"  {row['scenario']:3s} {row['backend']:7s} N={row['N']:5d} truth={row['K_true']} mean={row['mean_widths']} "
              f"exact={row['exact_width_mean']:.2f} L1={row['dimension_l1_error_mean']:.2f} "
              f"latent={row['latent_accuracy_mean']:.3f} iters={row['iterations_mean']:.1f} "
              f"({row['successful_fits']}/{row['fits']} ok, {row['fit_seconds_mean']:.1f}s)")


if __name__ == "__main__":
    main()
