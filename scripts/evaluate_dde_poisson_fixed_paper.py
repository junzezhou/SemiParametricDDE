"""Poisson DDE fixed-width paper study (D=2..5) with per-layer recovery metrics.

Same design and seed convention as ``evaluate_dde_normal_fixed_paper.py``
(``build_true_params`` with ``J = 3 K_1`` and ``max_val = 4``,
``pen_list = [N**(2/8)] * D``, one draw, oracle-labelled spectral
initialization, seeds ``20260827 + 100000 D + 1000 N + replicate``) with the
Poisson observed layer (``generate_X_poisson``) and the manuscript
convergence settings ``tol = 0.01`` (postprocessed parameter-update RMSE),
``max_it = 50``.  ``tau`` defaults to 0.3 (the MATLAB rule ``3 N**-0.3``
is available with ``--tau-rule matlab``).  The kernel defaults
(``sa_burn_in=10``, ``fixed_init_binarize="anchor_mixture"``,
``fixed_warm_start="layerwise"``) can be overridden for ablations.
Records, per replicate, the overall and per-layer graph accuracy,
coefficient RMSE and latent-state accuracy.  Checkpointed and resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dde import fit
from dde.identifiability import build_true_params
from dde.simulation import generate_X_poisson

SCENARIOS = {
    "D2": {"D": 2, "K_list": (6, 2), "J": 18},
    "D3": {"D": 3, "K_list": (18, 6, 2), "J": 54},
    "D4": {"D": 4, "K_list": (54, 18, 6, 2), "J": 162},
    "D5": {"D": 5, "K_list": (162, 54, 18, 6, 2), "J": 486},
}
SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000, 16000)
_TRUTH = {}


def _truth(name):
    if name not in _TRUTH:
        spec = SCENARIOS[name]
        prop, B_list, G_list, _ = build_true_params(list(spec["K_list"]), spec["J"], 4.0)
        _TRUTH[name] = (np.asarray(prop, float), [np.asarray(B, float) for B in B_list],
                        [np.asarray(G, bool) for G in G_list], None)
    return _TRUTH[name]


def run_one(task):
    name, N, replicate, settings = task
    spec = SCENARIOS[name]
    D = spec["D"]
    prop_t, B_t, G_t, _ = _truth(name)
    seed = settings["base_seed"] + 100_000 * D + 1_000 * int(N) + int(replicate)
    row = {"scenario": name, "D": D, "J": spec["J"], "K_list": json.dumps(list(spec["K_list"])),
           "N": int(N), "replicate": int(replicate), "seed": seed, "status": "error", "error": ""}
    started = time.perf_counter()
    try:
        X, A_true = generate_X_poisson(int(N), prop_t, B_t, random_state=seed)
        np.random.seed(seed)
        tau = float(settings["tau"]) if settings["tau_rule"] == "fixed" else float(3.0 * N ** -0.3)
        with threadpool_limits(limits=1):
            result = fit(X, method="DDE", D=D, family="Poisson", dimension_method="fixed",
                         K_list=list(spec["K_list"]), pen_list=[float(N) ** (2 / 8)] * D,
                         tau=tau, C=1, tol=settings["tol"], max_it=settings["max_it"],
                         sa_burn_in=settings["sa_burn_in"], fixed_init_binarize=settings["binarize"],
                         fixed_warm_start=settings["warm_start"], initialize="oracle",
                         B_list_true=B_t, G_list_true=G_t, verbose=False)
        layer_acc, layer_rmse, correct, edges, sq = [], [], 0, 0, []
        for B_hat, B_true, G_true in zip(result.B_list, B_t, G_t):
            graph_hat = np.asarray(B_hat)[:, 1:] != 0.0
            layer_acc.append(float(np.mean(graph_hat == G_true)))
            correct += int(np.sum(graph_hat == G_true)); edges += G_true.size
            err = (np.asarray(B_hat) - B_true).ravel() ** 2
            layer_rmse.append(float(np.sqrt(np.mean(err)))); sq.append(err)
        all_err = np.concatenate(sq + [(result.prop - prop_t) ** 2])
        row.update(status="ok", iterations=int(result.diagnostics["iterations"]),
                   graph_accuracy=correct / edges, layer_graph_accuracy=json.dumps(layer_acc),
                   B_rmse=float(np.sqrt(np.mean(np.concatenate(sq)))), layer_parameter_rmse=json.dumps(layer_rmse),
                   parameter_rmse=float(np.sqrt(np.mean(all_err))),
                   prop_mse=float(np.mean((result.prop - prop_t) ** 2)),
                   latent_accuracy=json.dumps([float(np.mean(np.asarray(A) == np.asarray(At))) for A, At in zip(result.A_list, A_true)]),
                   warm_start=json.dumps(result.diagnostics.get("warm_start")))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - started
    return row


def _summarize(rows, out_dir):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(r["scenario"], int(r["N"]))].append(r)
    out = []
    for (name, N), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        acc = np.array([json.loads(r["layer_graph_accuracy"]) for r in rs])
        rmse = np.array([json.loads(r["layer_parameter_rmse"]) for r in rs])
        lat = np.array([json.loads(r["latent_accuracy"]) for r in rs])
        out.append({"scenario": name, "D": rs[0]["D"], "J": rs[0]["J"], "K_list": rs[0]["K_list"], "N": N,
                    "replicates": len(rs),
                    "graph_accuracy_mean": float(np.mean([r["graph_accuracy"] for r in rs])),
                    "graph_accuracy_sd": float(np.std([r["graph_accuracy"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
                    "B_rmse_mean": float(np.mean([r["B_rmse"] for r in rs])),
                    "parameter_rmse_mean": float(np.mean([r["parameter_rmse"] for r in rs])),
                    "layer_graph_accuracy_mean": json.dumps(acc.mean(0).tolist()),
                    "layer_parameter_rmse_mean": json.dumps(rmse.mean(0).tolist()),
                    "latent_accuracy_mean": json.dumps(lat.mean(0).tolist()),
                    "iterations_mean": float(np.mean([r["iterations"] for r in rs])),
                    "seconds_mean": float(np.mean([r["seconds"] for r in rs]))})
    if out:
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--sample-sizes", nargs="+", type=int, default=list(SAMPLE_SIZES))
    p.add_argument("--replicates", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=20260827)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--tau-rule", choices=["fixed", "matlab"], default="fixed", help="fixed: --tau; matlab: 3 N^-0.3")
    p.add_argument("--max-it", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01, help="postprocessed parameter-update RMSE tolerance (manuscript 0.01)")
    p.add_argument("--sa-burn-in", type=int, default=10)
    p.add_argument("--binarize", choices=["anchor_mixture", "projection"], default="anchor_mixture")
    p.add_argument("--warm-start", choices=["layerwise", "none"], default="layerwise")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"base_seed": args.base_seed, "tau": args.tau, "tau_rule": args.tau_rule, "max_it": args.max_it, "tol": args.tol,
                "sa_burn_in": args.sa_burn_in, "binarize": args.binarize, "warm_start": args.warm_start}
    (args.output_dir / "metadata.json").write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "design": f"Poisson: build_true_params(K_list, J=3K1, 4.0); pen N^(2/8); tau {args.tau if args.tau_rule == 'fixed' else '3N^-0.3'}; tol {args.tol}; max_it {args.max_it}; sa_burn_in {args.sa_burn_in}; binarize {args.binarize}; warm_start {args.warm_start}; oracle init; seeds 20260827+100000D+1000N+rep"}, indent=2))
    checkpoint = args.output_dir / "replicates.jsonl"
    rows = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()] if checkpoint.exists() else []
    done = {(r["scenario"], int(r["N"]), int(r["replicate"])) for r in rows if r["status"] == "ok"}
    tasks = [(s, N, rep, settings) for s in args.scenarios for N in args.sample_sizes
             for rep in range(args.replicates) if (s, N, rep) not in done]
    tasks.sort(key=lambda t: (-SCENARIOS[t[0]]["D"], -t[1]))
    print(f"{len(done)} cached, {len(tasks)} to run", flush=True)

    def store(row):
        rows.append(row)
        with checkpoint.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{row['scenario']} N={row['N']} rep={row['replicate']}: {row['status']} {row.get('graph_accuracy', ''):} {row['seconds']:.1f}s {row['error']}", flush=True)

    if args.workers <= 1:
        for t in tasks:
            store(run_one(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(run_one, t) for t in tasks]):
                store(fut.result())
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with (args.output_dir / "replicates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)
    for s in _summarize(rows, args.output_dir):
        print(f"  {s['scenario']} N={s['N']}: acc={s['graph_accuracy_mean']:.4f} layers={s['layer_graph_accuracy_mean']} rmse={s['layer_parameter_rmse_mean']} ({s['replicates']} reps, {s['seconds_mean']:.1f}s)")


if __name__ == "__main__":
    main()
