"""Run the Big Five real-data CopDDE-CSP analysis with dde.

The workflow mirrors ``DDE-Copula-m/Scripts/B5.m`` as closely as the package
interface allows:

* read WASU0003_B5_OUTPUT.sav from ``Data/``;
* keep raw columns 3:102 and drop raw columns 46 and 53;
* build rank-likelihood inputs before filling missing responses with 3;
* use two-layer maximal widths [floor(J / 3), floor(floor(J / 3) / 3)];
* select CSP hyperparameters by predictive pMSE over the original grid.

Use ``--smoke`` for a fast interface check before launching the full grid.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DDE_ROOT = REPO_ROOT / "dde"
DATA_DIR = REPO_ROOT / "Data"
DEFAULT_SAV = DATA_DIR / "WASU0003_B5_OUTPUT.sav"
DEFAULT_OUT_DIR = REPO_ROOT / "Results" / "realdata_b5_copula_csp"

RAW_BIGFIVE_START = 3
RAW_BIGFIVE_END = 102
RAW_DROP_COLS = {46, 53}
EXTRA_VARS = ["gender", "race", "educ", "marstat", "employ"]
IDEOLOGY_VAR = "ideo5"

PAPER_TAUS = [0.1, 0.2, 0.3, 0.4]
PAPER_T_SPIKES = [0.02, 0.04, 0.06, 0.08]
PAPER_TEMPS = [0.7, 0.8, 0.9]


@dataclass
class DataBundle:
    raw: pd.DataFrame
    selected_names: list[str]
    selected_labels: dict[str, str]
    X_rank_source: np.ndarray
    X_filled: np.ndarray
    R: np.ndarray
    Rlevels: np.ndarray
    K_list: list[int]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def _ensure_dde_on_path() -> None:
    src = str(DDE_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _read_sav_via_r(sav_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read SPSS data and labels through R/haven.

    pyreadstat is intentionally not a package dependency. R/haven is available
    in the project environment and preserves the SPSS labels used by the
    codebook.
    """
    sav_path = sav_path.resolve()
    if not sav_path.exists():
        raise FileNotFoundError(f"Cannot find SPSS data file: {sav_path}")

    with tempfile.TemporaryDirectory(prefix="b5_sav_") as tmp:
        tmp_dir = Path(tmp)
        data_csv = tmp_dir / "data.csv"
        labels_csv = tmp_dir / "labels.csv"
        r_script = tmp_dir / "read_b5.R"
        r_script.write_text(
            textwrap.dedent(
                f"""
                suppressPackageStartupMessages(library(haven))
                d <- read_sav({json.dumps(str(sav_path))})
                labels <- data.frame(
                  name = names(d),
                  label = vapply(d, function(x) {{
                    y <- attr(x, "label")
                    if (is.null(y)) "" else as.character(y)
                  }}, character(1)),
                  stringsAsFactors = FALSE
                )
                write.csv(as.data.frame(d), {json.dumps(str(data_csv))}, row.names = FALSE, na = "")
                write.csv(labels, {json.dumps(str(labels_csv))}, row.names = FALSE, na = "")
                """
            ),
            encoding="utf-8",
        )
        subprocess.run(["Rscript", str(r_script)], check=True)
        data = pd.read_csv(data_csv)
        labels = pd.read_csv(labels_csv)
    return data, labels


def _make_rank_inputs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    R = np.full(X.shape, np.nan, dtype=float)
    Rlevels = np.zeros(X.shape[1], dtype=int)
    for j in range(X.shape[1]):
        col = X[:, j]
        obs = ~np.isnan(col)
        if not np.any(obs):
            continue
        values, inverse = np.unique(col[obs], return_inverse=True)
        R[obs, j] = inverse + 1
        Rlevels[j] = len(values)
    return R, Rlevels


def load_b5_data(sav_path: Path) -> DataBundle:
    raw, labels = _read_sav_via_r(sav_path)
    raw_cols = list(range(RAW_BIGFIVE_START, RAW_BIGFIVE_END + 1))
    selected_raw_cols = [col for col in raw_cols if col not in RAW_DROP_COLS]
    selected_idx = [col - 1 for col in selected_raw_cols]
    selected_names = raw.columns[selected_idx].tolist()
    selected_labels = dict(zip(labels["name"], labels["label"]))

    X_rank_source = raw.loc[:, selected_names].to_numpy(dtype=float)
    R, Rlevels = _make_rank_inputs(X_rank_source)
    X_filled = np.where(np.isnan(X_rank_source), 3.0, X_rank_source)

    J = X_filled.shape[1]
    K1_max = J // 3
    K2_max = K1_max // 3
    return DataBundle(
        raw=raw,
        selected_names=selected_names,
        selected_labels={name: selected_labels.get(name, "") for name in selected_names},
        X_rank_source=X_rank_source,
        X_filled=X_filled,
        R=R,
        Rlevels=Rlevels,
        K_list=[K1_max, K2_max],
    )


def paper_candidates(
    *,
    smoke: bool = False,
    optimizer_maxiter: int | None = None,
    convergence_tol: float | None = None,
) -> list[dict[str, Any]]:
    taus = PAPER_TAUS
    t_values = PAPER_T_SPIKES
    temps = PAPER_TEMPS
    if smoke:
        taus = [PAPER_TAUS[0]]
        t_values = PAPER_T_SPIKES[:2]
        temps = [PAPER_TEMPS[0]]

    candidates = []
    for tau, t1, t2, temp in product(taus, t_values, t_values, temps):
        candidate: dict[str, Any] = {
            "name": f"tau{tau:g}_t1{t1:g}_t2{t2:g}_temp{temp:g}",
            "C": 1,
            "it": 50,
            "tau": tau,
            "t_spike": [t1, t2],
            "temp": temp,
            "verbose": False,
        }
        if optimizer_maxiter is not None:
            candidate["optimizer_maxiter"] = int(optimizer_maxiter)
        if convergence_tol is not None:
            candidate["convergence_tol"] = float(convergence_tol)
        if smoke:
            candidate["it"] = 1
            candidate["optimizer_maxiter"] = 1
        candidates.append(candidate)
    return candidates


def selection_summary(result: Any) -> dict[str, Any]:
    cv_table = result.diagnostics["cv_table"]
    best_index = int(result.diagnostics["hyperparameter_selection"].best_index)
    best_row = dict(cv_table[best_index])
    B_list = result.B_list or []
    active = [int(np.any(np.asarray(B)[:, 1:] != 0.0, axis=0).sum()) for B in B_list]
    return {
        "best_index": best_index,
        "best_row": best_row,
        "num_active_by_layer": active,
        "iterations": result.diagnostics.get("iterations"),
        "K_list": result.diagnostics.get("K_list"),
    }


def _anchor_rows_for_B(
    B: np.ndarray,
    *,
    layer: int,
    row_names: list[str],
    row_labels: dict[str, str] | None = None,
    top_n: int,
) -> list[dict[str, Any]]:
    B = np.asarray(B, dtype=float)
    active_cols = np.where(np.any(B[:, 1:] != 0.0, axis=0))[0] + 1
    rows: list[dict[str, Any]] = []
    for col in active_cols:
        other_cols = [c for c in active_cols if c != col]
        if other_cols:
            scores = np.maximum(0.0, np.min(B[:, [col]] - B[:, other_cols], axis=1))
        else:
            scores = np.maximum(0.0, B[:, col])
        order = np.argsort(-scores)[: min(top_n, B.shape[0])]
        for rank, row_idx in enumerate(order, start=1):
            item = row_names[row_idx]
            rows.append(
                {
                    "layer": layer,
                    "active_column": int(col),
                    "latent_index": int(col),
                    "rank": int(rank),
                    "row_index": int(row_idx + 1),
                    "item": item,
                    "label": "" if row_labels is None else row_labels.get(item, ""),
                    "anchor_score": float(scores[row_idx]),
                    "loading": float(B[row_idx, col]),
                }
            )
    return rows


def anchor_tables(result: Any, data: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    B1 = np.asarray(result.B_list[0], dtype=float)
    B2 = np.asarray(result.B_list[1], dtype=float)
    b1_rows = _anchor_rows_for_B(
        B1,
        layer=1,
        row_names=data.selected_names,
        row_labels=data.selected_labels,
        top_n=5,
    )
    b2_rows = _anchor_rows_for_B(
        B2,
        layer=2,
        row_names=[f"A1_{i}" for i in range(1, B2.shape[0] + 1)],
        row_labels=None,
        top_n=3,
    )
    return pd.DataFrame(b1_rows), pd.DataFrame(b2_rows)


def active_latent_frame(result: Any, raw: pd.DataFrame) -> pd.DataFrame:
    A_sample_long = result.diagnostics.get("A_sample_long")
    if A_sample_long is None:
        raise ValueError("Result diagnostics do not contain A_sample_long.")

    B1 = np.asarray(result.B_list[0], dtype=float)
    B2 = np.asarray(result.B_list[1], dtype=float)
    active_A1 = np.where(np.any(B1[:, 1:] != 0.0, axis=0))[0]
    active_A2 = np.where(np.any(B2[:, 1:] != 0.0, axis=0))[0]

    A1_mean = np.mean(np.asarray(A_sample_long[0], dtype=float), axis=2)
    A2_mean = np.mean(np.asarray(A_sample_long[1], dtype=float), axis=2)
    parts = []
    if active_A1.size:
        parts.append(pd.DataFrame(A1_mean[:, active_A1 + 1], columns=[f"A1_{i + 1}" for i in active_A1]))
    if active_A2.size:
        parts.append(pd.DataFrame(A2_mean[:, active_A2 + 1], columns=[f"A2_{i + 1}" for i in active_A2]))
    if not parts:
        latent = pd.DataFrame(index=np.arange(raw.shape[0]))
    else:
        latent = pd.concat(parts, axis=1)

    extras = raw.loc[:, EXTRA_VARS + [IDEOLOGY_VAR]].reset_index(drop=True)
    return pd.concat([latent.reset_index(drop=True), extras], axis=1)


def ideology_tree_summary(latent: pd.DataFrame, *, random_state: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = latent[IDEOLOGY_VAR].isin([1, 5])
    df = latent.loc[idx].copy()
    y = df.pop(IDEOLOGY_VAR).astype(int)
    X = df
    categorical_cols = [col for col in EXTRA_VARS if col in X.columns]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("tree", DecisionTreeClassifier(max_leaf_nodes=11, random_state=random_state)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[1, 5])
    summary = {
        "n_extreme": int(df.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "labels": [1, 5],
        "confusion_matrix": cm.tolist(),
    }
    pred_df = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": pred})
    return pred_df, summary


def write_outputs(
    *,
    out_dir: Path,
    data: DataBundle,
    result: Any,
    config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_table = pd.DataFrame(result.diagnostics["cv_table"])
    timing_table = pd.DataFrame(result.diagnostics.get("timing_table", []))
    summary = selection_summary(result)
    b1_anchors, b2_anchors = anchor_tables(result, data)
    latent = active_latent_frame(result, data.raw)
    tree_pred, tree_summary = ideology_tree_summary(latent, random_state=int(config["random_state"]))

    cv_table.to_csv(out_dir / "b5_copula_csp_cv_table.csv", index=False)
    timing_table.to_csv(out_dir / "b5_copula_csp_timing_table.csv", index=False)
    b1_anchors.to_csv(out_dir / "b5_copula_csp_B1_anchors.csv", index=False)
    b2_anchors.to_csv(out_dir / "b5_copula_csp_B2_anchors.csv", index=False)
    latent.to_csv(out_dir / "b5_copula_csp_active_latents.csv", index=False)
    tree_pred.to_csv(out_dir / "b5_copula_csp_ideology_tree_predictions.csv", index=False)

    metadata = {
        "config": config,
        "data": {
            "n": int(data.X_filled.shape[0]),
            "j": int(data.X_filled.shape[1]),
            "selected_names": data.selected_names,
            "excluded_raw_columns": sorted(RAW_DROP_COLS),
            "excluded_variables": ["BIGFIVE_44", "BIGFIVE_51"],
            "total_missing_selected_cells": int(np.isnan(data.X_rank_source).sum()),
            "rlevels": data.Rlevels.tolist(),
            "K_list": data.K_list,
        },
        "selection": summary,
        "ideology_tree": tree_summary,
    }
    (out_dir / "b5_copula_csp_summary.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default),
        encoding="utf-8",
    )
    with (out_dir / "b5_copula_csp_result.pkl").open("wb") as f:
        pickle.dump(result, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sav-path", type=Path, default=DEFAULT_SAV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny paper-grid subset with one SAEM iteration.")
    parser.add_argument("--n-repeats", type=int, default=None, help="Override pMSE synthetic repetitions.")
    parser.add_argument("--pmse-n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=20260712)
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--convergence-tol", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-5, help="Normal initialization epsilon; B5.m uses 1e-5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dde_on_path()
    from dde import select_hyperparameters
    from dde.algorithms.copula import normal_initialize_copula_dde
    from dde.copula import simulate_gaussian_mixture

    data = load_b5_data(args.sav_path)
    Z_init = simulate_gaussian_mixture(
        data.X_filled,
        data.X_filled.shape[1],
        random_state=int(args.random_state),
    )
    init = normal_initialize_copula_dde(
        data.X_filled,
        data.K_list,
        Z=Z_init,
        K1_max=data.K_list[0],
        random_state=int(args.random_state),
        epsilon=float(args.epsilon),
    )
    candidates = paper_candidates(
        smoke=args.smoke,
        optimizer_maxiter=args.optimizer_maxiter,
        convergence_tol=args.convergence_tol,
    )
    # 25 synthetic data sets per candidate, as in the manuscript (Section 5.2); was 50
    n_repeats = args.n_repeats if args.n_repeats is not None else (1 if args.smoke else 25)

    config = {
        "sav_path": args.sav_path,
        "out_dir": args.out_dir,
        "smoke": bool(args.smoke),
        "n_candidates": len(candidates),
        "n_repeats": int(n_repeats),
        "pmse_n_estimators": int(args.pmse_n_estimators),
        "random_state": int(args.random_state),
        "optimizer_maxiter": args.optimizer_maxiter,
        "convergence_tol": float(args.convergence_tol),
        "epsilon": float(args.epsilon),
        "shared_initialization": True,
        "paper_grid": {
            "taus": PAPER_TAUS,
            "t_spikes": PAPER_T_SPIKES,
            "temps": PAPER_TEMPS,
        },
    }

    print(
        "Loaded B5 data:",
        f"N={data.X_filled.shape[0]}",
        f"J={data.X_filled.shape[1]}",
        f"K_list={data.K_list}",
        f"missing={int(np.isnan(data.X_rank_source).sum())}",
    )
    print(f"Evaluating {len(candidates)} candidates with n_repeats={n_repeats}.")

    result = select_hyperparameters(
        data.X_filled,
        D=2,
        model="copula",
        dimension_method="csp",
        criterion="pmse",
        candidates=candidates,
        fixed_hp={
            "K_list": data.K_list,
            "Z": Z_init,
            "R": data.R,
            "Rlevels": data.Rlevels,
            "prop_in": init["prop"],
            "gamma_in": init["gamma"],
            "B_list_in": init["B_list"],
            "A_list_in": init["A_list"],
        },
        n_repeats=int(n_repeats),
        pmse_n_estimators=int(args.pmse_n_estimators),
        refit=True,
        random_state=int(args.random_state),
    )
    write_outputs(out_dir=args.out_dir, data=data, result=result, config=config)

    summary = selection_summary(result)
    print("Best candidate:")
    print(json.dumps(summary["best_row"], indent=2, default=_json_default))
    print("Active nodes by layer:", summary["num_active_by_layer"])
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
