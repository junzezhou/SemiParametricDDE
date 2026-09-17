"""TIMSS response-time-only SemiDDE analysis.

This script evaluates the package SemiDDE path on the continuous TIMSS response
time variables under the paper's latent dimensions K_list=[7, 1].
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
DDE_ROOT = REPO_ROOT / "dde"
DATA_DIR = REPO_ROOT / "Deep-Discrete-Encoders" / "real data" / "TIMSS" / "dataset"
DEFAULT_OUT_DIR = REPO_ROOT / "Results" / "realdata_timss_semidde_rt"

SKILL_NAMES = [
    "Number",
    "Algebra",
    "Geometry",
    "DataProb",
    "Knowing",
    "Applying",
    "Reasoning",
]
PAPER_TABLE_S19 = {
    1: [0.83, 0.72, 0.88, 0.86, 0.87],
    2: [0.80, 0.72, 0.85, 0.84, 0.85],
    3: [0.76, 0.74, 0.85, 0.84, 0.84],
    4: [0.69, 0.67, 0.76, 0.75, 0.78],
}


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


def load_timss_rt() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(DATA_DIR / "TIMSS.csv", header=None).to_numpy(float)
    q_raw = pd.read_csv(DATA_DIR / "Q.csv", header=None, encoding="utf-8-sig").to_numpy(float)
    survey = pd.read_csv(DATA_DIR / "survey_response.csv", header=None).to_numpy(float).reshape(-1)
    response_time = raw[:, 29:58]
    q_matrix = q_raw[:, 1:]
    if response_time.shape != (435, 29):
        raise ValueError(f"Unexpected TIMSS response-time shape: {response_time.shape}.")
    if q_matrix.shape != (29, 7):
        raise ValueError(f"Unexpected Q-matrix shape after dropping item index: {q_matrix.shape}.")
    if not np.all(np.isfinite(response_time)):
        raise ValueError("Response-time matrix contains non-finite values.")
    return response_time, q_matrix, survey


def make_candidates(n: int, d: int, *, wide: bool = False) -> list[dict[str, Any]]:
    from dde import auto_grid

    if not wide:
        candidates = auto_grid(n, d)
    else:
        candidates = auto_grid(
            n,
            d,
            lambda_exponents=(0.0, 1 / 8, 2 / 8, 3 / 8),
            tau_exponents=(-1 / 8, -2 / 8, -3 / 8),
            tau_values=(0.1, 0.2, 0.3),
        )
    for idx, candidate in enumerate(candidates):
        candidate["name"] = (
            f"cand{idx:02d}_lambda{candidate.get('lambda_exp')}_tau"
            f"{candidate.get('tau_exp', 'fixed')}_{candidate['tau']:.6g}"
        )
    return candidates


def align_a1_to_q(result: Any, q_matrix: np.ndarray) -> dict[str, Any]:
    b1 = np.asarray(result.B_list[0], dtype=float)[:, 1:]
    a1 = np.asarray(result.A_list[0], dtype=float)
    weights = np.abs(b1).T @ q_matrix
    row_ind, col_ind = linear_sum_assignment(-weights)
    estimated_for_skill = np.full(q_matrix.shape[1], -1, dtype=int)
    for est_col, skill_col in zip(row_ind, col_ind):
        estimated_for_skill[skill_col] = est_col

    a1_aligned = np.zeros((a1.shape[0], q_matrix.shape[1]), dtype=float)
    orientation = np.ones(q_matrix.shape[1], dtype=int)
    loading_summary = np.zeros(q_matrix.shape[1], dtype=float)
    for skill_col, est_col in enumerate(estimated_for_skill):
        if est_col < 0:
            continue
        support = q_matrix[:, skill_col] > 0
        mean_loading = float(np.mean(b1[support, est_col])) if np.any(support) else float(np.mean(b1[:, est_col]))
        loading_summary[skill_col] = mean_loading
        if mean_loading < 0:
            orientation[skill_col] = -1
            a1_aligned[:, skill_col] = 1.0 - a1[:, est_col]
        else:
            a1_aligned[:, skill_col] = a1[:, est_col]

    b2 = np.asarray(result.B_list[1], dtype=float)
    if b2.shape[1] > 1 and estimated_for_skill.min() >= 0:
        b2_aligned_slope = b2[estimated_for_skill, 1]
        oriented_b2_slope = b2_aligned_slope * orientation
        a2_orientation = 1 if float(np.mean(oriented_b2_slope)) >= 0 else -1
    else:
        a2_orientation = 1

    a2 = np.asarray(result.A_list[1], dtype=float)
    a2_display = a2[:, [0]] if a2_orientation > 0 else 1.0 - a2[:, [0]]
    return {
        "A1_aligned": a1_aligned,
        "A2_display": a2_display,
        "estimated_for_skill": estimated_for_skill,
        "orientation": orientation,
        "loading_summary": loading_summary,
        "a2_orientation": int(a2_orientation),
        "alignment_weights": weights,
    }


def latent_by_survey(
    survey: np.ndarray,
    a2_display: np.ndarray,
    a1_aligned: np.ndarray,
) -> pd.DataFrame:
    valid = np.isfinite(survey)
    rows = []
    for cat in [1, 2, 3, 4]:
        idx = valid & (survey == cat)
        row = {
            "survey_response": int(cat),
            "n": int(idx.sum()),
            "A2_1": float(np.mean(a2_display[idx, 0])) if idx.any() else np.nan,
        }
        for skill_idx, skill_name in enumerate(SKILL_NAMES):
            row[f"A1_{skill_name}"] = (
                float(np.mean(a1_aligned[idx, skill_idx])) if idx.any() else np.nan
            )
        paper_values = PAPER_TABLE_S19[cat]
        row["paper_A2_1"] = paper_values[0]
        for offset, skill_name in enumerate(SKILL_NAMES[:4], start=1):
            row[f"paper_A1_{skill_name}"] = paper_values[offset]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wide", action="store_true", help="Use an expanded penalty/tau grid.")
    parser.add_argument("--max-outer-iter", type=int, default=10)
    parser.add_argument("--max-inner-iter", type=int, default=50)
    parser.add_argument("--mc-samples", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=5)
    parser.add_argument("--ebic-gamma", type=float, default=1.0)
    args = parser.parse_args()

    _ensure_dde_on_path()
    from dde import select_hyperparameters

    y, q_matrix, survey = load_timss_rt()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_candidates(y.shape[0], 2, wide=args.wide)

    config = {
        "data": "TIMSS response time columns 30:58 from TIMSS.csv",
        "Y_shape": list(y.shape),
        "survey_valid_n": int(np.isfinite(survey).sum()),
        "K_list": [7, 1],
        "D": 2,
        "criterion": "ebic",
        "ebic_gamma": args.ebic_gamma,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "max_outer_iter": args.max_outer_iter,
        "max_inner_iter": args.max_inner_iter,
        "mc_samples": args.mc_samples,
        "random_state": args.random_state,
    }
    (args.out_dir / "timss_semidde_rt_config.json").write_text(
        json.dumps(config, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"TIMSS SemiDDE RT EBIC run: Y={y.shape}, candidates={len(candidates)}")
    result = select_hyperparameters(
        y,
        D=2,
        model="semi",
        dimension_method="fixed",
        criterion="ebic",
        ebic_gamma=args.ebic_gamma,
        fixed_hp={"K_list": [7, 1]},
        candidates=candidates,
        base_fit_kwargs={
            "C": 1,
            "tol": 0.0,
            "max_outer_iter": args.max_outer_iter,
            "max_inner_iter": args.max_inner_iter,
            "max_enum_bits": 12,
            "mc_samples": args.mc_samples,
            "g_update_rho": 1.0,
            "verbose": False,
        },
        n_splits=2,
        random_state=args.random_state,
        refit=True,
    )

    cv_table = pd.DataFrame(result.diagnostics["cv_table"])
    cv_table.to_csv(args.out_dir / "timss_semidde_rt_cv_table.csv", index=False)
    timing_table = pd.DataFrame(result.diagnostics.get("timing_table", []))
    timing_table.to_csv(args.out_dir / "timss_semidde_rt_timing_table.csv", index=False)

    alignment = align_a1_to_q(result, q_matrix)
    summary = latent_by_survey(survey, alignment["A2_display"], alignment["A1_aligned"])
    summary.to_csv(args.out_dir / "timss_semidde_rt_table_s19_comparison.csv", index=False)

    pd.DataFrame(
        {
            "skill_index": np.arange(1, len(SKILL_NAMES) + 1),
            "skill_name": SKILL_NAMES,
            "estimated_A1_column": alignment["estimated_for_skill"] + 1,
            "orientation": alignment["orientation"],
            "mean_loading_on_q_support": alignment["loading_summary"],
        }
    ).to_csv(args.out_dir / "timss_semidde_rt_q_alignment.csv", index=False)
    pd.DataFrame(
        alignment["alignment_weights"],
        index=[f"est_A1_{i}" for i in range(1, 8)],
        columns=SKILL_NAMES,
    ).to_csv(args.out_dir / "timss_semidde_rt_alignment_weights.csv")

    with (args.out_dir / "timss_semidde_rt_result.pkl").open("wb") as f:
        pickle.dump(result, f)

    best_index = int(result.diagnostics["hyperparameter_selection"].best_index)
    summary_json = {
        "best_index": best_index,
        "best_row": cv_table.iloc[best_index].to_dict(),
        "iterations": result.diagnostics.get("iterations"),
        "prop": np.asarray(result.prop).tolist(),
        "A_shapes": [list(np.asarray(a).shape) for a in result.A_list],
        "B_shapes": [list(np.asarray(b).shape) for b in result.B_list],
        "a2_orientation": alignment["a2_orientation"],
    }
    (args.out_dir / "timss_semidde_rt_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print("Best EBIC row:")
    print(cv_table.iloc[[best_index]].to_string(index=False))
    print("Aligned Table S.19 comparison:")
    print(summary.to_string(index=False))
    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
