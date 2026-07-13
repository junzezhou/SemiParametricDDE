"""Build a compact comparison against CopDDE reported simulation targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "Results" / "copdde_D_parameter_tests"
SWEEPS = RESULTS / "sweeps"


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def d2_j100_comparison():
    ours = _load_csv(RESULTS / "copdde_D_summary_paper_paper_j.csv")
    if ours.empty:
        return pd.DataFrame()
    ours = ours.loc[ours["label"].eq("J100_paper")].copy()
    extra = _load_csv(SWEEPS / "copdde_d2_sweep_summary_d2_J100_N16000_r5_combined.csv")
    if not extra.empty:
        extra = extra.rename(
            columns={
                "ACC_G1_full_mean_mean": "ACC_G1_full_mean",
                "ACC_G2_full_mean_mean": "ACC_G2_full_mean",
                "num_active_1_mean_mean": "num_active_1_mean",
                "num_active_2_mean_mean": "num_active_2_mean",
            }
        )
        common_cols = [c for c in ours.columns if c in extra.columns]
        ours = pd.concat([ours, extra[common_cols]], ignore_index=True)
    paper = pd.DataFrame(
        {
            "N": [500, 1000, 2000, 4000, 8000, 16000],
            "paper_ACC_G1": [0.977, 0.994, 0.997, 1.000, 1.000, 1.000],
            "paper_ACC_G2": [0.974, 0.985, 0.990, 1.000, 1.000, 1.000],
            "paper_K1": [8.140, 9.020, 9.150, 9.980, 10.000, 10.000],
            "paper_K2": [3.300, 2.950, 2.990, 3.000, 3.020, 3.010],
        }
    )
    merged = paper.merge(ours, on="N", how="left")
    out = pd.DataFrame(
        {
            "experiment": "D2_J100_Table1",
            "N": merged["N"],
            "metric": "ACC_G1",
            "paper_value": merged["paper_ACC_G1"],
            "semidde_value": merged["ACC_G1_full_mean"],
            "direction": "higher_or_equal",
        }
    )
    rows = [out]
    rows.append(
        pd.DataFrame(
            {
                "experiment": "D2_J100_Table1",
                "N": merged["N"],
                "metric": "ACC_G2",
                "paper_value": merged["paper_ACC_G2"],
                "semidde_value": merged["ACC_G2_full_mean"],
                "direction": "higher_or_equal",
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "experiment": "D2_J100_Table1",
                "N": merged["N"],
                "metric": "K1_active_abs_error",
                "paper_value": np.abs(merged["paper_K1"] - 10.0),
                "semidde_value": np.abs(merged["num_active_1_mean"] - 10.0),
                "direction": "lower_or_equal",
            }
        )
    )
    rows.append(
        pd.DataFrame(
            {
                "experiment": "D2_J100_Table1",
                "N": merged["N"],
                "metric": "K2_active_abs_error",
                "paper_value": np.abs(merged["paper_K2"] - 3.0),
                "semidde_value": np.abs(merged["num_active_2_mean"] - 3.0),
                "direction": "lower_or_equal",
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


def d3_figure_c10_comparison():
    # Values are read approximately from Figure C.10 in
    # Papers/References/CopDDE.pdf. The plotted figure is the only reported
    # source for these D=3 MSE curves, so these targets should be treated as
    # visual reference values rather than exact table entries.
    paper = pd.DataFrame(
        {
            "N": [500, 1000, 2000, 4000, 8000, 16000],
            "paper_MSE_B1": [0.031, 0.023, 0.023, 0.014, 0.007, 0.004],
            "paper_MSE_B2": [0.36, 0.29, 0.235, 0.213, 0.198, 0.207],
            "paper_MSE_B3": [1.66, 1.72, 1.79, 1.767, 1.774, 1.782],
        }
    )
    frames = []
    for tag in [
        "matlab_blocks_t1_002_t3_001_N500_r5",
        "matlab_blocks_t1_002_t3_001_N1000_r3",
        "matlab_blocks_t1_002_t3_001_N2000_r3",
        "matlab_blocks_t1_002_t3_001_N4000_r3",
        "matlab_blocks_t1_002_t3_001_N8000_r3",
        "matlab_blocks_t1_002_t3_001_N16000_r2",
    ]:
        df = _load_csv(SWEEPS / f"copdde_d3_sweep_summary_{tag}.csv")
        if df.empty:
            continue
        df = df.loc[
            df["variant"].eq("mathematical__tspike_0.02_0.03_0.001")
            & df["backend"].eq("copdde_D_csp")
        ].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    ours = pd.concat(frames, ignore_index=True)
    merged = paper.merge(ours, on="N", how="left")
    rows = []
    for metric in ["MSE_B1", "MSE_B2", "MSE_B3"]:
        rows.append(
            pd.DataFrame(
                {
                    "experiment": "D3_FigureC10",
                    "N": merged["N"],
                    "metric": metric,
                    "paper_value": merged[f"paper_{metric}"],
                    "semidde_value": merged[f"{metric}_mean"],
                    "direction": "lower_or_equal",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def mark_status(df):
    df = df.copy()
    strict_high = df["direction"].eq("higher_or_equal") & (df["semidde_value"] >= df["paper_value"])
    strict_low = df["direction"].eq("lower_or_equal") & (df["semidde_value"] <= df["paper_value"])
    df["strictly_meets_or_beats"] = strict_high | strict_low

    same_high = df["direction"].eq("higher_or_equal") & (df["semidde_value"] >= df["paper_value"] - 0.012)
    same_low = df["direction"].eq("lower_or_equal") & (df["semidde_value"] <= 1.10 * df["paper_value"])
    k_metric = df["metric"].str.endswith("_active_abs_error")
    same_low = same_low | (
        df["direction"].eq("lower_or_equal")
        & k_metric
        & (df["semidde_value"] <= df["paper_value"] + 0.20)
    )
    df["same_level_or_better"] = same_high | same_low
    df.loc[df["semidde_value"].isna(), ["strictly_meets_or_beats", "same_level_or_better"]] = False
    df["ratio_semidde_to_paper"] = df["semidde_value"] / df["paper_value"]
    return df


def main():
    out_dir = RESULTS / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat([d2_j100_comparison(), d3_figure_c10_comparison()], ignore_index=True)
    comparison = mark_status(comparison)
    path = out_dir / "copdde_report_comparison.csv"
    comparison.to_csv(path, index=False)
    print(f"Comparison: {path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
