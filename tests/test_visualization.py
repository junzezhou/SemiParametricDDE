import numpy as np

from semidde.results import DDEResult
from semidde.visualization import visualization_report
from semidde.visualization.copula import make_copula_visualization_report
from semidde.visualization.semi import make_semi_visualization_report


def _result(model):
    return DDEResult(
        model=model,
        prop=np.array([0.55]),
        B_list=[
            np.array([[0.0, 0.7, 0.0], [0.1, 0.0, -0.6], [0.2, 0.4, 0.0]]),
            np.array([[0.0, 1.0], [0.1, -0.5]]),
        ],
        gamma=np.ones(3),
        A_list=[
            np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float),
            np.array([[1], [0], [1], [0]], dtype=float),
        ],
        diagnostics={
            "iterations": 2,
            "K_list": (2, 1),
            "cv_table": [{"candidate_index": 0, "mean_score": -1.0, "criterion": "smoke"}],
        },
    )


def test_copula_visualization_smoke(tmp_path):
    paths = make_copula_visualization_report(
        _result("copula"),
        output_dir=tmp_path,
        latent_frame={"A1_1": [1, 0, 1, 0], "A1_2": [0, 1, 1, 0], "label": ["a", "a", "b", "b"]},
        label_col="label",
    )
    assert "summary" in paths
    assert all((tmp_path / name).exists() for name in [p.split("/")[-1] for p in paths.values()])


def test_semi_visualization_smoke(tmp_path):
    paths = make_semi_visualization_report(
        _result("semi"),
        output_dir=tmp_path,
        latent_frame={"A1_1": [1, 0, 1, 0], "A1_2": [0, 1, 1, 0], "skill": ["n", "n", "g", "g"]},
        label_col="skill",
    )
    assert "summary" in paths
    assert all((tmp_path / name).exists() for name in [p.split("/")[-1] for p in paths.values()])


def test_visualization_report_unified_entrypoint(tmp_path):
    paths = visualization_report(
        _result("copula"),
        output_dir=tmp_path,
        latent_frame={"A1_1": [1, 0, 1, 0], "A1_2": [0, 1, 1, 0]},
    )
    assert "summary" in paths
    assert "latent_profiles" in paths
