import numpy as np

from dde.results import DDEResult
import pytest

from dde.visualization import anchor_scores, get_anchors, plot_weights
from dde.visualization.legacy import (
    make_copula_visualization_report,
    make_semi_visualization_report,
    plot_model_summary,
    visualization_report,
)


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
    assert "model_summary" in paths
    assert all((tmp_path / name).exists() for name in [p.split("/")[-1] for p in paths.values()])


def test_semi_visualization_smoke(tmp_path):
    paths = make_semi_visualization_report(
        _result("semi"),
        output_dir=tmp_path,
        latent_frame={"A1_1": [1, 0, 1, 0], "A1_2": [0, 1, 1, 0], "skill": ["n", "n", "g", "g"]},
        label_col="skill",
    )
    assert "model_summary" in paths
    assert all((tmp_path / name).exists() for name in [p.split("/")[-1] for p in paths.values()])


def test_visualization_report_unified_entrypoint(tmp_path):
    paths = visualization_report(
        _result("copula"),
        output_dir=tmp_path,
        latent_frame={"A1_1": [1, 0, 1, 0], "A1_2": [0, 1, 1, 0]},
    )
    assert "model_summary" in paths
    assert "latent_profiles" in paths


def test_anchor_scores_follow_the_relative_loading_definition():
    B = np.array([[0.0, 0.7, 0.0], [0.1, 0.0, -0.6], [0.2, 0.4, 0.1], [0.0, 0.9, 0.2]])
    scores = anchor_scores(B)
    # r_jk = max(min_{l != k}(b_jk - b_jl), 0)
    np.testing.assert_allclose(scores[:, 0], [0.7, 0.6, 0.3, 0.7])
    np.testing.assert_allclose(scores[:, 1], [0.0, 0.0, 0.0, 0.0])
    single = anchor_scores(np.array([[0.0, 0.5], [0.0, -0.2]]))
    np.testing.assert_allclose(single[:, 0], [0.5, 0.0])
    with pytest.raises(ValueError):
        anchor_scores(np.array([[1.0], [2.0]]))


def test_get_anchors_ranks_variables_and_uses_names_and_csp_indices():
    result = _result("semi")
    anchors = get_anchors(result, n_anchors=2, variable_names=["a", "b", "c"])
    assert list(anchors.columns) == ["latent_feature", "variable", "variable_index", "rank", "score", "loading"]
    first = anchors[anchors["latent_feature"] == 1]
    assert list(first["variable"]) == ["a", "b"]
    assert list(first["rank"]) == [1, 2]
    second = anchors[anchors["latent_feature"] == 2]
    assert list(second["variable"]) == []  # feature 2 has no positive relative loading
    csp = _result("semi")
    csp.diagnostics["active_columns_original_index"] = {"layer_1": [2, 5], "layer_2": [3]}
    labelled = get_anchors(csp, n_anchors=1)
    assert sorted(labelled["latent_feature"].unique()) == [2]
    assert labelled.iloc[0]["variable"] == "y1"
    with pytest.raises(ValueError):
        get_anchors(result, n_anchors=0)
    with pytest.raises(ValueError):
        get_anchors(result, variable_names=["a", "b"])


def test_plot_weights_returns_figures_and_saves_files(tmp_path):
    result = _result("copula")
    result.diagnostics["active_columns_original_index"] = {"layer_1": [1, 4], "layer_2": [2]}
    figures = plot_weights(result, output_dir=tmp_path, variable_names=["x", "y", "z"])
    assert sorted(figures) == [1, 2]
    assert (tmp_path / "weights_layer1.png").exists() and (tmp_path / "weights_layer2.png").exists()
    axis = figures[1].axes[0]
    assert [t.get_text() for t in axis.get_xticklabels()] == ["A1_1", "A1_4"]
    assert [t.get_text() for t in axis.get_yticklabels()] == ["x", "y", "z"]
    single = plot_weights(result, layer=2, include_intercept=True)
    assert [t.get_text() for t in single.axes[0].get_xticklabels()] == ["intercept", "A2_2"]
    assert [t.get_text() for t in single.axes[0].get_yticklabels()] == ["A1_1", "A1_4"]
    with pytest.raises(ValueError):
        plot_weights(result, layer=3)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_legacy_names_are_reachable_with_a_deprecation_warning():
    import dde.visualization as viz
    with pytest.warns(DeprecationWarning):
        assert viz.plot_model_summary is plot_model_summary
    # visualization_report is a primary interface (manuscript Appendix C)
    assert "visualization_report" in viz.__all__
