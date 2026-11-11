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


def _csp_result(model="DDE"):
    """Two active columns of four in layer 1, one of two in layer 2."""
    result = _result(model)
    result.diagnostics.update(
        dimension_method="csp",
        original_K_list=(4, 2),
        active_columns_original_index={"layer_1": [1, 3], "layer_2": [2]},
        family="normal",
    )
    return result


def test_visualization_report_supports_every_model(tmp_path):
    from dde.visualization import visualization_report as report

    expected = {"B1_heatmap", "B2_heatmap", "item_cluster_heatmap", "latent_profiles",
                "latent_profiles_prevalence", "model_summary", "top_loadings"}
    paths = report(_result("DDE"), output_dir=tmp_path / "dde")
    assert set(paths) == expected
    assert all(name.startswith(str(tmp_path / "dde" / "dde_")) for name in paths.values())
    semi = report(_result("SemiDDE"), output_dir=tmp_path / "semi")
    assert set(semi) == (expected | {"posterior_quality_table"})
    copula = report(_result("CopDDE"), output_dir=tmp_path / "copula",
                    theme_table={"theme": ["a"], "count": [1]})
    assert set(copula) == (expected | {"theme_table"})
    with pytest.raises(ValueError, match="DDE"):
        report(_result("other"), output_dir=tmp_path)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_variable_names_rename_recorded_column_names(tmp_path):
    result = _result("DDE")
    result.diagnostics["variable_names"] = ["BIGFIVE_1", "BIGFIVE_2", "BIGFIVE_3"]
    figure = plot_weights(result, layer=1)
    assert [t.get_text() for t in figure.axes[0].get_yticklabels()] == ["BIGFIVE_1", "BIGFIVE_2", "BIGFIVE_3"]
    figure = plot_weights(result, layer=1, variable_names={"BIGFIVE_1": "Name"})
    assert [t.get_text() for t in figure.axes[0].get_yticklabels()] == ["Name", "BIGFIVE_2", "BIGFIVE_3"]
    anchors = get_anchors(result, n_anchors=1, variable_names={"BIGFIVE_1": "Name"})
    assert anchors.iloc[0]["variable"] == "Name"
    with pytest.raises(ValueError, match="BIGFIVE_9"):
        plot_weights(result, layer=1, variable_names={"BIGFIVE_9": "Name"})
    # without recorded names the defaults are y1, y2, ... and can be renamed too
    plain = plot_weights(_result("DDE"), layer=1, variable_names={"y2": "Name"})
    assert [t.get_text() for t in plain.axes[0].get_yticklabels()] == ["y1", "Name", "y3"]
    import matplotlib.pyplot as plt
    plt.close("all")


def test_csp_weights_end_in_a_column_counting_the_shrunk_columns(tmp_path):
    result = _csp_result()
    figures = plot_weights(result)
    labels = [t.get_text() for t in figures[1].axes[0].get_xticklabels()]
    assert labels == ["A1_1", "A1_3", "+2 shrunk"]
    assert "4 before CSP shrinkage" in figures[1].axes[0].get_title()
    assert [t.get_text() for t in figures[2].axes[0].get_xticklabels()] == ["A2_2", "+1 shrunk"]
    plain = plot_weights(result, layer=1, show_shrunk=False)
    assert [t.get_text() for t in plain.axes[0].get_xticklabels()] == ["A1_1", "A1_3"]
    fixed = plot_weights(_result("DDE"), layer=1)  # no original widths: nothing added
    assert [t.get_text() for t in fixed.axes[0].get_xticklabels()] == ["A1_1", "A1_2"]
    paths = visualization_report(result, output_dir=tmp_path, feature_names={"y1": "Name"})
    assert "B1_heatmap" in paths
    import matplotlib.pyplot as plt
    plt.close("all")


def test_report_top_loadings_include_the_first_latent_column(monkeypatch, tmp_path):
    from dde.visualization import report as report_module

    frames = []
    original = report_module._plot_bar_frame

    def capture(df, *args, **kwargs):
        frames.append(df.copy())
        return original(df, *args, **kwargs)

    monkeypatch.setattr(report_module, "_plot_bar_frame", capture)
    report_module.plot_model_summary(_csp_result(), output_dir=tmp_path, feature_names=["a", "b", "c"])
    top = frames[0]
    first = top[top["latent"] == "A1_1"]
    # column A1_1 loads 0.7 on "a" and 0.4 on "c" (B1 above)
    assert list(first.sort_values("rank")["row"])[:2] == ["a", "c"]
    assert set(top["latent"]) == {"A1_1", "A1_3", "A2_2"}
    import matplotlib.pyplot as plt
    plt.close("all")


def test_fit_records_data_frame_column_names():
    pd = pytest.importorskip("pandas")
    from dde import fit

    rng = np.random.default_rng(0)
    Y = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"BIGFIVE_{j}" for j in range(1, 7)])
    np.random.seed(0)
    result = fit(Y, model="DDE", family="Normal", D=1, dimension_method="fixed", K_list=[2],
                 pen_list=[1.0], tau=0.3, max_it=3, verbose=False)
    assert result.diagnostics["variable_names"] == list(Y.columns)
    anchors = get_anchors(result, n_anchors=1, variable_names={"BIGFIVE_1": "Name"})
    assert set(anchors["variable"]) <= set(["Name"] + list(Y.columns[1:]))


def test_model_summary_reports_the_selection_criterion_on_its_own_scale():
    from dde.visualization.report import _selection_score_row

    label, value = _selection_score_row({"criterion": "ebic", "ebic": 81.2, "mean_score": -0.0812})
    assert label.startswith("best EBIC") and value == "81.2"
    label, value = _selection_score_row({"criterion": "pmse", "mean_pmse": 0.0134, "mean_score": -0.0134})
    assert label.startswith("best pMSE") and value == "0.0134"
    assert _selection_score_row({"criterion": "bic", "mean_score": -3.5})[0] == "best score"
