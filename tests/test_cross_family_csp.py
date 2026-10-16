"""Cross-family conformance tests for the shared CSP control path."""

from importlib import import_module

import numpy as np

from dde.algorithms import normal_csp
from dde.algorithms.csp import csp_active_mask, update_csp_layer


copula_csp = import_module("dde.algorithms.get_SAEM_RL_CSP_D")


def _initial_shared_state():
    """Return one deliberately mixed-strength three-layer CSP state."""
    rng = np.random.default_rng(20260830)
    J = 30
    widths = (8, 4, 2)
    row_widths = (J, *widths[:-1])
    column_scales = (
        np.array([1.00, 0.75, 0.40, 0.15, 0.05, 0.01, 0.00, 0.00]),
        np.array([0.80, 0.20, 0.02, 0.00]),
        np.array([0.50, 0.00]),
    )

    B_list = []
    for rows, width, scales in zip(row_widths, widths, column_scales):
        B = rng.normal(scale=0.25, size=(rows, width + 1))
        B[:, 1:] *= scales[None, :]
        B_list.append(B)

    omega = [
        np.linspace(1.0, width + 1.0, width + 1, dtype=float)
        for width in widths
    ]
    omega = [value / value.sum() for value in omega]
    slab_scale = [
        np.linspace(0.6, 1.4, width, dtype=float) for width in widths
    ]
    return J, widths, B_list, omega, slab_scale


def _run_shared_csp_evolution(layer_update):
    """Run the family adapter's CSP update through identical gated states."""
    J, widths, B_list, omega, slab_scale = _initial_shared_state()
    spike_scale = np.array([0.02, 0.03, 0.01])
    random = np.random.default_rng(91427)
    num_active = np.asarray(widths, dtype=int)
    history = []

    for _ in range(3):
        allocation = [None] * len(widths)
        penalty_scale = [None] * len(widths)

        # Both family kernels update from the deepest layer toward the data.
        for d in range(len(widths) - 1, -1, -1):
            num_rows = J if d == 0 else int(num_active[d - 1])
            (
                allocation[d],
                penalty_scale[d],
                slab_scale[d],
                omega[d],
            ) = layer_update(
                B_list[d],
                omega[d],
                spike_scale[d],
                slab_scale[d],
                num_rows,
                rng=random,
            )

        active_masks = [csp_active_mask(value) for value in allocation]
        num_active = np.asarray(
            [int(np.sum(mask)) for mask in active_masks], dtype=int
        )

        # Mirror the shared layer-gating rule before the following CSP sweep.
        for d in range(len(widths) - 1):
            inactive = np.flatnonzero(~active_masks[d])
            if inactive.size:
                B_list[d + 1][inactive, :] = 0.0

        history.append(
            {
                "allocation": [value.copy() for value in allocation],
                "active_masks": [value.copy() for value in active_masks],
                "penalty_scale": [value.copy() for value in penalty_scale],
                "slab_scale": [value.copy() for value in slab_scale],
                "omega": [value.copy() for value in omega],
                "num_active": num_active.copy(),
                "B_list": [value.copy() for value in B_list],
            }
        )

    return history


def _assert_nested_arrays_equal(normal_history, copula_history):
    assert len(normal_history) == len(copula_history)
    for normal_step, copula_step in zip(normal_history, copula_history):
        assert normal_step.keys() == copula_step.keys()
        for key in normal_step:
            normal_value = normal_step[key]
            copula_value = copula_step[key]
            if isinstance(normal_value, list):
                assert len(normal_value) == len(copula_value)
                for normal_array, copula_array in zip(normal_value, copula_value):
                    np.testing.assert_array_equal(normal_array, copula_array)
            else:
                np.testing.assert_array_equal(normal_value, copula_value)


def _recording_layer_update(calls):
    """Return a deterministic CSP update used to inspect family call sites."""

    def update(B_d, omega, spike_scale, slab_scale, num_rows, *, rng=None, **kwargs):
        B_d = np.asarray(B_d, dtype=float)
        omega = np.asarray(omega, dtype=float)
        slab_scale = np.asarray(slab_scale, dtype=float)
        calls.append(
            {
                "B_d": B_d.copy(),
                "omega": omega.copy(),
                "spike_scale": float(spike_scale),
                "slab_scale": slab_scale.copy(),
                "num_rows": int(num_rows),
            }
        )

        width = B_d.shape[1] - 1
        allocation = np.arange(1, width + 1, dtype=int)
        allocation[::2] += 1
        active = csp_active_mask(allocation)
        slab_update = slab_scale + np.where(active, 0.25, 0.0)
        column_scale = np.where(active, slab_update, float(spike_scale))
        penalty_scale = np.broadcast_to(
            column_scale[None, :], (B_d.shape[0], width)
        ).copy()
        omega_update = np.arange(1, width + 2, dtype=float)
        omega_update /= omega_update.sum()
        return allocation, penalty_scale, slab_update, omega_update

    return update


def _one_iteration_inputs():
    rng = np.random.default_rng(3319)
    N, J = 10, 18
    widths = (6, 2)
    return {
        "X": rng.normal(size=(N, J)),
        "prop": np.full(widths[-1], 0.5),
        "gamma": np.ones(J),
        "B_list": [
            rng.normal(scale=0.15, size=(J, widths[0] + 1)),
            rng.normal(scale=0.15, size=(widths[0], widths[1] + 1)),
        ],
        "A_list": [
            rng.integers(0, 2, size=(N, width), dtype=np.uint8)
            for width in widths
        ],
    }


def _copy_arrays(values):
    return [np.asarray(value).copy() for value in values]


def test_normal_and_copula_csp_paths_are_deterministically_identical():
    """CSP controls must evolve identically outside family likelihood code."""
    assert normal_csp.update_csp_layer is update_csp_layer
    assert copula_csp._update_csp_layer is update_csp_layer

    normal_history = _run_shared_csp_evolution(normal_csp.update_csp_layer)
    copula_history = _run_shared_csp_evolution(copula_csp._update_csp_layer)

    _assert_nested_arrays_equal(normal_history, copula_history)


def test_family_kernels_initialize_and_call_csp_controls_identically(monkeypatch):
    """The real family kernels must present the same first-sweep CSP state."""
    inputs = _one_iteration_inputs()
    normal_calls = []
    copula_calls = []
    monkeypatch.setattr(
        normal_csp, "update_csp_layer", _recording_layer_update(normal_calls)
    )
    monkeypatch.setattr(
        copula_csp, "_update_csp_layer", _recording_layer_update(copula_calls)
    )

    normal_raw = normal_csp.fit_normal_csp(
        inputs["X"],
        inputs["prop"],
        inputs["gamma"],
        _copy_arrays(inputs["B_list"]),
        _copy_arrays(inputs["A_list"]),
        C=1,
        it=1,
        t_spike=[0.02, 0.03],
        temp=0.9,
        tau=0.3,
        optimizer_maxiter=1,
        mstep_solver="lbfgsb",
        penalty_scaling_power=0.0,
        random_state=1817,
        verbose=False,
    )

    np.random.seed(1817)
    copula_raw = copula_csp.get_SAEM_RL_CSP_D(
        X=inputs["X"],
        Z=inputs["X"].copy(),
        R=np.zeros_like(inputs["X"]),
        Rlevels=np.zeros(inputs["X"].shape[1], dtype=int),
        prop_in=inputs["prop"],
        gamma_in=inputs["gamma"],
        B_list_in=_copy_arrays(inputs["B_list"]),
        A_list_in=_copy_arrays(inputs["A_list"]),
        C=1,
        it=1,
        t_spike=[0.02, 0.03],
        temp=0.9,
        tau=0.3,
        optimizer_maxiter=1,
        verbose=False,
    )

    _assert_nested_arrays_equal(normal_calls, copula_calls)
    for normal_key, copula_key in (
        ("allocation", "z_cusp"),
        ("penalty_scale", "theta"),
        ("slab_scale", "theta_slab"),
        ("omega", "omega"),
    ):
        for normal_value, copula_value in zip(
            normal_raw[normal_key], copula_raw[copula_key]
        ):
            np.testing.assert_array_equal(normal_value, copula_value)
    np.testing.assert_array_equal(normal_raw["num_active"], copula_raw["num_act"])
