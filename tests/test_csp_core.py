import numpy as np
from scipy.special import gammaln

from dde.algorithms.csp import (
    csp_active_mask,
    maximal_identifiable_widths,
    update_csp_layer,
)


def _legacy_update_csp_layer(B_d, omega, theta_inf, theta_slab, num_rows):
    """Frozen copy of the pre-extraction CopDDE update for equivalence QA."""
    Kd = B_d.shape[1] - 1
    L = Kd + 1
    abs_sum = np.sum(np.abs(B_d[:, 1:]), axis=0)
    log_dens_spike = -num_rows * np.log(2.0 * theta_inf) - abs_sum / theta_inf
    log_dens_slab = (
        np.log(5.0)
        - num_rows * np.log(2.0)
        + gammaln(1.0 + num_rows)
        - (1.0 + num_rows) * np.log(5.0 + abs_sum)
    )
    log_omega = np.log(np.maximum(omega, np.finfo(float).tiny))

    prob_mat = np.empty((Kd, L), dtype=float)
    for h in range(Kd):
        log_probs = log_omega.copy()
        log_probs[: h + 1] += log_dens_spike[h]
        log_probs[h + 1 :] += log_dens_slab[h]
        probs = np.exp(log_probs - np.max(log_probs))
        prob_mat[h, :] = probs / np.sum(probs)

    z = np.argmax(prob_mat, axis=1) + 1
    alpha = Kd
    v = np.zeros(L, dtype=float)
    for ell in range(1, Kd + 1):
        n_l = np.sum(z == ell)
        n_gtl = np.sum(z > ell)
        v[ell - 1] = np.random.beta(1.0 + n_l, alpha + n_gtl)
    v[L - 1] = 1.0
    omega = v * np.concatenate(([1.0], np.cumprod(1.0 - v[:-1])))

    theta_slab = theta_slab.copy()
    for h in range(Kd):
        if z[h] > (h + 1):
            shape_post = 1.0 + num_rows
            rate_post = 5.0 + np.sum(np.abs(B_d[:, h + 1]))
            theta_slab[h] = 1.0 / np.random.gamma(
                shape_post, 1.0 / rate_post
            )

    theta_col = np.where(z <= np.arange(1, Kd + 1), theta_inf, theta_slab)
    theta = np.broadcast_to(theta_col[None, :], (B_d.shape[0], Kd)).copy()
    return z, theta, theta_slab, omega


def test_extracted_csp_update_is_fixed_seed_equivalent_to_legacy_code():
    rng = np.random.default_rng(20260828)
    B = rng.normal(scale=0.15, size=(12, 7))
    omega = np.linspace(1.0, 7.0, 7)
    omega /= omega.sum()
    slab = rng.uniform(0.5, 1.5, size=6)

    np.random.seed(9917)
    expected = _legacy_update_csp_layer(B, omega, 0.03, slab, 12)
    np.random.seed(9917)
    actual = update_csp_layer(B, omega, 0.03, slab, 12)

    for actual_item, expected_item in zip(actual, expected):
        assert np.array_equal(actual_item, expected_item)


def test_csp_active_mask_uses_allocation_greater_than_column_index():
    allocation = np.array([2, 1, 4, 4, 7])
    assert np.array_equal(
        csp_active_mask(allocation),
        np.array([True, False, True, False, True]),
    )


def test_csp_allocation_separates_strong_columns_from_zero_columns():
    B = np.zeros((30, 5), dtype=float)
    B[:3, 1] = 1.0
    B[:3, 2] = 1.0
    np.random.seed(12)
    allocation, _, _, _ = update_csp_layer(
        B,
        np.ones(5) / 5.0,
        0.02,
        np.ones(4),
        30,
    )
    assert np.array_equal(
        csp_active_mask(allocation),
        np.array([True, True, False, False]),
    )


def test_maximal_identifiable_widths_uses_recursive_floor():
    assert maximal_identifiable_widths(486, 5) == (162, 54, 18, 6, 2)
    assert maximal_identifiable_widths(50, 2) == (16, 5)
