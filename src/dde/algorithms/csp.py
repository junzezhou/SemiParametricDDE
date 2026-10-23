"""Shared cumulative-shrinkage-process primitives for DDE estimators.

The CopDDE and parametric-DDE CSP paths use the scale parameterization

    beta | scale ~ Laplace(0, scale),

so the adaptive lasso weight is ``1 / scale``.  This module deliberately keeps
the update order and random-number calls of the original CopDDE implementation;
the extraction is therefore numerically reproducible under a fixed NumPy seed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.special import gammaln


_FLOAT_TINY = np.finfo(float).tiny


def maximal_identifiable_widths(J: int, D: int) -> tuple[int, ...]:
    """Return the maximal width sequence satisfying the DDE 3-to-1 rule.

    The observed dimension is not included in the returned tuple.  Thus,
    ``maximal_identifiable_widths(486, 5)`` is ``(162, 54, 18, 6, 2)``.
    """
    J = int(J)
    D = int(D)
    if J < 1:
        raise ValueError("J must be a positive integer.")
    if D < 1:
        raise ValueError("D must be a positive integer.")

    widths = []
    lower_width = J
    for d in range(D):
        width = lower_width // 3
        if width < 1:
            raise ValueError(
                f"D={D} is too deep for J={J} under the DDE identifiability "
                f"rule; layer {d + 1} would have maximal width zero."
            )
        widths.append(width)
        lower_width = width
    return tuple(widths)


def csp_active_mask(allocation: Sequence[int]) -> np.ndarray:
    """Return the CSP slab indicator ``allocation[k] > k + 1``."""
    z = np.asarray(allocation, dtype=int).reshape(-1)
    return z > np.arange(1, z.size + 1)


def update_csp_layer(
    B_d,
    omega,
    spike_scale: float,
    slab_scale,
    num_rows: int,
    *,
    rng: Optional[object] = None,
    slab_shape: float = 1.0,
    slab_rate: float = 5.0,
):
    """Perform the CopDDE CSP allocation, stick, and slab-scale update.

    Parameters follow the original package implementation.  ``spike_scale``
    and ``slab_scale`` are Laplace scales, not rates.  When ``rng`` is omitted,
    NumPy's legacy global RNG is used so the extracted implementation remains
    fixed-seed equivalent to the original CopDDE function.

    The slab places a Gamma(``slab_shape``, ``slab_rate``) hyperprior on the
    Laplace rate ``1 / slab_scale`` of each slab column; the CopDDE values are
    shape 1 and rate 5.  The slab marginal density of a column with L1 norm
    ``S`` over ``num_rows`` rows is then
    ``rate^shape / Gamma(shape) * Gamma(shape + num_rows) / (2^num_rows
    (rate + S)^(shape + num_rows))`` and the slab rate posterior is
    Gamma(``shape + num_rows``, ``rate + S``), so larger ``shape`` or smaller
    ``rate`` make the slab penalty stronger and more adaptive to weak columns.
    """
    B_d = np.asarray(B_d, dtype=float)
    omega = np.asarray(omega, dtype=float).reshape(-1)
    slab_scale = np.asarray(slab_scale, dtype=float).reshape(-1).copy()
    spike_scale = float(spike_scale)
    num_rows = int(num_rows)

    if B_d.ndim != 2 or B_d.shape[1] < 2:
        raise ValueError("B_d must be a 2D coefficient matrix with an intercept column.")
    Kd = B_d.shape[1] - 1
    L = Kd + 1
    if omega.shape != (L,):
        raise ValueError(f"omega shape {omega.shape} must be {(L,)}.")
    if slab_scale.shape != (Kd,):
        raise ValueError(f"slab_scale shape {slab_scale.shape} must be {(Kd,)}.")
    if not np.isfinite(spike_scale) or spike_scale <= 0:
        raise ValueError("spike_scale must be finite and positive.")
    if num_rows < 0:
        raise ValueError("num_rows must be a nonnegative integer.")
    slab_shape = float(slab_shape)
    slab_rate = float(slab_rate)
    if not np.isfinite(slab_shape) or slab_shape <= 0:
        raise ValueError("slab_shape must be finite and positive.")
    if not np.isfinite(slab_rate) or slab_rate <= 0:
        raise ValueError("slab_rate must be finite and positive.")

    random = np.random if rng is None else rng
    abs_sum = np.sum(np.abs(B_d[:, 1:]), axis=0)
    log_dens_spike = (
        -num_rows * np.log(2.0 * spike_scale) - abs_sum / spike_scale
    )
    log_dens_slab = (
        slab_shape * np.log(slab_rate)
        - gammaln(slab_shape)
        - num_rows * np.log(2.0)
        + gammaln(slab_shape + num_rows)
        - (slab_shape + num_rows) * np.log(slab_rate + abs_sum)
    )
    log_omega = np.log(np.maximum(omega, _FLOAT_TINY))

    allocation_prob = np.empty((Kd, L), dtype=float)
    for h in range(Kd):
        log_probs = log_omega.copy()
        log_probs[: h + 1] += log_dens_spike[h]
        log_probs[h + 1 :] += log_dens_slab[h]
        probs = np.exp(log_probs - np.max(log_probs))
        allocation_prob[h, :] = probs / np.sum(probs)

    allocation = np.argmax(allocation_prob, axis=1) + 1
    alpha = Kd
    v = np.zeros(L, dtype=float)
    for ell in range(1, Kd + 1):
        n_l = np.sum(allocation == ell)
        n_gtl = np.sum(allocation > ell)
        v[ell - 1] = random.beta(1.0 + n_l, alpha + n_gtl)
    v[L - 1] = 1.0
    omega_update = v * np.concatenate(([1.0], np.cumprod(1.0 - v[:-1])))

    for h in range(Kd):
        if allocation[h] > (h + 1):
            shape_post = slab_shape + num_rows
            rate_post = slab_rate + np.sum(np.abs(B_d[:, h + 1]))
            slab_scale[h] = 1.0 / random.gamma(
                shape_post, 1.0 / rate_post
            )

    column_scale = np.where(
        csp_active_mask(allocation), slab_scale, spike_scale
    )
    penalty_scale = np.broadcast_to(
        column_scale[None, :], (B_d.shape[0], Kd)
    ).copy()
    return allocation, penalty_scale, slab_scale, omega_update


def _canonicalize_csp_layer_order(prop, B_list, A_list):
    """Order latent coordinates by loading strength without changing the model.

    The cumulative-shrinkage prior is ordered, whereas spectral/varimax
    initialization is permutation invariant.  A simultaneous permutation of
    a layer's incoming columns, persistent states, and outgoing rows preserves
    every model probability while placing the strongest coordinates first.
    """
    prop = np.asarray(prop, dtype=float).copy()
    B_list = [np.asarray(B, dtype=float).copy() for B in B_list]
    A_list = [np.asarray(A).copy() for A in A_list]
    permutations = []
    D = len(B_list)
    for d in range(D):
        strength = np.sum(np.abs(B_list[d][:, 1:]), axis=0)
        permutation = np.argsort(-strength, kind="stable")
        permutations.append(permutation)
        B_list[d][:, 1:] = B_list[d][:, permutation + 1]
        A_list[d] = A_list[d][:, permutation]
        if d + 1 < D:
            B_list[d + 1] = B_list[d + 1][permutation, :]
        else:
            prop = prop[permutation]
    return prop, B_list, A_list, permutations


__all__ = [
    "csp_active_mask",
    "maximal_identifiable_widths",
    "update_csp_layer",
]
