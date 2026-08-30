"""Validation helpers for list-based DDE parameter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DDEListShape:
    """Shape metadata inferred from a list-based DDE parameter set."""

    D: int
    N: int
    J: int
    K_list: tuple[int, ...]


def validate_K_list_identifiability(
    K_list: Sequence[int],
    *,
    D: Optional[int] = None,
    J: Optional[int] = None,
) -> tuple[int, ...]:
    """Validate the DDE layer-width identifiability ordering."""
    K_tuple = tuple(int(k) for k in K_list)
    if D is not None and len(K_tuple) != int(D):
        raise ValueError(f"K_list length {len(K_tuple)} must match D={int(D)}.")
    if len(K_tuple) < 1:
        raise ValueError("K_list must contain at least one layer width.")
    if any(k < 1 for k in K_tuple):
        raise ValueError("K_list entries must be positive integers.")
    if J is not None and int(J) < 3 * K_tuple[0]:
        raise ValueError(
            "Observed dimension J must satisfy J >= 3 * K_list[0] "
            f"under the DDE identifiability rule; got J={int(J)} and K_list={K_tuple}."
        )
    for d in range(len(K_tuple) - 1):
        if K_tuple[d] < 3 * K_tuple[d + 1]:
            raise ValueError(
                "K_list must satisfy K_list[d] >= 3 * K_list[d + 1] "
                f"for every adjacent layer; got {K_tuple}."
            )
    return K_tuple


def _is_sequence_like(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _as_2d_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {arr.shape}.")
    return arr


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")


def _check_binary(arr: np.ndarray, name: str, tol: float) -> None:
    if not np.all((np.abs(arr) <= tol) | (np.abs(arr - 1.0) <= tol)):
        raise ValueError(f"{name} must contain binary 0/1 values.")


def validate_dde_lists(
    observed,
    B_list,
    *,
    A_list=None,
    prop=None,
    gamma=None,
    K_list: Optional[Sequence[int]] = None,
    observed_name: str = "observed",
    require_binary_A: bool = False,
    require_finite: bool = True,
    probability_tol: float = 0.0,
    binary_tol: float = 0.0,
) -> DDEListShape:
    """Validate the shared list-based DDE layer convention.

    The convention is ordered from the shallowest latent layer to the deepest:

    ``B_list[0]`` has shape ``(J, K_1 + 1)`` and maps ``A_1`` to the observed
    Gaussian/latent-Gaussian layer.  For ``d >= 1``, ``B_list[d]`` has shape
    ``(K_d, K_{d+1} + 1)`` and maps parent layer ``A_{d+1}`` to child layer
    ``A_d``.  ``prop`` is the independent Bernoulli vector for ``A_D``.

    Parameters
    ----------
    observed:
        Observed or latent-Gaussian matrix with shape ``(N, J)``.
    B_list:
        Sequence of coefficient matrices following the convention above.
    A_list:
        Optional sequence of latent layer matrices, with ``A_list[d]`` shaped
        ``(N, K_{d+1})`` in 1-based notation.
    prop:
        Optional deepest-layer Bernoulli probability vector, shape ``(K_D,)``.
    gamma:
        Optional observed-layer variance vector, shape ``(J,)``.
    K_list:
        Optional explicit latent widths.  When supplied, it must match the
        widths implied by ``B_list``.
    require_binary_A:
        If True, require every entry in ``A_list`` to be 0 or 1.
    require_finite:
        If True, require supplied numeric arrays to contain finite values.

    Returns
    -------
    DDEListShape
        Inferred ``D``, ``N``, ``J``, and ``K_list`` metadata.
    """
    X = _as_2d_array(observed, observed_name)
    N, J = X.shape
    if require_finite:
        _check_finite(X, observed_name)

    if not _is_sequence_like(B_list):
        raise ValueError("B_list must be a non-empty sequence of 2D arrays.")
    B_arrays = [_as_2d_array(B, f"B_list[{d}]") for d, B in enumerate(B_list)]
    D = len(B_arrays)
    if D < 1:
        raise ValueError("B_list must contain at least one latent layer.")
    if require_finite:
        for d, B in enumerate(B_arrays):
            _check_finite(B, f"B_list[{d}]")

    inferred_K = [int(B_arrays[0].shape[1] - 1)]
    if B_arrays[0].shape[0] != J:
        raise ValueError(
            f"B_list[0] shape {B_arrays[0].shape} must have {J} rows to match "
            f"{observed_name}.shape[1]."
        )
    if inferred_K[0] < 0:
        raise ValueError("B_list[0] must have at least one intercept column.")

    for d in range(1, D):
        expected_rows = inferred_K[d - 1]
        if B_arrays[d].shape[0] != expected_rows:
            raise ValueError(
                f"B_list[{d}] shape {B_arrays[d].shape} must have {expected_rows} "
                f"rows to match K_{d}."
            )
        K_parent = int(B_arrays[d].shape[1] - 1)
        if K_parent < 0:
            raise ValueError(f"B_list[{d}] must have at least one intercept column.")
        inferred_K.append(K_parent)

    inferred_K_tuple = tuple(inferred_K)
    validate_K_list_identifiability(inferred_K_tuple, J=J)
    if K_list is not None:
        K_tuple = validate_K_list_identifiability(K_list, D=D, J=J)
        if K_tuple != inferred_K_tuple:
            raise ValueError(
                f"K_list {K_tuple} does not match widths implied by B_list "
                f"{inferred_K_tuple}."
            )

    if A_list is not None:
        if not _is_sequence_like(A_list):
            raise ValueError("A_list must be a sequence of 2D arrays.")
        A_arrays = [_as_2d_array(A, f"A_list[{d}]") for d, A in enumerate(A_list)]
        if len(A_arrays) != D:
            raise ValueError(f"A_list length {len(A_arrays)} must match D={D}.")
        for d, A in enumerate(A_arrays):
            expected = (N, inferred_K[d])
            if A.shape != expected:
                raise ValueError(f"A_list[{d}] shape {A.shape} must be {expected}.")
            if require_finite:
                _check_finite(A, f"A_list[{d}]")
            if require_binary_A:
                _check_binary(A, f"A_list[{d}]", binary_tol)

    if prop is not None:
        prop_arr = np.asarray(prop, dtype=float).reshape(-1)
        expected = inferred_K[-1]
        if prop_arr.shape != (expected,):
            raise ValueError(f"prop shape {prop_arr.shape} must be ({expected},).")
        if require_finite:
            _check_finite(prop_arr, "prop")
        lo = 0.0 - probability_tol
        hi = 1.0 + probability_tol
        if np.any((prop_arr < lo) | (prop_arr > hi)):
            raise ValueError("prop entries must be probabilities in [0, 1].")

    if gamma is not None:
        gamma_arr = np.asarray(gamma, dtype=float).reshape(-1)
        if gamma_arr.shape != (J,):
            raise ValueError(f"gamma shape {gamma_arr.shape} must be ({J},).")
        if require_finite:
            _check_finite(gamma_arr, "gamma")
        if np.any(gamma_arr <= 0):
            raise ValueError("gamma entries must be positive.")

    return DDEListShape(D=D, N=N, J=J, K_list=inferred_K_tuple)
