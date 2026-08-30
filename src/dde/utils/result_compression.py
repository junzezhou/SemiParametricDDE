"""Helpers for compressing fitted CSP DDE results to active latent columns."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _copy_list(arrays):
    if arrays is None:
        return None
    return [np.asarray(arr).copy() for arr in arrays]


def _active_masks_from_B(B_list, *, active_tol: float, min_active: bool):
    B_arrays = [np.asarray(B, dtype=float) for B in B_list]
    D = len(B_arrays)
    masks = [np.ones(B_arrays[d].shape[1] - 1, dtype=bool) for d in range(D)]
    forced = [False] * D

    changed = True
    while changed:
        changed = False
        new_masks = []
        new_forced = []
        for d, B in enumerate(B_arrays):
            rows = np.ones(B.shape[0], dtype=bool) if d == 0 else masks[d - 1]
            slopes = B[rows, 1:]
            active = np.any(np.abs(slopes) > active_tol, axis=0)
            was_forced = False
            if min_active and active.size and not np.any(active):
                norms = np.linalg.norm(slopes, axis=0) if slopes.size else np.zeros(active.size)
                active[int(np.argmax(norms))] = True
                was_forced = True
            new_masks.append(active)
            new_forced.append(was_forced)
        for old, new in zip(masks, new_masks):
            if old.shape != new.shape or np.any(old != new):
                changed = True
                break
        masks = new_masks
        forced = new_forced
    return masks, forced


def _compress_A_sample_long(A_sample_long, masks):
    if A_sample_long is None:
        return None
    compressed = []
    for samples, mask in zip(A_sample_long, masks):
        arr = np.asarray(samples)
        if arr.ndim >= 2 and arr.shape[1] == mask.size + 1:
            keep = np.concatenate(([True], mask))
            compressed.append(arr[:, keep, ...].copy())
        else:
            compressed.append(arr.copy())
    return compressed


def compress_csp_result_components(
    *,
    prop,
    B_list,
    A_list,
    diagnostics: Dict[str, Any] | None = None,
    raw: Any = None,
    active_masks=None,
    active_tol: float = 1e-12,
    min_active: bool = True,
) -> Dict[str, Any]:
    """Compress DDE components to CSP-active latent columns.

    The returned components use compact active columns only.  A 1-based mapping
    back to the original latent column numbers is stored in
    ``diagnostics["active_columns_original_index"]``.
    """
    B_arrays = _copy_list(B_list)
    A_arrays = _copy_list(A_list)
    prop_arr = None if prop is None else np.asarray(prop, dtype=float).reshape(-1).copy()
    diagnostics_out: Dict[str, Any] = dict(diagnostics or {})

    if not B_arrays:
        return {
            "prop": prop_arr,
            "B_list": B_arrays,
            "A_list": A_arrays,
            "diagnostics": diagnostics_out,
            "raw": raw,
        }

    original_K_list = tuple(int(B.shape[1] - 1) for B in B_arrays)
    if active_masks is None:
        masks, forced = _active_masks_from_B(
            B_arrays, active_tol=active_tol, min_active=min_active
        )
    else:
        if len(active_masks) != len(B_arrays):
            raise ValueError("active_masks length must match B_list length.")
        masks = []
        for d, (mask, B) in enumerate(zip(active_masks, B_arrays)):
            mask_array = np.asarray(mask, dtype=bool).reshape(-1)
            expected = B.shape[1] - 1
            if mask_array.shape != (expected,):
                raise ValueError(
                    f"active_masks[{d}] shape {mask_array.shape} must be {(expected,)}."
                )
            masks.append(mask_array.copy())
        forced = [False] * len(B_arrays)
    active_index = {
        f"layer_{d + 1}": (np.flatnonzero(mask) + 1).astype(int).tolist()
        for d, mask in enumerate(masks)
    }

    B_compressed = []
    for d, B in enumerate(B_arrays):
        row_mask = slice(None) if d == 0 else masks[d - 1]
        col_mask = np.concatenate(([True], masks[d]))
        B_compressed.append(B[row_mask, :][:, col_mask].copy())

    A_compressed = None
    if A_arrays is not None:
        A_compressed = [A[:, masks[d]].copy() for d, A in enumerate(A_arrays)]

    prop_compressed = prop_arr
    if prop_arr is not None:
        prop_compressed = prop_arr[masks[-1]].copy()

    compressed_K_list = tuple(int(np.sum(mask)) for mask in masks)
    diagnostics_out["original_K_list"] = tuple(
        diagnostics_out.get("original_K_list", original_K_list)
    )
    diagnostics_out["K_list"] = compressed_K_list
    diagnostics_out["active_columns_original_index"] = active_index
    if any(forced):
        diagnostics_out["compression_forced_min_active"] = {
            f"layer_{d + 1}": bool(value) for d, value in enumerate(forced)
        }

    raw_out = raw
    if isinstance(raw, dict):
        raw_out = dict(raw)
        raw_out["prop"] = prop_compressed
        raw_out["B_list"] = B_compressed
        if A_compressed is not None:
            raw_out["A_list"] = A_compressed
        if "A_sample_long" in raw_out:
            raw_out["A_sample_long"] = _compress_A_sample_long(raw_out["A_sample_long"], masks)

    if "A_sample_long" in diagnostics_out:
        diagnostics_out["A_sample_long"] = _compress_A_sample_long(
            diagnostics_out["A_sample_long"], masks
        )
    if "num_active" in diagnostics_out:
        diagnostics_out["num_active"] = np.asarray(compressed_K_list, dtype=int)

    return {
        "prop": prop_compressed,
        "B_list": B_compressed,
        "A_list": A_compressed,
        "diagnostics": diagnostics_out,
        "raw": raw_out,
    }


__all__ = ["compress_csp_result_components"]
