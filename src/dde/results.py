"""Result containers shared by the package-level fitting API."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DDEResult:
    """Standard result object returned by :func:`dde.fit`.

    The algorithm-specific raw return value is kept in ``raw`` so that the
    first package version stays faithful to the research implementations while
    we gradually standardize outputs.
    """

    model: str
    K_list: Optional[List[int]] = None
    prop: Optional[np.ndarray] = None
    B_list: Optional[List[np.ndarray]] = None
    gamma: Optional[np.ndarray] = None
    A_list: Optional[List[np.ndarray]] = None
    transformed: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


def data_column_names(Y):
    """Column names of a data frame (``None`` for arrays), as strings."""
    columns = getattr(Y, "columns", None)
    if columns is None:
        return None
    return [str(name) for name in columns]


def record_column_names(result, names):
    """Store the data's column names in ``diagnostics["variable_names"]``."""
    if names is not None and result is not None:
        result.diagnostics = dict(result.diagnostics or {})
        result.diagnostics.setdefault("variable_names", list(names))
    return result
