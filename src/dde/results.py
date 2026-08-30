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
