import numpy as np
import pytest

from semidde import DDEResult, fit
from semidde.utils import TLP, binary, logistic, thres


def test_public_imports():
    assert fit is not None
    assert DDEResult.__name__ == "DDEResult"


def test_shared_utils():
    assert TLP(np.array([2.0, -0.5]), 1.0) == pytest.approx(1.5)
    assert np.array_equal(thres(np.array([0.1, 2.0]), 0.5), np.array([0.0, 2.0]))
    assert binary([0, 1, 2], 2).shape == (3, 2)
    assert logistic(0.0) == pytest.approx(0.5)


def test_fit_requires_initialized_args():
    Y = np.zeros((5, 18))
    with pytest.raises(ValueError, match="initialized algorithm arguments"):
        fit(Y, model="semi", K_list=[6, 2])

    with pytest.raises(ValueError):
        fit(Y, model="copula", K_list=[6, 2])
