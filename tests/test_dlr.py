import numpy as np
import pytest

from sbw_host import dlr as D


def test_dlr_circular_is_separation_over_radius():
    assert D.d_dlr(3.0, 4.0, 2.0, 2.0, 0.0) == pytest.approx(2.5)


def test_dlr_along_axes():
    # a=4 along +u, b=1 along +v
    assert D.d_dlr(4.0, 0.0, 4.0, 1.0, 0.0) == pytest.approx(1.0)
    assert D.d_dlr(0.0, 1.0, 4.0, 1.0, 0.0) == pytest.approx(1.0)
    assert D.d_dlr(0.0, 4.0, 4.0, 1.0, 0.0) == pytest.approx(4.0)
    # rotate the galaxy by 90 deg: now the major axis is along v
    assert D.d_dlr(0.0, 4.0, 4.0, 1.0, 90.0) == pytest.approx(1.0)


def test_dlr_assignment_threshold():
    assert list(D.dlr_assignment([2.0, 0.5, 3.0])) == [0.0, 1.0, 0.0]
    assert list(D.dlr_assignment([5.0, 6.0])) == [0.0, 0.0]
    assert list(D.dlr_assignment([])) == []
