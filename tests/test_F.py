import pytest

from mpi_list.F import AlgFn, F

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

def test_AlgFn():
    def const(x):
        def f(y):
            return x
        return AlgFn(f)

    u = const(10) + const(1) == const(11)
    assert u('ok') == True
