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

class UV:
    def __init__(self,u,v):
        self.u = u
        self.v = v

def test_lookup():
    K = F('u')
    assert isinstance(K, AlgFn)
    assert K(UV(1,2)) == 1

def test_lookup_chain():
    z = lambda x: x*2
    K = F.u.v(4) + F('v')
    assert isinstance(K, AlgFn)
    assert K(UV(UV(1,z), 2)) == 10
