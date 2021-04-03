import pytest

import numpy as np
from mpi_list import Context
from mpi_list.reducer import Reducer, CommReducer

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"


def test():
    C = Context()

    x0 = np.zeros(10)
    def add(x,y):
        x += y
    R = Reducer(add, x0)
    R(np.ones(10))
    CommReducer(C,R)()
    if C.rank == 0:
        print(x0[0], x0[-1])

def test2():
    C = Context()

    x0 = [0]
    def add(x,y):
        x[0] += y[0]
    R = Reducer(add, x0)
    R([10])
    CommReducer(C,R)()
    if C.rank == 0:
        print(x0[0])

if __name__=="__main__":
    "Allow tests to be run stand-alone using mpirun."
    test()
    test2()
