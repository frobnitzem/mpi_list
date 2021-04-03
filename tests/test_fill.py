import pytest

import numpy as np
from mpi_list.fill import fill

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

def check_fill(delta, sends):
    x = delta.copy()
    # follow each send
    for sno in sends:
        for i,j,n in sno:
            assert x[i] >= n, "Sending non-existent items"
            x[i] -= n
            x[j] += n
    assert np.all(x == 0), "Improper ending state"

def test_fill():
    ans = fill(np.array([0])) # test trivial case
    print(ans)

    # test a known case
    delta = np.array([-2,3,-1,1, 1,-5,3])
    sends = fill(delta)
    print(delta)
    print(sends)
    check_fill(delta, sends)

    for M in range(2,100,3): # range of sizes
        for j in range(10): # tests per size
            delta = np.random.randint(-10,11, size=M)
            v = np.random.randint(M)
            delta[v] -= delta.sum() # deposit excess

            sends = fill( delta )
            check_fill(delta, sends)
