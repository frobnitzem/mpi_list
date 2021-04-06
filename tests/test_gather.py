import pytest

from mpi_list.dfm import DFM, Context

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

import numpy as np

def concat(ll):
    ans = []
    for a in ll:
        ans.extend(a)
    return ans

def test_group(N=0, M=1):
    C = Context()

    def groups(e, out):
        key = e % M
        if key not in out:
            out[key] = []
        out[key].append(e)
    dfm = C \
      . iterates(N) \
      . group( groups, lambda x: x, M )

    assert dfm.len() == min(N,M)

    i0 = C.rank * (M//C.procs) + min(C.rank, M%C.procs)
    i1 = (C.rank+1) * (M//C.procs) + min(C.rank+1, M%C.procs)
    for eg in dfm.E:
        for e in eg:
            loc = e%M
            assert i0 <= loc
            assert loc < i1

    # use flatMap and check sizes again
    fdf = dfm.flatMap(lambda x: x).len()
    assert fdf == N

def test_repartition(N=0, M=1):
    C = Context()

    start = C \
      . iterates(N) \
      . map( lambda x: np.ones((x,4)) )

    K = start.map( len ).reduce(lambda a,b: a+b, 0)
    if C.rank == 0:
        assert K == N*(N-1)//2

    dfm = start \
      . repartition(len,
                    lambda df,rng: [df[r0:r1] for r0,r1 in rng],
                    np.vstack, M)

    K = dfm.map( len ).reduce(lambda a,b: a+b, 0)
    if C.rank == 0:
        print(N,K,[len(e) for e in dfm.E])
        assert K == N*(N-1)//2

    for e in dfm.E:
        assert len(e) >= N//M

def test_all():
    test_group(10, 1)
    test_group(1, 10)
    test_group(100, 10)
    test_group(101, 14)
    test_group(10, 100)

    test_repartition(10, 1)
    test_repartition(10, 2)
    test_repartition(10, 20)
    test_repartition(101, 20)

# TODO: add test that repartitions random sizes per rank
# and double-checks output size, partitions, etc.

if __name__=="__main__":
    test_all()
