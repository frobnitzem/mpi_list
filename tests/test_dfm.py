import pytest

from mpi_list.dfm import DFM, Context

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

def test_dfm(N=97):
    import numpy as np
    C = Context()

    dfm = C \
      . iterates(N) \
      . map( lambda x: np.ones(x) ) \

    assert dfm.len() == N
    assert len(dfm.E) >= N // C.procs

    # Note: What follows requires correct ordering of iterates:

    ans = dfm . collect()
    if C.rank == 0:
        for i in range(N):
            assert len(ans[i]) == i

    dfm = dfm.map( lambda x: x.sum() )

    ans = dfm . collect()
    if C.rank == 0:
        for i in range(N):
            assert ans[i] == i

def test_head(N=107, M=3):
    import numpy as np
    C = Context()

    h = C \
      . iterates(N) \
      . head(M)
    assert len(h) == min(N, M)
    assert h[0] == 0
    assert h[-1] == len(h)-1

def test_reduce(N=101):
    C = Context()

    dfm = C . iterates(N) . map(lambda x: [x])

    s = [0]
    def add(a,b):
        a[0] += b[0]

    ans = dfm.reduce(add, s)
    assert ans[0] == N*(N-1) // 2

    s = []
    def append(a,b):
        a.extend(b)

    ans = dfm.reduce(append, s)
    assert len(ans) == N
    if N >= 1:
        assert ans[0] == 0 and ans[-1] == N-1

def test_filter(N=223):
    C = Context()

    dfm = C . iterates(N)

    for n in [1, 2, 3, 7, 20]:
      ans = dfm . filter(lambda x: x % n == n-1)
      assert ans.len() == N // n

def test_flatmap():
    C = Context()

    dfm = C . iterates(100) \
            . flatMap(lambda x: [a for a in str(x)])

    ans = dfm.collect()
    if C.rank == 0:
        N = 2*100 - 10
        assert len(ans) == N
        assert ans[0] == '0'
        assert ans[-2] == '9'
        assert ans[-1] == '9'

def test_nodemap(N=100):
    C = Context()

    def test_fn(rank, elems):
        assert rank == C.rank
        return [ (rank, len(elems)) ]

    dfm = C . iterates(N) \
            . nodeMap( test_fn )

    assert dfm.len() == C.procs

    v = dfm.head()
    if N > 0:
        assert v[0][0] == 0 and v[0][1] == (N // C.procs) + (N % C.procs != 0)

def test_combinations():
    test_dfm(0)
    test_dfm(1)
    test_dfm(10)
    test_dfm(101)
    test_dfm(997)

    test_head(100, 10)

    test_reduce(0)
    test_reduce(1)
    test_reduce(16)

    test_filter(0)
    test_filter(1)
    test_filter(21)
    test_filter(311)

    test_flatmap()
    test_nodemap(100)

if __name__=="__main__":
    "Allow tests to be run stand-alone using mpirun."
    test_combinations()
