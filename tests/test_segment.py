import pytest

from mpi_list.segment import even_spread, cumsum, segments

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

def test_spread():
    with pytest.raises(AssertionError) as excinfo:
        even_spread(1,0)
    for M in [1,4,21]:
        assert len(even_spread(10,M)) == M

def test_segments(blks=[], oblks=[]):
    sched = segments(cumsum(blks), cumsum(oblks))
    inp = [0]*len(blks)
    out = [0]*len(oblks)

    for c in sched:
        n = c.d1-c.d0
        assert n > 0
        assert c.s1-c.s0 == n
        assert inp[c.src] == c.s0
        assert out[c.dst] == c.d0
        inp[c.src] = c.s1
        out[c.dst] = c.d1

    # all inputs used
    for i in range(len(blks)):
        assert inp[i] == blks[i]
    # all outputs are the desired sizes
    for i in range(len(oblks)):
        assert out[i] == oblks[i]

def test_segments_e(blks=[], N=0):
    oblk = even_spread(sum(i for i in blks), N)
    return test_segments(blks, oblk)

def test_all():
    blks = [100,30,10,0,33,4,201]
    test_segments_e(blks, 1)
    test_segments_e(blks, 5)
    test_segments_e(blks, 10)
    test_segments_e(blks, 201)
    test_segments([76, 12, 441, 864, 12, 42], [65, 124, 247, 800, 211])

