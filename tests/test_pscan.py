import pytest

from mpi_list.pscan import psched, slices_to_sched
from mpi_list import Context

__author__ = "David M. Rogers"
__copyright__ = "Oak Ridge National Lab"
__license__ = "MIT"

def test_sch(n=11):
    sch = slices_to_sched( psched(n) )
    lst = [i for i in range(n)]
    for i,j in sch:
        lst[j] += lst[i]
    for i,n in enumerate(lst):
        assert n == i*(i+1)//2

def test_multiple():
    for n in [0,1,2, 5, 8, 10, 20, 48, 71, 145]:
        test_sch(n)

def test_scan(N=12):
    C = Context()
    lst = C.iterates(N).scan(lambda a,b: a+b).collect()

    if C.rank == 0:
        for i,n in enumerate(lst):
            assert n == i*(i+1)//2

def test_mscan():
    for n in [0,1,5, 12, 32, 48, 120, 211]:
        test_scan(n)

if __name__=="__main__":
    test_mscan()
