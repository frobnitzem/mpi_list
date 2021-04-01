#!/usr/bin/env python3

from dfm import Context
from reducer import Reducer, CommReducer
import numpy as np

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

if __name__ == "__main__":
    test()
    test2()
