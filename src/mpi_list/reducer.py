from math import log
import numpy as np

# fn should operate by modifying its first argument
# at the end of the reduction, `data` will hold the answer.
class Reducer:
    def __init__(self, fn, zero):
        self.fn = fn      #  *a,a -> ()
        self.data = zero  #  a

    def __call__(self, data2):
        self.fn(self.data, data2)

class CommReducer:
    def __init__(self, C, R):
        self.comm = C.comm
        self.rank = C.rank
        self.procs = C.procs
        self.MPI = C.MPI
        self.R = R

    def __call__(self):
        n = self.procs
        step = 1
        lev = 0
        while step < self.procs:
            lev += 1
            if self.rank % step != 0:
                break

            if self.rank % (2*step) == 0:
                if self.rank + step < self.procs:
                    #print(f"{self.rank} <- {self.rank+step}")
                    self.recv(self.rank+step, lev)
            else:
                #print(f"{self.rank} -> {self.rank-step}")
                self.send(self.rank-step, lev)
                break

            #for i in range(0, self.procs, 2*step):
            #    self.join2(i, i+step)
            step *= 2

    def recv(self, j, lev):
        if isinstance(self.R.data, np.ndarray):
            return self.fast_recv(j, lev)
        self.R( self.comm.recv(source=j, tag=lev) )

    def fast_recv(self, j, lev):
        len1 = (1<<30) - 1
        nchunk = (self.R.data.nbytes + len1) >> 30

        dst = bytearray(self.R.data.nbytes) # MPI's nbytes is stored in an int!
        obj = memoryview(dst)
        for k in range(nchunk):
            end = min((k+1)<<30, self.R.data.nbytes)
            self.comm.Recv([obj[k<<30 : end], end-(k<<30), self.MPI.BYTE], source=j, tag=100*lev+k)
        #self.comm.Recv([dst, self.R.data.nbytes, MPI.BYTE], source=j, tag=lev)
        self.R( np.frombuffer(dst, dtype=self.R.data.dtype) )

    def send(self, i, lev):
        if isinstance(self.R.data, np.ndarray):
            return self.fast_send(i, lev)
        self.comm.send(self.R.data, dest=i, tag=lev)

    def fast_send(self, i, lev):
        len1 = (1<<30) - 1
        nchunk = (self.R.data.nbytes + len1) >> 30

        obj = self.R.data.tobytes()
        for k in range(nchunk):
            end = min((k+1)<<30, self.R.data.nbytes)
            self.comm.Send([obj[k<<30 : end], end-(k<<30), self.MPI.BYTE], dest=i, tag=100*lev+k)
        #self.comm.Send([obj, self.R.data.nbytes, MPI.BYTE], dest=i, tag=lev)

