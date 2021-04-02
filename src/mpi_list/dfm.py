from .fill import fill
from .reducer import Reducer, CommReducer

# Distributed Free Monoid = A list of something.
class DFM:
    def __init__(self, ctxt, elems):
        self.C = ctxt
        self.E = elems

    # Number of elements in DFM
    # returns the total size to every process
    def len(self):
        return self.C.comm.allreduce(len(self.E))

    # Map over elements.
    # f : elem -> new elem
    def map(self, f):
        return DFM(self.C, [f(e) for e in self.E])
    
    # Filter, removing some elements.
    # f : elem -> bool
    def filter(self, f):
        return DFM(self.C, [e for e in self.E if f(e)])

    # Map over elements and concatenate all results.
    # f : elem -> [new elem]
    def flatMap(self, f): # applyM
        ans = []
        for e in self.E:
            ans.extend( f(e) )
        return DFM(self.C, ans)

    # Apply an associative, pairwise reduction to the dataset.
    # The result is sent to all nodes if distribute=True
    # otherwise, it is present only on the root node.
    #
    # Note: x0 must be an object that can be updated in-place
    # It must be initialized to a value representing
    # the starting value for a single MPI rank.
    #
    # It will contain an undefined value on return of this function,
    # having to do with the reduction order.
    #
    # The function f must operate in-place, storing
    # its result on the left.
    # f : *a, a -> ()
    def reduce(self, f, x0, distribute=True):
        R = Reducer(f, x0)
        for e in self.E:
            R(e)
        CommReducer(self.C,R)()
        if distribute:
            x0 = self.C.comm.bcast(x0)
        return x0

    # Collect all the elements to the root node.
    # This *should* be the same as reduce(extend, [], distribute=False)
    def collect(self):
        lE = self.C.comm.gather(self.E)
        if self.C.rank != 0:
            return lE
        ans = []
        for x in lE:
            ans.extend(x)
        return ans

    # map over every rank - f is called
    # once on every node with arguments
    #  - rank : int
    #  - E : list containing all local elements
    # f must return a list
    def nodeMap(self, f):
        ans = f(self.C.rank, self.E)
        assert isinstance(ans, list), f"nodeMap: f must return a list (got {type(f)})"
        return DFM(self.C, f(self.C.rank, self.E))

    # Distribute the first n elements to all nodes
    # (useful for interactive debugging)
    def head(self, n=10):
        # create dfm with length of each rank
        ans = []
        root = 0
        while len(ans) < n:
            if root >= self.C.procs:
                break
            data = None
            if root == self.C.rank:
                data = self.E[:min(len(self.E), n - len(ans))]
            ans.extend( self.C.comm.bcast(data, root=root) )
            root += 1
        return ans

    ## re-group elements locally
    #def gather(self, concat, tgt, sz=len):
    #    #
    #    newE = []
    #    return DFM(self.C, newE)

# create a global context
# The context holds its DFMs
class Context:
    def __init__(self):
        from mpi4py import MPI
        self.rdds = [] # link to all DFMs
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.procs = self.comm.Get_size()
        self.MPI = MPI

    # Create a DFM from a sequence of numbers.
    def iterates(self, n, robin=False):
        if robin: # round-robin is simpler, but destroys ordering
            return DFM(self, list(range(self.rank, n, self.procs)))
        blk = n // self.procs
        extra = n % self.procs
        elapsed = min(self.rank, extra) # extra elements prior to rank
        extra1 = self.rank < extra # do I have an extra element?
        i0 = blk*self.rank + elapsed
        return DFM(self, list(range(i0, i0+blk+extra1)))
