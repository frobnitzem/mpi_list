from .fill import fill
from .reducer import Reducer, CommReducer
from .pscan import psched

class DFM:
    """Distributed Free Monoid = A list of something.

    All method calls are parallel, and must be called
    by every rank.

    Attributes:
        C: Reference to the Context object
        E: List of local elements.

    """
    def __init__(self, C, E):
        self.C = C
        self.E = E

    def len(self):
        """Number of elements in DFM (returned to all ranks)

        Returns:
            int : total size
        """
        return self.C.comm.allreduce(len(self.E))

    def map(self, f):
        """Map over elements.

        Args:
            f: function of type = elem -> new elem

        Returns:
            new DFM

        """
        return DFM(self.C, [f(e) for e in self.E])
    
    def filter(self, f):
        """Filter, removing some elements.

        Args:
            f: function of type = elem -> bool

        Returns:
            new DFM

        """
        return DFM(self.C, [e for e in self.E if f(e)])

    def flatMap(self, f): # applyM
        """Map over elements and concatenate all results.

        Args:
            f: function of type = elem -> [new elem]

        Returns:
            new DFM

        """
        ans = []
        for e in self.E:
            ans.extend( f(e) )
        return DFM(self.C, ans)

    def reduce(self, f, x0, distribute=True):
        """Reduce the dataset to a value.

        Apply an associative, pairwise reduction to the dataset.
        The result is sent to all ranks if distribute=True
        otherwise, it is present only on the root rank (rank 0).

        The function f must operate in-place, storing
        its result on the left, for example::

            def f(a,b):
                a[0] = b[0]

        This is indicated in the function's type with *elem.

        Each rank calls `f(x0, e)` on all its elements,
        then does a fan-in reduction on x0.  You can
        technically implement a different function for each
        phase by detecting whether `e` has the type
        of x0 or the type of an element.
        In fact, this is the only way to get around the
        restriction that `type(x0) == type(e)`.

        Note:
            `x0` must be an object that can be updated in-place.
            On return, `x0` will contain an undefined value.

            It must be initialized to a value representing
            the starting value for a single MPI rank.


        Args:
            f: a function modifying its first argument, type = *elem, elem -> ()
            x0: the "zero" value of the first argument
            distribute: Distribute the answer from rank 0 to all ranks?

        Returns:
            elem

        """
        R = Reducer(f, x0)
        for e in self.E:
            R(e)
        CommReducer(self.C,R)()
        if distribute:
            x0 = self.C.comm.bcast(x0)
        return x0

    def scan(self, f):
        """Perform a parallel prefix-scan on the dataset.

        Args:
            f: an associative, pairwise function of type = elem, elem -> elem

        Returns:
            DFM containing [e0, f(e0,e1), f(e0,f(e1,e2)), ...]

        """

        rank = self.C.rank
        procs = self.C.procs

        # compute local prefix-sum
        pre = []
        if len(self.E) > 0:
            pre = [self.E[0]]
            for i in range(1, len(self.E)):
                pre.append(f(pre[i-1], self.E[i]))

        if procs == 1:
            return DFM(self.C, pre)

        last = []
        if len(pre) > 0:
            last = [ pre[-1] ]

        # send last val. to rank+1 nbr
        if rank % 2 == 0: # even ranks send first
            if rank != procs-1:
                self.C.comm.send(last, dest=rank+1, tag=10)
            if rank == 0:
                recv = []
            else:
                recv = self.C.comm.recv(source=rank-1, tag=11)
        else: # odd ranks recv first
            recv = self.C.comm.recv(source=rank-1, tag=10)
            if rank != procs-1:
                self.C.comm.send(last, dest=rank+1, tag=11)
        last = recv

        # ranks 1, ..., procs-1 participate in prefix scan
        if rank > 0:
            vrank = rank-1 # virtual rank numbering
            sch = psched(procs-1)
            for i,sl in enumerate(sch):
                off = sl.step//2
                # sending rank?
                if vrank >= sl.start \
                       and vrank < sl.stop \
                       and (vrank - sl.start)%sl.step == 0:
                    self.C.comm.send(last, dest=rank+off, tag=i)
                # receiving rank?
                elif vrank >= sl.start+off \
                       and (vrank - sl.start-off)%sl.step == 0:
                    u = self.C.comm.recv(source=rank-off, tag=i)
                    if len(last) == 0:
                        last = u
                    elif len(u) != 0:
                        last = [ f(u[0], last[0]) ]
                    # else u == [] and last remains unchanged

        # distribute incoming prefix scan (if non-empty)
        if len(last) > 0:
            for i in range(len(pre)):
                pre[i] = f(last[0], pre[i])

        return DFM(self.C, pre)

    def collect(self, root=0):
        """Collect all the elements to the root rank.

        This is equivalent to reduce(extend, [], distribute=False),
        but uses MPI_Gather.

        Note:
            Even though non-root ranks receive None, they *still*
            have call this function.
            Check the dataset size before calling this, since
            this may cause a memory error.

        Args:
            root: The rank receiving the collected results.

        Returns:
            List of elems if rank == root, None otherwise.

        """

        lE = self.C.comm.gather(self.E, root=root)
        if self.C.rank != root:
            return None
        ans = []
        for x in lE:
            ans.extend(x)
        return ans

    def nodeMap(self, f):
        """map over the MPI ranks.

        The input function, f, is called
        once on every rank with arguments:
        * rank : int
        * E : list containing all local elements
        
        Note:
            f must return a list

        Args:
            f: function of type = int, elem -> [new elems]

        Returns:
            DFM

        """

        ans = f(self.C.rank, self.E)
        assert isinstance(ans, list), f"nodeMap: f must return a list (got {type(f)})"
        return DFM(self.C, f(self.C.rank, self.E))

    def head(self, n=10):
        """Distribute the first n elements to all ranks
        (useful for interactive debugging)

        Returns:
            first n values, [elem]
        """

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

class Context:
    """Global context
    
    The context is a convenient place to store MPI attributes.

    Attributes:
        rank:  rank of the current process (0, 1, ..., procs-1)
        procs: number of MPI ranks
        comm:  MPI.COMM_WORLD
        MPI:   mpi4py's MPI module

    """
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.procs = self.comm.Get_size()
        self.MPI = MPI

    def iterates(self, n, robin=False):
        """Create a DFM from a sequence of numbers.

        Args:
            n:     The number of iterates
            robin: If True, assignments are done round-robin,
                   so rank 0 will have 0, procs, 2*procs, ...
                   rank 1 will have 1, procs+1, 2*procs+1, ...

        Returns:
            DFM holding numbers 0, 1, ..., n-1

        """
        if robin: # round-robin is simpler, but destroys ordering
            return DFM(self, list(range(self.rank, n, self.procs)))
        blk = n // self.procs
        extra = n % self.procs
        elapsed = min(self.rank, extra) # extra elements prior to rank
        extra1 = self.rank < extra # do I have an extra element?
        i0 = blk*self.rank + elapsed
        return DFM(self, list(range(i0, i0+blk+extra1)))
