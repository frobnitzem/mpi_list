def gather_partitions(C, dP, N):
    """Gather together all the elements whose
    target sequence number is in the current
    rank's domain (seq0 <= seq < seq1)

    seq = 0, 1, ..., N-1
    seq0 = (rank+0) * (N//procs) + min(N%procs, rank+0)
    seq1 = (rank+1) * (N//procs) + min(N%procs, rank+1)

    Note:
        This must be called by all ranks.

    Args:
        C: Context
        dP: {seq : e'} from current rank
        N: Number of output elements

    Returns:
        [e'] belonging to current rank [ len <= seq1-seq0 ]
    """
    sets = [[] for i in range(C.procs)] # output sets going to each rank
    bs = (N+C.procs-1) // C.procs # max blk sz
    bs0 = N//C.procs
    for seq, p in dP.items():
        j = seq // bs # lower bound on j
        while seq > (j+1)*bs0 + min(N % C.procs, j+1):
            j += 1
        sets[j].append( (seq,p) )

    out = [ sets[C.rank] ] # local data skips MPI
    for root in range(C.procs):
        if C.rank == root:
            # ans : [ [(i,p) belonging to self] ]
            out.extend( C.comm.gather([], root) )
        else:
            C.comm.gather(sets[root], root)

    # re-assemble local partitions
    ans = []
    for r in out:
        for sp in r:
            ans.extend(sp)
    # ans is now a list of (i,p) belonging to this rank
    # need to sort it
    perm = [(s[0],i) for i,s in enumerate(s)]
    perm.sort()
    return [ ans[i][1] for j,i in perm ]

def send_chunks(comm, lst, dst, tag, max_elems=100000):
    for i,n in enumerate(range(0, len(lst), max_elems)):
        m = min(n+max_elems, len(lst))
        comm.send(lst[n:m], dest=dst, tag=tag+100*i)

def recv_chunks(comm, lst, off, src, tag, max_elems=100000):
    for i,n in enumerate(range(off, len(lst), max_elems)):
        m = min(n+max_elems, len(lst))
        lst[n:m] = comm.recv(source=src, tag=tag+100*i)

def send_items(C, items, sched):
    """Send the indicated items to the set of destinations.

    Args:
        C: Context
        items: local items to send
        sched: list of (src,dst) pairs for all sends involving this rank

    Returns:
        list of items received
    """
    i = 0
    sends = []
    recvs = []
    for k,src,dst in enumerate(sched):
        if src == C.rank:
            assert i < len(items), "Too many sends requested."
            req = C.comm.isend(items[i], dest=dst, tag=k)
            sends.append(req)
            i += 1
        elif dst == C.rank:
            req = C.comm.irecv(source=src, tag=k)
            recvs.append(req)
    assert i == len(items), "Some items were not sent!"

    ans = []
    for r in recvs:
        ans.append( r.wait() )
    for s in sends:
        s.wait()
    return ans
