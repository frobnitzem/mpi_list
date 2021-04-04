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
        result = [[e'] with a given sequence number]
        limited to sequence numbers belonging to current rank.
        The sequence numbers of each sub-list
        are sorted ascending, but not provided.
        Note: len(result) <= seq1 - seq0
    """
    sets = [[] for i in range(C.procs)] # output sets going to each rank
    bs = (N+C.procs-1) // C.procs # max blk sz
    bs0 = N//C.procs
    for seq, p in dP.items():
        j = seq // bs # lower bound on j
        while seq >= (j+1)*bs0 + min(N % C.procs, j+1):
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
            ans.append(sp)
    # ans is now a list of (i,p) belonging to this rank
    # need to sort it
    if len(ans) == 0:
        return []

    perm = [(s[0],i) for i,s in enumerate(ans)]
    perm.sort()

    # gather together all elements with the same idx (s[0] above)
    # into grps, a list of lists -- one per cval
    grps = []
    cval = None # current output idx
    clist = []  # elems at idx
    for key, i in perm:
        if cval is None:
            cval = key
        elif key != cval:
            grps.append(clist)
            clist = []
            cval = key
        clist.extend(ans[i][1]) # all elements of key from a rank
    if cval is not None:
        grps.append(clist)

    return grps

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
        sched: list of (tag,src,dst) pairs for all sends involving this rank

    Returns:
        [ [items received with idx] over all idx-s ]
    """
    i = 0
    sends = []
    recvs = [] # nested list of recv-s, grouped by idx
    dgrp = [] # group of recv-s to the same idx
    cidx = None
    for tag,src,dst,idx in sched:
        if src == C.rank:
            assert i < len(items), "Too many sends requested."
            req = C.comm.isend(items[i], dest=dst, tag=tag)
            sends.append(req)
            i += 1
        elif dst == C.rank:
            if cidx != idx:
                if len(dgrp) > 0:
                    recvs.append(dgrp)
                dgrp = []
                cidx = idx
            req = C.comm.irecv(source=src, tag=tag)
            dgrp.append(req)
    if len(dgrp) > 0:
        recvs.append(dgrp)
    assert i == len(items), "Some items were not sent!"

    ans = []
    for dgrp in recvs:
        v = []
        for r in dgrp:
            v.append( r.wait() )
        ans.append(v)
    for s in sends:
        s.wait()
    return ans
