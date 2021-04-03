# Parallel prefix scan code

def psched(n):
    """Create a prefix scan schedule for n ranks.

    The result is encoded as a list of slices -- one slice for
    the reduction carried out at each level of the scan.

    If needed, full list of (from,to) ranks can be recovered using the
    `slices_to_sched` function.

    Args:
        n: integer number of items to scan

    Returns:
        [slice]: Each slice contains (start,stop,step)
                 for stepping through `from` ranks.
                 `to` ranks are always at i + step//2

    """
    sch = []
    skip = 1
    while 2*skip-1 < n:
        # i+skip < n
        sch.append(slice(skip-1,n-skip,2*skip))
        #sch.extend([(i,i+skip) for i in range(skip-1,n-skip,2*skip)])
        skip *= 2
    while 3*skip > n:
        skip = skip // 2
    #from math import log, floor
    #skip = 2**int(floor(log(n/3.0)/log(2.0)))
    while skip >= 1:
        # i+skip < n
        sch.append(slice(2*skip-1,n-skip,2*skip))
        #sch.extend([(i,i+skip) for i in range(2*skip-1,n-skip,2*skip)])
        skip = skip // 2
    return sch

def slices_to_sched(sl):
   """Create a concrete schedule from a list of slices
      returned by `psched`.

   """
   sch = []
   for s in sl:
       step = s.step // 2
       sch.extend([(i,i+step) for i in range(s.start, s.stop, s.step)])
   return sch
