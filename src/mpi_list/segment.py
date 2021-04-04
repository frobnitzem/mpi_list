def even_spread(M, N):
    """Return a list of target sizes for an even spread.

    Output sizes are either M//N or M//N+1

    Args:
        M: number of elements
        N: number of partitons

     Returns:
        target_sizes : [int]
              len(target_sizes) == N
              sum(target_sizes) == M

    """
    if N == 0:
        assert M == 0
        return []
    tgt = [ M//N ]*N
    for i in range(M%N):
        tgt[i] += 1
    return tgt

def cumsum(blks):
    csum = [0]
    for i in range(len(blks)):
        csum.append(csum[i]+blks[i])
    return csum

class Cxn:
    def __init__(self, src, dst, s0,s1, d0,d1):
        self.src = src
        self.dst = dst
        self.s0, self.s1 = s0, s1
        self.d0, self.d1 = d0, d1
    def __str__(self):
        return f"src[{self.src}][{self.s0}:{self.s1}] ~> dst[{self.dst}][{self.d0}:{self.d1}]"
    def __repr__(self):
        return f"Cxn({self.src},{self.dst},{self.s0},{self.s1},{self.d0},{self.d1})"

def segments(src, dst):
    """List out corresponding segments of `src` and `dst`.

    Note:
        src[0] == 0
        dst[0] == 0
        src[-1] == dst[-1]

    Args:
        src: [int] ascending sequence of starting offsets
        dst: [int] ascending sequence of starting offsets

    Returns:
        [Cxn]

    """
    assert src[0] == 0 and dst[0] == 0
    assert src[-1] == dst[-1], f"Input and output sizes ({src[-1]} and {dst[-1]}) don't match."
    ans = []
    idx = 0  # current global index
    i, j = 1,1 # next blk of src, dst to check
    while i < len(src) and j < len(dst):
        end = min(src[i], dst[j])

        if end-idx > 0:
            ans.append( Cxn(i-1,j-1,
                            idx-src[i-1],end-src[i-1],
                            idx-dst[j-1],end-dst[j-1])
                      )
        if end == src[i]:
            i += 1
        if end == dst[j]:
            j += 1
        idx = end

    return ans

def segments_e(blks, N):
    # Compute segments for mapping N even groups
    # (see segments and even_spread)
    #
    oblk = even_spread(sum(i for i in blks), N)
    return segments(cumsum(blks), cumsum(oblk))

if __name__=="__main__":
    M = 200
    a = even_spread(M, 6)
    b = even_spread(M, 9)
    print(a)
    print(b)
    ans = segments(cumsum(a), cumsum(b))
    for g in ans:
        print(g)

    print()
    ans = segments(cumsum(b), cumsum(a))
    for g in ans:
        print(g)
