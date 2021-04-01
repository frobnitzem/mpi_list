# Implement nearest fill rule

# Return the fan-in / fan-out communication strategy for achieving delta == 0
# 
# \param  delta = items - target
# \return [ [(src,dst,count)] ] = sending schedule at each level
def fill(delta):
    N = len(delta)
    assert sum(i for i in delta) == 0

    lev = [ delta ] # surplus items at each level
    sends = []

    level = 0 # target level for sends
    skip = 1  # skip for addressing within target level
    while N > 1:
        sno = []
        nlev = []

        odd = len(lev[-1]) % 2
        for i in range(0,len(lev[-1])-odd, 2):
            c0 = lev[-1][i]
            c1 = lev[-1][i+1]
            if c1 > 0: # surplus items
                sno.append( ((i+1)*skip,i*skip,c1) )
            nlev.append( c0+c1 )

        if odd: # copy-through
            nlev.append(lev[-1][-1])

        if len(sno) > 0:
            sends.append(sno)
        lev.append(nlev) # surplus/deficit at ea. level
        N = (N+1)//2
        skip *= 2
        level += 1
        assert N == len(nlev)
        assert level+1 == len(lev)

    assert len(lev[-1]) == 1 and lev[-1][0] == 0

    # N = 1
    while level > 0:
        N *= 2
        skip = skip//2 # skip for addressing within target level
        level -= 1 # target level for sends
        sno = []

        odd = len(lev[level]) % 2
        for i in range(0, len(lev[level])-odd, 2):
            c0 = lev[level][i]
            c1 = lev[level][i+1]
            if c1 < 0:
                sno.append( (i*skip, (i+1)*skip, -c1) )

        if len(sno) > 0:
            sends.append(sno)

    return sends

