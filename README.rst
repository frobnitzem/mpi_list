========
mpi list
========

This package implements the `DFM` class.

The `DFM` is a useful abstraction for working with
lists distributed over a set of MPI ranks.
The acronym stands for distributed free monoid,
which is just a fancy way to say it's a list.

If you're familiar with spark, it's like an RDD,
but only holds a list.

Quick Start
===========

.. code-block::

    from mpi_list import Context, DFM

    C = Context() # calls MPI_Init via mpi4py

    # After each of the three lines below:
    #  1. each rank now has 1000//C.procs consecutive numbers
    #  2. each rank now has a list of strings
    #  3. only numbers containing a '2' remain
    dfm = C . iterates(1000) \
            . map(lambda i: f"String {i}") \
            . filter(lambda s: '2' in s)

    if C.rank == 0:
        # Caution! Uncommenting this will deadlock your program.
        # Collective calls must be called by all ranks!
        #print( dfm . head(10) )
        pass

    # This is OK, since all ranks now have 'ans'
    ans = dfm.head(10)
    if C.rank == 0:
        print( ans )

    ans = dfm . filter(lambda s: len(s) <= len("String nn")) \
              . collect()
    if ans is not None: # only rank 0 gets "collect"
        print( ans )

Launch your program with `mpirun python my_prog.py`.

If you're using a supercomputer, consider installing
[spindle](https://computing.llnl.gov/projects/spindle/software)
and then use `spindle mpirun python my_prog.py`.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
