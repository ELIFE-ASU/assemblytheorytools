# Forward enumeration

This example shows how assembly forward enumeration can be performed.
Forward enumeration systematically explores the assembly space constructed from a pool of objects: starting from a set
of starting structures, it generates everything reachable in one joining step, then repeats.

`enum_forward.py` starts from four amino acids — glycine, alanine, serine and proline — and drives
`att.enumerate_neighborhood` over them to map the space around that pool.

Two things to watch:

- The space grows combinatorially with each generation, so start small and strip hydrogens
  (`att.remove_hydrogen_from_graph`) before enumerating.
- Deduplication is by graph isomorphism, which is a pairwise check against everything found so far. This, rather than
  the generation step itself, is what dominates the runtime as the pool grows.

See the [neighbourhood enumeration guide](https://assemblytheorytools.readthedocs.io/en/latest/guide/enumeration.html)
for the underlying functions.
