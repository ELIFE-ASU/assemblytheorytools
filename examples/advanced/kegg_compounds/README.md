In this example the code demonstrates how to calculate
the assembly of KEGG compounds using the CBR-db,
a curated biochemical database that integrates and refines data
from KEGG and ATLAS databases to support precise analyses of
biochemical reaction data. Here, there is an example of how to run the
assembly calculation for a list of KEGG compound IDs.
Furthermore, there is an example of how to run them in parallel
on a HPC cluster to speed up the calculations.

`kegg_c_complexity_matrix.py` computes the hydrogen-stripped assembly index
of KEGG compounds alongside several other molecular complexity scores
(Bertz, Böttcher, Wiener, Balaban, spacial score, Proudfoot and MC1), then
plots every pair of scores against each other to compare how they relate.