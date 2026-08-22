# Reassembly

In this example, the reassembly of molecules is calculated using the Reassembler algorithm.
The Reassembler reconstructs molecules from their assembly building blocks by applying SMARTS reaction templates,
allowing for the exploration of chemical space and the generation of novel compounds.

Unlike the assembly index — which asks for the *shortest* construction of one known target — reassembly asks what is
*reachable* from a starting pool. It enumerates a space rather than optimising within one.

## `reassemble_aa.py`

Reassembles four amino acids (glycine, alanine, serine and proline). Converts each SMILES to an RDKit molecule with
explicit hydrogens, reports their molecular weights, and runs the reassembly to see what the shared building blocks can
produce.

## `reassemble_vo_space.py`

Explores the space spanned by the virtual objects of a single molecule. Starting from glycine with explicit hydrogens,
it combines the molecules into one superstructure with `att.combine_mols` and reassembles from there. This is the more
interesting variant: the starting pool is exactly the set of intermediates the assembly pathway already built, so the
result shows what *else* those same building blocks could have made.

See the [reassembly guide](https://assemblytheorytools.readthedocs.io/en/latest/guide/reassembly.html) for the class API
these scripts drive.
