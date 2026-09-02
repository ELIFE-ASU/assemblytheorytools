# Forward enumeration

This example shows one forward assembly join. Upward enumeration takes two
graphs and generates every valid result of identifying compatible vertices
between them; it does not add a single bond.

`enum_forward.py` first calculates the joint pathway of glycine, alanine, serine
and proline. It selects the `CN` and `C=O` virtual objects by their SMILES (the
returned list order is not stable), calls `att.enumerate_up` once, and draws the
new objects beneath those two inputs.

Two things to watch:

- `enumerate_up` can contain isomorphic duplicates. Use
  `att.enumerate_neighborhood` when you need a colour-aware deduplicated
  neighbourhood and join-index records.
- Repeating the operation over successive generations grows combinatorially,
  so start from small hydrogen-stripped fragments.

See the [neighbourhood enumeration guide](../../../docs/source/guide/enumeration.md)
for the underlying functions.
