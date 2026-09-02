# Miscellaneous examples

Standalone scripts that do not fit the other advanced categories. Each one runs on its own.

## `CFG_string_comparison.py`

Compares the exact string assembly index against the approximate context-free-grammar method from `assemblycfg`, and
plots the two against each other. Use this to judge how much accuracy the fast approximate route costs on your kind of
input before relying on it at scale.

## `circle_assembly_plot_example.py`

Demonstrates `att.plot_assembly_circle`, which arranges objects in concentric circles by assembly index — the innermost
ring holds the simplest objects, the outermost the most complex. It takes a square adjacency matrix and one assembly
index per node; where `adj_matrix[i, j] >= 1`, an arrow is drawn from node `i` to node `j`. This is the clearest way to
show many objects and their relationships in one figure.

## `figure5.py`

Reproduces Figure 5 of the ATT paper. Builds N-bit adder circuits by stitching together N copies of a full-adder graph,
then calculates their assembly indices. Because each adder is a literal repetition of the same sub-circuit, this is a
clean demonstration of how the index rewards reuse: the index grows far more slowly than the circuit size.

## `metabolic_pathway.py`

Constructs a graph of a metabolic pathway from its metabolites and reaction connections, and calculates assembly
properties over it. An example of applying the machinery to a biological network rather than a single molecule.

## `rna_string.py`

Treats RNA sequences as strings and compares their fast CFG/RePair approximate
assembly-index upper bounds with those of random strings drawn from the same
nucleotide pool. These are not exact assembly indices; the gap is a null-model
comparison for strings of the same length and composition.
