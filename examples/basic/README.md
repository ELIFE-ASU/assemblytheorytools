# Basic examples

These small scripts demonstrate one ATT feature at a time. Run them from this
directory so generated plots and calculator artifacts stay together:

```bash
cd examples/basic
python 1_simple_molecule_example.py
```

## `1_simple_molecule_example.py`

Calculates benzene's molecular assembly index and prints its virtual objects.
This is the shortest end-to-end molecular example.

## `2_simple_string_example.py`

Calculates the undirected string assembly index, virtual objects, and pathway
for `abracadabra` through the molecular-graph mode.

## `3_simple_arb_graph.py`

Builds a labelled NetworkX graph, draws it, and calculates its assembly index.
It shows the required node and edge `color` attributes for non-molecular input.

## `4_joint_pathway.py`

Compares glycine and alanine with
`att.calculate_assembly_index_similarity`, Bertz complexity, and Tanimoto
similarity. It then calculates their joint assembly pathway and saves molecule-
and graph-style pathway plots. The optional metro renderer runs only on Linux.

## `5_joint_undirected_string_example.py`

Calculates the joint undirected assembly index of `abracadabra` and `abra` via
the molecular-graph string mode, sharing repeated substrings across the inputs.

## `6_parallel_calculations.py`

Calculates hydrogen-stripped assembly indices for a batch of SMILES strings
with `att.calculate_assembly_index_parallel`.

## `7_pathway_vis.py`

Plots the glycine/alanine joint pathway from both RDKit-molecule and NetworkX
inputs. The optional metro renderer runs only on Linux.

## `8_recursive_ma_example.py`

Builds a synthetic MS/MS fragmentation tree and estimates a distribution of
molecular assembly values with `att.MAEstimator`.

## `9_rust_backend.py`

Exercises the Rust-backed index, minimum-depth, graph-inspection, and search
APIs. It reconstructs and plots a minimum pathway when the installed
`assembly-theory` release supports pathway output.
