# Arbitrary graphs

The calculator works on labelled simple undirected graphs. This makes assembly
index available for networks, lattices and other structures with no chemical
interpretation.

## Building a conforming graph

{func}`~assemblytheorytools.tools_graph.write_ass_graph_file` validates the
calculator's input requirements before writing a file:

1. Node indices start at 0 and are contiguous.
2. Every node carries a `color` attribute — a nonempty string without whitespace.
3. Every edge carries an integer `color` attribute from 1 through 32767.
4. The graph has at most 32767 vertices and has no directed edges, parallel
   edges or self loops.

Missing or invalid attributes raise `ValueError` with the affected node or
edge. The graph name must fit on one line. NumPy integer colours are accepted;
strings, floats and booleans are rejected as edge colours.

```python
import networkx as nx
import assemblytheorytools as att

graph = nx.Graph()
graph.add_node(0, color="0")
graph.add_node(1, color="1")
graph.add_node(2, color="2")
graph.add_node(3, color="3")

graph.add_edge(0, 1, color=1)
graph.add_edge(1, 2, color=1)
graph.add_edge(2, 3, color=1)

ai, virt_obj, pathway = att.calculate_assembly_index(graph)
print(ai)   # 2
```

A four-node path takes two joins: build a two-edge fragment, then join the
remaining elementary edge. Because the node colours are distinct, this example
does not rely on repeated interchangeable subgraphs.

Node colours partition the vertices into types — two nodes with the same colour
are interchangeable, two with different colours are not. Give every node the
same colour for an unlabelled graph, or distinct colours to forbid all
substitution.

## Node-label normalisation

Two graphs that are isomorphic but numbered differently must give the same
index. {func}`~assemblytheorytools.tools_graph.canonicalize_node_labels`
renumbers the nodes, in their current iteration order, to contiguous integers
starting at zero, and
`calculate_assembly_index(..., canonicalize=True)` (the default) applies it for
you. Despite the historical function name, this is input normalisation rather
than a canonical graph-isomorphism labelling:

```python
scrambled = att.scramble_node_indices(graph)
att.is_graph_isomorphic(graph, scrambled)                 # True
att.calculate_assembly_index(scrambled)[0]                # 2, same as before
```

{func}`~assemblytheorytools.tools_graph.scramble_node_indices` is useful in
tests to confirm a result does not depend on node numbering.

{func}`~assemblytheorytools.tools_graph.is_graph_isomorphic` compares topology
only. When node or edge colours matter, call `networkx.is_isomorphic` with
categorical `node_match` and `edge_match` functions, as the neighbourhood
enumerator does internally.

## Graph utilities

{mod}`assemblytheorytools.tools_graph` carries the operations needed to prepare
and dissect these graphs:

| Task | Function |
| --- | --- |
| Drop hydrogens | {func}`~assemblytheorytools.tools_graph.remove_hydrogen_from_graph` |
| Split a disconnected graph | {func}`~assemblytheorytools.tools_graph.get_disconnected_subgraphs` |
| Join two graphs | {func}`~assemblytheorytools.tools_graph.join_graphs`, {func}`~assemblytheorytools.tools_graph.compose_graphs` |
| Isomorphism check | {func}`~assemblytheorytools.tools_graph.is_graph_isomorphic` |
| Read/write GraphML | {func}`~assemblytheorytools.tools_graph.write_graphml`, {func}`~assemblytheorytools.tools_graph.read_graphml` |

Note that `write_graphml`/`read_graphml` are the right way to persist these
graphs: NetworkX's own pickling does not guarantee attribute round-tripping
across versions.

## Crystal structures

{func}`~assemblytheorytools.tools_cell.cif_to_nx` reads the primitive cell
from a CIF file with {func}`~assemblytheorytools.tools_cell.read_cif_file` and
builds a graph with {func}`~assemblytheorytools.tools_cell.cell_to_nx`, which
also accepts any periodic ASE `Atoms` object:

```python
graph = att.cif_to_nx("structure.cif")            # wrap-around supercell graph
graph.graph["reps"]                                 # e.g. (2, 1, 2)
ai, virt_obj, pathway = att.calculate_assembly_index(graph)

atoms = att.read_cif_file("structure.cif")
graph = att.cell_to_nx(atoms, reps=(2, 2, 2), cutoff_mult=1.2)
```

The nodes are the atoms of `reps` copies of the cell and the edges are the
bonds found under the supercell's periodic boundaries, so every atom keeps its
full coordination and the graph has no surface. `reps=None` (the default)
picks the smallest tiling that guarantees a simple graph; an explicit `reps`
that would need a self-loop (an atom bonded to its own image) or a parallel
edge (a pair bonded through two images) raises `ValueError`. Each node records
`cell_index` (the atom in the input cell) and `image` (the integer cell
shift), and the graph attributes `reps`, `cutoff_mult`, `periodic`, `cell` and
`pbc` record the model so that it can be reported with the result; `cif_to_nx`
additionally stores the file path in `source`. `atoms_to_nx` is for molecules:
it ignores the cell and warns when given a periodic `Atoms` object.

`cif_to_nx(..., periodic=False)` instead returns a finite open cluster: the
central cell of a `reps` tiling (default `(3, 3, 3)`) plus its first bonded
shell, with a `shell` node attribute (0 for the central cell, 1 for the
shell). Surface atoms of that cluster are under-coordinated, which is why the
periodic graph is the default. {func}`~assemblytheorytools.tools_cell.tile_cell`
and {func}`~assemblytheorytools.tools_cell.tile_cell_shells` expose the same
tiling as `Atoms` objects.

Two atoms are bonded when their distance is below `cutoff_mult` times the sum
of their covalent radii (ASE's natural cutoffs); the same criterion drives
{func}`~assemblytheorytools.tools_cell.get_bonding_config`,
{func}`~assemblytheorytools.tools_cell.find_clusters` and the tiling
functions. Every edge is assigned bond order `1`; `cif_to_nx` does not call
`guess_bond_orders`, whose molecular valence model does not suit crystals.

Keep the following in mind when interpreting results:

* The graph is the bond graph of a finite torus, so the assembly index is a
  property of the chosen `reps` and `cutoff_mult`. Report both.
* Covalent radii are a crude criterion for ionic and metallic contacts and
  can produce very dense graphs for metal-rich minerals.
* `read_cif_file` warns when sites have fractional or mixed occupancy. ASE
  keeps every such site with its majority species, so split sites overlap
  and inflate coordination numbers.
* Molecular and ionic crystals give disconnected covalent graphs; with the
  default `joint_corr=True`, `calculate_assembly_index` subtracts one less than
  the number of components that contain at least one bond. Isolated atoms — a
  bare counter-ion, for instance — add no joining operations and are not
  counted.
* The Rust backend refuses graphs with more than 999 atoms or bonds and the
  C++ calculator more than 32767 vertices; large `reps` reach these limits
  quickly.
* `write_graphml` cannot serialise the tuple metadata; drop `image`, `reps`,
  `cell` and `pbc` before exporting.

The CIF conversion is still experimental and emits a `UserWarning` on every
call.

## Plotting

```python
att.plot_graph(graph)
att.plot_mol_graph(graph)          # molecule-style rendering
att.plot_interactive_graph(graph)  # pyvis; writes interactive_graph.html
                                   # (plus a lib/ folder) into the working directory
```

## See also

* {doc}`../api/tools_graph` — graph conversion and manipulation.
* {doc}`../api/tools_cell` — crystal structures and periodic cells.
* {doc}`../api/tools_plotting` — plotting functions.
* {doc}`enumeration` — exploring the neighbourhood of a graph.
