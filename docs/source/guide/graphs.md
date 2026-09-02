# Arbitrary graphs

The calculator works on any labelled undirected graph, not just molecules. This
makes assembly index available for networks, lattices and other structures that
have no chemical interpretation.

## Building a conforming graph

The calculator input requires three rules. The writer type-checks colour
attributes that are present, but does not reliably reject every missing
attribute before invoking the calculator, so validate custom graphs explicitly:

{func}`~assemblytheorytools.tools_graph.write_ass_graph_file`:

1. Node indices start at 0 and are contiguous.
2. Every node carries a `color` attribute — any string label without spaces.
3. Every edge carries a `color` attribute that is an **integer**, starting at 1.

Rule 3 is the one that bites: a string edge colour raises `AssertionError:
Edge color for edge (0, 1) is not an integer.`

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
| Induced subgraph of the highest-degree nodes | {func}`~assemblytheorytools.tools_graph.top_n_degree_subgraph` |

Note that `write_graphml`/`read_graphml` are the right way to persist these
graphs: NetworkX's own pickling does not guarantee attribute round-tripping
across versions.

## Crystal structures

{func}`~assemblytheorytools.tools_cell.cif_to_nx` reads a CIF file and produces
an experimental graph. It reads the cell, expands it with
{func}`~assemblytheorytools.tools_cell.tile_cell`, and infers connectivity with
{func}`~assemblytheorytools.tools_cell.get_bonding_config`:

```python
graph = att.cif_to_nx("structure.cif", reps=(3, 3, 3), cutoff_mult=1.2)
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
```

Because a crystal is periodic, what gets analysed is the finite tiled cell
chosen by `reps`; `cutoff_mult` controls the natural-distance bonding cutoff.
Every inferred edge is currently assigned bond order `1` — `cif_to_nx` does
not call `guess_bond_orders` or prune the tiled graph. The result and its
assembly index therefore depend on both parameters, which should be reported
alongside any result.

## Plotting

```python
att.plot_graph(graph)
att.plot_mol_graph(graph)          # molecule-style rendering
att.plot_interactive_graph(graph)  # pyvis, opens in a browser
```

## See also

* {doc}`../api/tools_graph` — graph conversion and manipulation.
* {doc}`../api/tools_cell` — crystal structures and periodic cells.
* {doc}`../api/tools_plotting` — plotting functions.
* {doc}`enumeration` — exploring the neighbourhood of a graph.
