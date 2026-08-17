# Arbitrary graphs

The calculator works on any labelled undirected graph, not just molecules. This
makes assembly index available for networks, lattices and other structures that
have no chemical interpretation.

## Building a conforming graph

Three rules, enforced by
{func}`~assemblytheorytools.tools_graph.write_ass_graph_file`:

1. Node indices start at 0 and are contiguous.
2. Every node carries a `color` attribute — any label.
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

A four-node path takes two joins: build a two-edge fragment, then reuse it.

Node colours partition the vertices into types — two nodes with the same colour
are interchangeable, two with different colours are not. Give every node the
same colour for an unlabelled graph, or distinct colours to forbid all
substitution.

## Canonical labelling

Two graphs that are isomorphic but numbered differently must give the same
index. {func}`~assemblytheorytools.tools_graph.canonicalize_node_labels`
renumbers a graph into canonical form, and
`calculate_assembly_index(..., canonicalize=True)` (the default) applies it for
you:

```python
scrambled = att.scramble_node_indices(graph)
att.is_graph_isomorphic(graph, scrambled)                 # True
att.calculate_assembly_index(scrambled)[0]                # 2, same as before
```

{func}`~assemblytheorytools.tools_graph.scramble_node_indices` is useful in
tests to confirm a result does not depend on node numbering.

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
| Densest subgraph | {func}`~assemblytheorytools.tools_graph.top_n_degree_subgraph` |

Note that `write_graphml`/`read_graphml` are the right way to persist these
graphs: NetworkX's own pickling does not guarantee attribute round-tripping
across versions.

## Crystal structures

{func}`~assemblytheorytools.tools_cell.cif_to_nx` reads a CIF file and produces
a conforming graph, going through {func}`~assemblytheorytools.tools_cell.read_cif_file`,
{func}`~assemblytheorytools.tools_cell.find_clusters` and
{func}`~assemblytheorytools.tools_cell.guess_bond_orders`:

```python
graph = att.cif_to_nx("structure.cif")
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
```

Because a crystal is periodic, what gets analysed is a finite cluster cut from
the lattice. {func}`~assemblytheorytools.tools_cell.tile_cell` and
{func}`~assemblytheorytools.tools_cell.tile_cell_shells` control how much of the
lattice is included — the index depends on that choice, so state the tiling
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
