# Pathways

The third value returned by
{func}`~assemblytheorytools.assembly.calculate_assembly_index` is the assembly
pathway: a {class}`~networkx.DiGraph` whose nodes are virtual objects and whose
edges are joining operations, directed from inputs to output. Elementary parts
have in-degree zero; the target has out-degree zero.

```python
import assemblytheorytools as att

graph = att.smi_to_nx("NCC(=O)O.CC(N)C(=O)O")
ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

print(pathway.number_of_nodes(), pathway.number_of_edges())
```

## Assembly depth

{func}`~assemblytheorytools.construction.assign_levels` annotates each node with
its assembly depth: 0 for an elementary part, otherwise one more than its
deepest predecessor.

:::{warning}
`assign_levels` visits nodes in **insertion order** and reads each
predecessor's level as it goes, so a predecessor must already have been
assigned one. The pathway returned by `calculate_assembly_index` is not in
topological order, and calling `assign_levels` on it directly raises
`KeyError: 'level'`.
:::

Re-order the graph first:

```python
import networkx as nx

ordered = nx.DiGraph()
ordered.add_nodes_from((n, pathway.nodes[n]) for n in nx.topological_sort(pathway))
ordered.add_edges_from(pathway.edges(data=True))

att.assign_levels(ordered)
print(sorted({d["level"] for _, d in ordered.nodes(data=True)}))   # [0, 1, 2, 3, 4]
```

{func}`~assemblytheorytools.construction.get_vos_on_layer` then extracts the
virtual objects at a given depth, and
{func}`~assemblytheorytools.tools_graph.longest_path_length` gives the depth of
the whole pathway.

## Plotting

{func}`~assemblytheorytools.tools_plotting.plot_pathway` is the main entry
point. `plot_type` selects the renderer:

```python
import matplotlib.pyplot as plt

att.plot_pathway(pathway, plot_type="mol")     # molecule structures
att.plot_pathway(pathway, plot_type="graph")   # graph diagrams
plt.show()
```

It returns the `(figure, axes)` pair, so the result can be saved or composed
into a larger figure:

```python
fig, ax = att.plot_pathway(pathway, plot_type="graph")
fig.savefig("pathway.svg")
```

Layout is the hard part of these diagrams; the `layout_style` argument selects
between the crossing-minimisation layouts in
{mod}`assemblytheorytools.tools_plotting`
({func}`~assemblytheorytools.tools_plotting.multipartite_layout_crossmin`,
{func}`~assemblytheorytools.tools_plotting.multipartite_layout_crossmin_long`
and {func}`~assemblytheorytools.tools_plotting.multipartite_layout_sa`, a
simulated-annealing variant). Set `auto_fig_size=True` to size the canvas to the
pathway rather than fixing it up front.

Alternative renderings:

* {func}`~assemblytheorytools.tools_plotting.plot_pathway_mid_arrow` — arrows
  drawn at edge midpoints, which reads better on wide pathways.
* {func}`~assemblytheorytools.tools_plotting.plot_digraph_metro` — metro-map
  style diagram. Writes to a file rather than returning a figure, and is
  Linux-only.
* {func}`~assemblytheorytools.tools_plotting.plot_assembly_circle` — circular
  layout taking an adjacency matrix and per-node indices, used for comparing
  many objects at once.

## Alternative shortest pathways

A shortest pathway is rarely unique.
{func}`~assemblytheorytools.find_other_paths.all_shortest_paths` enumerates the
alternatives for a molecule:

```python
mol = att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
paths = att.all_shortest_paths(mol)
print(len(paths))
```

The count itself is informative: a molecule with many distinct shortest
pathways is less constrained in how it can be assembled than one with a single
route.

## Parsing a saved pathway

When a calculation was run with `save_dir=True`, or its output was archived, the
pathway file can be re-parsed without recomputing:

```python
pathway, vo_list = att.parse_pathway_file("pathway.json", vo_type="smiles")
```

{func}`~assemblytheorytools.construction.parse_pathway_file` accepts
`vo_type="smiles"` or `"graph"` and can return the pathway log as a third value
with `log=True`. It is a thin wrapper over
{class}`~assemblytheorytools.construction.AssemblyConstruction`, which is
available directly for finer control:

```python
from assemblytheorytools.construction import AssemblyConstruction
```

Note the submodule import — `AssemblyConstruction` is not re-exported at the
package root.

For string pathways, use
{func}`~assemblytheorytools.construction.parse_string_pathway_file`.

## See also

* {doc}`../api/construction` — pathway parsing, levelling and conversion.
* {doc}`../api/find_other_paths` — alternative shortest pathways.
* {doc}`../api/tools_plotting` — every plotting function.
