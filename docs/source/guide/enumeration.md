# Neighbourhood enumeration

{mod}`assemblytheorytools.neighborhood_enumeration` generates the graphs that
lie one assembly join away from a given structure. An upward join identifies
(coalesces) compatible vertices from two graphs; a downward join partitions all
of a graph's edges between two connected subgraphs. These are assembly-space
operations, not single-bond edits.

## One step up

{func}`~assemblytheorytools.neighborhood_enumeration.enumerate_up` joins two
graphs by identifying vertices with the same `color` in every valid way:

```python
import networkx as nx
import assemblytheorytools as att


def path_graph(n_edges):
    g = nx.Graph()
    g.add_edges_from([(k, k + 1) for k in range(n_edges)])
    for i in g.nodes:
        g.nodes[i]["color"] = "C"
    for e in g.edges:
        g.edges[e]["color"] = 1
    return g


ups = att.enumerate_up(path_graph(2), path_graph(2))
print(len(ups))   # 19
```

Nineteen joins, but many are isomorphic. Deduplicate to get the distinct
structures:

```python
unique = [ups[0]]
for g in ups[1:]:
    if not any(nx.is_isomorphic(g, u) for u in unique):
        unique.append(g)

print(len(unique))   # 5
```

The inputs may include isolated nodes. A join is possible when the two graphs
have compatible vertex colours and, when valence checking is enabled, enough
remaining valence at the vertices being identified.

`obey_valence=True` (the default) refuses joins that would exceed an atom's
valence, using a standard table. Pass `custom_valence_table` to override it, or
`obey_valence=False` for non-chemical graphs where valence is meaningless.
`allow_dots` mainly affects downward partitions: when it is `False`, the two
parts must also form a connected union. Each individual part remains connected.

## One step down

{func}`~assemblytheorytools.neighborhood_enumeration.enumerate_down` returns the
ways **all** edges can be partitioned into two non-empty connected subgraphs,
as lists of edge partitions:

```python
att.enumerate_down(path_graph(2))
# [[[(0, 1)], [(1, 2)]]]
```

## The full neighbourhood

{func}`~assemblytheorytools.neighborhood_enumeration.enumerate_neighborhood`
combines both directions over a list of graphs and deduplicates by graph
isomorphism, matching node and edge colours:

```python
result = att.enumerate_neighborhood(
    [att.remove_hydrogen_from_graph(att.smi_to_nx("CCO"))])

print(sorted(result))
# ['N_graphs', 'down_jos', 'input_graphs', 'up_jos']
print(len(result["N_graphs"]))   # 8
```

The returned dictionary holds:

`input_graphs`
: The graphs that were passed in.

`N_graphs`
: The neighbourhood — every distinct graph one assembly join away.

`up_jos` and `down_jos`
: Index triples recording each join. A down triple `(n1, n2, s)` says
  neighbourhood graphs `n1` and `n2` join to input `s`; an up triple
  `(s1, s2, n)` says inputs `s1` and `s2` join to neighbour `n`.

Because deduplication is by isomorphism rather than node numbering, the result
does not depend on how the inputs were labelled — passing
{func}`~assemblytheorytools.tools_graph.scramble_node_indices` versions of the
same graphs gives an equivalent answer.

## Cost

The neighbourhood grows quickly with graph size, and the isomorphism check is
pairwise against everything found so far. Strip hydrogens first
({func}`~assemblytheorytools.tools_graph.remove_hydrogen_from_graph`) and start
with small structures; `examples/advanced/forward_enumeration/` shows how to
drive this over successive generations to explore an assembly space.

## See also

* {doc}`../api/neighborhood_enumeration` — the enumeration functions.
* {doc}`reassembly` — generating molecules from fragments with reaction templates.
* {doc}`../api/tools_graph` — isomorphism and relabelling helpers.
