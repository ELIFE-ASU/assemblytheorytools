# Neighbourhood enumeration

{mod}`assemblytheorytools.neighborhood_enumeration` generates the graphs that
lie one edit away from a given structure — the structures reachable by adding or
removing a single bond. This is how the local assembly landscape around an
object is explored.

## One step up

{func}`~assemblytheorytools.neighborhood_enumeration.enumerate_up` joins two
graphs in every way a single new bond allows:

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

Both input graphs need at least one edge — joining against a single isolated
node yields nothing, since there is no bond to form.

`obey_valence=True` (the default) refuses joins that would exceed an atom's
valence, using a standard table. Pass `custom_valence_table` to override it, or
`obey_valence=False` for non-chemical graphs where valence is meaningless.
`allow_dots=True` permits disconnected results.

## One step down

{func}`~assemblytheorytools.neighborhood_enumeration.enumerate_down` returns the
ways a graph can be split by removing a bond, as lists of edge partitions:

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
: The neighbourhood — every distinct graph one edit away.

`up_jos` and `down_jos`
: The joining operations that produce each neighbour, as index tuples
  identifying which inputs combined and how.

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
