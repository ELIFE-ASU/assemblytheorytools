"""
Enumeration of the graph neighbourhood of a molecule.

This module generates graphs one assembly join away from given structures.
``enumerate_up`` identifies compatible vertices of two graphs,
``enumerate_down`` partitions a graph's full edge set between two connected
subgraphs, and ``enumerate_neighborhood`` combines both directions. Results are
deduplicated by colour-aware graph isomorphism.
"""

import itertools
import sys
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from rdkit import Chem

from .tools_graph import canonicalize_node_labels

node_match = nx.algorithms.isomorphism.categorical_node_match('color', None)
edge_match = nx.algorithms.isomorphism.categorical_edge_match('color', None)
ptable = Chem.GetPeriodicTable()


def enumerate_neighborhood(
    graphs: List[nx.Graph],
    obey_valence: bool = True,
    allow_dots: bool = True,
    debug: bool = False,
    custom_valence_table: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Generate the neighborhood of input graphs in assembly space.

    Computes the set of graphs that are one assembly joining operation away
    from the input graphs. A down join partitions every edge of an input between
    two connected subgraphs; an up join identifies compatible vertices between
    two inputs. Results are deduplicated by graph isomorphism.

    Parameters
    ----------
    graphs : list of networkx.Graph
        Input graphs to compute the neighborhood for. Assumed to be distinct.
    obey_valence : bool, optional
        If True, enforce valence constraints when creating up joins,
        by default True.
    allow_dots : bool, optional
        If True, allow disconnected graphs in the output, by default True.
    debug : bool, optional
        If True, enable debug mode for additional output, by default False.
    custom_valence_table : dict or None, optional
        Custom valence table mapping atom symbols to valence values.
        Example: custom_valence_table={'P': 3, 'S': 4}
        Atoms absent from the custom table use RDKit default valences.
        Defaults to None.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - "input_graphs" : list of networkx.Graph
            Canonicalized versions of input graphs.
        - "N_graphs" : list of networkx.Graph
            Graphs in the neighborhood, unique up to isomorphism.
        - "down_jos" : set of tuple
            Down join operations as (n1, n2, s) where n1, n2 are indices
            in N_graphs and s is an index in input_graphs.
        - "up_jos" : set of tuple
            Up join operations as (s1, s2, n) where s1, s2 are indices
            in input_graphs and n is an index in N_graphs.

    Warnings
    --------
    Input graphs are assumed to be distinct. Duplicate graphs may lead
    to redundant computations.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> result = att.enumerate_neighborhood(
    ...     [att.remove_hydrogen_from_graph(att.smi_to_nx("CCO"))])
    >>> sorted(result)
    ['N_graphs', 'down_jos', 'input_graphs', 'up_jos']
    >>> len(result["N_graphs"])
    8

    ``N_graphs`` is the neighbourhood -- every distinct graph one assembly join
    away.
    ``up_jos`` and ``down_jos`` are the joining operations that produce each
    neighbour. Deduplication is by graph isomorphism rather than node
    numbering, so relabelling the inputs gives an equivalent answer.

    Strip hydrogens first: the neighbourhood grows quickly with graph size.
    Isomorphism checks compare candidates sharing the same graph hash.
    """

    graphs = [canonicalize_node_labels(graph) for graph in graphs]
    down_partitions = [enumerate_down(graph, allow_dots=allow_dots) for graph in graphs]
    up_graphs = {
        (first, second): enumerate_up(
            graphs[first], graphs[second],
            obey_valence=obey_valence,
            allow_dots=allow_dots,
            debug=debug,
            custom_valence_table=custom_valence_table,
        )
        for first, second in itertools.combinations_with_replacement(range(len(graphs)), 2)
    }
    neighbors = []
    buckets = {}

    def neighbor_index(graph: nx.Graph) -> int:
        """Find or store a representative, using hashes to narrow comparisons."""
        graph_hash = weisfeiler_lehman_graph_hash(
            graph, node_attr="color", edge_attr="color"
        )
        candidates = buckets.setdefault(graph_hash, [])
        for index in candidates:
            if nx.is_isomorphic(
                graph, neighbors[index], node_match=node_match, edge_match=edge_match
            ):
                return index

        index = len(neighbors)
        neighbors.append(graph.copy())
        candidates.append(index)
        return index

    down_jos = set()
    for source, graph in enumerate(graphs):
        for partition in down_partitions[source]:
            parts = [graph.edge_subgraph(edges) for edges in partition]
            if not allow_dots and not all(nx.is_connected(part) for part in parts):
                print(
                    "Warning: A disconnected graph was found in a down join operation. "
                    "This should never happen. Please report this bug."
                )
                continue
            first, second = sorted(neighbor_index(part) for part in parts)
            down_jos.add((first, second, source))

    up_jos = set()
    for (first, second), joined_graphs in up_graphs.items():
        for graph in joined_graphs:
            if not nx.is_connected(graph):
                print(
                    "Warning: A disconnected graph was found in an up join operation. "
                    "This should never happen. Please report this bug."
                )
                sys.exit()
            up_jos.add((first, second, neighbor_index(graph)))

    return {
        "input_graphs": graphs,
        "N_graphs": neighbors,
        "down_jos": down_jos,
        "up_jos": up_jos,
    }


def enumerate_down(
    graph: nx.Graph, allow_dots: bool = True,
) -> List[List[List[Tuple[Any, Any]]]]:
    """
    Enumerate all edge partitions of a graph into two connected subgraphs.

    Computes the power set of edges and filters for edge sets where both
    the subgraph induced by these edges and its complement are connected.
    Uses brute force enumeration and may be slow for large graphs.

    Parameters
    ----------
    graph : networkx.Graph
        Input connected graph to partition.
    allow_dots : bool, optional
        If True, allows disconnected unions of partitions, by default True.

    Returns
    -------
    list of list
        List of partition pairs, where each element is [subset, complement].
        Each subset is a list of edges forming a connected subgraph, and
        complement is the list of remaining edges also forming a connected
        subgraph.

    Notes
    -----
    This is a brute-force method that enumerates all possible edge subsets.
    Time complexity is ``O(2^|E|)``, where ``|E|`` is the number of edges.

    Examples
    --------
    >>> import networkx as nx
    >>> import assemblytheorytools as att
    >>> graph = nx.Graph()
    >>> graph.add_edges_from([(0, 1), (1, 2)])
    >>> for i in graph.nodes:
    ...     graph.nodes[i]["color"] = "C"
    >>> for e in graph.edges:
    ...     graph.edges[e]["color"] = 1
    >>> att.enumerate_down(graph)
    [[[(0, 1)], [(1, 2)]]]

    The two-edge path splits one way: into its two single edges.
    """
    partition_pairs = []
    edges = list(graph.edges())
    if not edges:
        return partition_pairs

    # Fix one edge in the first part to avoid enumerating both orientations.
    anchor_edge = tuple(sorted(edges[0]))
    for size in range(len(edges) - 1):
        for selected in itertools.combinations(edges[1:], size):
            subset = [tuple(sorted(edge)) for edge in selected] + [anchor_edge]
            subgraph = graph.edge_subgraph(subset)
            if not nx.is_connected(subgraph):
                continue

            selected_edges = set(subset)
            complement = [
                edge for edge in edges if tuple(sorted(edge)) not in selected_edges
            ]
            other = graph.edge_subgraph(complement)
            if not nx.is_connected(other):
                continue
            if not allow_dots and not nx.is_connected(nx.compose(subgraph, other)):
                continue
            partition_pairs.append([subset, complement])
    return partition_pairs


def get_valence(
    atom_symbol: str,
    ptable: Chem.rdchem.PeriodicTable = ptable,
    custom_valence_table: Optional[Dict[str, int]] = None,
) -> int:
    """
    Get the default valence of an atom based on its chemical symbol.

    Retrieves the valence from a custom table if provided, otherwise uses
    RDKit's periodic table default valence values.

    Parameters
    ----------
    atom_symbol : str
        The chemical symbol of the atom (e.g., "C" for carbon, "O" for oxygen).
    ptable : rdkit.Chem.rdchem.PeriodicTable, optional
        RDKit PeriodicTable object for looking up default valences.
        Defaults to the global `ptable` instance.
    custom_valence_table : dict or None, optional
        Custom valence mapping {atom_symbol: valence}. If provided and
        contains the atom symbol, this value takes precedence, by default None.

    Returns
    -------
    int
        The default or custom valence of the atom.

    Raises
    ------
    ValueError
        If the atom symbol is invalid or not recognized by the periodic table.
    """
    if custom_valence_table and atom_symbol in custom_valence_table:
        return custom_valence_table[atom_symbol]
    return ptable.GetDefaultValence(atom_symbol)


def enumerate_up(
    graph1: nx.Graph,
    graph2: nx.Graph,
    obey_valence: bool = True,
    allow_dots: bool = True,
    debug: bool = False,
    custom_valence_table: Optional[Dict[str, int]] = None,
) -> List[nx.Graph]:
    """
    Enumerate graphs formed by joining two input graphs.

    Computes all possible graphs that can be created by identifying (merging)
    vertices of the same color between two input graphs. Optionally enforces
    chemical valence constraints and filters for connected graphs.

    Parameters
    ----------
    graph1 : networkx.Graph
        First input graph with 'color' node attributes.
    graph2 : networkx.Graph
        Second input graph with 'color' node attributes.
    obey_valence : bool, optional
        If True, enforces valence rules for atoms (prevents overbonding),
        by default True.
    allow_dots : bool, optional
        If True, allows disconnected output graphs, by default True.
    debug : bool, optional
        If True, prints detailed debugging information, by default False.
    custom_valence_table : dict or None, optional
        Custom valence table mapping atom symbols to valence values.
        Example: custom_valence_table={'P': 3, 'S': 4}
        Atoms absent from the custom table use RDKit default valences.
        Defaults to None.

    Returns
    -------
    list of networkx.Graph
        List of graphs formed by valid vertex identifications between
        graph1 and graph2.

    Raises
    ------
    ValueError
        If nodes lack 'color' attributes when obey_valence is True.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Enumerate vertex colors shared by both graphs.
    2. For each color, enumerate valid vertex identification combinations.
    3. Filter combinations that produce multi-edges or violate valence rules.
    4. Compute the outer product of combinations across all colors.
    5. Filter out combinations that create multi-edges.
    6. Generate output graphs from the valid vertex identifications.

    An input may be an isolated node. A join is possible when the inputs share
    a compatible vertex colour and, when enabled, the identified vertices have
    sufficient remaining valence.

    Examples
    --------
    >>> import networkx as nx
    >>> import assemblytheorytools as att
    >>> def path_graph(n_edges):
    ...     g = nx.Graph()
    ...     g.add_edges_from([(k, k + 1) for k in range(n_edges)])
    ...     for i in g.nodes:
    ...         g.nodes[i]["color"] = "C"
    ...     for e in g.edges:
    ...         g.edges[e]["color"] = 1
    ...     return g
    >>> ups = att.enumerate_up(path_graph(2), path_graph(2))
    >>> len(ups)
    19

    Many of those are isomorphic; deduplicate for the distinct structures:

    >>> unique = [ups[0]]
    >>> for g in ups[1:]:
    ...     if not any(nx.is_isomorphic(g, u) for u in unique):
    ...         unique.append(g)
    >>> len(unique)
    5
    """

    if obey_valence:
        if debug:
            print("Checking valence budgets...")
        valence_budgets = [np.zeros(len(graph)) for graph in (graph1, graph2)]
        for g_idx, graph in enumerate((graph1, graph2)):
            budget = valence_budgets[g_idx]
            for node, data in graph.nodes(data=True):
                if 'color' not in data:
                    raise ValueError(
                        f"Node {node} does not have a color attribute. "
                        "Please add a color attribute to the nodes."
                    )
                budget[node] = get_valence(
                    data['color'], custom_valence_table=custom_valence_table
                )
                if debug:
                    print(
                        f"Node {node} in graph {g_idx + 1} has color {data['color']} "
                        f"and valence budget {budget[node]}"
                    )
                for edge in graph.edges(node):
                    budget[node] -= graph.edges[edge]['color']
                if budget[node] < 0:
                    print(
                        f"Warning: Node {node} in graph {g_idx + 1} is overbonded. "
                        "Skipping this graph."
                    )
                    return []
        if debug:
            print(f"Valence budgets for graph1: {valence_budgets[0]}")
            print(f"Valence budgets for graph2: {valence_budgets[1]}")
        if any(sum(budget) == 0 for budget in valence_budgets):
            if debug:
                print(
                    "No valence budget left in (at least) one of the graphs. "
                    "Returning empty list."
                )
            return []

    colors1 = {data['color'] for _, data in graph1.nodes(data=True)}
    colors2 = {data['color'] for _, data in graph2.nodes(data=True)}

    combinations = {}
    for color in colors1 & colors2:
        nodes1 = [
            node for node in graph1
            if graph1.nodes[node]['color'] == color
            and (not obey_valence or valence_budgets[0][node] > 0)
        ]
        nodes2 = [
            node for node in graph2
            if graph2.nodes[node]['color'] == color
            and (not obey_valence or valence_budgets[1][node] > 0)
        ]
        valid_identifications = {
            (node1, node2)
            for node1, node2 in itertools.product(nodes1, nodes2)
            if not obey_valence
            or get_valence(color, custom_valence_table=custom_valence_table)
            - valence_budgets[0][node1] <= valence_budgets[1][node2]
        }

        if debug:
            print(f"Number of valid identifications = {len(valid_identifications)}")

        # Parallel edges arise when adjacent pairs in both graphs are merged.
        g1_check_edges = {
            tuple(sorted(edge)) for edge in itertools.combinations(nodes1, 2)
            if graph1.has_edge(*edge)
        }
        g2_check_edges = {
            tuple(sorted(edge)) for edge in itertools.combinations(nodes2, 2)
            if graph2.has_edge(*edge)
        }

        valid_color_maps = set()
        if valid_identifications:
            for k in range(min(len(nodes1), len(nodes2)) + 1):
                # Fix the first tuple's order so each matching is visited once.
                for node1_subset in itertools.combinations(nodes1, k):
                    for node2_perm in itertools.permutations(nodes2, k):
                        candidate = frozenset(zip(node1_subset, node2_perm))
                        if (
                            candidate <= valid_identifications
                            and conditional_check_multi_edge_generation(
                                candidate, g1_check_edges, g2_check_edges
                            )
                        ):
                            valid_color_maps.add(candidate)
        combinations[color] = valid_color_maps

    valid_maps = map_outer_product(combinations, graph1, graph2)

    if debug:
        print("Combination keys: ", combinations.keys())
        print("Combination items: ", list(combinations.values()))
        print(f"Number of valid color-specific maps = {sum(map(len, combinations.values()))}")
        print(f"Number of valid maps = {len(valid_maps)}")

    output_graphs = []
    for vertex_map in valid_maps:
        joined = map_application(vertex_map, graph1, graph2)
        if not nx.is_connected(joined):
            print(
                "Warning: A disconnected graph was formed in an up join operation. "
                "This should never happen. Please report this bug."
            )
            for name, graph in (
                ("Graph1", graph1), ("Graph2", graph2), ("Joined graph", joined)
            ):
                print(
                    f"{name} has {graph.number_of_nodes()} nodes "
                    f"and {graph.number_of_edges()} edges."
                )
                print(f"{name} nodes data: {graph.nodes(data=True)}")
                print(f"{name} edges data: {graph.edges(data=True)}")
            print(f"Vertex identification map: {vertex_map}")
            sys.exit()
        output_graphs.append(joined)
    return output_graphs


def map_outer_product(
    combinations: Dict[str, Set[FrozenSet[Tuple[int, int]]]],
    graph1: nx.Graph,
    graph2: nx.Graph,
) -> List[Set[Tuple[int, int]]]:
    """
    Combine color-specific maps into valid vertex identification maps.

    Enumerates the Cartesian product of valid color-specific maps and filters
    out those that would create multi-edges in the joined graph.

    Parameters
    ----------
    combinations : dict
        Dictionary mapping colors to sets of valid vertex identification maps
        for that color. Format: {color: {frozenset((node1, node2), ...)}}.
    graph1 : networkx.Graph
        First input graph with 'color' node attributes.
    graph2 : networkx.Graph
        Second input graph with 'color' node attributes.

    Returns
    -------
    list of set or set of frozenset
        List of valid complete vertex identification maps, where each map
        is a set of (graph1_node, graph2_node) tuples.

    Notes
    -----
    Special case: If only one color exists, returns the valid maps for that
    color directly without computing the outer product. The empty map is
    removed from that set in place.
    """

    # Preserve the single-color fast path's in-place removal and set return.
    if len(combinations) == 1:
        valid_maps = next(iter(combinations.values()))
        valid_maps.discard(frozenset())
        return valid_maps

    nonempty = {color: maps for color, maps in combinations.items() if maps}

    def cross_color_edges(graph: nx.Graph) -> Set[Tuple[Any, Any]]:
        """Within-color edges have already been checked for each partial map."""
        return {
            tuple(sorted((u, v))) for u, v in graph.edges()
            if graph.nodes[u]['color'] != graph.nodes[v]['color']
            and graph.nodes[u]['color'] in nonempty
            and graph.nodes[v]['color'] in nonempty
        }

    g1_check_edges = cross_color_edges(graph1)
    g2_check_edges = cross_color_edges(graph2)
    valid_maps = []
    for color_maps in itertools.product(*nonempty.values()):
        candidate = set(itertools.chain.from_iterable(color_maps))
        if candidate and conditional_check_multi_edge_generation(
            candidate, g1_check_edges, g2_check_edges
        ):
            valid_maps.append(candidate)

    return valid_maps


def conditional_check_multi_edge_generation(
    candidate_map: Union[Set, FrozenSet],
    g1_check_edges: Union[List, Set],
    g2_check_edges: Union[List, Set],
) -> bool:
    """
    Check if a vertex identification map would create multi-edges.

    Validates that a candidate vertex identification mapping between two graphs
    would not produce multi-edges (parallel edges) when the graphs are joined.

    Parameters
    ----------
    candidate_map : set or frozenset
        Set of vertex identification pairs (graph1_node, graph2_node).
    g1_check_edges : list of tuple
        Edges in graph1 to check for potential multi-edge conflicts.
        Each edge's endpoints should be in sorted order.
    g2_check_edges : list of tuple
        Edges in graph2 to check for potential multi-edge conflicts.
        Each edge's endpoints should be in sorted order.

    Returns
    -------
    bool
        True if the mapping is valid (no multi-edges created), False otherwise.

    Notes
    -----
    Multi-edges occur when two vertices that are connected in one graph
    get identified with two vertices that are also connected in the other graph.
    """

    g1_vertices = sorted(u for u, _ in candidate_map)
    g2_vertices = sorted(v for _, v in candidate_map)

    g1_edges_to_check = [
        edge for edge in itertools.combinations(g1_vertices, 2) if edge in g1_check_edges
    ]
    g2_edges_to_check = [
        edge for edge in itertools.combinations(g2_vertices, 2) if edge in g2_check_edges
    ]

    return not any(
        ((u1, u2) in candidate_map and (v1, v2) in candidate_map)
        or ((u1, v2) in candidate_map and (v1, u2) in candidate_map)
        for (u1, v1), (u2, v2) in itertools.product(g1_edges_to_check, g2_edges_to_check)
    )


def map_application(
    vertex_map: Iterable[Tuple[int, int]],
    graph1: nx.Graph,
    graph2: nx.Graph,
) -> nx.Graph:
    """
    Apply vertex identification map to join two graphs.

    Creates a joined graph by composing two input graphs and then contracting
    (identifying) vertex pairs specified in the mapping. Node labels in graph2
    are incremented to avoid collisions before composition.

    Parameters
    ----------
    vertex_map : iterable of tuple
        Vertex identification mapping as (graph1_node, graph2_node) pairs.
        Each pair specifies two nodes that should be merged in the output.
    graph1 : networkx.Graph
        First input graph.
    graph2 : networkx.Graph
        Second input graph (node labels will be incremented internally).

    Returns
    -------
    networkx.Graph
        Joined graph with vertices identified according to the map.

    Raises
    ------
    ValueError
        If the joined graph has an unexpected number of edges, indicating
        a potential bug in the vertex identification process.

    Notes
    -----
    The function performs vertex identification by contracting nodes, which
    merges two vertices into one while preserving all incident edges.
    """

    n1 = graph1.number_of_nodes()
    shifted_graph2 = nx.relabel_nodes(graph2, lambda node: node + n1, copy=True)
    joined_graph = nx.compose(graph1, shifted_graph2)

    for v1, v2 in vertex_map:
        nx.contracted_nodes(joined_graph, v1, v2 + n1, copy=False)

    edges1, edges2 = graph1.number_of_edges(), graph2.number_of_edges()
    if joined_graph.number_of_edges() != edges1 + edges2:
        raise ValueError(
            f"The joined graph has the wrong number of edges, "
            f"{joined_graph.number_of_edges()} =/= {edges1} + {edges2}. "
            "This is probably a bug. Please report it."
        )

    # Contraction metadata refers to labels that will no longer exist.
    for _, data in joined_graph.nodes(data=True):
        data.pop('contraction', None)
    return nx.convert_node_labels_to_integers(joined_graph, first_label=0)
