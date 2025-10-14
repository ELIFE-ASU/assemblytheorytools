import itertools
from typing import List

import networkx as nx
import numpy as np
from rdkit import Chem

from .tools_graph import canonicalize_node_labels

node_match = nx.algorithms.isomorphism.categorical_node_match('color', None)
edge_match = nx.algorithms.isomorphism.categorical_edge_match('color', None)
ptable = Chem.GetPeriodicTable()


def enumerate_neighborhood(graphs: List[nx.Graph],
                           obey_valence: bool = True,
                           allow_dots: bool = True,
                           debug=False,
                           custom_valence_table=None):
    """
    Generate the neighborhood of input graphs in assembly space.
    
    Computes the set of graphs that are one joining operation away from the
    input graphs. This includes both "down joins" (decompositions) and "up joins"
    (combinations). Results are deduplicated by graph isomorphism.
    
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
        If None, uses RDKit default valences, by default None.
    
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
    """

    # Canonicalize the input graphs
    graphs = [canonicalize_node_labels(graph) for graph in graphs]

    # Enumerate graphs that can form the input graphs in one joining operation (down join)
    down_partitions = dict()
    for s, graph in enumerate(graphs):
        down_partitions[s] = enumerate_down(graph, allow_dots=allow_dots)

    # Enumerate the graphs that can be formed by joining two input graphs (up join)
    up_graphs = dict()
    for i, graph1 in enumerate(graphs):
        for j, graph2 in enumerate(graphs[i:]):
            up_graphs[(i, i + j)] = enumerate_up(graph1,
                                                 graph2,
                                                 obey_valence=obey_valence,
                                                 allow_dots=allow_dots,
                                                 debug=debug,
                                                 custom_valence_table=custom_valence_table)

    # Mod out the down join operations by isomorphism
    N_graphs = []
    down_jos = set()
    for s in range(len(graphs)):
        for down_partition in down_partitions[s]:
            jo = [-1, -1, s]
            g1 = graphs[s].edge_subgraph(down_partition[0])
            g2 = graphs[s].edge_subgraph(down_partition[1])

            # Filter out disconnected graphs if not allowed
            if not allow_dots and (not nx.is_connected(g1) or not nx.is_connected(g2)):
                print(
                    "Warning: A disconnected graph was found in a down join operation. This should never happen. Please report this bug.")
                continue

            for in_idx, g_part in enumerate([g1, g2]):
                found = False
                for n_idx, g in enumerate(N_graphs):
                    if nx.is_isomorphic(g_part, g, node_match=node_match, edge_match=edge_match):
                        jo[in_idx] = n_idx
                        found = True
                        break
                if not found:
                    N_graphs.append(g_part.copy())
                    jo[in_idx] = len(N_graphs) - 1

            if jo[0] <= jo[1]:
                down_jos.add(tuple(jo))
            else:
                down_jos.add(tuple([jo[1], jo[0], s]))

    # Mod out the up join operations by isomorphism
    up_jos = set()
    for (i, j), up_graphs_list in up_graphs.items():
        for up_graph in up_graphs_list:
            # Filter out disconnected graphs if not allowed
            if not allow_dots and not nx.is_connected(up_graph):
                print(
                    "Warning: A disconnected graph was found in an up join operation. This should never happen. Please report this bug.")
                continue

            jo = [i, j, -1]
            found = False
            for idx, g in enumerate(N_graphs):
                if nx.is_isomorphic(up_graph, g, node_match=node_match, edge_match=edge_match):
                    jo[2] = idx
                    found = True
                    break
            if not found:
                N_graphs.append(up_graph.copy())
                jo[2] = len(N_graphs) - 1

            up_jos.add(tuple(jo))  # jo's are already canonicalized

    return {"input_graphs": graphs, "N_graphs": N_graphs, "down_jos": down_jos, "up_jos": up_jos}


def enumerate_down(graph: nx.Graph, allow_dots: bool = True):
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
    Time complexity is O(2^|E|) where |E| is the number of edges.
    """
    partition_pairs = []
    edges = list(graph.edges())
    if not edges:
        return partition_pairs
    anchor_edge = tuple(sorted(edges[0]))  # Pick an arbitrary edge as the anchor
    iterable_edges = edges[1:]  # Exclude the anchor edge from the combinations
    for i in range(0, len(edges) - 1):
        for subset in itertools.combinations(iterable_edges, i):
            subset = list(subset)
            subset = [tuple(sorted(edge)) for edge in subset]
            subset.append(anchor_edge)  # Include the anchor edge in the subset
            subgraph = graph.edge_subgraph(subset)
            if nx.is_connected(subgraph):
                complement = graph.copy()
                complement.remove_edges_from(subset)
                for node in list(complement.nodes):  # Remove degree 0 nodes from the complement
                    if len(list(complement.neighbors(node))) == 0:
                        complement.remove_node(node)
                if nx.is_connected(complement):
                    # If not allowing dots, skip if the union is disconnected
                    if not allow_dots and not nx.is_connected(nx.compose(subgraph, complement)):
                        continue
                    partition_pairs.append([list(subset), list(complement.edges())])
    return partition_pairs


def get_valence(atom_symbol: str, ptable: Chem.rdchem.PeriodicTable = ptable, custom_valence_table=None) -> int:
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
        return custom_valence_table[atom_symbol]  # Return the custom valence if provided and available.
    else:
        return ptable.GetDefaultValence(atom_symbol)  # Return the default valence for the atomic number.


def enumerate_up(graph1: nx.Graph,
                 graph2: nx.Graph,
                 obey_valence: bool = True,
                 allow_dots: bool = True,
                 debug: bool = False,
                 custom_valence_table=None):
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
        Custom valence mapping {atom_symbol: valence}. If None, uses
        RDKit default valences, by default None.
    
    Returns
    -------
    list of networkx.Graph
        List of graphs formed by valid vertex identifications between
        graph1 and graph2.
    
    Notes
    -----
    Algorithm:
    
    1. Enumerate vertex colors shared by both graphs
    2. For each color, enumerate valid vertex identification combinations
    3. Filter combinations that produce multi-edges or violate valence rules
    4. Compute outer product of combinations across all colors
    5. Filter out combinations that create multi-edges
    6. Generate output graphs from valid vertex identifications
    
    Raises
    ------
    ValueError
        If nodes lack 'color' attributes when obey_valence is True.
    """

    # # Copy graphs for safety
    # graph1 = graph1.copy()
    # graph2 = graph2.copy()

    # Check that we have the information for valence checks
    if obey_valence:
        if debug:
            print("Checking valence budgets...")
        valence_budgets = [np.zeros(graph1.number_of_nodes()), np.zeros(graph2.number_of_nodes())]
        for g_idx, graph in enumerate([graph1, graph2]):
            for node in graph.nodes:
                if 'color' not in graph.nodes[node]:
                    raise ValueError(
                        f"Node {node} does not have a color attribute. Please add a color attribute to the nodes.")
                valence_budgets[g_idx][node] = get_valence(graph.nodes[node]['color'],
                                                           custom_valence_table=custom_valence_table)
                if debug:
                    print(
                        f"Node {node} in graph {g_idx + 1} has color {graph.nodes[node]['color']} and valence budget {valence_budgets[g_idx][node]}")
                for edge in graph.edges(node):
                    valence_budgets[g_idx][node] -= graph.edges[edge][
                        'color']  # This assumes 1=single bond, 2=double bond, etc.
                if valence_budgets[g_idx][node] < 0:
                    print(f"Warning: Node {node} in graph {g_idx + 1} is overbonded. Skipping this graph.")
                    return []
        if debug:
            print(f"Valence budgets for graph1: {valence_budgets[0]}")
            print(f"Valence budgets for graph2: {valence_budgets[1]}")
        if sum(valence_budgets[0]) == 0 or sum(valence_budgets[1]) == 0:  # No valence budget left
            if debug:
                print("No valence budget left in (at least) one of the graphs. Returning empty list.")
            return []

    # Get the colors of the nodes in graph1 and graph2
    colors1 = set([graph1.nodes[node]['color'] for node in graph1.nodes])
    colors2 = set([graph2.nodes[node]['color'] for node in graph2.nodes])
    shared_colors = colors1.intersection(colors2)  # Get the shared colors

    combinations = dict()  # Elements of this dict are formated like {color:[list of valid vertex identification maps within this color]}
    for color in shared_colors:
        valid_color_maps = set()  # Elements will be lists of tuples, where each tuple is a pair of nodes to be identified
        # Get the nodes of the shared color in graph1 and graph2
        if obey_valence:
            nodes1 = [node for node in graph1.nodes if
                      graph1.nodes[node]['color'] == color and 0 < valence_budgets[0][node]]
            nodes2 = [node for node in graph2.nodes if
                      graph2.nodes[node]['color'] == color and 0 < valence_budgets[1][node]]
        else:
            nodes1 = [node for node in graph1.nodes if graph1.nodes[node]['color'] == color]
            nodes2 = [node for node in graph2.nodes if graph2.nodes[node]['color'] == color]

        valid_identifications = []  # This will be a list of tuples, where each tuple is a pair of nodes that could be identified
        # Get all the valid identifications
        if obey_valence:  # If obey_valence is True, we will only consider identifications that do not exceed the valence budget
            for node1 in nodes1:
                for node2 in nodes2:
                    if get_valence(color, custom_valence_table=custom_valence_table) - valence_budgets[0][node1] <= \
                            valence_budgets[1][node2]:
                        # These identification tuples will always be written as (v1,v2) where v1 is from graph1 and v2 is from graph2
                        valid_identifications.append((node1, node2))
        else:
            for node1 in nodes1:
                for node2 in nodes2:
                    valid_identifications.append((node1, node2))

        if debug:
            print(f"Number of valid identifications = {len(valid_identifications)}")

        # We only form multi-edges by identifying a pair of adjacent vertices in graph1 with a pair of adjacent vertices in graph2,
        # so we will check for this condition

        g1_check_edges = []
        for u, v in itertools.combinations(nodes1, 2):
            if graph1.has_edge(u, v):
                g1_check_edges.append(tuple(sorted((u, v))))
        g2_check_edges = []
        for u, v in itertools.combinations(nodes2, 2):
            if graph2.has_edge(u, v):
                g2_check_edges.append(tuple(sorted((u, v))))

        g1_check_edges = set(g1_check_edges)  # Sets have faster membership testing than lists
        g2_check_edges = set(g2_check_edges)

        # Now we will enumerate the combinations of possible valid vertex identifications
        if valid_identifications:  # Skip this color if there are no valid identifications
            for k in range(min(len(nodes1), len(nodes2)) + 1):  # k sets how many identification we will perform
                for node1_perm in itertools.permutations(nodes1, k):
                    for node2_perm in itertools.permutations(nodes2, k):
                        candidate_color_map = frozenset(zip(node1_perm, node2_perm))
                        # Check that every pair is in valid_identifications
                        valid = False  # This is the default value if the candidate map doesn't make it through the next if statement
                        if all(pair in valid_identifications for pair in candidate_color_map):
                            valid = conditional_check_multi_edge_generation(candidate_color_map,
                                                                            g1_check_edges,
                                                                            g2_check_edges)  # Check for multi-edges
                        if valid:  # This candidate color map is valid, so we will add it to the list of valid color maps
                            valid_color_maps.add(candidate_color_map)

        combinations[color] = valid_color_maps  # Now we have every valid color map restricted to this color

    # Now we will enumerate the outer product of these combinations
    valid_maps = map_outer_product(combinations, graph1, graph2)

    if debug:
        print(f"Number of combinations = {len(combinations)}")
        print(combinations.keys())
        print([combinations[key] for key in combinations.keys()])
        print(f"Number of valid color-specific maps = {sum(len(combinations[key]) for key in combinations.keys())}")
        print(f"Number of valid maps = {len(valid_maps)}")

    # Now we need to generate the output graphs from the set of valid maps
    output_graphs = []
    for m in valid_maps:
        joined = map_application(m, graph1, graph2)
        if not allow_dots and not nx.is_connected(joined):
            continue
        output_graphs.append(joined)
    return output_graphs


def map_outer_product(combinations, graph1, graph2):
    """
    Compute valid vertex identification maps from outer product of color-specific maps.
    
    Enumerates the Cartesian product of valid color-specific vertex identification
    maps and filters out those that would create multi-edges in the joined graph.
    
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
    list of set
        List of valid complete vertex identification maps, where each map
        is a set of (graph1_node, graph2_node) tuples.
    
    Notes
    -----
    Special case: If only one color exists, returns the valid maps for that
    color directly without computing the outer product.
    """
    # If there is only one color, we can just return the valid maps for that color
    if len(combinations) == 1:
        return combinations[list(combinations.keys())[0]]
    valid_maps = []  # This will be the list of valid maps
    # Remove colors with empty sets
    filtered_combinations = {color: maps for color, maps in combinations.items() if maps}
    colors = list(filtered_combinations.keys())
    lists_of_maps = [filtered_combinations[color] for color in colors]

    # We only need to worry about edges that connect two different colors
    g1_check_edges = []
    for edge in graph1.edges():
        if graph1.nodes[edge[0]]['color'] != graph1.nodes[edge[1]]['color']:
            if graph1.nodes[edge[0]]['color'] in colors and graph1.nodes[edge[1]]['color'] in colors:
                g1_check_edges.append(tuple(sorted(edge)))
    g2_check_edges = []
    for edge in graph2.edges():
        if graph2.nodes[edge[0]]['color'] != graph2.nodes[edge[1]]['color']:
            if graph2.nodes[edge[0]]['color'] in colors and graph2.nodes[edge[1]]['color'] in colors:
                g2_check_edges.append(tuple(sorted(edge)))

    # Now we will enumerate the outer product of these combinations
    for candidate_map in itertools.product(
            *lists_of_maps):  # THIS IS WHERE THE BUG IS HAPPENING, itertools is passing []
        candidate_map = set(itertools.chain.from_iterable(
            candidate_map))  # This should flatten the tuple of sets of tuples into a single set of tuples
        valid = conditional_check_multi_edge_generation(candidate_map, g1_check_edges, g2_check_edges)

        if valid:
            valid_maps.append(candidate_map)
    return valid_maps


def conditional_check_multi_edge_generation(candidate_map, g1_check_edges, g2_check_edges):
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
        Should only include edges connecting different colors.
    g2_check_edges : list of tuple
        Edges in graph2 to check for potential multi-edge conflicts.
        Should only include edges connecting different colors.
    
    Returns
    -------
    bool
        True if the mapping is valid (no multi-edges created), False otherwise.
    
    Notes
    -----
    Multi-edges occur when two vertices that are connected in one graph
    get identified with two vertices that are also connected in the other graph.
    """

    g1_vertices = sorted([pair[0] for pair in candidate_map])
    g2_vertices = sorted([pair[1] for pair in candidate_map])

    g1_edges_to_check = [edge for edge in itertools.combinations(g1_vertices, 2) if edge in g1_check_edges]
    g2_edges_to_check = [edge for edge in itertools.combinations(g2_vertices, 2) if edge in g2_check_edges]

    for g1_edge in g1_edges_to_check:
        for g2_edge in g2_edges_to_check:
            if ((g1_edge[0], g2_edge[0]) in candidate_map and (g1_edge[1], g2_edge[1]) in candidate_map) or \
                    ((g1_edge[0], g2_edge[1]) in candidate_map and (g1_edge[1], g2_edge[0]) in candidate_map):
                return False
    return True


def map_application(map, graph1, graph2):
    """
    Apply vertex identification map to join two graphs.
    
    Creates a joined graph by composing two input graphs and then contracting
    (identifying) vertex pairs specified in the mapping. Node labels in graph2
    are incremented to avoid collisions before composition.
    
    Parameters
    ----------
    map : iterable of tuple
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
    g1 = graph1.copy()
    # We are incrementing the node labels of graph2 by n1 to avoid collisions
    g2 = nx.relabel_nodes(graph2, lambda x: x + n1, copy=True)
    joined_graph = nx.compose(g1, g2)

    for v1, v2 in map:
        joined_graph = nx.contracted_nodes(joined_graph, v1, v2 + n1)  # Vertex identification!

    if len(joined_graph.edges()) != len(g1.edges()) + len(g2.edges()):
        raise ValueError(
            f"The joined graph has the wrong number of edges, {len(joined_graph.edges())} =/= {len(g1.edges())} + {len(g2.edges())}. This is probably a bug. Please report it.")

    return joined_graph
