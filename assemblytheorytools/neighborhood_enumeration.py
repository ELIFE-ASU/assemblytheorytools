# This script will take a molgraph and return the neighborhood in assembly space with connectivity information

import itertools
import random
from typing import List
from rdkit import Chem
import networkx as nx
import numpy as np

node_match = nx.algorithms.isomorphism.categorical_node_match('color', None)
edge_match = nx.algorithms.isomorphism.categorical_edge_match('color', None)
ptable = Chem.GetPeriodicTable()


def enumerate_neighborhood(graphs: List[nx.Graph], obey_valence: bool = True, allow_dots: bool = True):
    """
    Generate the neighborhood of the input graphs in assembly space;
    this is the set of graphs one joining operation away.

    WARNING: Input graphs are assumed to be distinct.

    Returns a dictionary with the following keys:
    - "input_graphs": the same list of graphs input to the function
    - "N_graphs": a list of graphs in the neighborhood (unique up to isomorphism)
    - "down_jos": a set of `down' join operations with elements like (n1,n2,s)
       where n1 and n2 are indices in the N_input list and s is an index in the input_graphs list
    - "up_jos": a set of `up' join operations with elements like (s1,s2,n),
         where s1 and s2 are indices in the input_graphs list and n is an index in the N_graphs list
    - "allow_dots": a boolean indicating whether disconnected graphs are allowed in the output
    """

    # Canonicalize the input graphs
    graphs = [nxG_canonicalize(graph) for graph in graphs]

    # Enumerate graphs that can form the input graphs in one joining operation (down join)
    down_partitions = dict()
    for s, graph in enumerate(graphs):
        down_partitions[s] = enumerate_down(graph, allow_dots=allow_dots)

    # Enumerate the graphs that can be formed by joining two input graphs (up join)
    up_graphs = dict()
    for i, graph1 in enumerate(graphs):
        for j, graph2 in enumerate(graphs[i:]):
            up_graphs[(i, i + j)] = enumerate_up(graph1, graph2, obey_valence=obey_valence, allow_dots=allow_dots)

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
    Given a connected graph, this function will enumerate the power set of edges,
    and filter for the edge sets where the subgraph restricted to these edges, and its complement,
    are both connected. This is a brute force method, and will be slow for large graphs.
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


def get_valence(atom_symbol: str, ptable: Chem.rdchem.PeriodicTable = ptable) -> int:
    """
    Get the default valence of an atom based on its symbol.

    Args:
        atom_symbol (str): The chemical symbol of the atom (e.g., "C" for carbon, "O" for oxygen).
        ptable (Chem.rdchem.PeriodicTable, optional): An RDKit PeriodicTable object.
            Defaults to the global `ptable` instance.

    Returns:
        int: The default valence of the atom.

    Raises:
        ValueError: If the atom symbol is invalid or not recognized by the periodic table.
    """
    atomic_num = ptable.GetAtomicNumber(atom_symbol)  # Get the atomic number of the atom.
    return ptable.GetNOuterElecs(atomic_num)  # Return the default valence for the atomic number.


def enumerate_up(graph1: nx.Graph, graph2: nx.Graph, obey_valence: bool = True, allow_dots: bool = True, debug: bool = False):
    """
    Given two graphs, graph1 and graph2, this function will return the set of graphs that can be formed by
    a joining operation acting on graph1 and graph2. If obey_valence is True, then the we will resctict ourselves
    to only those graphs that obey the valence rules of the atoms in the graph.

    The algorithm is as follows:
    1. Enumerate the vertex colors shared by graph1 and graph2.
    2. For each shared color, enumerate the combinations of possible valid vertex identifications.
    3. Filter out combinations that produce multi-edges or break valence rules (if obey_valence is True).
    4. Enumerate the outer product of these combinations (skipping the invalid combination of no identification).
    5. Filter out combinations that produce multi-edges.
    6. Generate the output graphs from the set valid of vertex identification combinations.
    """
    

    # # Copy graphs for safety
    # graph1 = graph1.copy()
    # graph2 = graph2.copy()
    
    # Check that we have the information for valence checks
    if obey_valence:
        valence_budgets = [np.zeros(graph1.number_of_nodes()), np.zeros(graph2.number_of_nodes())]
        for g_idx, graph in enumerate([graph1, graph2]):
            for node in graph.nodes:
                if 'color' not in graph.nodes[node]:
                    raise ValueError(
                        f"Node {node} does not have a color attribute. Please add a color attribute to the nodes.")
                valence_budgets[g_idx][node] = get_valence(graph.nodes[node]['color'])
                for edge in graph.edges(node):
                    valence_budgets[g_idx][node] -= graph.edges[edge]['color']  # This assumes 1=single bond, 2=double bond, etc.

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
                    if get_valence(color) - valence_budgets[0][node1] <= valence_budgets[1][node2]:
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
                            valid = conditional_check_multi_edge_generation(candidate_color_map, g1_check_edges,
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
    for map in valid_maps:
        joined = map_application(map, graph1, graph2)
        if not allow_dots and not nx.is_connected(joined):
            continue
        output_graphs.append(joined)
    return output_graphs


def map_outer_product(combinations, graph1, graph2): # BUG LOOKS TO BE HAPPENING HERE
    """
    This function will return the valid maps from the outer product of the valid color-specific maps.
    """

    if len(combinations) == 1:
        return combinations[list(combinations.keys())[0]]  # If there is only one color, we can just return the valid maps for that color
    valid_maps = []  # This will be the list of valid maps
    # Remove colors with empty sets
    filtered_combinations = {color: maps for color, maps in combinations.items() if maps}
    colors = list(filtered_combinations.keys())
    lists_of_maps = [filtered_combinations[color] for color in colors]
    
    # We only need to worry about edges that connect two different colors
    g1_check_edges = []  # This will be the set of edges in graph1 that we'll need to worry about generating multi-edges
    for edge in graph1.edges():
        if graph1.nodes[edge[0]]['color'] != graph1.nodes[edge[1]]['color']:
            if graph1.nodes[edge[0]]['color'] in colors and graph1.nodes[edge[1]]['color'] in colors:
                g1_check_edges.append(tuple(sorted(edge)))
    g2_check_edges = []  # This will be the set of edges in graph2 that we'll need to worry about generating multi-edges
    for edge in graph2.edges():
        if graph2.nodes[edge[0]]['color'] != graph2.nodes[edge[1]]['color']:
            if graph2.nodes[edge[0]]['color'] in colors and graph2.nodes[edge[1]]['color'] in colors:
                g2_check_edges.append(tuple(sorted(edge)))

    # Now we will enumerate the outer product of these combinations
    for candidate_map in itertools.product(*lists_of_maps): # THIS IS WHERE THE BUG IS HAPPENING, itertools is passing []
        candidate_map = set(itertools.chain.from_iterable(candidate_map))  # This should flatten the tuple of sets of tuples into a single set of tuples
        valid = conditional_check_multi_edge_generation(candidate_map, g1_check_edges, g2_check_edges)

        if valid:
            valid_maps.append(candidate_map)
    return valid_maps


def conditional_check_multi_edge_generation(candidate_map, g1_check_edges, g2_check_edges):
    """
    Check that the candidate map does not generate multi-edges in the joined graph.
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
    This function will apply the map to the two graphs and return the resulting graph.
    The map is a list of tuples, where each tuple is a pair of nodes to be identified.
    A tuple in the map will be of the form (v1, v2), where v1 is from graph1 and v2 is from graph2.
    """

    n1 = graph1.number_of_nodes()
    g1 = graph1.copy()
    g2 = nx.relabel_nodes(graph2, lambda x: x + n1,
                          copy=True)  # We are incrementing the node labels of graph2 by n1 to avoid collisions
    joined_graph = nx.compose(g1, g2)

    for v1, v2 in map:
        joined_graph = nx.contracted_nodes(joined_graph, v1, v2 + n1)  # Vertex identification!

    if len(joined_graph.edges()) != len(g1.edges()) + len(g2.edges()):
        raise ValueError(
            f"The joined graph has the wrong number of edges, {len(joined_graph.edges())} =/= {len(g1.edges())} + {len(g2.edges())}. This is probably a bug. Please report it.")

    return joined_graph


def nxG_canonicalize(G):
    """
    This function takes a networkx graph and makes its node labels 
    into a sequence of integers from 0 to n-1, where n is the number of nodes.
    Input graph will potentially skip some numbers, so this function will
    relabel the nodes to be a sequence of integers from 0 to n-1.

    This will make the code more stable.
    """
    # Get the current node labels
    current_labels = list(G.nodes())
    # Create a mapping from current labels to new labels
    label_mapping = {current_labels[i]: i for i in range(len(current_labels))}
    # Relabel the graph
    G = nx.relabel_nodes(G, label_mapping)
    return G
