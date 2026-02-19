from typing import List

import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att


def test_get_graph_charges():
    """
    Test the calculation of formal charges for nodes in a molecular graph.

    This function performs the following steps:
    1. Creates a molecular graph using `att.ph_2p_graph()`.
    2. Calculates the formal charges of the graph's nodes using `att.get_graph_charges()`.
    3. Prints the calculated charges.
    4. Asserts that the calculated charges match the expected values.

    Asserts:
        - The calculated charges are equal to [2, 0].

    Notes:
        - The graph represents a molecule with two nodes, where the expected charges are predefined.
    """
    print(flush=True)
    print('Testing charged case', flush=True)
    graph = att.ph_2p_graph()
    charges = att.get_graph_charges(graph)
    print("Charges of the graph:", charges, flush=True)
    assert charges == [2, 0]


def test_smi_to_nx_conversion():
    """
    Test the conversion of a SMILES string to a NetworkX graph and back to a SMILES string.

    This function performs the following steps:
    1. Converts a SMILES string to a NetworkX graph.
    2. Converts the NetworkX graph back to a SMILES string.
    3. Asserts that the original SMILES string and the converted SMILES string are equal.

    Asserts:
        - The converted SMILES string is equal to the original SMILES string.
    """
    print(flush=True)
    smi = "[H]O[H]"
    graph = att.smi_to_nx(smi)
    smi_out = att.nx_to_smi(graph)
    assert smi_out == smi, f"Expected {smi}, but got {smi_out}"


def test_inchi_to_nx_conversion():
    """
    Test the conversion of an InChI string to a NetworkX graph and back to an InChI string.

    This function performs the following steps:
    1. Converts an InChI string to a NetworkX graph.
    2. Converts the NetworkX graph back to an InChI string.
    3. Checks if the original InChI string and the converted InChI string are equal.

    Asserts:
        - The converted InChI string is equal to the original InChI string.
    """
    print(flush=True)
    inchi = "InChI=1S/H2O/h1H2"
    graph = att.inchi_to_nx(inchi)
    inchi_out = att.nx_to_inchi(graph)
    assert inchi_out == inchi, f"Expected {inchi}, but got {inchi_out}"


def test_join_graphs():
    """
    Test the functionality of joining and splitting molecular graphs.

    This function performs the following steps:
    1. Creates two molecular graphs from SMILES strings.
    2. Joins the two graphs into a single graph.
    3. Asserts that the joined graph has the correct number of nodes and edges.
    4. Splits the joined graph back into its disconnected subgraphs.
    5. Asserts that the split subgraphs have the correct number of nodes and edges.
    6. Verifies that the original graphs are isomorphic to the split subgraphs.

    Asserts:
        - The joined graph has 5 nodes and 3 edges.
        - The split subgraphs have the correct number of nodes and edges.
        - The original graphs are isomorphic to the split subgraphs.
    """
    print(flush=True)
    # Create a molecular graph for water
    g1 = att.smi_to_nx('[H][O][H]')
    # Create a molecular graph for oxygen
    g2 = att.smi_to_nx('[O][O]')
    # Join the two graphs into a single graph
    joined = att.join_graphs([g1, g2])
    assert joined.number_of_nodes() == 5
    assert joined.number_of_edges() == 3

    # Split the joined graph back into its components
    g1_split, g2_split = att.get_disconnected_subgraphs(joined)
    assert g1_split.number_of_nodes() == 3
    assert g1_split.number_of_edges() == 2
    assert g2_split.number_of_nodes() == 2
    assert g2_split.number_of_edges() == 1

    # Check that the original graphs are equal to the split graphs
    assert nx.is_isomorphic(g1, g1_split)
    assert nx.is_isomorphic(g2, g2_split)


def test_compose_graphs():
    print(flush=True)
    # Create a molecular graph for water
    g1 = att.smi_to_nx('[H][O][H]')
    # Create a molecular graph for oxygen
    g2 = att.smi_to_nx('[O][O]')
    # Compose the two graphs into a single graph
    composed = att.compose_graphs([g1, g2])
    assert composed.number_of_nodes() == 3
    assert composed.number_of_edges() == 2


def test_set_graph_layer():
    print(flush=True)
    # Create a molecular graph for water
    g1 = att.smi_to_nx('[H][O][H]')
    # Create a molecular graph for oxygen
    g2 = att.smi_to_nx('[O][O]')

    # create a directed graph and add the two graphs as nodes
    g = nx.DiGraph()
    g.add_node(0, graph=g1)
    g.add_node(1, graph=g2)
    g.add_edge(0, 1)

    g = att.set_graph_layer(g)

    # Fixed assert statement
    assert all(g.nodes[node]['layer'] == 1 or g.nodes[node]['layer'] == 0 for node in g.nodes)


def test_strip_digraph_layer():
    print(flush=True)
    smis = ['CC(OC)C=C',
            'CC(OC)C',
            'CCC']
    graphs = [att.smi_to_nx(smi) for smi in smis]
    pathway = att.calculate_assembly_index_pairwise_joint(graphs, settings={'strip_hydrogen': True})
    pathway = att.strip_digraph_layer(pathway, 0)
    # Check that the first layer has been stripped
    assert all(pathway.nodes[node]['layer'] > 0 for node in pathway.nodes)


def top_n_degree_subgraph(G: nx.DiGraph, n: int, must_keep: List[nx.Graph]) -> nx.DiGraph:
    G = G.copy()

    symbols_g = set()
    for node in G.nodes():
        vo = G.nodes[node].get('vo')
        for node, data in vo.nodes(data=True):
            symbols_g.add(data['color'])

    symbols_ref = set()
    for vo in must_keep:
        for node, data in vo.nodes(data=True):
            symbols_ref.add(data['color'])

    if 'H' in symbols_ref and 'H' not in symbols_g:
        must_keep = [att.remove_hydrogen_from_graph(g) for g in must_keep]

    degrees = ((u, G.in_degree(u) + G.out_degree(u)) for u in G.nodes())
    top_nodes = {u for u, _ in sorted(degrees, key=lambda x: x[1], reverse=True)[:n]}
    keep_nodes = set()
    for node in G.nodes():
        vo = G.nodes[node].get('vo')
        if vo is None:
            continue
        for g in must_keep:
            if nx.is_isomorphic(vo, g):
                keep_nodes.add(node)
                break
    return G.subgraph(top_nodes | keep_nodes)


def test_top_n_degree_subgraph():
    print(flush=True)
    smis = ['CC(OC)C=C',
            'CC(OC)C',
            'CC(OC)CCC',
            'CCC']
    graphs = [att.smi_to_nx(smi) for smi in smis]

    joined_graph = att.join_graphs(graphs)
    pathway = att.calculate_assembly_index(joined_graph, strip_hydrogen=True)[-1]
    att.plot_pathway(pathway,
                     frame_on=True,
                     plot_type='mol',
                     fig_size=(14, 7),
                     layout_style='crossmin_long')
    plt.show()
    pathway = att.calculate_assembly_index(joined_graph, strip_hydrogen=True)[-1]
    subgraph = top_n_degree_subgraph(pathway, n=3, must_keep=graphs)
    att.plot_pathway(subgraph,
                     frame_on=True,
                     plot_type='mol',
                     fig_size=(14, 7),
                     layout_style='crossmin_long')
    plt.show()
    assert len(subgraph) == 5