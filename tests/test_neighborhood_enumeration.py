# This will use pytest to verify that the code is functioning properly.
import random

import networkx as nx
from rdkit import Chem

node_match = nx.algorithms.isomorphism.categorical_node_match('color', None)
edge_match = nx.algorithms.isomorphism.categorical_edge_match('color', None)
import assemblytheorytools as att


def test_enumerate_down():
    """
    Test the enumerate_down function
    """
    # Set random seed for reproducibility
    random.seed(42)
    # Create a simple test graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (3, 6)])

    # Get the partitions from the down_enum function
    partitions = att.enumerate_down(graph)
    assert len(partitions) == 21  # By hand, I think there are 21 partitions, before modding out by isomorphism
    # Check that the partitions are valid
    for partition in partitions:
        assert len(partition) == 2
        assert set(partition[0]).isdisjoint(partition[1])
        assert set(partition[0]) | set(partition[1]) == set(graph.edges())


def test_enumerate_down2():
    """
    Test the enumerate_down function
    """
    # Set random seed for reproducibility
    random.seed(42)
    # Create a simple test graph
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)])

    # Get the partitions from the down_enum function
    partitions = att.enumerate_down(graph)
    assert len(partitions) == 6  # By hand, I think there are 6 partitions, before modding out by isomorphism
    # Check that the partitions are valid
    for partition in partitions:
        assert len(partition) == 2
        assert len(partition[0]) + len(partition[1]) == len(graph.edges())


def test_enumerate_up_small():
    """
    Test the enumerate_up function on a small case where the exact answer is known by hand
    """
    # Set random seed for reproducibility
    random.seed(42)

    l = 2
    edges = [(k, k + 1) for k in range(l)]

    # These are our test input graphs
    g1 = nx.Graph()
    g1.add_edges_from(edges)
    g2 = nx.Graph()
    g2.add_edges_from(edges)

    # These are the expected output graphs
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    G5 = nx.Graph()
    G5.add_edges_from([(0, 4), (1, 4), (2, 4), (3, 4)])
    G6 = nx.Graph()
    G6.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2)])

    # Add colors to the nodes and edges
    for g in [g1, g2, G1, G2, G3, G4, G5, G6]:
        for i in g.nodes():
            g.nodes[i]['color'] = 'C'
        for e in g.edges():
            g.edges[e]['color'] = 1

    N_up_graphs = att.enumerate_up(g1, g2)
    print(f"Number of enum_up graphs before iso-check: {len(N_up_graphs)}")
    unique_graphs = [N_up_graphs[0]]
    for g in N_up_graphs[1:]:
        if not any(nx.is_isomorphic(g, unique_g) for unique_g in unique_graphs):
            unique_graphs.append(g)
    print(f"Number of unique graphs after iso-check: {len(unique_graphs)}")
    assert len(unique_graphs) == 6
    # Check that the output graphs are isomorphic to the expected graphs
    for expected_graph in [G1, G2, G3, G4, G5, G6]:
        assert any(nx.is_isomorphic(g, expected_graph) for g in unique_graphs)


def test_enumerate_up_runs():
    """
    Test the enumerate_up function with small graphs just to see if it works
    """
    # Set random seed for reproducibility
    random.seed(42)

    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2), (2, 3)])

    for g in [g1, g2]:
        for i in g.nodes():
            g.nodes[i]['color'] = 'C'
        for e in g.edges():
            g.edges[e]['color'] = 1

    N_up_graphs = att.enumerate_up(g1, g2)

    assert len(N_up_graphs) > 0


def test_enumerate_up():
    """
    Test the enumerate_neighborhood function with scrambled node indices and make sure output is equivalent
    """
    # Set random seed for reproducibility
    random.seed(42)

    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2), (2, 3)])

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2), (2, 3)])

    g1_scrambled = att.scramble_node_indices(g1.copy(), seed=42)
    g2_scrambled = att.scramble_node_indices(g2.copy(), seed=45)

    for g in [g1, g2, g1_scrambled, g2_scrambled]:
        for i in g.nodes():
            g.nodes[i]['color'] = 'C'
        for e in g.edges():
            g.edges[e]['color'] = 1

    output = att.enumerate_neighborhood([g1, g2])

    output_scrambled = att.enumerate_neighborhood([g1_scrambled, g2_scrambled])

    print("Scrambled graphs:")
    for i, g in enumerate(output_scrambled['input_graphs']):
        print(f"Graph {i}: {g}")
        print(f"Edges: {g.edges()}")
        print(f"Node colors: {g.nodes(data=True)}\n")
    # print(f"Number of graphs in neighborhood: {len(output['N_graphs'])}")
    print(f"Number of graphs in neighborhood (scrambled): {len(output_scrambled['N_graphs'])}")
    # print(f"Number of down jos: {len(output['down_jos'])}")
    print(f"Number of down jos (scrambled): {len(output_scrambled['down_jos'])}")
    # print(f"Number of up jos: {len(output['up_jos'])}")
    print(f"Number of up jos (scrambled): {len(output_scrambled['up_jos'])}")
    # print(f"Down jos: {output['down_jos']}")
    print(f"Down jos (scrambled): {output_scrambled['down_jos']}")
    # print(f"Up jos: {output['up_jos']}")
    print(f"Up jos (scrambled): {output_scrambled['up_jos']}")
    for i, g in enumerate(output_scrambled['N_graphs']):
        print(f"\nGraph {i}: {g}")
        print(f"Edges: {g.edges()}")
        print(f"Node colors: {g.nodes(data=True)}")

    for key in output.keys():
        assert len(output[key]) == len(output_scrambled[
                                           key]), f"Output lengths differ for key {key}: {len(output[key])} vs {len(output_scrambled[key])}"


def test_multicolored_graph_full_pipeline():
    """
    Test the enumerate_neighborhood function with a multicolored graph
    """
    # Set random seed for reproducibility
    random.seed(42)

    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2)])

    for i in g1.nodes():
        g1.nodes[i]['color'] = 'C'
    for e in g1.edges():
        g1.edges[e]['color'] = 1

    g2 = g1.copy()

    g1.nodes[0]['color'] = 'P'
    g2.nodes[1]['color'] = 'P'

    output = att.enumerate_neighborhood([g1, g2])

    g1_scrambled = att.scramble_node_indices(g1.copy(), seed=42)
    g2_scrambled = att.scramble_node_indices(g2.copy(), seed=42)

    output_scrambled = att.enumerate_neighborhood([g1_scrambled, g2_scrambled])

    print(f"Number of graphs in neighborhood: {len(output['N_graphs'])}")
    # print(f"Number of graphs in neighborhood (scrambled): {len(output_scrambled['N_graphs'])}")
    print(f"Number of down jos: {len(output['down_jos'])}")
    # print(f"Number of down jos (scrambled): {len(output_scrambled['down_jos'])}")
    print(f"Number of up jos: {len(output['up_jos'])}")
    # print(f"Number of up jos (scrambled): {len(output_scrambled['up_jos'])}")
    print(f"Down jos: {output['down_jos']}")
    # print(f"Down jos (scrambled): {output_scrambled['down_jos']}")
    print(f"Up jos: {output['up_jos']}")
    # print(f"Up jos (scrambled): {output_scrambled['up_jos']}")
    for i, g in enumerate(output['N_graphs']):
        print(f"\nGraph {i}: {g}")
        print(f"Edges: {g.edges()}")
        print(f"Node colors: {g.nodes(data=True)}")

    assert len(output['N_graphs']) == len(output_scrambled['N_graphs'])
    assert len(output['down_jos']) == len(output_scrambled['down_jos'])
    assert len(output['up_jos']) == len(output_scrambled['up_jos'])


def test_rTCA():
    """
    Just make sure the rTCA example runs
    """

    # Set random seed for reproducibility
    random.seed(1)

    # rTCA smiles
    smiles_dict = {"ACE": "CC(=O)[O-]",
                   "PYR": "CC(=O)C(=O)[O-]",
                   "FUM": "C(=C/C(=O)[O-])\C(=O)[O-]",
                   "MAL": "C(C(C(=O)O)O)C(=O)O",
                   "OXA": "C(C(=O)C(=O)O)C(=O)O",
                   "SUC": "C(CC(=O)[O-])C(=O)[O-]",
                   "AKG": "C(CC(=O)O)C(=O)C(=O)O",
                   "CAC": "C(/C(=C/C(=O)O)/C(=O)O)C(=O)O",
                   "CIT": "C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O",
                   "ISC": "C(C(C(C(=O)O)O)C(=O)O)C(=O)O",
                   "OXS": "C(C(C(=O)C(=O)O)C(=O)O)C(=O)O"
                   }

    rTCA_keys = list(smiles_dict.keys())
    smiles_list = [smiles_dict[key] for key in rTCA_keys]

    mols = [att.smi_to_mol(smile) for smile in smiles_list]

    # Convert the RDKit molecules to NetworkX graphs and remove hydrogen atoms
    graphs = [att.remove_hydrogen_from_graph(att.mol_to_nx(mol)) for mol in mols]
    # Limit the number of graphs for testing
    output = att.enumerate_neighborhood(graphs[:3])  # Limit to 3 graphs for test speed
    assert len(output['N_graphs']) > 0


def test_enum_up_C_2_O():
    g = nx.Graph()
    g.add_edges_from([(0, 1), ])
    g.nodes[0]['color'] = 'C'
    g.nodes[1]['color'] = 'O'
    g.edges[(0, 1)]['color'] = 2

    up_graphs = att.enumerate_up(g, g, 1, 1)

    print(f"Neighborhood+ size = {len(up_graphs)}")

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])
    g2.nodes[1]['color'] = 'C'
    g2.nodes[0]['color'] = 'O'
    g2.nodes[2]['color'] = 'O'
    g2.edges[(0, 1)]['color'] = 2
    g2.edges[(1, 2)]['color'] = 2

    flag = False
    for gN in up_graphs:
        if nx.is_isomorphic(gN, g2, node_match=node_match, edge_match=edge_match):
            flag = True
            break

    assert flag


def test_C_2_O():
    g = nx.Graph()
    g.add_edges_from([(0, 1), ])
    g.nodes[0]['color'] = 'C'
    g.nodes[1]['color'] = 'O'
    g.edges[(0, 1)]['color'] = 2

    output = att.enumerate_neighborhood([g])

    print("Neighborhood size:")
    print(len(output["N_graphs"]))

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])
    g2.nodes[1]['color'] = 'C'
    g2.nodes[0]['color'] = 'O'
    g2.nodes[2]['color'] = 'O'
    g2.edges[(0, 1)]['color'] = 2
    g2.edges[(1, 2)]['color'] = 2

    flag = False
    for gN in output["N_graphs"]:
        if nx.is_isomorphic(gN, g2, node_match=node_match, edge_match=edge_match):
            flag = True
            break

    assert flag


def test_enum_up_S_2_O():
    g = nx.Graph()
    g.add_edges_from([(0, 1), ])
    g.nodes[0]['color'] = 'S'
    g.nodes[1]['color'] = 'O'
    g.edges[(0, 1)]['color'] = 2

    up_graphs = att.enumerate_up(g, g, 1, 1, 1)

    print(f"Neighborhood+ size = {len(up_graphs)}")

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])
    g2.nodes[1]['color'] = 'S'
    g2.nodes[0]['color'] = 'O'
    g2.nodes[2]['color'] = 'O'
    g2.edges[(0, 1)]['color'] = 2
    g2.edges[(1, 2)]['color'] = 2

    flag = False
    for gN in up_graphs:
        if nx.is_isomorphic(gN, g2, node_match=node_match, edge_match=edge_match):
            flag = True
            break

    assert flag


def test_S_2_O():
    g = nx.Graph()
    g.add_edges_from([(0, 1), ])
    g.nodes[0]['color'] = 'S'
    g.nodes[1]['color'] = 'O'
    g.edges[(0, 1)]['color'] = 2

    output = att.enumerate_neighborhood([g])

    print("Neighborhood size:")
    print(len(output["N_graphs"]))

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])
    g2.nodes[1]['color'] = 'S'
    g2.nodes[0]['color'] = 'O'
    g2.nodes[2]['color'] = 'O'
    g2.edges[(0, 1)]['color'] = 2
    g2.edges[(1, 2)]['color'] = 2

    flag = False
    for gN in output["N_graphs"]:
        if nx.is_isomorphic(gN, g2, node_match=node_match, edge_match=edge_match):
            flag = True
            break

    assert flag


def test_input_valence():
    """
    Check that when given saturated input graphs, the neighborhood enumeration with obey_valence=True returns no new graphs.
    """
    input_smis = ["N#N", "C#O", "O=O", "[H][H]"]
    input_gs = [att.smi_to_nx(smi, add_hydrogens=False, sanitize=False) for smi in input_smis]

    out = att.enumerate_neighborhood(input_gs, obey_valence=True)
    assert len(out['up_jos']) == 0
    assert len(out['down_jos']) == 0 # This is expected since these are all single edge graphs
