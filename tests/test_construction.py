import networkx as nx

import assemblytheorytools as att


def test_assign_levels():
    """
    Test the `assign_levels` function with a directed graph.

    This function performs the following steps:
    1. Creates a directed graph.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes and adds them to the graph.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes with their levels and add them to the graph
    nodes = {"CC": 0, "C=C": 0, "CO": 0, "CC=C": 1, "OCC=C": 2}
    graph.add_nodes_from(nodes)

    # Define edges and add them to the graph
    edges = [("CC", "CC=C"), ("C=C", "CC=C"), ("CO", "OCC=C"), ("CC=C", "OCC=C")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node]["level"] == level, \
            f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_linear_chain():
    """
    Test the `assign_levels` function with a linear chain graph.

    This function performs the following steps:
    1. Creates a directed graph representing a linear chain of nodes.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes to form a linear chain.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node][
                   "level"] == level, f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_empty_graph():
    """
    Test the `assign_levels` function with an empty graph.

    This function performs the following steps:
    1. Creates an empty directed graph.
    2. Calls the `assign_levels` function on the empty graph.
    3. Asserts that the graph remains empty after the function call.

    Asserts:
        - The graph has no nodes after calling `assign_levels`.
    """
    print(flush=True)
    # Create an empty directed graph
    graph = nx.DiGraph()
    # Assign levels to the empty graph
    att.assign_levels(graph)
    # Verify that the graph has no nodes
    assert len(graph.nodes) == 0, "Empty graph should have no nodes."


def test_convert_digraph_vo_to_target():
    smi = att.pubchem_name_to_smi('diethyl phthalate')
    print(f"SMILES: {smi}", flush=True)
    graph = att.smi_to_nx(smi, sanitize=True, add_hydrogens=True)

    pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)[2]
    pathway = att.convert_digraph_vo_to_target(pathway)
    smis = []
    for node in pathway.nodes():
        smis.append(pathway.nodes[node]['vo'])
    print(smis, flush=True)
    ref_smi = ['CC',
               'CCO',
               'CO',
               'C=O',
               'CC(=O)O',
               'C=CC(=O)O',
               'C=C',
               'CC=CC(=O)O',
               'CC=CC(=O)OCC',
               'CC=C(C)C(=O)OCC',
               'C=CC=C(C)C(=O)OCC',
               'CCOC(=O)C1=C(C(=O)OCC)C=CC=C1']

    assert att.check_elements(smis, ref_smi)


def test_get_vos_on_layer():
    print(flush=True)
    smis = ['CC(OC)C=C',
            'CC(OC)C',
            'CCC']
    graphs = [att.smi_to_nx(smi) for smi in smis]
    # combine the graphs into one graph
    combined = att.join_graphs(graphs)
    pathway = att.calculate_assembly_index(combined, strip_hydrogen=True)[-1]
    vos_layer_0 = att.get_vos_on_layer(pathway, 0)
    print("VOs on layer 0:", vos_layer_0, flush=True)
    assert len(vos_layer_0) == 3

    vos_layer_range = att.get_vos_on_layer(pathway, [0, 1])
    print("VOs on layers 0 and 1:", vos_layer_range, flush=True)
    assert len(vos_layer_range) == 2

    vos_layer_all = att.get_vos_on_layer(pathway, 'all')
    print("VOs on all layers:", vos_layer_all, flush=True)
    assert len(vos_layer_all) == 4
