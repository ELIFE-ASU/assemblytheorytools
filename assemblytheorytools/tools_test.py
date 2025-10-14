import networkx as nx


def check_elements(input_list, reference_list):
    """
    Check if all elements in the input list are present in the reference list.

    Parameters
    ----------
    input_list : list
        The list of elements to check.
    reference_list : list
        The list of reference elements.

    Returns
    -------
    bool
        True if all elements in input_list are in reference_list, False otherwise.
    """
    # Handle an empty list case
    if not input_list:
        return False

        # Check if all elements are in reference_list
    return all(item in reference_list for item in input_list)


def print_graph_details(graph):
    """
    Print the details of a graph, including node indices, node colours, edge connections, and edge colours.

    Parameters
    ----------
    graph : networkx.Graph
        The graph whose details are to be printed.

    Returns
    -------
    None
        This function prints information and does not return a value.
    """
    print("{", flush=True)
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        edge_colors = [graph.get_edge_data(*edge)['color'] for edge in edge_connections]
        print(f"({node_index}, {node_color}): {edge_connections}, {edge_colors}", flush=True)
    print("}", flush=True)


def water_graph() -> nx.Graph:
    """
    Constructs a graph representation of a water molecule.

    The graph consists of three nodes representing the atoms in a water molecule:
    one oxygen (O) and two hydrogens (H). Edges represent bonds between the atoms,
    with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the water molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="O")  # Oxygen atom
    graph.add_node(1, color="H")  # Hydrogen atom
    graph.add_node(2, color="H")  # Hydrogen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=1)  # Bond between oxygen and first hydrogen
    graph.add_edge(0, 2, color=1)  # Bond between oxygen and second hydrogen
    return graph


def phosphine_graph() -> nx.Graph:
    """
    Constructs a graph representation of a phosphine molecule.

    The graph consists of four nodes representing the atoms in a phosphine molecule:
    one phosphorus (P) and three hydrogens (H). Edges represent bonds between the atoms,
    with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="P")  # Phosphorus atom
    graph.add_node(1, color="H")  # Hydrogen atom
    graph.add_node(2, color="H")  # Hydrogen atom
    graph.add_node(3, color="H")  # Hydrogen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=1)  # Bond between phosphorus and first hydrogen
    graph.add_edge(0, 2, color=1)  # Bond between phosphorus and second hydrogen
    graph.add_edge(0, 3, color=1)  # Bond between phosphorus and third hydrogen
    return graph


def ph_2p_graph() -> nx.Graph:
    """
    Constructs a graph representation of a simple phosphine-like molecule.
    
    The system is +2 charged, with two phosphorus atoms and one hydrogen atom.
    The graph consists of two nodes representing the atoms: one phosphorus (P) 
    and one hydrogen (H). An edge represents the bond between the phosphorus 
    and hydrogen atoms, with the bond type indicated by an edge attribute.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine-like molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="P")  # Phosphorus atom
    graph.add_node(1, color="H")  # Hydrogen atom

    # Add edge with bond type attribute
    graph.add_edge(0, 1, color=1)  # Bond between phosphorus and hydrogen
    return graph


# add a function that creates a CO2 graph
def co2_graph() -> nx.Graph:
    """
    Constructs a graph representation of a carbon dioxide (CO2) molecule.

    The graph consists of three nodes representing the atoms in a CO2 molecule:
    one carbon (C) and two oxygens (O). Edges represent bonds between the atoms,
    with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the CO2 molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="C")  # Carbon atom
    graph.add_node(1, color="O")  # Oxygen atom
    graph.add_node(2, color="O")  # Oxygen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=2)  # Double bond between carbon and first oxygen
    graph.add_edge(0, 2, color=2)  # Double bond between carbon and second oxygen
    return graph
