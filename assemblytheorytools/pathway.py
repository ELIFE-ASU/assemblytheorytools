import json
import os

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .tools_graph import nx_to_mol, get_disconnected_subgraphs


def convert_edge_color(edge_color):
    """
    Convert an edge color descriptor to its corresponding numerical value.

    Args:
        edge_color (str or int): The descriptor of the edge color (e.g., 'single', ....) or an integer value.

    Returns:
        int: The numerical value corresponding to the edge color.
    """
    edge_color_map = {
        "single": 1,
        "double": 2,
        "triple": 3,
        "quadruple": 4,
        "quintuple": 5,
        "ionic": 6,
    }

    if edge_color in edge_color_map:
        return edge_color_map[edge_color]
    else:
        return int(edge_color)


def add_nodes_edges(graph, vertices, vertex_colors, edges, edge_colors):
    """
    Add nodes and edges to a NetworkX graph with specified colors.

    Args:
        graph (networkx.Graph): The graph to which nodes and edges will be added.
        vertices (list): A list of vertices to be added to the graph.
        vertex_colors (list): A list of colors corresponding to each vertex.
        edges (list): A list of edges to be added to the graph.
        edge_colors (list): A list of colors corresponding to each edge.

    Returns:
        None
    """
    # Use zip to iterate over vertices and their colors
    for node, color in zip(vertices, vertex_colors):
        graph.add_node(node, color=color)
    # Convert edges to tuples to ensure they are hashable
    edges = [tuple(edge) for edge in edges]
    for edge, edge_color in zip(edges, edge_colors):
        graph.add_edge(*edge, color=convert_edge_color(edge_color))


def add_graph(graph_data):
    """
    Create a NetworkX graph from the provided graph data and return its connected subgraphs.

    Args:
        graph_data (dict): A dictionary containing the graph data with keys 'Vertices', 'VertexColours', 'Edges', and 'EdgeColours'.

    Returns:
        list: A list of subgraphs, each representing a connected component.
    """
    graph = nx.Graph()
    # Add nodes and edges to the graph
    add_nodes_edges(
        graph,
        graph_data['Vertices'],
        graph_data['VertexColours'],
        graph_data['Edges'],
        graph_data['EdgeColours']
    )
    # Return connected subgraphs
    return get_disconnected_subgraphs(graph)


def get_conversion_dict(data):
    """
    Extract vertex and edge color mappings from the provided graph data.

    Args:
        data (dict): A dictionary containing the graph data with a key 'file_graph'.

    Returns:
        tuple: A tuple containing two dictionaries:
            - vert_col_dict (dict): A dictionary mapping vertices to their colors.
            - edge_col_dict (dict): A dictionary mapping edges (as tuples) to their colors.
    """
    # Extract data from the 'file_graph' key
    graph_data = data['file_graph'][0]
    # Create a dictionary mapping vertices to their colors
    vert_col_dict = {vertex: color for vertex, color in zip(graph_data['Vertices'], graph_data['VertexColours'])}
    # Create a dictionary mapping edges (as tuples) to their colors
    edge_col_dict = {tuple(edge): color for edge, color in zip(graph_data['Edges'], graph_data['EdgeColours'])}
    return vert_col_dict, edge_col_dict


def extract_duplicates(edges, vert_col_dict, edge_col_dict):
    """
    Extract unique vertices and their colors, and get the colors of the edges.

    Args:
        edges (list): A list of edges, where each edge is represented as a tuple of vertices.
        vert_col_dict (dict): A dictionary mapping vertices to their colors.
        edge_col_dict (dict): A dictionary mapping edges (as tuples) to their colors.

    Returns:
        tuple: A tuple containing:
            - list: A list of unique vertices.
            - list: A list of colors corresponding to the unique vertices.
            - list: The original list of edges.
            - list: A list of colors corresponding to the edges.
    """
    # Extract unique vertices
    verts = {v for edge in edges for v in edge}
    verts_c = [vert_col_dict[v] for v in verts]

    # Get colors of the edges directly without intermediate variable
    edges_c = [edge_col_dict[tuple(edge)] for edge in edges]

    return list(verts), verts_c, edges, edges_c


def get_pathway_to_graph(file_path):
    """
    Load graph data from a JSON file and create NetworkX graphs for different sections.

    Args:
        file_path (str): The path to the JSON file containing the graph data.

    Returns:
        dict: A dictionary containing NetworkX graphs for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """
    # Load data from the JSON file
    with open(os.path.abspath(file_path), 'r') as file:
        data = json.load(file)

    # Get vertex and edge color mappings
    vert_col_dict, edge_col_dict = get_conversion_dict(data)
    graphs = {}

    # List of keys that use the add_graph function directly
    direct_graph_keys = ['file_graph', 'remnant']
    for key in direct_graph_keys:
        if key in data:
            graphs[key] = add_graph(data[key][0])

    # Process 'duplicates' if present
    if 'duplicates' in data:
        duplicate_graphs = []
        for dup in data['duplicates']:
            graph = nx.Graph()
            # Extract and add nodes and edges
            add_nodes_edges(
                graph,
                *extract_duplicates(dup['Left'], vert_col_dict, edge_col_dict)
            )
            duplicate_graphs.append(graph)
        graphs['duplicates'] = duplicate_graphs

    # Process 'removed_edges' if present
    if 'removed_edges' in data:
        removed_graph = nx.Graph()
        add_nodes_edges(
            removed_graph,
            *extract_duplicates(data['removed_edges'], vert_col_dict, edge_col_dict)
        )
        graphs['removed_edges'] = get_disconnected_subgraphs(removed_graph)

    return graphs


def get_pathway_to_mol(file_path):
    """
    Convert graph data from a JSON file to RDKit molecule objects.

    Args:
        file_path (str): The path to the JSON file containing the graph data.

    Returns:
        dict: A dictionary containing RDKit molecule objects for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """
    graphs = get_pathway_to_graph(file_path)
    out_dict = {}
    # Convert each section to RDKit molecule objects and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in graphs:
            out_dict[key] = [nx_to_mol(g) for g in graphs[key]]
    return out_dict


def get_pathway_to_inchi(file_path):
    """
    Convert graph data from a JSON file to InChI strings.

    Args:
        file_path (str): The path to the JSON file containing the graph data.

    Returns:
        dict: A dictionary containing InChI strings for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """
    graphs = get_pathway_to_graph(file_path)
    out_dict = {}
    # Convert each section to InChI and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in graphs:
            out_dict[key] = [Chem.MolToInchi(nx_to_mol(g)) for g in graphs[key]]
    return out_dict


def get_pathway_to_smi(file_path):
    """
    Convert graph data from a JSON file to SMILES strings.

    Args:
        file_path (str): The path to the JSON file containing the graph data.

    Returns:
        dict: A dictionary containing SMILES strings for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """
    graphs = get_pathway_to_graph(file_path)
    out_dict = {}
    # Convert each section and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in graphs:
            out_dict[key] = [Chem.MolToSmiles(nx_to_mol(g)) for g in graphs[key]]
    return out_dict


def get_mol_pathway_to_inchi(pathway):
    """
    Convert a pathway of RDKit molecule objects to InChI strings.

    Args:
        pathway (dict): A dictionary containing RDKit molecule objects for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.

    Returns:
        dict: A dictionary containing InChI strings for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """

    # Detect the file_graph data type
    dtype = type(pathway['file_graph'][0])

    out_dict = {}
    # Convert each section to InChI and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in pathway:
            if dtype == nx.Graph:
                out_dict[key] = [Chem.MolToInchi(nx_to_mol(g)) for g in pathway[key]]
            else:
                out_dict[key] = [Chem.MolToInchi(g) for g in pathway[key]]
    return out_dict


def get_mol_pathway_to_smi(pathway):
    """
    Convert a pathway of RDKit molecule objects to SMILES strings.

    Args:
        pathway (dict): A dictionary containing RDKit molecule objects for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.

    Returns:
        dict: A dictionary containing SMILES strings for different sections such as 'file_graph', 'remnant', 'duplicates', and 'removed_edges'.
    """
    # Detect the file_graph data type
    dtype = type(pathway['file_graph'][0])

    out_dict = {}
    # Convert each section and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in pathway:
            if dtype == nx.Graph:
                out_dict[key] = [Chem.MolToSmiles(nx_to_mol(g)) for g in pathway[key]]
            else:
                out_dict[key] = [Chem.MolToSmiles(g) for g in pathway[key]]
    return out_dict


def convert_pathway_dict_to_list(in_dict):
    """
    Convert a dictionary of pathways to a list.

    Args:
        in_dict (dict): A dictionary where keys are section names and values are lists of pathways.

    Returns:
        list: A list containing all pathways from the input dictionary.
    """
    in_list = []
    for key in in_dict:
        in_list.extend(in_dict[key])
    return in_list
