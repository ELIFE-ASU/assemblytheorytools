import json
import os

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .graphtools import nx_to_mol


def get_disconnected_subgraphs(graph):
    # Return subgraphs of connected components without copying if not necessary
    return [graph.subgraph(c) for c in nx.connected_components(graph)]


def convert_edge_color(edge_color):
    # Use the predefined edge_color_map
    edge_color_map = {"single": 1, "double": 2, "triple": 3}
    return edge_color_map[edge_color]


def add_nodes_edges(graph, vertices, vertex_colors, edges, edge_colors):
    # Use zip to iterate over vertices and their colors
    for node, color in zip(vertices, vertex_colors):
        graph.add_node(node, color=color)
    # Convert edges to tuples to ensure they are hashable
    edges = [tuple(edge) for edge in edges]
    for edge, edge_color in zip(edges, edge_colors):
        graph.add_edge(*edge, color=convert_edge_color(edge_color))


def add_graph(graph_data):
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
    # Extract data from the 'file_graph' key
    graph_data = data['file_graph'][0]
    # Create a dictionary mapping vertices to their colors
    vert_col_dict = {vertex: color for vertex, color in zip(graph_data['Vertices'], graph_data['VertexColours'])}
    # Create a dictionary mapping edges (as tuples) to their colors
    edge_col_dict = {tuple(edge): color for edge, color in zip(graph_data['Edges'], graph_data['EdgeColours'])}
    return vert_col_dict, edge_col_dict


def extract_duplicates(edges, vert_col_dict, edge_col_dict):
    # Extract unique vertices
    verts = {v for edge in edges for v in edge}
    verts_c = [vert_col_dict[v] for v in verts]

    # Get colors of the edges directly without intermediate variable
    edges_c = [edge_col_dict[tuple(edge)] for edge in edges]

    return list(verts), verts_c, edges, edges_c


def create_graphs_from_data(file_path):
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


def get_pathway_to_inchi(file_path):
    graphs = create_graphs_from_data(file_path)
    out_dict = {}
    # Convert each section to inchi and store in out_dict
    for key in ['file_graph', 'remnant', 'duplicates', 'removed_edges']:
        if key in graphs:
            out_dict[key] = [Chem.MolToInchi(nx_to_mol(g)) for g in graphs[key]]
    return out_dict
