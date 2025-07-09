import json
import argparse
import networkx as nx

def parse_pathway_file(data):
    """
    Parse the pathway JSON file
    Returns a dict for each section.
    """
    parsed_pathway = {}

    file_graphs = []
    for idx, fg in enumerate(data.get('file_graph', [])):
        file_graphs.append({
            'vertices': fg.get('Vertices', []),
            'edges': fg.get('Edges', []),
            'vertex_colours': fg.get('VertexColours', []),
            'edge_colours': fg.get('EdgeColours', []),
        })
    parsed_pathway['file_graph'] = file_graphs

    remnants = []
    for idx, rem in enumerate(data.get('remnant', [])):
        remnants.append({
            'vertices': rem.get('Vertices', []),
            'edges': rem.get('Edges', []),
            'vertex_colours': rem.get('VertexColours', []),
            'edge_colours': rem.get('EdgeColours', []),
        })
    parsed_pathway['remnant'] = remnants

    duplicates = []
    for dup in data.get('duplicates', []):
        duplicates.append({
            'left_edges': dup.get('Left', []),
            'right_edges': dup.get('Right', [])
        })
    parsed_pathway['duplicates'] = duplicates

    parsed_pathway['removed_edges'] = data.get('removed_edges', [])

    return parsed_pathway

def calculateJO(data):
    """
    Calculates the raw MA and the JO index
    Auto-corrects for joint assembly spaces containing disjoint molecules
    Returns the raw MA and JO index
    """
    edge_set = set()
    edge_list = []
    edges = [fg["edges"] for fg in data["file_graph"]]

    original_graph = nx.Graph()

    for edge in edges[0]:
        tuple_edge = tuple(edge)
        edge_set.add(tuple_edge)
        edge_list.append(tuple_edge)
    
    original_graph.add_edges_from(edge_list)
    original_cc = nx.number_connected_components(original_graph)

    JO_correction = 0
    MA = original_graph.number_of_edges() - original_cc

    right_duplicates = [ dup_dict["right_edges"] for dup_dict in data["duplicates"] ]
    
    for fragment in right_duplicates:

        MA -= (len(fragment) - 1)

        fragment_atom_set = set()
        remnant_atom_set = set()
        remnant_graph = nx.Graph()
        remnant_edge_list = []

        for edge in fragment:
            fragment_atom_set.add(edge[0])
            fragment_atom_set.add(edge[1])
            edge_set.remove(tuple(edge))

        for edge in edge_set:
            remnant_atom_set.add(edge[0])
            remnant_atom_set.add(edge[1])
            remnant_edge_list.append(tuple(edge))

        remnant_graph.add_edges_from(remnant_edge_list)
        remnant_cc = nx.number_connected_components(remnant_graph)
        delta_cc = max(remnant_cc - original_cc, 0)
        original_cc = remnant_cc
        JO_correction += max(0, len(fragment_atom_set & remnant_atom_set) - 1) - delta_cc

    return MA, MA + JO_correction
        

def main():
    parser = argparse.ArgumentParser(description='Parse a graph JSON file.')
    parser.add_argument('json_file', help='Path to the JSON file to parse')
    args = parser.parse_args()

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parsed_pathway = parse_pathway_file(data)
    MA, JO_index = calculateJO(parsed_pathway)
    print("Raw MA: ", MA, "\nJO index: ", JO_index)


if __name__ == '__main__':
    main()
