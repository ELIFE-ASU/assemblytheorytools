def check_elements(input_list, reference_list):
    """
    Check if all elements in the input list are present in the reference list.

    Args:
        input_list (list): The list of elements to check.
        reference_list (list): The list of reference elements.

    Returns:
        bool: True if all elements in input_list are in reference_list, False otherwise.
    """
    # Handle empty list case
    if not input_list:
        return False

        # Check if all elements are in reference_list
    return all(item in reference_list for item in input_list)


def print_graph_details(graph):
    """
    Print the details of a graph, including node indices, node colors, edge connections, and edge colors.

    Args:
        graph (networkx.Graph): The graph whose details are to be printed.

    Returns:
        None
    """
    print("{", flush=True)
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        edge_colors = [graph.get_edge_data(*edge)['color'] for edge in edge_connections]
        print(f"({node_index}, {node_color}): {edge_connections}, {edge_colors}", flush=True)
    print("}", flush=True)



