import networkx as nx

def get_level(G: nx.DiGraph, node: str) -> int | None:
    """
    Returns the level of a node in a graph.
    
    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges represent assembly steps.
    node : str
        The node for which to determine the assembly depth.
    Returns
    -------
    int
        The assembly depth of the node, or None if the depth cannot be determined.
    Raises
    ------
    ValueError
        If the node is not present in the graph.
    """
    preds = [
        edge[0] for edge in list(G.in_edges(node))
    ] 
    if len(preds) > 0:
        return max([G.nodes[pred]["level"] for pred in preds]) + 1
    elif len(preds) == 0:
        return 0
    no_pred = [pred for pred in preds if "level" not in G.nodes[pred]]
    try:
        return get_level(G, no_pred[0])
    except Exception:
        return None

def assign_levels(G: nx.DiGraph, inplace: bool=True) -> None | nx.DiGraph:
    """
    Assigns assembly depth to nodes in a graph.
    For consistency, assembly depth is referred to as "level" in this context.
    
    
    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges represent assembly steps.
    Returns
    -------
    None or networkx.DiGraph
        If inplace is True, modifies the graph in place and returns None.
        If inplace is False, returns a new graph with updated node attributes.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph G must be a directed graph (DiGraph).")
    if not inplace:
        G = G.copy()

    for node in G.nodes:
        if not list(G.predecessors(node)):
            G.nodes[node].update({"level": 0})
        else:
            _ = get_level(G, node)
            G.nodes[node].update({"level": get_level(G, node)})

    if not inplace:
        return G
    return None