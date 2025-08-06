import json
import networkx as nx
import os

def immediate_predecessors(data, interval):
    """    
    For example, if we have the abracadabra data, and we want the whole interval,
    then it will return ["abra", "c", "a", "d", "abra"].
    
    Args:
        data (json): The pathway data from assemblycpp.
        interval (tuple): A tuple of the form (start, length) indicating the interval.
    
    Returns:
        list: A list of strings representing the immediate predecessors in the pathway.
    """
    output = []

    c_idx = interval[0]
    while c_idx < sum(interval):
        parent = ""
        for dup in data["duplicates"]:
            if dup["Left"][1] < interval[1]: # Make sure the duplicate can fit in the interval
                if c_idx in range(dup["Left"][0], sum(dup["Left"])): # If duplicate contains c_idx
                    if dup["Left"][1] > len(parent): # If this duplicate is larger than the current parent
                        if dup["Left"][0] >= interval[0] and sum(dup["Left"]) <= sum(interval): # If this duplicate is within the interval
                            parent = data["file_graph"][0]["Fragments"][0][dup["Left"][0]:sum(dup["Left"])]
                elif c_idx in range(dup["Right"][0], sum(dup["Right"])): # now check the right copy
                    if dup["Right"][1] > len(parent):
                        if dup["Right"][0] >= interval[0] and sum(dup["Right"]) <= sum(interval): 
                            parent = data["file_graph"][0]["Fragments"][0][dup["Right"][0]:sum(dup["Right"])]
        if parent == "":
            output.append(data["file_graph"][0]["Fragments"][0][c_idx])
            c_idx += 1
        else:
            output.append(parent)
            c_idx += len(parent)
    
    return output


def build_str(interval, data, path):
    """
    Builds the string from the pathway data and adds it to the path.
    
    Args:
        str (str): The string to build.
        data (json): The pathway data from assemblycpp.
        path (networkx.DiGraph): The current pathway graph.
    
    Returns:
        networkx.DiGraph: Updated pathway with the string added.
    """

    ledger = immediate_predecessors(data, interval)
    c_idx = interval[0]
    for sub_str in ledger:
        if sub_str not in path.nodes:
            path = build_str([c_idx, c_idx+len(sub_str)], data, path) # Recursively build the duplicate strings if not already in the path
        c_idx += len(sub_str)
    
    str_in_progress = ledger[0]
    for idx in range(1, len(ledger)): # Builds string from left to right
        str_in_progress_new = str_in_progress + ledger[idx]
        if str_in_progress_new not in path.nodes: # Only relevant if the path is not minimum
            path.add_node(str_in_progress_new)
        if (str_in_progress, str_in_progress_new) not in path.edges: # Only relevant if the path is not minimum
            path.add_edge(str_in_progress, str_in_progress_new)
        if (ledger[idx], str_in_progress_new) not in path.edges: # Only relevant if the path is not minimum
            path.add_edge(ledger[idx], str_in_progress_new)
        str_in_progress = str_in_progress_new
    return path


def parse_string_pathway_file(file_path_pathway):
    """
    Parses a pathway file and returns the pathway as a list of virtual objects.
    
    Args:
        file_path_pathway (str): Path to the pathway file.
    
    Returns:
        VOs: dict of virtual objects in calculated pathway.
        path: networkx.DiGraph representing the pathway.
    """
    if not os.path.isfile(file_path_pathway):
        raise FileNotFoundError(f"Pathway file not found: {file_path_pathway}")
    else:
        # Load the pathway file
        with open(file_path_pathway) as f:
            data = json.load(f)
   
    VOs = dict()
    VOs["file_string"] = data["file_graph"][0]["Fragments"][0]
    VOs["remnant"] = data["remnant"][0]["Fragments"]
    dups = [data["duplicates"][i]["Left"] for i in range(len(data["duplicates"]))]
    VOs["duplicates"] = [VOs["file_string"][dup[0]:sum(dup)] for dup in dups]
    
    path = nx.DiGraph()

    # We will build the string from left to right, constructing duplicates as needed
    for char in list(set(VOs["file_string"])):
        path.add_node(char) # Add units

    path = build_str([0, len(VOs["file_string"])], data, path) # Build the string from the pathway data
    
    return VOs, path 
