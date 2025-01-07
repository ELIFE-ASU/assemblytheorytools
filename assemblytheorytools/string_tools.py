import numpy as np
import networkx as nx

def load_fasta(file_path):
    """
    Load a FASTA file and return its contents as a single string.

    Args:
        file_path (str): The path to the FASTA file.

    Returns:
        str: The contents of the FASTA file as a single string.
    """
    # Load the file contents into a NumPy array
    fasta_array = np.genfromtxt(file_path, dtype=str, delimiter='\n', comments='>')
    # Join the array elements into a single string
    fasta_content = ''.join(fasta_array)
    return fasta_content


def prep_joint_string_ai(input_list):
    """
    Compute the joint assembly index of strings by concatenating them with unique delimiters.

    Args:
        input_list (list of str): List of input strings to be concatenated.

    Raises:
        ValueError: If an empty string is found in the input list.

    Returns:
        tuple: A tuple containing the concatenated string and a list of unique delimiters used.
    """
    if "" in input_list:
        raise ValueError("Empty string in input list")

    # Build a string of all the inputs separated by unique dummy characters
    delimiters = []
    amalgam_string = input_list[0]
    for string in input_list[1:]:
        unique_char = get_unique_char(amalgam_string + string)
        amalgam_string += unique_char + string
        delimiters.append(unique_char)

    # To use this to calculate joint assembly index, use formula:
    # ai(amalgam_string) - 2 * len(delimiters) = joint_ai(input_list)
    # delimiters can be used to process the pathway
    return amalgam_string, delimiters


def get_unique_char(input_str):
    """
    Find a unique character that is not present in the input string.

    Args:
        input_str (str): The input string to check against.

    Returns:
        str: A unique character not present in the input string.
    """
    for i in range(1,
                   1114111):  # Max Unicode code point is 0x10FFFF, which is 1114111 in decimal, start at 1 to skip null
        char = chr(i)
        if char not in input_str:
            return char


def get_dirstr_molecule(dir_str):
    """
    Make a molecule that corresponds to a directed string. The string will have the same assembly index
    as the molecular graph, and the paths will correspond as well.

    Args:
        dir_str (str): The directed string.

    Returns:
        graph: A networkx graph of the corresponding molecule.
    """
    
    graph = nx.Graph()
    graph.add_node(0, label=0)
    graph.add_node(1,label=0)
    graph.add_edge(0,1,label=dir_str[0])
    for i in range(1, len(dir_str)):
        graph.add_node(i+1, label=0)
        graph.add_edge(i, i+1,label=dir_str[i])

    return graph


def get_undir_str_molecule(undir_str):
    """
    Make a molecule that corresponds to an undirected string. The string will have the same assembly index
    as the molecular graph, and the paths will correspond as well.

    Args:
        undir_str (str): The undirected string.

    Returns:
        graph: A networkx graph of the corresponding molecule.
    """
    blank = get_unique_char(undir_str)
    graph = nx.Graph()
    graph.add_node(0, label=blank)
    graph.add_node(1, label=undir_str[0])
    graph.add_node(2, label=blank)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    for i in range(1, len(undir_str)):
        graph.add_node(2*i+1, label=undir_str[i])
        graph.add_edge(2*i, 2*i+1)
        graph.add_node(2*i+2, label=blank)
        graph.add_edge(2*i+1, 2*i+2)

    return graph
