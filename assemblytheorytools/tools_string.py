"""
String assembly helpers.

This module supports assembly index calculations on sequences rather than
molecules. It loads FASTA files, concatenates strings with unique delimiters for
joint assembly calculations, generates random test strings, and builds the
directed and undirected graph representations of a string.
"""

import networkx as nx
import random
import string
from typing import List


def load_fasta(file_path: str) -> str:
    """
    Load a FASTA file and return its contents as a single string.

    This function ignores header lines (starting with '>') and
    concatenates all sequence lines.

    Parameters
    ----------
    file_path : str
        The path to the FASTA file.

    Returns
    -------
    str
        The contents of the FASTA file as a single string with all
        sequence lines concatenated.
    """
    sequence_content = ""

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            # Skip header lines that start with '>'
            if not line.startswith(">"):
                sequence_content += line

    return sequence_content


def prep_joint_string_ai(input_list: list[str]) -> tuple[str, list[str]]:
    """Combine a list of strings by concatenating them with unique delimiters.

    Parameters
    ----------
    input_list : list[str]
        A list of input strings to be concatenated.

    Returns
    -------
    tuple[str, list[str]]
        A tuple containing:

        - The concatenated string.
        - A list of the unique delimiters used.

        The joint assembly index can be calculated using the formula:
        ``ai(amalgam_string) - 2 * len(delimiters) = joint_ai(input_list)``

    Raises
    ------
    ValueError
        If an empty string is found in the input list.
    """
    if not input_list:
        raise ValueError("Input list cannot be empty")
    if "" in input_list:
        raise ValueError("Empty string in input list")

    # Build a string of all the inputs separated by unique fake characters
    delimiters: List[str] = []
    reserved_chars = "".join(input_list)
    amalgam_string: str = input_list[0]
    for item in input_list[1:]:
        # Reserve characters from every input up front. Looking only at the
        # strings processed so far allowed an early delimiter to collide with
        # a character in a later string.
        unique_char = get_unique_char(reserved_chars + "".join(delimiters))
        amalgam_string += unique_char + item
        delimiters.append(unique_char)

    return amalgam_string, delimiters


def get_unique_char(input_str: str) -> str:
    """Find a unique character that is not present in the given input string.

    This function first attempts to find a unique character from the set of
    printable ASCII characters. If no unique character is found, it falls
    back to searching a broader range of Unicode characters.

    Parameters
    ----------
    input_str : str
        The input string to check for unique characters.

    Returns
    -------
    str
        A character that is not present in the input string.

    Raises
    ------
    ValueError
        If no unique character can be found within the specified ranges of
        characters.
    """
    # Try ASCII printable characters first
    for char in string.printable:
        if char not in input_str and char != ' ':
            return char

    # Try a broader range of Unicode characters (excluding surrogates and control chars)
    for codepoint in range(0x00A1, 0x2FFF):  # Example: Latin-1 Supplement to CJK Radicals
        char = chr(codepoint)
        if char.isprintable() and char not in input_str:
            return char

    # Raise an error if no unique character is found
    raise ValueError("Ran out of delimiter symbols. Try broadening the range of allowable symbols.")


def get_undir_str_molecule(
        undir_str: str, debug: bool = False
) -> tuple[nx.Graph, dict[str, str]]:
    """Create a molecular graph from an undirected string.

    The resulting molecular graph has the same assembly index as the string,
    and the paths correspond between the two.

    Parameters
    ----------
    undir_str : str
        The undirected string to convert.
    debug : bool, optional
        If ``True``, print debug information. Defaults to ``False``.

    Returns
    -------
    tuple[nx.Graph, dict[str, str]]
        A tuple containing:

        - A NetworkX graph of the corresponding molecule.
        - A dictionary mapping characters to edge colors (as strings).
    """

    # Create a dictionary to map each unique character in the undirected string to a unique edge colour
    edge_color_dict: dict[str, str] = {}
    for i, char in enumerate(sorted(set(undir_str))):
        edge_color_dict[char] = str(i + 1)

    # If debug is enabled, print the edge colour dictionary
    if debug:
        print("Edge color dict:", flush=True)
        print(edge_color_dict, flush=True)

    # Initialise the graph and add the first two nodes with a 'null' colour
    blank = 'null'
    graph = nx.Graph()
    graph.add_node(0, color=blank)
    graph.add_node(1, color=blank)

    # Add the first edge with the colour corresponding to the first character in the undirected string
    graph.add_edge(0, 1, color=int(edge_color_dict[undir_str[0]]))

    # Iterate through the rest of the undirected string, adding nodes and edges to the graph
    for i in range(1, len(undir_str)):
        graph.add_node(i + 1, color=blank)
        graph.add_edge(i, i + 1, color=int(edge_color_dict[undir_str[i]]))

    # Return the graph and the edge colour dictionary
    return graph, edge_color_dict


def get_dir_str_molecule(dir_str: str) -> nx.Graph:
    """Create a molecular graph from a directed string.

    The assembly index of the string is determined by the molecular graph,
    and the shortest paths correspond.

    Parameters
    ----------
    dir_str : str
        The directed string to convert.

    Returns
    -------
    nx.Graph
        A NetworkX graph of the corresponding molecule.
    """
    blank = 'null'
    graph = nx.Graph()
    graph.add_node(0, color=blank)
    graph.add_node(1, color=dir_str[0])
    graph.add_node(2, color=blank)
    graph.add_edge(0, 1, color=1)
    graph.add_edge(1, 2, color=2)
    for i in range(1, len(dir_str)):
        graph.add_node(2 * i + 1, color=dir_str[i])
        graph.add_edge(2 * i, 2 * i + 1, color=1)
        graph.add_node(2 * i + 2, color=blank)
        graph.add_edge(2 * i + 1, 2 * i + 2, color=2)

    return graph


def generate_random_strings(n_pool: int, n_length: int) -> list[str]:
    """
    Generate a list of random strings of a specified length.

    This function creates `n_pool` random strings, each of length `n_length`,
    using lowercase letters.

    Parameters
    ----------
    n_pool : int
        The number of random strings to generate.
    n_length : int
        The length of each random string.

    Returns
    -------
    list[str]
        A list of randomly generated strings.
    """
    # Define the character set to include lowercase letters
    chars = string.ascii_lowercase

    # Generate a list of random strings using the specified character set
    return [''.join(random.choices(chars, k=n_length)) for _ in range(n_pool)]
