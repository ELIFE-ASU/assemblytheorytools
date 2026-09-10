"""
String assembly helpers.

This module supports assembly index calculations on sequences rather than
molecules. It loads FASTA files, concatenates strings with unique delimiters for
joint assembly calculations, generates random test strings, and builds the
directed and undirected graph representations of a string.
"""

import random
import string

import networkx as nx


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
    with open(file_path, "r") as file:
        lines = (line.strip() for line in file)
        return "".join(line for line in lines if not line.startswith(">"))


def prep_joint_string_ai(input_list: list[str]) -> tuple[str, list[str]]:
    """
    Combine a list of strings by concatenating them with unique delimiters.

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
        If the input list is empty or contains an empty string.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.prep_joint_string_ai(["abracadabra", "abra"])
    ('abracadabra0abra', ['0'])

    The separator is chosen by :func:`get_unique_char` so that it cannot
    appear in any of the inputs.
    """
    if not input_list:
        raise ValueError("Input list cannot be empty")
    if "" in input_list:
        raise ValueError("Empty string in input list")

    # Reserve every input character before choosing the first delimiter.
    reserved_chars = "".join(input_list)
    delimiters: list[str] = []
    parts = [input_list[0]]
    for item in input_list[1:]:
        delimiter = get_unique_char(reserved_chars)
        reserved_chars += delimiter
        delimiters.append(delimiter)
        parts.extend((delimiter, item))

    return "".join(parts), delimiters


def get_unique_char(input_str: str) -> str:
    """
    Find a unique character that is not present in the given input string.

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

    Examples
    --------
    This returns a character *absent* from the input, not the input's
    alphabet. It is how :func:`prep_joint_string_ai` picks a safe separator:

    >>> import assemblytheorytools as att
    >>> att.get_unique_char("abracadabra")
    '0'
    >>> att.get_unique_char("abc0")
    '1'
    """
    used_chars = set(input_str)
    for char in string.printable:
        if char != " " and char not in used_chars:
            return char

    for codepoint in range(0x00A1, 0x2FFF):
        char = chr(codepoint)
        if char.isprintable() and char not in used_chars:
            return char

    raise ValueError(
        "Ran out of delimiter symbols. Try broadening the range of allowable symbols."
    )


def get_undir_str_molecule(
    undir_str: str, debug: bool = False
) -> tuple[nx.Graph, dict[str, str]]:
    """
    Create a molecular graph from an undirected string.

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

    Raises
    ------
    IndexError
        If ``undir_str`` is empty.
    """
    edge_color_dict = {
        char: str(index)
        for index, char in enumerate(sorted(set(undir_str)), start=1)
    }
    if debug:
        print("Edge color dict:", flush=True)
        print(edge_color_dict, flush=True)

    if not undir_str:
        raise IndexError("string index out of range")

    graph = nx.Graph()
    graph.add_nodes_from(range(len(undir_str) + 1), color="null")
    graph.add_edges_from(
        (index, index + 1, {"color": int(edge_color_dict[char])})
        for index, char in enumerate(undir_str)
    )
    return graph, edge_color_dict


def get_dir_str_molecule(dir_str: str) -> nx.Graph:
    """
    Create a molecular graph from a directed string.

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

    Raises
    ------
    IndexError
        If ``dir_str`` is empty.
    """
    if not dir_str:
        raise IndexError("string index out of range")

    graph = nx.Graph()
    graph.add_node(0, color="null")
    for index, char in enumerate(dir_str):
        node = 2 * index + 1
        graph.add_node(node, color=char)
        graph.add_node(node + 1, color="null")
        graph.add_edge(node - 1, node, color=1)
        graph.add_edge(node, node + 1, color=2)
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

    Examples
    --------
    Useful as a null model: a random string has almost no repetition to
    exploit, so its assembly index sits close to its length.

    >>> import assemblytheorytools as att
    >>> pool = att.generate_random_strings(3, 8)
    >>> len(pool), {len(s) for s in pool}
    (3, {8})

    An 11-character random string scores 10, against 7 for
    ``abracadabra``; the difference is what the internal structure buys.
    """
    return [
        "".join(random.choices(string.ascii_lowercase, k=n_length))
        for _ in range(n_pool)
    ]
