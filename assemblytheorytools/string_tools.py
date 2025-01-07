import numpy as np


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
