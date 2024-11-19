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

# Hack to compute joint assembly index of strings
def prep_joint_string_ai(input_list, reversible = False, timeout = 10.0):
    if "" in input_list:
        raise ValueError("Empty string in input list")
    # Build a string of all the inputs seperated by unique dummy characters
    delimiters = []
    amalgam_string = input_list[0]
    for string in input_list[1:]:
        unique_char = get_unique_char(amalgam_string + string)
        amalgam_string += unique_char + string
        delimiters.append(unique_char)
    
    # To use this to calculate joint assembly index, use formula:
    # ai(amalgum_string)-2*len(delimiters) = joint_ai(input_list)
    # 
    # delimiters can be used to process the pathway 
    return amalgam_string, delimiters

def get_unique_char(input_str):
    for i in range(1, 1114111):  # Max Unicode code point is 0x10FFFF, which is 1114111 in decimal, start at 1 to skip null
        char = chr(i)
        if char not in input_str:
            return char
