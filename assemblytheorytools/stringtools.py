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
