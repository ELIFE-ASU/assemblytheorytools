import json
import os

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .tools_graph import nx_to_mol, get_disconnected_subgraphs, bond_order_assout_to_int



def convert_pathway_dict_to_list(in_dict):
    """
    Convert a dictionary of pathways to a list.

    Args:
        in_dict (dict): A dictionary where keys are section names and values are lists of pathways.

    Returns:
        list: A list containing all pathways from the input dictionary.
    """
    in_list = []
    for key in in_dict:
        in_list.extend(in_dict[key])
    return in_list
