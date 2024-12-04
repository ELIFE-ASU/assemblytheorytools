import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.GraphDescriptors import BertzCT


def molecular_weight(mol):
    """
    Calculates the molecular weight of a molecule.

    Molecular weight is the sum of the atomic weights of all the atoms in a molecule.
    It is a simple measure of the molecule's size and is often used to estimate
    properties like solubility and boiling point.

    :param mol: An RDKit molecule object.
    :return: The molecular weight.
    """
    return Descriptors.MolWt(mol)


def bertz_complexity(mol):
    """
    Calculates the Bertz complexity of a molecule.

    Bertz complexity is a topological index that combines both bond and atom
    information. It is a measure of the structural complexity of the molecule
    and is used to compare the complexity of different molecules.

    :param mol: An RDKit molecule object.
    :return: The Bertz complexity.
    """
    return BertzCT(mol)


def wiener_index(mol):
    """
    Calculates the Wiener index of a molecule.

    Wiener index is a topological descriptor calculated as the sum of the shortest
    path lengths between all pairs of atoms in a molecule. It is a measure of the
    molecule's branching and can be used to estimate various physicochemical properties.

    :param mol: An RDKit molecule object.
    :return: The Wiener index.
    """
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    G = nx.Graph()

    # Add nodes
    for i in range(len(distance_matrix)):
        G.add_node(i)

    # Add edges
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    return nx.wiener_index(G)


def balaban_index(mol):
    """
    Calculates the Balaban index of a molecule.

    Balaban index is a topological descriptor that quantifies the degree of
    branching in a molecule. It is useful for predicting physicochemical
    properties and biological activities of molecules.

    :param mol: An RDKit molecule object.
    :return: The Balaban index.
    """
    return Descriptors.BalabanJ(mol)


def randic_index(mol):
    """
    Calculates the Randic index of a molecule.

    Randic index is a topological descriptor calculated by summing the
    inverse square roots of the product of the degrees of connected atom pairs.
    It is used for the study of molecular structure-activity
    relationships and the prediction of physicochemical properties.

    :param mol: An RDKit molecule object.
    :return: The Randic index.
    """
    G = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degrees = [sum(row) for row in G]
    randic_sum = 0
    for i, row in enumerate(G):
        for j, val in enumerate(row):
            if val == 1:
                randic_sum += 1 / (degrees[i] * degrees[j]) ** 0.5
    return randic_sum / 2


def kirchhoff_index(mol):
    """
    Calculates the Kirchhoff index of a molecule.

    Kirchhoff index is a topological index calculated as the sum of the effective
    resistances between all pairs of vertices in the molecular graph. It is used for
    predicting physicochemical properties and molecular activities.

    :param mol: An RDKit molecule object.
    :return: The Kirchhoff index.
    """
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.float64)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    pseudo_inverse_laplacian = np.linalg.pinv(laplacian_matrix)
    diagonal_elements = np.diagonal(pseudo_inverse_laplacian)

    kirchhoff_sum = 0
    for i in range(len(diagonal_elements)):
        for j in range(i + 1, len(diagonal_elements)):
            kirchhoff_sum += (diagonal_elements[i] + diagonal_elements[j] - 2 * pseudo_inverse_laplacian[i, j])

    return kirchhoff_sum


def spacial_score(mol, normalize=False):
    """
    Calculates the spacial score of a molecule. https://github.com/frog2000/Spacial-Score

    Spacial score is a descriptor that quantifies the spatial arrangement
    of atoms in a molecule. It can be used to predict various molecular properties.

    :param mol: An RDKit molecule object.
    :param normalize: A boolean indicating whether to normalize the score.
    :return: The spacial score of the molecule.
    """
    return rdkit.Chem.SpacialScore.SPS(mol, normalize)
