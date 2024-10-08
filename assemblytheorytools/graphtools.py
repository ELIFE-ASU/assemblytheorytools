import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .moltools import standardize_mol


def nx_to_mol(graph):
    # Create an editable RDKit molecule
    mol = Chem.RWMol()
    # Dictionary to map node identifiers to atom indices in the RDKit molecule
    node_to_idx = {}

    # Add atoms to the molecule
    for node, data in graph.nodes(data=True):
        # Get the atomic symbol from the node's 'color' attribute, default to 'C' if not present
        atom_symbol = data.get('color', 'C')
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        node_to_idx[node] = idx

    # Add bonds to the molecule
    for u, v, data in graph.edges(data=True):
        # Get the bond order from the edge's 'color' attribute, default to 1 if not present
        bond_order = data.get('color', 1)
        # Map the bond order to RDKit's bond types
        bond_type = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
        }.get(bond_order, Chem.rdchem.BondType.SINGLE)
        # Add the bond to the molecule
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Sanitize the molecule to generate implicit hydrogens and conformations
    standardize_mol(mol)

    # Return the immutable Mol object
    return mol.GetMol()


def mol_to_nx(mol):
    graph = nx.Graph()
    converter = {Chem.rdchem.BondType.SINGLE: 1,
                 Chem.rdchem.BondType.DOUBLE: 2,
                 Chem.rdchem.BondType.TRIPLE: 3,
                 Chem.rdchem.BondType.AROMATIC: 4}

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       color=atom.GetSymbol())

    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       color=converter[bond.GetBondType()])
    return graph


def write_ass_graph_file(graph, file_name="graph_info"):
    # Get the number of vertices
    num_vertices = graph.number_of_nodes()
    # Get the edges
    edges = list(graph.edges())
    # Get vertex colors
    vertex_colors = nx.get_node_attributes(graph, 'color')
    # Get edge colors
    edge_colors = nx.get_edge_attributes(graph, 'color')
    # Write the information to a file
    with open(file_name, 'w') as f:
        f.write(f"{graph.name}\n")
        f.write(f"{num_vertices}\n")
        f.write(" ".join([f"{e + 1}" for edge in edges for e in edge]) + "\n")
        f.write(" ".join([f"{color}" for node, color in vertex_colors.items()]) + "\n")
        f.write(" ".join([f"{color}" for node, color in edge_colors.items()]))


def graph_equal(graph1, graph2):
    # Function to determine if two graphs are isomorphic
    if prelim_graph_equal(graph1, graph2) == False:
        return False
    else:
        # Check if the two graphs are isomorphic
        return are_isomorphic_networkx(graph1, graph2)


def are_isomorphic_networkx(graph1, graph2):
    G1 = nx.Graph()
    G2 = nx.Graph()

    # Add nodes and edges with attributes to G1
    for node, color in zip(graph1.nodes, graph1.node_colors):
        G1.add_node(node, color=color)
    for edge, color in zip(graph1.edges, graph1.edge_colors):
        G1.add_edge(*edge, color=color)

    # Add nodes and edges with attributes to G2
    for node, color in zip(graph2.nodes, graph2.node_colors):
        G2.add_node(node, color=color)
    for edge, color in zip(graph2.edges, graph2.edge_colors):
        G2.add_edge(*edge, color=color)

    # Create a matcher object and use it to check isomorphism
    matcher = GraphMatcher(G1, G2, node_match=lambda n1, n2: n1['color'] == n2['color'],
                           edge_match=lambda e1, e2: e1['color'] == e2['color'])
    return matcher.is_isomorphic()


def prelim_graph_equal(graph1, graph2):
    # Function to determine if two graphs are preliminarily equal (i.e., same number of nodes and edges)
    if len(graph1.nodes) != len(graph2.nodes):
        return False
    if len(graph1.edges) != len(graph2.edges):
        return False
    # Check that the counts of edge colors and node colors are the same too
    for color in list(set(graph1.edge_colors)):
        if graph1.edge_colors.count(color) != graph2.edge_colors.count(color):
            return False
    for color in list(set(graph1.node_colors)):
        if graph1.node_colors.count(color) != graph2.node_colors.count(color):
            return False

    return True
