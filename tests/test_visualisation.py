import os
import platform
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import assemblytheorytools as att


def test_plot_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_mol_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    fig, ax = att.plot_mol_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_interactive_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    att.plot_interactive_graph(graph)
    assert os.path.isfile('interactive_graph.html'), "Failed to generate the file."
    os.remove('interactive_graph.html')
    # remove the folder lib
    if os.path.exists('lib'):
        shutil.rmtree('lib')


def test_plot_digraph():
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_digraph_metro_calc():
    print(flush=True)
    if platform.system().lower() == "linux":
        # Define the SMILES string for glycine
        smi = "C(C(=O)O)N"

        # Convert to Mol object
        mol = att.smi_to_mol(smi)
        # Compute the assembly index and associated data
        _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)
        att.plot_digraph_metro(pathway)
        assert os.path.isfile('metro.png'), "Failed to generate the file."
        assert os.path.isfile('metro.svg'), "Failed to generate the file."
        os.remove('metro.png')
        os.remove('metro.svg')

        # Convert to Graph
        graph = att.smi_to_nx(smi)
        # Compute the assembly index and associated data
        _, _, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
        att.plot_digraph_metro(pathway)
        assert os.path.isfile('metro.png'), "Failed to generate the file."
        assert os.path.isfile('metro.svg'), "Failed to generate the file."
        os.remove('metro.png')
        os.remove('metro.svg')
    else:
        print("Skipping test_plot_digraph_metro_calc: not running on Linux.", flush=True)


def test_plot_digraph_topological():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_graph(pathway, layout='topological')
    plt.show()

    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_pathway_mol():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_pathway(pathway, plot_type='mol')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."

    # Convert the SMILES string to an RDKit Mol object
    graph = att.smi_to_nx(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    fig, ax = att.plot_pathway(pathway, plot_type='graph')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=False)

    fig, ax = att.plot_pathway(pathway, plot_type='atoms')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_assembly_circle():
    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']
    os.environ["ASS_PATH"] = "/Users/ejanin/Desktop/assemblycpp/assemblyCpp_linux_v5_combined"
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj_matrix[i, i + 1] = 1  # Chain
    adj_matrix[0, 4] = 1  # b -> ba (branch)
    adj_matrix[1, 6] = 1  # a -> baa (branch)
    adj_matrix[3, 5] = 1  # c -> dc (branch)

    labels = nodes
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.png'

    fig, ax = att.plot_assembly_circle(
        nodes=nodes,
        adj_matrix=adj_matrix,
        labels=labels,
        node_size=node_size,
        arrow_size=arrow_size,
        node_color=node_color,
        edge_color=edge_color,
        fig_size=fig_size,
        filename=filename
    )

    assert os.path.isfile('circle_plot.png'), "Failed to generate the file."
    os.remove('circle_plot.png')


from typing import Tuple, Dict, List
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw


def show_common_bonds(
        smiles_a: str,
        smiles_b: str,
        legends: List[str] | None = None,
        common_bond_color: Tuple[float, float, float] = (0.1, 0.8, 0.1),
        common_atom_color: Tuple[float, float, float] = (0.1, 0.8, 0.1),
        size: Tuple[int, int] = (700, 350),
        timeout_s: int = 5,
        ring_matches_ring_only: bool = True,
        complete_rings_only: bool = True,
):
    if legends is None:
        legends = ["A", "B"]
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        raise ValueError("One or both SMILES strings could not be parsed by RDKit.")

    # standardize molecule layouts for better visualization
    mol_a = att.standardize_mol(mol_a, add_hydrogens=False)
    mol_b = att.standardize_mol(mol_b, add_hydrogens=False)

    # Compute MCS (Maximum Common Substructure)
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = int(timeout_s)
    mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrderExact
    mcs_params.RingMatchesRingOnly = bool(ring_matches_ring_only)
    mcs_params.CompleteRingsOnly = bool(complete_rings_only)

    mcs_res = rdFMCS.FindMCS([mol_a, mol_b], mcs_params)
    if not mcs_res.smartsString:
        # No overlap found; draw without highlights.
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    mcs_mol = Chem.MolFromSmarts(mcs_res.smartsString)
    if mcs_mol is None:
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    match_a = mol_a.GetSubstructMatch(mcs_mol)
    match_b = mol_b.GetSubstructMatch(mcs_mol)

    if not match_a or not match_b:
        # If MCS SMARTS can't be matched back (rare), draw without highlights.
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    # Map MCS bonds to bond indices in each molecule
    def mcs_bond_indices(parent_mol, match: Tuple[int, ...]) -> List[int]:
        bond_idxs = []
        for b in mcs_mol.GetBonds():
            a1 = match[b.GetBeginAtomIdx()]
            a2 = match[b.GetEndAtomIdx()]
            pb = parent_mol.GetBondBetweenAtoms(a1, a2)
            if pb is not None:
                bond_idxs.append(pb.GetIdx())
        return bond_idxs

    common_bonds_a = mcs_bond_indices(mol_a, match_a)
    common_bonds_b = mcs_bond_indices(mol_b, match_b)

    common_atoms_a = list(match_a)
    common_atoms_b = list(match_b)

    # Color dictionaries for RDKit drawing
    bond_colors_a: Dict[int, Tuple[float, float, float]] = {i: common_bond_color for i in common_bonds_a}
    bond_colors_b: Dict[int, Tuple[float, float, float]] = {i: common_bond_color for i in common_bonds_b}
    atom_colors_a: Dict[int, Tuple[float, float, float]] = {i: common_atom_color for i in common_atoms_a}
    atom_colors_b: Dict[int, Tuple[float, float, float]] = {i: common_atom_color for i in common_atoms_b}

    img = Draw.MolsToGridImage(
        [mol_a, mol_b],
        molsPerRow=2,
        subImgSize=(size[0] // 2, size[1]),
        legends=legends,
        highlightBondLists=[common_bonds_a, common_bonds_b],
        highlightBondColors=[bond_colors_a, bond_colors_b],
        highlightAtomLists=[common_atoms_a, common_atoms_b],
        highlightAtomColors=[atom_colors_a, atom_colors_b],
        useSVG=False,  # set True if you prefer SVG output
    )
    return img


def test_show_common_bonds():
    print(flush=True)

    mols_str = ["codeine",
                "morphine"]

    smis = [att.pubchem_name_to_smi(name) for name in mols_str]
    img = show_common_bonds(*smis, legends=mols_str)
    assert img is not None, "Failed to generate the image."
    img.show()
