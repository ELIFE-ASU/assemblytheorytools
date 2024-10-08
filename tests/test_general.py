import pytest
import assemblytheorytools as att




def test_version():
    assert att.__version__ == '0.0.01'

def test_ass_graph():
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # Convert the system into graphs
    graph = att.mol_to_nx(mol)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(graph)
    # Compare to the hand calculated value
    assert ai == 2
    assert att.is_graph_isomorphic(graph, path["file_graph"][0])

def test_ass_mol():
    from rdkit.Chem import AllChem as Chem
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == path["file_graph"][0]


# def test_joint_ass_mol():
#     from rdkit import Chem
#     from rdkit.Chem import AllChem as Chem
#     # Calculating the joint MA using mol approach
#     molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
#     ps = Chem.SmilesParserParams()
#     ps.removeHs = False
#     combined_mol = Chem.MolFromSmiles(molecules, ps)
#     # Chem.Kekulize(combined_mol)
#
#     # Calculate the assembly index
#     ai, path = att.calculate_assembly_index(combined_mol)
#     print(f"Assembly index: {ai}", flush=True)
#     print(f"Pathway: {path}", flush=True)
#     # Compare to the hand calculated value
#     assert ai == 11



def test_joint_ass_mol():
    import networkx as nx
    from rdkit.Chem import AllChem as Chem
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the smile to mol
    mols = [att.smi_to_mol(smile) for smile in molecules]
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value

    assert ai == 11
    # assert att.split_mols(mol) == path["file_graph"][0]

def test_joint_ass_graph():
    import networkx as nx
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the smile to mol
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Convert the system into graphs
    graphs = [att.mol_to_nx(mol) for mol in mols]
    # Join the graphs
    graphs_joint = nx.disjoint_union_all(graphs)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(graphs_joint)
    # Compare to the hand calculated value
    out_graph = nx.disjoint_union_all(path["file_graph"])
    assert ai == 11
    assert att.is_graph_isomorphic(graphs_joint, out_graph)