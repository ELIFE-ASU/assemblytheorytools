import os
import shutil

import networkx as nx
import numpy as np
import ase.io
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def list_subdirs(directory, target="ai_calc"):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(target)]


def print_graph_details(graph):
    print("{", flush=True)
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        print(f"({node_index}, {node_color}): {edge_connections}", flush=True)
    print("}", flush=True)


def test_ass_graph():
    smi_in = "[H]C#C[H]"
    # Convert all the smile to mol
    mol = att.smi_to_mol(smi_in)
    # Convert the system into graphs
    graph = att.mol_to_nx(mol)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(graph)
    # Get the input graph from the output dict
    input_graph = path["file_graph"][0]
    smi_out = Chem.MolToSmiles(att.nx_to_mol(input_graph))
    # Compare to the hand calculated value
    assert ai == 2  # Check the assembly index
    assert att.is_graph_isomorphic(graph, input_graph)  # Check the output graph is the same as the input
    assert smi_in == smi_out  # Check the graph conversion to and from RDKit


def test_ass_mol_file():
    smi_in = "[H]C#C[H]"
    # Convert all the smile to mol
    mol = att.smi_to_mol(smi_in)
    # write the mol file
    mol_file = "tmp.mol"
    att.write_v2k_mol_file(mol, mol_file)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol_file)

    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == path["file_graph"][0]


def test_ass_mol():
    smi_in = "[H]C#C[H]"
    # Convert all the smile to mol
    mol = att.smi_to_mol(smi_in)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(path["file_graph"][0])


def test_compare_ass_graph_mol_file_mol():
    smis = ["c1ccccc1", "[BH-]1-[NH+]=[BH-]-[NH+]=[BH-]-[NH+]=1"]
    for smi_in in smis:
        # Convert all the smile to mol
        mol = att.smi_to_mol(smi_in)
        # Convert the system into graphs
        graph = att.mol_to_nx(mol)
        # write the mol file
        mol_file = "tmp.mol"
        att.write_v2k_mol_file(mol, mol_file)

        # Graph
        ai_graph, _ = att.calculate_assembly_index(graph)
        # Mol file
        ai_mol_file, _ = att.calculate_assembly_index(mol_file)
        # Mol
        ai_mol, _ = att.calculate_assembly_index(mol)

        assert ai_graph == ai_mol_file == ai_mol


def test_big_chungus():
    mol_file = os.path.abspath("data/mol_files/big_chungus.mol")
    # Get the mol object
    mol = att.molfile_to_mol(mol_file)
    # Convert the system into graphs
    graph = att.mol_to_nx(mol)

    # Graph
    ai_graph, _ = att.calculate_assembly_index(graph, timeout=1000.0)
    # Mol file
    ai_mol_file, _ = att.calculate_assembly_index(mol_file, timeout=1000.0)
    # Mol
    ai_mol, _ = att.calculate_assembly_index(mol, timeout=1000.0)

    assert ai_graph == ai_mol_file == ai_mol == 8


def test_taxol_file():
    mol_file = os.path.abspath("data/mol_files/taxol.mol")
    # Mol file
    ai_mol_file, _ = att.calculate_assembly_index(mol_file, timeout=1000.0)

    assert ai_mol_file == 23


def test_hydrogen_stripping():
    mol_file = os.path.abspath("data/mol_files/alanine.mol")
    # Get the mol object
    mol = att.molfile_to_mol(mol_file)
    mol = att.smi_to_mol("C[C@@H](C(=O)O)N")

    # Convert the system into graphs
    graph = att.mol_to_nx(mol)
    # Graph
    ai_graph, _ = att.calculate_assembly_index(att.remove_hydrogen_from_graph(graph))
    # Mol file
    ai_mol_file, _ = att.calculate_assembly_index(mol_file)
    # Mol
    ai_mol, _ = att.calculate_assembly_index(Chem.RemoveHs(mol))

    assert ai_graph == ai_mol_file == ai_mol == 4

    # Test the manual case
    # Graph
    ai_graph, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
    # Mol file
    ai_mol_file, _ = att.calculate_assembly_index(mol_file, strip_hydrogen=True)
    # Mol
    ai_mol, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    assert ai_graph == ai_mol_file == ai_mol == 4


def test_ass_mol_debug():
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol, debug=True)
    # Get the path of the created file
    dir_list = list_subdirs(os.getcwd())
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(path["file_graph"][0])
    assert len(dir_list) == 1
    # Clean up
    shutil.rmtree(dir_list[0])


def test_joint_ass_mol():
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the smile to mol
    mols = [att.smi_to_mol(smile) for smile in molecules]
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value
    out_mol = Chem.MolToInchi(att.split_mols(mol)[0])
    assert ai == 11
    assert out_mol == Chem.MolToInchi(path["file_graph"][0])


def test_joint_ass_graph():
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


def test_all_paths_simple():
    mol = att.smi_to_mol("C#CCC=C")
    paths = att.all_shortest_paths(mol, f_graph_care=False)
    expected = ['InChI=1S/CH4/h1H4',
                'InChI=1S/C2H2/c1-2/h1-2H',
                'InChI=1S/C2H4/c1-2/h1-2H2',
                'InChI=1S/C2H6/c1-2/h1-2H3',
                'InChI=1S/C3H8/c1-3-2/h3H2,1-2H3',
                'InChI=1S/C5H6/c1-3-5-4-2/h1,4H,2,5H2']
    for p in paths:
        assert p in expected


def test_node_scramble():
    smi_in = "[H]OC(=O)C([H])(N([H])C(=O)C([H])([H])N([H])[H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H]"
    # smi_in = "CCC"
    # Convert all the smile to mol
    mol = att.smi_to_mol(smi_in)
    # Convert the system into graphs
    graph = att.mol_to_nx(mol, add_hydrogens=False)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(graph)
    # Get the input graph from the output dict
    input_graph = path["file_graph"][0]
    att.plot_mol_graph(input_graph, f_labs=True, filename="graph")
    smi_out = Chem.MolToSmiles(att.nx_to_mol(input_graph))

    graph_sc = att.scramble_node_indices(graph)
    # Calculate the assembly index
    ai_sc, path_sc = att.calculate_assembly_index(graph_sc)
    # Get the input graph from the output dict
    input_graph = path_sc["file_graph"][0]
    att.plot_mol_graph(input_graph, f_labs=True, filename="scrambled")
    smi_out_sc = Chem.MolToSmiles(att.nx_to_mol(input_graph))

    # assert ai == ai_sc
    assert smi_out == smi_out_sc  # Check the graph conversion to and from RDKit


def test_str_ass():
    s_inpt = "abracadabra"
    ai, _ = att.calculate_assembly_index(s_inpt)
    ai_ref = 7
    assert ai == ai_ref


def test_hand_graph():
    print("This is a hand construction graph test", flush=True)
    # Create a ring graph with 8 nodes
    G = nx.cycle_graph(8)
    # Set the labels of the nodes to be "C" - a carbon atom
    nx.set_node_attributes(G, "C", "color")
    # Set the edge labels to be "1" - a single bond
    nx.set_edge_attributes(G, 1, "color")
    print("input", flush=True)
    print_graph_details(G)

    ai, path = att.calculate_assembly_index(G)
    # Convert the dict to a list
    path = att.convert_pathway_dict_to_list(path)
    print("output", flush=True)
    print(f"Ass index = {ai}", flush=True)
    # Remove the first pathway
    path.pop(0)
    for i, p in enumerate(path):
        print(f"Pathway object = {i}", flush=True)
        print_graph_details(p)

    assert ai == 3


def test_cif_loading():
    print(flush=True)
    target_dir = "data/cif_files/"

    dirs = att.file_list_all(os.path.expanduser(target_dir))
    print(dirs)
    for file in dirs:
        print(file, flush=True)
        if file in ['data/cif_files/Attakolite_0.cif', 'data/cif_files/Wodginite_3.cif']:
            # Attakolite_0 invalid spacegroup C 1 2/m 1
            # Wodginite_3 invalid spacegroup C 1 2/c 1
            continue
        # input mol file
        atoms = att.read_cif_file(file)
        tmp_file = file.split('.')[0] + ".mol"
        att.atoms_to_mol_file(atoms, fname=tmp_file)
        atoms2 = ase.io.read(tmp_file)
        os.remove(tmp_file)
        # check that the atoms are the same
        assert np.allclose(atoms.get_positions(), atoms2.get_positions(), rtol=1e-04, atol=1e-04)
        assert np.allclose(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())


def test_cif_ai():
    print(flush=True)
    target_dir = "data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(target_dir))
    file = dirs[0]

    # input mol file
    atoms = att.read_cif_file(file)
    tmp_file = file.split('.')[0] + ".mol"
    att.atoms_to_mol_file(atoms, fname=tmp_file)
    ai_mol, _ = att.calculate_assembly_index(tmp_file, joint_corr=False)

    os.remove(tmp_file)

    graph = att.atoms_to_nx(atoms)
    ai_graph, _ = att.calculate_assembly_index(graph, joint_corr=False)

    assert ai_mol == ai_graph == 4
