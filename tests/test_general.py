import os
import shutil

import networkx as nx
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def list_subdirs(directory, target="ai_calc"):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(target)]


def test_version():
    assert att.__version__ == '0.0.01'


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
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # write the mol file
    mol_file = "tmp.mol"
    att.write_v2k_mol_file(mol, mol_file)
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol_file)
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == path["file_graph"][0]

    # Remove the files
    tmp_file = os.path.splitext(mol_file)[0]
    os.remove(tmp_file + ".mol")
    os.remove(tmp_file + ".err")
    os.remove(tmp_file + ".out")
    os.remove(tmp_file + "Out")
    os.remove(tmp_file + "Pathway")


def test_ass_mol():
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(path["file_graph"][0])


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


def print_graph_details(graph):
    print("{", flush=True)
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        print(f"({node_index}, {node_color}): {edge_connections}", flush=True)
    print("}", flush=True)


def test_hand_graph():
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


test_ass_mol()
