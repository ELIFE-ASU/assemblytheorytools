from log_parsing import encode_path_data
from assembly_pathv3 import check_edge_in_list, assemblyConstruction, fix_repeated_equiv
from path_db_parsing import parse_db_path
# import igraph
import os
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Draw
import json


def encode_go_output(go_output):
    # open text file
    text_file = open("temp.txt", "w")
    # write string to file
    text_file.write(go_output)
    text_file.close()
    encode_path_data("temp.txt")
    os.remove("temp.txt")
    pathfile = open("encoded_temp.txt", "r")
    length = len(pathfile.readlines())
    pathfile.close()
    pathfile = open("encoded_temp.txt", "r")
    ma = pathfile.readline()[:-1]
    runtime = pathfile.readline()[:-1]
    path = ""
    for i in range(length - 2):
        if i == length - 3:
            path = path + pathfile.readline()
        path = path + pathfile.readline()[:-1] + "\n"
    pathfile.close()
    os.remove("encoded_temp.txt")
    return path[:-1]


def draw_pathway(pathway, mode):
    v, e, v_l, e_l, remnant_e, equivalences, duplicates = parse_pathway(pathway)
    remnant_e, duplicates, equivalences = fix_repeated_equiv(
        remnant_e, duplicates, equivalences, e
    )  # Fixing sometimes errors in the go output
    construction_object = assemblyConstruction(
        v, e, v_l, e_l, remnant_e, equivalences, duplicates
    )

    try:
        construction_object.generate_pathway()
        construction_object.plot_pathway(mode)
        pathway_success = True
    except ValueError as e:
        pathway_success = False
    except IndexError as e:
        pathway_success = False

    return pathway_success, construction_object


def parse_pathway_file_ian(file):
    f = open(file)
    data = json.load(f)
    v = data["file_graph"][0]['Vertices']
    e = data["file_graph"][0]['Edges']
    v_l = data["file_graph"][0]['VertexColours']
    e_l = data["file_graph"][0]['EdgeColours']
    remnant_e = data["remnant"][0]["Edges"] + data["removed_edges"]
    duplicates = [[dup["Right"], dup['Left']] for dup in data["duplicates"]]
    equivalences = [[1, 1]]
    mode = 1
    remnant_e, duplicates, equivalences = fix_repeated_equiv(
        remnant_e, duplicates, equivalences, e
    )  # Fixing sometimes errors in the go output
    construction_object = assemblyConstruction(
        v, e, v_l, e_l, remnant_e, equivalences, duplicates
    )
    try:
        construction_object.generate_pathway()
        construction_object.plot_pathway(mode)
        pathway_success = True
    except ValueError as e:
        pathway_success = False
        print("error")
    except IndexError as e:
        pathway_success = False
        print("error")

    return construction_object, pathway_success


def parse_pathway(pathway):
    filename = "temp.txt"
    pathway_file = open(filename, "w")
    # write string to file
    pathway_file.write("10\n0.1\n" + pathway)
    # close file
    pathway_file.close()

    v, e, v_c, e_c, er, equivalences, repeated = parse_db_path(filename)
    os.remove("temp.txt")

    return v, e, v_c, e_c, er, equivalences, repeated


def transfrom_bond_str(bond):
    if bond == 1.0:
        return "single"
    if bond == 2.0:
        return "double"
    if bond == 3.0:
        return "triple"
    return "error"


def mol2tables(mol, str=False):
    atoms_info = [(atom.GetIdx(), atom.GetSymbol()) for atom in mol.GetAtoms()]
    if str:
        bonds_info = [
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                transfrom_bond_str(bond.GetBondTypeAsDouble()),
            )
            for bond in mol.GetBonds()
        ]
    else:
        bonds_info = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
            for bond in mol.GetBonds()
        ]
    tables = (atoms_info, bonds_info)
    return tables


def tables2graph(tables):
    atoms_info, bonds_info = tables
    graph = igraph.Graph()
    for atom_info in atoms_info:
        graph.add_vertex(atom_info[0], AtomicSymbole=atom_info[1])
    for bond_info in bonds_info:
        graph.add_edge(bond_info[0], bond_info[1], BondTypeAsDouble=bond_info[2])
    return graph


def plot_graph(graph):
    layout = graph.layout_graphopt()
    color_dict_vertex = {"C": "black", "O": "red", "N": "blue", "P": "purple", "S": "orange"}
    color_dict_edge = {1.0: "black", 2.0: "green"}
    my_plot = igraph.Plot()
    my_plot.add(
        graph,
        layout=layout,
        bbox=(300, 300),
        margin=20,
        vertex_color=[color_dict_vertex[atom] for atom in graph.vs["AtomicSymbole"]],
        vertex_size=[10 for v in graph.vs],
        edge_color=[color_dict_edge[ed] for ed in graph.es["BondTypeAsDouble"]],
    )
    return my_plot


def transfrom_bond(bond):
    if bond == 1.0:
        return Chem.rdchem.BondType.SINGLE
    if bond == 2.0:
        return Chem.rdchem.BondType.DOUBLE
    if bond == 3.0:
        return Chem.rdchem.BondType.TRIPLE
    return "error"


def tables2mol(tables):
    atoms_info, bonds_info = tables
    emol = RWMol()
    for v in atoms_info:
        emol.AddAtom(Chem.Atom(v[1]))
    for e in bonds_info:
        emol.AddBond(e[0], e[1], transfrom_bond(e[2]))
    mol = emol.GetMol()
    return mol


def transfrom_bond_float(bond):
    if bond == "single":
        return 1.0
    if bond == "double":
        return 2.0
    if bond == "triple":
        return 3.0
    return "error"


def highlight_edges_from_molecule(steps_mod, original_m, e, label):
    highlight_bonds_total = []
    highlight_atoms_total = []
    legends = []
    for j, dup in enumerate(steps_mod):
        highlight_bonds = []
        highlight_atoms_pre = []
        for i, bond in enumerate(e):
            if check_edge_in_list(list(bond), dup):
                highlight_bonds.append(i)
                highlight_atoms_pre.append(bond)
        highlight_atoms = {item for sublist in highlight_atoms_pre for item in sublist}
        highlight_bonds_total.append(highlight_bonds)
        highlight_atoms_total.append(highlight_atoms)
        if isinstance(label, str):
            legends.append("{} {}".format(label, j + 1))
        else:
            legends.append("{}".format(label[j]))
    return highlight_atoms_total, highlight_bonds_total, legends
