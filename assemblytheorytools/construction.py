import copy
import json
import os

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol


def transform_array(target_array, comp_array, source_val, target_val, new_val, pairs_list):
    """
    Transforms the target array by replacing specific values based on the comparison array and pairs list.

    This function iterates over the comparison array and updates the target array by replacing elements
    that match the target value and source value with a new value, according to the pairs list.

    :param target_array: List of lists, where each sublist represents an edge in the target array.
    :param comp_array: List of lists, where each sublist represents an edge in the comparison array.
    :param source_val: The source value to be replaced.
    :param target_val: The target value to be replaced.
    :param new_val: The new value to replace the source and target values.
    :param pairs_list: List of pairs that determine valid replacements.
    :return: The modified target array with updated values.
    """
    for i, edge in enumerate(comp_array):
        if edge[0] == target_val and [source_val, edge[1]] in pairs_list:
            target_array[i] = [new_val, edge[1]]
        elif edge[1] == target_val and [edge[0], source_val] in pairs_list:
            target_array[i] = [edge[0], new_val]
    return target_array


def repeated_sizes(repeated):
    """
    Returns a sorted list of unique sizes of the second element in each tuple in the repeated list.

    :param repeated: List of tuples, where each tuple contains two elements.
    :return: Sorted list of unique sizes of the second element in each tuple.
    """
    rep = sorted(set(len(rep[1]) for rep in repeated))
    return rep


def equal_list(list_a, list_b):
    """
    Compares two lists of lists and checks if they contain the same elements.

    This function converts each sublist in the input lists to a set of tuples and compares them.
    It returns True if both lists contain the same sets of tuples, otherwise False.

    :param list_a: First list of lists to compare.
    :param list_b: Second list of lists to compare.
    :return: True if both lists contain the same sets of tuples, otherwise False.
    """
    return set(row for row in list_a) == set(row for row in list_b)


def check_edge_in_list(edges, list_in):
    """
    Checks if a given list of edges is present in any of the lists within a list of lists.

    This function iterates over each list in the input list of lists and uses the `equal_list` function
    to check if any of these lists contain the same elements as the given list of edges.

    :param edges: List of edges to check.
    :param list_in: List of lists, where each sublist is a list of edges.
    :return: True if the given list of edges is present in any of the lists within the input list of lists, otherwise False.
    """
    return any(equal_list(l, edges) for l in list_in)


def equivalence(remnant_pieces, equivalences):
    """
    Applies equivalence transformations to the remnant pieces based on the provided equivalences.

    This function creates a deep copy of the remnant pieces and iterates through each edge in each piece.
    If an edge's vertex matches any vertex in the equivalences list, it replaces the vertex with the corresponding
    equivalent vertex.

    :param remnant_pieces: List of lists, where each sublist represents a piece containing edges.
    :param equivalences: List of pairs, where each pair represents an equivalence between two vertices.
    :return: A deep copy of the remnant pieces with applied equivalence transformations.
    """
    pieces_copy = copy.deepcopy(remnant_pieces)
    equivalences_array = np.array(equivalences)
    equivalences_list = equivalences_array[:, 1].tolist()

    for piece in pieces_copy:
        for edge in piece:
            if edge[0] in equivalences_list:
                edge[0] = equivalences[equivalences_list.index(edge[0])][0]
            if edge[1] in equivalences_list:
                edge[1] = equivalences[equivalences_list.index(edge[1])][0]

    return pieces_copy


def fix_repeated_equiv(edge_list, repeated_equiv, equivalences, edge_pairs):
    """
    Fixes repeated equivalences in the edge list by transforming the edges based on the provided equivalences.

    This function identifies and resolves repeated equivalences in the edge list. It updates the edge list and
    repeated equivalences by applying transformations based on the equivalences and edge pairs.

    :param edge_list: List of edges to be transformed.
    :param repeated_equiv: List of repeated equivalences to be fixed.
    :param equivalences: List of equivalences to be applied.
    :param edge_pairs: List of valid edge pairs for transformations.
    :return: Tuple containing the updated edge list, repeated equivalences, and equivalences.
    """
    global new_val, target_val, source_val
    equivalences = np.unique(equivalences, axis=0).tolist()
    equiv_np = np.array(equivalences)
    sorted_eq = equiv_np[equiv_np[:, 0].argsort()] if len(equiv_np) != 0 else equiv_np

    repeated_eq_1 = [array.tolist() for i, array in enumerate(sorted_eq) if
                     array[1] in np.concatenate((sorted_eq[:, 1][:i], sorted_eq[:, 1][i + 1:]), axis=0)]
    repeated_eq_2 = [array.tolist() for i, array in enumerate(sorted_eq) if
                     array[0] in np.concatenate((sorted_eq[:, 0][:i], sorted_eq[:, 0][i + 1:]), axis=0)]

    inter = np.intersect1d(repeated_eq_1, repeated_eq_2).tolist()
    idx = [sorted_eq.tolist().index(rem) for rem in repeated_eq_1]
    remove = np.delete(sorted_eq, idx, axis=0)

    if repeated_eq_1:
        repeated_mod = [equivalence(rep, remove) for rep in repeated_equiv]
        trans_edges = equivalence([edge_list], remove)[0]
        if not repeated_eq_2 or (repeated_eq_2[0][0] == repeated_eq_2[0][0] and not inter):
            sort_repeated_eq_1 = np.array(repeated_eq_1)[np.array(repeated_eq_1)[:, 1].argsort()]
            add = []
            for i in range(len(sort_repeated_eq_1) // 2):
                new_val = equiv_np[equiv_np[:, 1].argsort()][-1][1] + 1 + i
                target_val, source_val = sort_repeated_eq_1[2 * i][1], sort_repeated_eq_1[2 * i][0]
                add += [[source_val, new_val], sort_repeated_eq_1[2 * i + 1].tolist()]
                edge_list = transform_array(edge_list, trans_edges, source_val, target_val, new_val, edge_pairs)
                for i, rep in enumerate(repeated_mod):
                    for j, _ in enumerate(rep):
                        repeated_equiv[i][j] = transform_array(repeated_equiv[i][j], repeated_mod[i][j], source_val,
                                                               target_val, new_val, edge_pairs)
            equivalences = remove.tolist() + add
        else:
            for item in repeated_eq_2:
                if inter[0] == item[0] and inter[1] != item[1]:
                    new_val, target_val, source_val = item[1], inter[1], item[0]
            equivalences = remove.tolist() + np.delete(repeated_eq_1, repeated_eq_1.index(inter), axis=0).tolist()
            edge_list = transform_array(edge_list, trans_edges, source_val, target_val, new_val, edge_pairs)
            for i, rep in enumerate(repeated_mod):
                for j, _ in enumerate(rep):
                    repeated_equiv[i][j] = transform_array(repeated_equiv[i][j], repeated_mod[i][j], source_val,
                                                           target_val, new_val, edge_pairs)

            sorted_eq = np.array(equivalences)[np.array(equivalences)[:, 0].argsort()]
            repeated_eq_1 = [array.tolist() for i, array in enumerate(sorted_eq) if
                             array[1] in np.concatenate((sorted_eq[:, 1][:i], sorted_eq[:, 1][i + 1:]), axis=0)]
            if repeated_eq_1:
                edge_list, repeated_equiv, equivalences = fix_repeated_equiv(edge_list, repeated_equiv, equivalences,
                                                                             edge_pairs)

    return edge_list, repeated_equiv, equivalences


def index_set(lists, list_in):
    """
    Finds the index of a list within a list of lists that matches the given list.

    This function converts the input list and each list within the list of lists to a set of tuples.
    It then checks if any of these sets match the set of the input list and returns the index (1-based) of the matching list.

    :param lists: List of lists to search within.
    :param list_in: List to find within the list of lists.
    :return: 1-based index of the matching list, or None if no match is found.
    """
    list_in_set = set(tuple(row) for row in list_in)
    for i, i_list in enumerate(lists):
        if set(tuple(row) for row in i_list) == list_in_set:
            return i + 1


def select_length(dict_array):
    """
    Takes a dictionary and returns the entry for the 'len' key.

    :param dict_array: Dictionary containing arrays lengths and indexes.
    :return: Entry for e['len'].
    """
    return dict_array["len"]


def transform_bond_string_float(bond):
    """
    Converts a bond type from string to float representation.

    This function takes a bond type as a string and returns its corresponding float value.
    If the bond type is not recognized, it returns an error string.

    :param bond: Bond type as a string
    :return: Float representation of the bond type
    """
    if bond == "single":
        return 1.0
    if bond == "double":
        return 2.0
    if bond == "triple":
        return 3.0
    return "error"


def transform_bond_float_rdkit(bond):
    """
    Converts a bond type from float to RDKit bond type.

    This function takes a bond type as a float and returns its corresponding RDKit bond type.
    If the bond type is not recognized, it returns an error string.

    :param bond: Bond type as a float
    :return: RDKit bond type
    """
    bond = float(bond)
    if bond == 1.0:
        return Chem.rdchem.BondType.SINGLE
    if bond == 2.0:
        return Chem.rdchem.BondType.DOUBLE
    if bond == 3.0:
        return Chem.rdchem.BondType.TRIPLE
    return "error"


def tables2mol(tables):
    """
    Converts atom and bond information into an RDKit molecule object.

    This function takes a tuple containing atom information and bond information,
    creates an RDKit molecule object, adds atoms and bonds to it, and returns the molecule.

    :param tables: A tuple containing two lists:
                   - atoms_info: List of tuples, where each tuple contains atom index and atom type.
                   - bonds_info: List of tuples, where each tuple contains bond start index, bond end index, and bond type.
    :return: An RDKit molecule object constructed from the provided atom and bond information.
    """
    atoms_info, bonds_info = tables
    edit_mol = RWMol()
    for v in atoms_info:
        edit_mol.AddAtom(Chem.Atom(v[1]))
    for e in bonds_info:
        edit_mol.AddBond(e[0], e[1], transform_bond_float_rdkit(e[2]))
    mol = edit_mol.GetMol()
    return mol


class AssemblyConstruction:
    def __init__(self, data, if_string=False):

        v = data["file_graph"][0]['Vertices']
        e = data["file_graph"][0]['Edges']
        v_l = data["file_graph"][0]['VertexColours']
        e_l = data["file_graph"][0]['EdgeColours']
        remnant_e = data["remnant"][0]["Edges"] + data["removed_edges"]
        duplicates = [[dup["Right"], dup['Left']] for dup in data["duplicates"]]
        equivalences = [[1, 1]]

        remnant_e, duplicates, equivalences = fix_repeated_equiv(remnant_e, duplicates, equivalences, e)
        self.v = v
        self.e = e
        self.v_l = v_l
        self.e_l = e_l
        self.remnant_e = remnant_e
        self.equivalences = equivalences
        self.duplicates = duplicates
        self.if_string = if_string
        # Construct the atoms list
        atoms_list = []
        atoms_list_index = []
        atoms_pre = []
        full_atoms_list = []
        for i, bond in enumerate(e):
            atom_list = [[v_l[bond[0]], v_l[bond[1]]], e_l[i]]
            atom_list_index = [bond[0], bond[1]]
            atom_set = [{v_l[bond[0]], v_l[bond[1]]}, e_l[i]]
            if not (atom_set in atoms_pre):
                atoms_pre.append(atom_set)
                atoms_list.append(atom_list)
                atoms_list_index.append(atom_list_index)
            full_atoms_list.append(atom_list)
        self.atoms = atoms_pre
        self.full_atoms_list = full_atoms_list
        self.atoms_list = atoms_list
        self.atoms_list_index = atoms_list_index

    def consistent_join(self, pieces_mod, steps_mod, repeated_mo1_cp, step, digraph, indexes):
        left_sort = [rep[0] for rep in repeated_mo1_cp]
        right_sort = [rep[1] for rep in repeated_mo1_cp]
        for pic in pieces_mod:
            for pic_i in pieces_mod:
                pic_r = np.reshape(pic, (np.shape(pic)[0] * 2))
                pic_i_r = np.reshape(pic_i, (np.shape(pic_i)[0] * 2))
                for idx, ed in enumerate(pic_i_r):
                    if ed in pic_r and not (pic == pic_i):

                        step = step + 1
                        if self.if_string:
                            steps_mod.append(np.sort(pic + pic_i, axis=0).tolist())
                        else:
                            steps_mod.append(pic + pic_i)
                        if not (len(pic) == 1):
                            if pic in left_sort:
                                digraph.append(
                                    [
                                        "step{}".format(indexes[left_sort.index(pic)]),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic in right_sort:
                                digraph.append(
                                    [
                                        "step{}".format(indexes[right_sort.index(pic)]),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic in steps_mod:
                                digraph.append(
                                    [
                                        "step{}".format(steps_mod.index(pic) + 1),
                                        "step{}".format(step),
                                    ]
                                )
                            else:
                                digraph.append(
                                    ["step{}".format("_error"), "step{}".format(step)]
                                )

                        else:
                            v_object1 = self.atoms.index(
                                [
                                    {self.v_l[pic[0][0]], self.v_l[pic[0][1]]},
                                    self.e_l[self.e.index(pic[0])],
                                ]
                            )
                            digraph.append(
                                ["virtual_object{}".format(v_object1), "step{}".format(step)]
                            )
                        if not (len(pic_i) == 1):
                            if pic_i in left_sort:
                                digraph.append(
                                    [
                                        "step{}".format(indexes[left_sort.index(pic_i)]),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic_i in right_sort:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            indexes[right_sort.index(pic_i)]
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic_i in steps_mod:
                                digraph.append(
                                    [
                                        "step{}".format(steps_mod.index(pic_i) + 1),
                                        "step{}".format(step),
                                    ]
                                )
                            else:
                                digraph.append(
                                    ["step{}".format("_error"), "step{}".format(step)]
                                )

                        else:
                            v_object1 = self.atoms.index(
                                [
                                    {self.v_l[pic_i[0][0]], self.v_l[pic_i[0][1]]},
                                    self.e_l[self.e.index(pic_i[0])],
                                ]
                            )
                            digraph.append(
                                ["virtual_object{}".format(v_object1), "step{}".format(step)]
                            )
                        pieces_mod.remove(pic)
                        pieces_mod.remove(pic_i)
                        if self.if_string:
                            pieces_mod.insert(0, np.sort(pic + pic_i, axis=0).tolist())
                        else:
                            pieces_mod.insert(0, pic + pic_i)
                        return pieces_mod, steps_mod, step, digraph
        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(self, pieces_mod, steps_mod, sorted_repeated_mod1, step, digraph):
        step_ind = [1 for i in range(len(sorted_repeated_mod1))]
        indexes = [0 for i in range(len(sorted_repeated_mod1))]
        sorted_repeated_mod1_cp = copy.deepcopy(sorted_repeated_mod1)
        left_sort = [rep[0] for rep in sorted_repeated_mod1_cp]
        right_sort = [rep[1] for rep in sorted_repeated_mod1_cp]
        while len(sorted_repeated_mod1) != 0:
            for j, repeat in enumerate(sorted_repeated_mod1_cp):
                if (not step_ind[j]) or repeated_sizes(sorted_repeated_mod1)[0] != len(
                        repeat[0]
                ):
                    continue
                if check_edge_in_list(repeat[1], pieces_mod) or check_edge_in_list(repeat[1], steps_mod):
                    pieces_mod.append(repeat[0])
                    sorted_repeated_mod1.remove(repeat)
                    step_ind[j] = 0
                    if index_set(steps_mod, repeat[1]) is not None:
                        indexes[j] = index_set(steps_mod, repeat[1])
                    else:
                        indexes[j] = indexes[index_set(left_sort, repeat[1]) - 1]
                else:
                    repeat_cp = copy.deepcopy(repeat[1])
                    pieces_mod_cp = copy.deepcopy(pieces_mod)
                    indices = []

                    for i, piece in enumerate(pieces_mod):
                        for k, rep in enumerate(repeat[1]):
                            if rep in piece:
                                indices.append(i)
                                for edge in piece:
                                    repeat_cp.remove(edge)
                                break
                        if len(repeat_cp) == 0:
                            break
                    if not indices:
                        continue
                    combined_pieces = [pieces_mod[i] for i in indices]

                    for idx in indices:
                        pieces_mod.remove(pieces_mod_cp[idx])
                    # We consistently join the remnant pieces in such a way that the constructed molecule never has
                    # disconnected pieces
                    while not (len(combined_pieces) == 1):
                        combined_pieces, steps_mod, step, digraph = self.consistent_join(
                            combined_pieces,
                            steps_mod,
                            sorted_repeated_mod1_cp,
                            step,
                            digraph,
                            indexes,
                        )
                    # Construct repeat[1]
                    pieces_mod.append(combined_pieces[0])
                    pieces_mod.append(repeat[0])
                    step_ind[j] = 0
                    indexes[j] = index_set(steps_mod, repeat[1])
                    sorted_repeated_mod1.remove(repeat)

        return pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes

    def generate_pathway(self):
        step = 0
        digraph = []
        steps = []
        pieces = [[edge] for edge in self.remnant_e]

        duplicates_mod = [equivalence(rep, self.equivalences) for rep in
                          self.duplicates] if self.equivalences else self.duplicates
        pieces_mod = equivalence(pieces, self.equivalences) if self.equivalences else pieces

        sizes = sorted([{"index": i, "len": len(repeat[0])} for i, repeat in enumerate(duplicates_mod)],
                       key=select_length)
        sorted_repeated_mod1 = [duplicates_mod[size["index"]] for size in sizes]

        pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes = self.repeated_construction(
            pieces_mod, steps, sorted_repeated_mod1, step, digraph
        )

        pieces_mod_cp = []
        while len(pieces_mod) != len(pieces_mod_cp):
            pieces_mod_cp = copy.deepcopy(pieces_mod)
            pieces_mod, steps_mod, step, digraph = self.consistent_join(
                pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes
            )

        self.steps = steps_mod
        self.digraph = digraph
        self.pieces_mod = pieces_mod

        return None

    def pathway_inchi_vo(self):
        molecules_vo = []
        inchi_list = []

        for atom in self.atoms_list:
            mol = tables2mol(([(0, atom[0][0]), (1, atom[0][1])], [(0, 1, transform_bond_string_float(atom[1]))]))
            molecules_vo.append(mol)
            inchi_list.append(Chem.MolToInchi(mol))

        steps_index_s = []
        vs_atoms = []

        for step in self.steps:
            indices = list(set(np.reshape(step, -1)))
            steps_index_s.append(
                [[indices.index(edge[0]), indices.index(edge[1]), self.e_l[self.e.index(edge)]] for edge in step])
            vs_atoms.append([self.v_l[at] for at in indices])

        molecules_steps = []

        for i, step in enumerate(steps_index_s):
            mol = tables2mol(([(i, at) for at in vs_atoms[i]],
                              [(edge[0], edge[1], transform_bond_string_float(edge[2])) for edge in step]))
            molecules_steps.append(mol)
            inchi_list.append(Chem.MolToInchi(mol))

        self.molecules_vo = molecules_vo
        self.molecules_steps = molecules_steps
        self.steps_indx_s = steps_index_s
        self.vs_atoms = vs_atoms

        return inchi_list

    def plot_pathway(self):
        pic_path = "path_images"
        os.makedirs(pic_path, exist_ok=True)
        for i, vo in enumerate(self.molecules_vo):
            Draw.MolToFile(vo, os.path.join(pic_path, "virtual_object{}.png").format(i))
        for i, step in enumerate(self.molecules_steps):
            Draw.MolToFile(step, os.path.join(pic_path, "step{}.png").format(i + 1))

        for dir_edge in self.digraph:
            print(dir_edge)

        return None


def generate_directional_graph(digraph):
    """
    Generates a directional NetworkX graph from the given digraph.

    :param digraph: List of tuples representing directed edges.
    :return: A directional NetworkX graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(digraph)
    return graph


def parse_pathway_file(file):
    # Load the pathway file
    with open(file) as f:
        data = json.load(f)
    # Make the construction object
    construction_object = AssemblyConstruction(data)
    construction_object.generate_pathway()
    inchi_list = construction_object.pathway_inchi_vo()

    # Generate the directional graph
    graph = generate_directional_graph(construction_object.digraph)

    return graph, inchi_list
