import copy
import json
import os
import shutil

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import RWMol


def transform_array(array, array_mod, fromm, repa, replace, e):
    """
    Transforms the input array by replacing elements based on specified conditions.

    This function iterates over `array_mod` and checks if the elements match the `repa` value.
    If a match is found, it further checks if a specific pair exists in `e` and then replaces
    the corresponding elements in the `array`.

    :param array: List of lists to be transformed.
    :param array_mod: List of lists used for comparison.
    :param fromm: The value to be checked in the first position of the pairs.
    :param repa: The value to be replaced if conditions are met.
    :param replace: The value to replace `repa` with.
    :param e: List of lists containing pairs to be checked against.
    :return: Transformed array.
    """
    for i, edge in enumerate(array_mod):
        if edge[0] == repa:
            if [fromm, edge[1]] in e:
                array[i] = [replace, edge[1]]
        if edge[1] == repa:
            if [edge[0], fromm] in e:
                array[i] = [edge[0], replace]
    return array


def repeated_sizes(repeated):
    """
    Takes a list of sorted equivalences and returns the set of lengths of the equivalence array.

    :param repeated: List of sorted equivalences
    :return: List of lengths of each equivalence
    """
    rep = list(set([len(rep[1]) for rep in repeated]))
    rep.sort()
    return rep


def equal_two(list_a, list_b):
    """
    Tests if two lists contain the same elements, setwise equality.

    This function takes two lists and checks if they contain the same elements,
    regardless of the order of the elements.

    :param list_a: Input list, e.g., [A, B, C, D]
    :param list_b: Input list, e.g., [A, C, B, D]
    :return: True if the two lists are equal setwise, False otherwise
    """
    if set(row for row in list_a) == set(row for row in list_b):
        return True
    else:
        return False


def equal(list_a, list_b):
    """
    Tests if two lists contain the same elements, setwise equality.

    This function takes two lists and checks if they contain the same elements,
    regardless of the order of the elements.

    :param list_a: Input list, e.g., [A, B, C, D]
    :param list_b: Input list, e.g., [A, C, B, D]
    :return: True if the two lists are equal setwise, False otherwise
    """
    if set(tuple(row) for row in list_a) == set(tuple(row) for row in list_b):
        return True
    else:
        return False


def check_edge_in_list(edges, list_in):
    """
    Checks if a list is contained in a list of lists.

    This function takes a list of lists and a list, and checks if the list is contained
    in the list of lists using setwise equality.

    :param list_in: Input list of lists, e.g., [[A, B, C, D], [A, C, B, D]]
    :param edges: Input list, e.g., [A, C, B, D]
    :return: True if the list is contained in the list of lists, False otherwise
    """
    for l in list_in:
        if equal(l, edges):
            return True
    return False


def equivalence(pieces, equivalences):
    """
    Transforms a list of pieces of the remnant graph to their equivalent set of pieces from the original graph.

    This function takes a list of pieces of the remnant graph that can have a label not found in the original graph,
    and transforms them to their equivalent set of pieces all from the original graph based on the provided equivalences.

    :param pieces: Input list of pieces
    :param equivalences: List of equivalences
    :return: Relabeled list of pieces
    """
    pieces_mod = copy.deepcopy(pieces)
    for j, piece in enumerate(pieces_mod):
        for i, edge in enumerate(piece):
            if edge[0] in np.array(equivalences)[:, 1]:
                pieces_mod[j][i][0] = equivalences[
                    np.array(equivalences)[:, 1].tolist().index(edge[0])
                ][0]
            if edge[1] in np.array(equivalences)[:, 1]:
                pieces_mod[j][i][1] = equivalences[
                    np.array(equivalences)[:, 1].tolist().index(edge[1])
                ][0]
    return pieces_mod


def fix_repeated_equiv(er, repeated, equivalences, e):
    """
    Fixes indexing issues in equivalences by transforming arrays based on specified conditions.

    This function attempts to fix indexing issues in the equivalences array. It iterates over the equivalences,
    identifies repeated elements, and transforms the arrays accordingly. This process is non-deterministic and
    may require multiple attempts to achieve a well-formatted pathway.

    :param er: List of edges to be transformed.
    :param repeated: List of repeated equivalences.
    :param equivalences: List of equivalences.
    :param e: List of edges containing pairs to be checked against.
    :return: Tuple containing the transformed edges, repeated equivalences, and updated equivalences.
    """
    equivalences = np.unique(equivalences, axis=0).tolist()
    equiv_np = np.array(equivalences)
    if len(equiv_np) != 0:
        sorted_eq = equiv_np[equiv_np[:, 0].argsort()]
    else:
        sorted_eq = equiv_np
    repeated_eq_1 = []
    for i, array in enumerate(sorted_eq):
        check = np.concatenate((sorted_eq[:, 1][0:i], sorted_eq[:, 1][i + 1:]), axis=0)
        if array[1] in check:
            repeated_eq_1.append(array.tolist())

    repeated_eq_2 = []
    for i, array in enumerate(sorted_eq):
        check_2 = np.concatenate(
            (sorted_eq[:, 0][0:i], sorted_eq[:, 0][i + 1:]), axis=0
        )
        if array[0] in check_2:
            repeated_eq_2.append(array.tolist())
    inter = np.intersect1d(repeated_eq_1, repeated_eq_2).tolist()
    idx = [sorted_eq.tolist().index(rem) for rem in repeated_eq_1]
    remove = np.delete(sorted_eq, idx, axis=0)
    if repeated_eq_1 != []:
        repeated_mod = [equivalence(rep, remove) for rep in repeated]
        er_mod = equivalence([er], remove)[0]
        if repeated_eq_2 == [] or (
                repeated_eq_2[0][0] == repeated_eq_2[0][0] and inter == []
        ):
            rep_eq_1_np = np.array(repeated_eq_1)
            sort_repeated_eq_1 = rep_eq_1_np[rep_eq_1_np[:, 1].argsort()]
            add = []
            for i in range(int(sort_repeated_eq_1.shape[0] / 2)):
                replace = equiv_np[equiv_np[:, 1].argsort()][-1][1] + 1 + i
                repa = sort_repeated_eq_1[2 * i][1]
                fromm = sort_repeated_eq_1[2 * i][0]
                add = add + [[fromm, replace], sort_repeated_eq_1.tolist()[2 * i + 1]]
                er = transform_array(er, er_mod, fromm, repa, replace, e)
                for i, rep in enumerate(repeated_mod):
                    for j, _ in enumerate(rep):
                        repeated[i][j] = transform_array(
                            repeated[i][j], repeated_mod[i][j], fromm, repa, replace, e
                        )
            equivalences = remove.tolist() + add
        else:
            for item in repeated_eq_2:
                if inter[0] == item[0] and inter[1] != item[1]:
                    replace = item[1]
                    repa = inter[1]
                    fromm = item[0]
            equivalences = (
                    remove.tolist()
                    + np.delete(repeated_eq_1, repeated_eq_1.index(inter), axis=0).tolist()
            )
            er = transform_array(er, er_mod, fromm, repa, replace, e)
            for i, rep in enumerate(repeated_mod):
                for j, _ in enumerate(rep):
                    repeated[i][j] = transform_array(
                        repeated[i][j], repeated_mod[i][j], fromm, repa, replace, e
                    )

            equiv_np = np.array(equivalences)
            sorted_eq = equiv_np[equiv_np[:, 0].argsort()]
            repeated_eq_1 = []
            for i, array in enumerate(sorted_eq):
                check = np.concatenate(
                    (sorted_eq[:, 1][0:i], sorted_eq[:, 1][i + 1:]), axis=0
                )
                if array[1] in check:
                    repeated_eq_1.append(array.tolist())
            if repeated_eq_1 != []:
                er, repeated, equivalences = fix_repeated_equiv(
                    er, repeated, equivalences
                )

    return er, repeated, equivalences


def index_set(lists, list_in):
    """
    Takes a list of list of lists and a list of lists, and returns the first index where the list and the element of the list of lists have set equality.

    This function iterates over a list of list of lists and checks if any of the lists within it have set equality with the provided list of lists.

    :param lists: Input list of list of lists
    :param list_in: Input list of lists
    :return: The first index where the list of lists appears in the list of list of lists with set equality, or None if not found.
    """
    for i, lista in enumerate(lists):
        if set(tuple(row) for row in list_in) == set(tuple(row) for row in lista):
            return i + 1


def select_length(e):
    """
    Takes a dictionary and returns the entry for the 'len' key.

    :param e: Dictionary containing arrays lengths and indexes.
    :return: Entry for e['len'].
    """
    return e["len"]


def transform_bond_float(bond):
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


def transform_bond(bond):
    """
    Converts a bond type from float to RDKit bond type.

    This function takes a bond type as a float and returns its corresponding RDKit bond type.
    If the bond type is not recognized, it returns an error string.

    :param bond: Bond type as a float
    :return: RDKit bond type
    """
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
    emol = RWMol()
    for v in atoms_info:
        emol.AddAtom(Chem.Atom(v[1]))
    for e in bonds_info:
        emol.AddBond(e[0], e[1], transform_bond(e[2]))
    mol = emol.GetMol()
    return mol


class AssemblyConstruction:
    def __init__(self, v, e, v_l, e_l, remnant_e, equivalences, duplicates, if_string=False):
        self.v = v
        self.e = e
        self.v_l = v_l
        self.e_l = e_l
        self.remnant_e = remnant_e
        self.equivalences = equivalences
        self.duplicates = duplicates
        self.ifstring = if_string
        atoms_list = []
        atoms_list_indx = []
        atoms_pre = []
        full_atoms_list = []
        for i, bond in enumerate(e):
            atom_list = [[v_l[bond[0]], v_l[bond[1]]], e_l[i]]
            atom_list_indx = [bond[0], bond[1]]
            atom_set = [{v_l[bond[0]], v_l[bond[1]]}, e_l[i]]
            if not (atom_set in atoms_pre):
                atoms_pre.append(atom_set)
                atoms_list.append(atom_list)
                atoms_list_indx.append(atom_list_indx)
            full_atoms_list.append(atom_list)
        self.atoms = atoms_pre
        self.full_atoms_list = full_atoms_list
        self.atoms_list = atoms_list
        self.atoms_list_index = atoms_list_indx

    def consistent_join(self, pieces_mod, steps_mod, repeated_mo1_cp, step, digraph, indexes):
        """final takes a set of graph pieces[[[18 25],[25 17]],[[12 18],[14 18],[18 17]],[[14 26]]] and "intelligently" join one pair of edges at a time depending
            if the join version is still a connected graph, for example [[18 25],[25 17],[14 26]] is NOT valid
            but [[14 26],[14 18],[18 17],[14 26]] would be

        :param pieces_mod: The current starting piece of edges [[[18 25],[25 17]],[[12 18],[14 18],[18 17]],[[14 26]]](modified by equivalence up to the current step)
        :param steps_mod: the current list of steps to construct all the pieces(modified by equivalence up to the current step)
        :param repeated_mo1_cp: Is the original copy of the duplicate edges before the recursive_join was performed
        :param step: the number of joins so far i.e. assembly index(up to the current step)
        :param digraph: list atom0-> step1,atom1-> step1, step1-> step2,atom1-> step2, atom0-> step3,atom0-> step3
        :param indexes: from the repeated array, at each entry is the index of the steps_mod
        :return pieces_mod: The finishing piece of edges, [[[18 25],[25 17]],[[12 18],[14 26],[14 18],[18 17]]]
        :return steps_mod: set of graph pieces after intelligent join
        :return step: outputs the assembly index N=3(up to the current step)
        :return digraph: list atom0-> step1,atom1-> step1, step1-> step2,atom1-> step2, atom0-> step3,atom0-> step3(up to the current step)
        """
        left_sort = [rep[0] for rep in repeated_mo1_cp]
        right_sort = [rep[1] for rep in repeated_mo1_cp]
        for pic in pieces_mod:
            for pic_i in pieces_mod:
                pic_r = np.reshape(pic, (np.shape(pic)[0] * 2))
                pic_i_r = np.reshape(pic_i, (np.shape(pic_i)[0] * 2))
                for idx, ed in enumerate(pic_i_r):
                    if ed in pic_r and not (pic == pic_i):

                        step = step + 1
                        if self.ifstring:
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
                            atom1 = self.atoms.index(
                                [
                                    {self.v_l[pic[0][0]], self.v_l[pic[0][1]]},
                                    self.e_l[self.e.index(pic[0])],
                                ]
                            )
                            digraph.append(
                                ["atom{}".format(atom1), "step{}".format(step)]
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
                            atom1 = self.atoms.index(
                                [
                                    {self.v_l[pic_i[0][0]], self.v_l[pic_i[0][1]]},
                                    self.e_l[self.e.index(pic_i[0])],
                                ]
                            )
                            digraph.append(
                                ["atom{}".format(atom1), "step{}".format(step)]
                            )
                        pieces_mod.remove(pic)
                        pieces_mod.remove(pic_i)
                        if self.ifstring:
                            pieces_mod.insert(0, np.sort(pic + pic_i, axis=0).tolist())
                        else:
                            pieces_mod.insert(0, pic + pic_i)
                        return pieces_mod, steps_mod, step, digraph
        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(self, pieces_mod, steps_mod, sorted_repeated_mod1, step, digraph):
        """repeated_construction takes list of sorted equivalences [[[[0,2],[0,15]],[[1,3],[1,16]]],[[[2,49],[31,49]],[[3,50],[37,50]]][[[63,75],[67,70],[70,75]],[[49,50],[50,53],[52,53]]]]
        and if the right side of the equivalence is on pieces_mod, it adds the left side to pieces_mod and captures the index for the entry of the equivalences list.
        If right side is not on the pieces_mod, it constructs it from the known pieces and/or equivalences(final function) and it adds the final piece the pieces_mod,
        and the index of the equivalence in the steps_mod.
        """
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
                    cum = [pieces_mod[i] for i in indices]

                    for idx in indices:
                        pieces_mod.remove(pieces_mod_cp[idx])
                    # We consistently join the remnant pieces in such a way that the constructed molecule never has disconnected pieces
                    while not (len(cum) == 1):
                        cum, steps_mod, step, digraph = self.consistent_join(
                            cum,
                            steps_mod,
                            sorted_repeated_mod1_cp,
                            step,
                            digraph,
                            indexes,
                        )
                    # Construct repeat[1]
                    pieces_mod.append(cum[0])
                    pieces_mod.append(repeat[0])
                    step_ind[j] = 0
                    indexes[j] = index_set(steps_mod, repeat[1])
                    sorted_repeated_mod1.remove(repeat)

        return pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes

    def generate_pathway(self):
        ## Construct Remnant Graph
        # The Remnant Graph are usually disjoint pieces
        step = 0
        # The digraph contains the information of the assembly path
        digraph = []
        # The steps contains all the new assembled pieces at each step
        steps = []
        # We construct each piece of the remnant graph one edge at a time
        pieces = [[edge] for edge in self.remnant_e]

        # change to equivalences
        if len(self.equivalences) == 0:
            duplicates_mod = self.duplicates
            pieces_mod = pieces
        else:
            # change to equivalences
            duplicates_mod = [equivalence(rep, self.equivalences) for rep in self.duplicates]
            pieces_mod = equivalence(pieces, self.equivalences)

        # We sort the arrays of repeated by size
        sizes = []
        for i, repeat in enumerate(duplicates_mod):
            sizes.append({"index": i, "len": len(repeat[0])})

        sizes.sort(key=select_length)

        sorted_repeated_mod1 = []
        for size in sizes:
            sorted_repeated_mod1.append(duplicates_mod[size["index"]])

        # We generate all the steps that are needed to construct the duplicates

        (
            pieces_mod,
            steps_mod,
            sorted_repeated_mod1_cp,
            step,
            digraph,
            indexes,
        ) = self.repeated_construction(
            pieces_mod, steps, sorted_repeated_mod1, step, digraph
        )

        ## Consistent Join of all Pieces
        # We consistently join the remnant pieces in such a way that the constructed molecule never has disconnected pieces
        pieces_mod_cp = []
        while not (len(pieces_mod) == len(pieces_mod_cp)):
            pieces_mod_cp = copy.deepcopy(pieces_mod)
            pieces_mod, steps_mod, step, digraph = self.consistent_join(
                pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes
            )

        self.steps = steps_mod
        self.digraph = digraph
        self.pieces_mod = pieces_mod

        return None

    def pathway_log(self, file_name="pathway_log"):
        pathway_file = open("{}.txt".format(file_name), "w")
        pathway_file.write("#####Graph#####\n")
        pathway_file.write(str(self.v) + "\n")
        pathway_file.write(str(self.e) + "\n")
        pathway_file.write(str(self.v_l) + "\n")
        pathway_file.write(str(self.e_l) + "\n")
        pathway_file.write("#####Atoms#####\n")
        for index, a in enumerate(self.atoms_list):
            pathway_file.write("atom{}={}\n".format(index, a))
        pathway_file.write("#####Steps#####\n")
        for index, ste in enumerate(self.steps):
            pathway_file.write("step{}={}\n".format(index + 1, ste))
        pathway_file.write("#####Digraph#####\n")
        for i in self.digraph:
            pathway_file.write(str(i) + "\n")
        # close file
        pathway_file.close()

        return None

    def pathway_inchi_fragments(self):
        molecules_atoms = []
        inchi_list = []
        for atom in self.atoms_list:
            molecules_atoms.append(
                tables2mol(
                    (
                        [(0, atom[0][0]), (1, atom[0][1])],
                        [(0, 1, transform_bond_float(atom[1]))],
                    )
                )
            )
            inchi_list.append(Chem.MolToInchi(molecules_atoms[-1]))
        steps_index_s = []
        vs_atoms = []
        vs_atoms_index = []
        for i, st in enumerate(self.steps):
            indices = list(
                set(np.reshape(self.steps[i], (np.shape(self.steps[i])[0] * 2)))
            )
            steps_index = []
            for edge in self.steps[i]:
                steps_index.append(
                    [
                        indices.index(edge[0]),
                        indices.index(edge[1]),
                        self.e_l[self.e.index(edge)],
                    ]
                )
            v_atoms = []
            vs_atom_index = []
            for at in indices:
                v_atoms.append(self.v_l[at])
                vs_atom_index.append(at)
            steps_index_s.append(steps_index)
            vs_atoms.append(v_atoms)
            vs_atoms_index.append(vs_atom_index)

        molecules_steps = []
        for i, step in enumerate(steps_index_s):
            molecules_steps.append(
                tables2mol(
                    (
                        [(i, at) for at in vs_atoms[i]],
                        [
                            (edge[0], edge[1], transform_bond_float(edge[2]))
                            for edge in step
                        ],
                    )
                )
            )
            inchi_list.append(Chem.MolToInchi(molecules_steps[-1]))

        self.molecules_atoms = molecules_atoms
        self.molecules_steps = molecules_steps
        self.steps_indx_s = steps_index_s
        self.vs_atoms = vs_atoms
        self.vs_atoms_indx = vs_atoms_index
        return inchi_list

    def plot_pathway(self, mode):
        _ = self.pathway_inchi_fragments()

        if mode == 1:
            if not os.path.exists("path-images/path"):
                os.makedirs("path-images/path")
            for i, atom in enumerate(self.molecules_atoms):
                Draw.MolToFile(atom, "path-images/path/atom{}.png".format(i))
            for i, step in enumerate(self.molecules_steps):
                Draw.MolToFile(step, "path-images/path/step{}.png".format(i + 1))
        ###### Draw Molecules Mode 1 #####
        if mode == 2:
            if not os.path.exists("path-images/path"):
                os.makedirs("path-images/path")
            mol = self.molecules_steps[-1]
            atoms_info = [
                (atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol())
                for atom in mol.GetAtoms()
            ]
            bonds_info = [
                [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                for bond in mol.GetBonds()
            ]

            highlight_bonds = []
            highlight_atoms = []
            for step in self.steps:
                highlight_step_bond = []
                highlight_step_atom = []
                for i, bond in enumerate(bonds_info):
                    if bond in step:
                        continue
                    highlight_step_bond.append(i)
                    highlight_step_atom.append(bond)
                highlight_bonds.append(highlight_step_bond)
                highlight_atoms.append(
                    {item for sublist in highlight_step_atom for item in sublist}
                )

            for step, _ in enumerate(self.steps):
                atom_cols = {}
                for i, at in enumerate(highlight_atoms[step]):
                    atom_cols[at] = (1.0, 1.0, 1.0)
                bond_cols = {}
                for i, bd in enumerate(highlight_bonds[step]):
                    bond_cols[bd] = (1.0, 1.0, 1.0)
                d = rdMolDraw2D.MolDraw2DCairo(
                    500, 500
                )  # or MolDraw2DCairo to get PNGs
                # d.drawOptions().useBWAtomPalette()
                d.drawOptions().noAtomLabels = True
                d.drawOptions().fillHighlights = True
                d.drawOptions().continuousHighlight = False
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d,
                    mol,
                    highlightBonds=highlight_bonds[step],
                    highlightBondColors=bond_cols,
                )
                d.WriteDrawingText("path-images/path/step{}.png".format(step + 1))
            for i, atom in enumerate(self.molecules_atoms):
                Draw.MolToFile(atom, "path-images/path/atom{}.png".format(i))

        if mode == 2 or mode == 1:
            shutil.copyfile(
                "klay/pathway-plot.html", "path-images/path/pathway-plot.html"
            )
            shutil.copyfile(
                "klay/cytoscape-klay.js", "path-images/path/cytoscape-klay.js"
            )

            f = open("path-images/path/pathway.js", "w")
            f.write(
                "document.addEventListener('DOMContentLoaded', function(){\nvar cy = window.cy = cytoscape({\ncontainer: document.getElementById('cy'),\nwheelSensitivity: 0.1,\nlayout: {name: 'klay'},\nstyle: [{selector: 'node',\nstyle: {shape: 'round-rectangle',\n'height': 70,\n'width': 70,\n'background-fit': 'cover',\n'border-color': '#000',\n'border-width': 3,\n'border-opacity': 0.5,\n'background-width': 35,\n'background-height': 35,\n'background-image-containment': 'over',}},\n{selector: 'edge',\nstyle: {\n'curve-style': 'bezier',\n'target-arrow-shape': 'triangle',\n'line-color': '#1aa7ec',\n'target-arrow-color': '#1aa7ec',\n'opacity': 0.5}},\n"
            )
            for index, _ in enumerate(self.atoms_list):
                f.write(
                    "{{selector: '#atom{}', style: {{'background-image': 'atom{}.png',}}}},\n".format(
                        index, index
                    )
                )
            for index, _ in enumerate(self.steps):
                f.write(
                    "{{selector: '#step{}', style: {{'background-image': 'step{}.png',}}}},\n".format(
                        index + 1, index + 1
                    )
                )
            f.write("],\n elements: {\n nodes: [\n")
            for index, _ in enumerate(self.atoms_list):
                f.write("  {{ data: {{ id: 'atom{}' }} }},\n".format(index))
            for index, _ in enumerate(self.steps):
                f.write("  {{ data: {{ id: 'step{}' }} }},\n".format(index + 1))
            f.write("],\n edges: [\n")
            for diredge in self.digraph:
                f.write(
                    "  {{ data: {{ source: '{}', target: '{}' }} }},\n".format(
                        diredge[0], diredge[1]
                    )
                )
            f.write("]\n}\n});\n});\n")
            f.close()

        return None


def parse_pathway_file(file):
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

    remnant_e, duplicates, equivalences = fix_repeated_equiv(remnant_e, duplicates, equivalences, e)
    construction_object = AssemblyConstruction(v, e, v_l, e_l, remnant_e, equivalences, duplicates)

    construction_object.generate_pathway()
    construction_object.plot_pathway(mode)

    return construction_object
