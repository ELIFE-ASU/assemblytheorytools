import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol


def select_length(e):
    """
    select_length takes a dictionary and return the entry for the 'len' entry

    :param e: dictionary of arrays lengths and indexes
    :return: entry for e['len']
    """
    return e["len"]


def equal(lista, listb):
    """
    equal takes two lists [A,B,C,D] and [A,C,B,D] and test if they contain the same elements, setwise equality

    :param lista: input list [A,B,C,D]
    :param listb: input list [A,C,B,D]
    :return: outputs if the two lists are equal or not
    """
    if set(tuple(row) for row in lista) == set(tuple(row) for row in listb):
        return True
    else:
        return False


def check_edge_in_lista(edges, lista):
    """
    check_edge_in_lista takes a list of lists [[A,B,C,D], [A,C,B,D]] and a list [A,C,B,D] and checks if the list is contained in the list of lists

    :param lista: input list of lists [[A,B,C,D], [A,C,B,D]]
    :param edges: input list [A,C,B,D]
    :return: outputs if the list is contained in the list of lists
    """
    for l in lista:
        if equal(l, edges):
            return True
    return False


def index_set(lists, listb):
    """
    index_set takes a list of list of lists [[[1,2],[2,3]],[[3,4],[5,6]]] and a list of lists [[2,3],[1,2]] and returns the first index where the list and the element of the list of list
        have a set equality

    :param lists: input list of list of lists [[[1,2],[2,3]],[[3,4],[5,6]]]
    :param listb: input list [[2,3],[1,2]]
    :return: outputs the first index for which the list  of lists appears in the list of list of lists in set equality
    """
    for i, lista in enumerate(lists):
        if set(tuple(row) for row in listb) == set(
                tuple(row) for row in lista
        ):
            return i + 1


def repeated_sizes(repeated):
    """
    repeated_sizes takes a list of sorted equivalences [[[[0, 2], [0, 15]], [[1, 3], [1, 16]]],[[[2, 49], [31, 49]], [[3, 50], [37, 50]]] [[[63, 75], [67, 70], [70, 75]], [[49, 50], [50, 53], [52, 53]]]],and returns the set of  lenghts of the equivalence array.

    :param repeated: [[[[0, 2], [0, 15]], [[1, 3], [1, 16]]],[[[2, 49], [31, 49]], [[3, 50], [37, 50]]] [[[63, 75], [67, 70], [70, 75]], [[49, 50], [50, 53], [52, 53]]]]
    :return: len of each equivalence [2,3]
    """
    rep = list(set([len(rep[1]) for rep in repeated]))
    rep.sort()
    return rep


def equivalence(pieces, equivalences):
    """
    equivalence transforms a list of pieces of the remnant graph that can have a label not found in the original graph, to thier equivalent set of pieces all from the original graph

    :param pieces: input list [[[9 17]],[[24 25],[25 26],[26 21]],[[28 29] [14 29]]
    :param equivalences: input list [[17 26],[19 27],[12 28],[18 29]]
    :return: outputs a relabeled list [[[9 17]],[[24 25],[25 17],[17 21]],[[12 18] [14 18]]
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


class assemblyConstruction:
    def __init__(
            self,
            v,
            e,
            v_l,
            e_l,
            remnant_e,
            equivalences,
            duplicates,
            ifstring=False,
    ):
        self.v = v
        self.e = e
        self.v_l = v_l
        self.e_l = e_l
        self.remnant_e = remnant_e
        self.equivalences = equivalences
        self.duplicates = duplicates
        self.ifstring = ifstring
        atoms_list = []
        atoms_pre = []
        full_atoms_list = []
        for i, bond in enumerate(e):
            atom_list = [[v_l[bond[0]], v_l[bond[1]]], e_l[i]]
            atom_set = [{v_l[bond[0]], v_l[bond[1]]}, e_l[i]]
            if not (atom_set in atoms_pre):
                atoms_pre.append(atom_set)
                atoms_list.append(atom_list)
            full_atoms_list.append(atom_list)
        self.atoms = atoms_pre
        self.full_atoms_list = full_atoms_list
        self.atoms_list = atoms_list

    def consistent_join(
            self, pieces_mod, steps_mod, repeated_mo1_cp, step, digraph, indexes
    ):
        """
        final takes a set of graph pieces[[[18 25],[25 17]],[[12 18],[14 18],[18 17]],[[14 26]]] and "intelligently" join one pair of edges at a time depending
            if the join version is still a connected graph, for example [[18 25],[25 17],[14 26]] is NOT valid
            but [[14 26],[14 18],[18 17],[14 26]] would be

        :param pieces_mod: The current starting piece of edges [[[18 25],[25 17]],[[12 18],[14 18],[18 17]],[[14 26]]](modified by equivalence up to the current step)
        :param steps_mod: the current list of steps to construct all the pieces(modified by equivalence up to the current step)
        :param repeated_mo1_cp: Is the original copy of the duplicate edges before the recursive_join was performed
        :param step: the number of joins so far i.e. assembly index(up to the to the current step)
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
            for pici in pieces_mod:
                pic_r = np.reshape(pic, (np.shape(pic)[0] * 2))
                pici_r = np.reshape(pici, (np.shape(pici)[0] * 2))
                for idx, ed in enumerate(pici_r):
                    if ed in pic_r and not (pic == pici):
                        step = step + 1
                        if self.ifstring:
                            steps_mod.append(
                                np.sort(pic + pici, axis=0).tolist()
                            )
                        else:
                            steps_mod.append(pic + pici)
                        if not (len(pic) == 1):
                            if pic in left_sort:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            indexes[left_sort.index(pic)]
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic in right_sort:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            indexes[right_sort.index(pic)]
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pic in steps_mod:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            steps_mod.index(pic) + 1
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            else:
                                digraph.append(
                                    [
                                        "step{}".format("_error"),
                                        "step{}".format(step),
                                    ]
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
                        if not (len(pici) == 1):
                            if pici in left_sort:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            indexes[left_sort.index(pici)]
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pici in right_sort:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            indexes[right_sort.index(pici)]
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            elif pici in steps_mod:
                                digraph.append(
                                    [
                                        "step{}".format(
                                            steps_mod.index(pici) + 1
                                        ),
                                        "step{}".format(step),
                                    ]
                                )
                            else:
                                digraph.append(
                                    [
                                        "step{}".format("_error"),
                                        "step{}".format(step),
                                    ]
                                )

                        else:
                            atom1 = self.atoms.index(
                                [
                                    {
                                        self.v_l[pici[0][0]],
                                        self.v_l[pici[0][1]],
                                    },
                                    self.e_l[self.e.index(pici[0])],
                                ]
                            )
                            digraph.append(
                                ["atom{}".format(atom1), "step{}".format(step)]
                            )
                        pieces_mod.remove(pic)
                        pieces_mod.remove(pici)
                        if self.ifstring:
                            pieces_mod.insert(
                                0, np.sort(pic + pici, axis=0).tolist()
                            )
                        else:
                            pieces_mod.insert(0, pic + pici)
                        return pieces_mod, steps_mod, step, digraph
        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(
            self, pieces_mod, steps_mod, sorted_repeated_mod1, step, digraph
    ):
        """
        repeated_construction takes list of sorted equivalences [[[[0,2],[0,15]],[[1,3],[1,16]]],[[[2,49],[31,49]],[[3,50],[37,50]]][[[63,75],[67,70],[70,75]],[[49,50],[50,53],[52,53]]]]
        and if the right side of the equivalence is on pices_mod, it adds the left side to pices_mod and captures the index for the entry of the equivalences list.
        If right side is not on the pieces_mod, it constructs it from the known pieces and/or equivalences(final function) and it adds the final pice the pieces_mod,
        and the index of the equivalence in the steps_mod.
        """
        step_ind = [1 for i in range(len(sorted_repeated_mod1))]
        indexes = [0 for i in range(len(sorted_repeated_mod1))]
        sorted_repeated_mod1_cp = copy.deepcopy(sorted_repeated_mod1)
        left_sort = [rep[0] for rep in sorted_repeated_mod1_cp]
        right_sort = [rep[1] for rep in sorted_repeated_mod1_cp]
        while len(sorted_repeated_mod1) != 0:
            for j, repeat in enumerate(sorted_repeated_mod1_cp):
                if (not step_ind[j]) or repeated_sizes(sorted_repeated_mod1)[
                    0
                ] != len(repeat[0]):
                    continue
                if check_edge_in_lista(
                        repeat[1], pieces_mod
                ) or check_edge_in_lista(repeat[1], steps_mod):
                    pieces_mod.append(repeat[0])
                    sorted_repeated_mod1.remove(repeat)
                    step_ind[j] = 0
                    if index_set(steps_mod, repeat[1]) != None:
                        indexes[j] = index_set(steps_mod, repeat[1])
                    else:
                        indexes[j] = indexes[
                            index_set(left_sort, repeat[1]) - 1
                            ]
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
                    if indices == []:
                        continue
                    cum = [pieces_mod[i] for i in indices]
                    for idx in indices:
                        pieces_mod.remove(pieces_mod_cp[idx])
                    counter = 0
                    while not (len(cum) == 1):
                        counter = counter + 1
                        if counter > 100:
                            raise ValueError("Cant join the pieces")
                        cum, steps_mod, step, digraph = self.consistent_join(
                            cum,
                            steps_mod,
                            sorted_repeated_mod1_cp,
                            step,
                            digraph,
                            indexes,
                        )
                    pieces_mod.append(cum[0])
                    pieces_mod.append(repeat[0])
                    step_ind[j] = 0
                    indexes[j] = index_set(steps_mod, repeat[1])
                    sorted_repeated_mod1.remove(repeat)

        return (
            pieces_mod,
            steps_mod,
            sorted_repeated_mod1_cp,
            step,
            digraph,
            indexes,
        )

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

        # Change to equivalences
        if len(self.equivalences) == 0:
            duplicates_mod = self.duplicates
            pieces_mod = pieces
        else:
            # change to equivalences
            duplicates_mod = [
                equivalence(rep, self.equivalences) for rep in self.duplicates
            ]
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
                pieces_mod,
                steps_mod,
                sorted_repeated_mod1_cp,
                step,
                digraph,
                indexes,
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

    def pathway_log_string(self) -> str:
        pathway_file = []
        pathway_file.append("#####Graph#####\n")
        pathway_file.append(str(self.v) + "\n")
        pathway_file.append(str(self.e) + "\n")
        pathway_file.append(str(self.v_l) + "\n")
        pathway_file.append(str(self.e_l) + "\n")
        pathway_file.append("#####Atoms#####\n")
        for index, a in enumerate(self.atoms_list):
            pathway_file.append("atom{}={}\n".format(index, a))
        pathway_file.append("#####Steps#####\n")
        for index, ste in enumerate(self.steps):
            pathway_file.append("step{}={}\n".format(index + 1, ste))
        pathway_file.append("#####Digraph#####\n")
        for i in self.digraph:
            pathway_file.append(str(i) + "\n")
        return "".join(pathway_file)

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
        steps_indx_s = []
        vs_atoms = []
        for i, st in enumerate(self.steps):
            indices = list(
                set(
                    np.reshape(self.steps[i], (np.shape(self.steps[i])[0] * 2))
                )
            )
            steps_indx = []
            for edge in self.steps[i]:
                steps_indx.append(
                    [
                        indices.index(edge[0]),
                        indices.index(edge[1]),
                        self.e_l[self.e.index(edge)],
                    ]
                )
            v_atoms = []
            for at in indices:
                v_atoms.append(self.v_l[at])

            steps_indx_s.append(steps_indx)
            vs_atoms.append(v_atoms)

        molecules_steps = []
        for i, step in enumerate(steps_indx_s):
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
        self.steps_indx_s = steps_indx_s
        self.vs_atoms = vs_atoms
        return inchi_list


def parse_pathway_file_ian(data):
    v = data["file_graph"][0]['Vertices']
    e = data["file_graph"][0]['Edges']
    v_l = data["file_graph"][0]['VertexColours']
    e_l = data["file_graph"][0]['EdgeColours']
    remnant_e = data["remnant"][0]["Edges"] + data["removed_edges"]
    duplicates = [[dup["Right"], dup['Left']] for dup in data["duplicates"]]
    equivalences = [[1, 1]]

    construction_object = assemblyConstruction(
        v, e, v_l, e_l, remnant_e, equivalences, duplicates
    )

    construction_object.generate_pathway()

    return construction_object


def transform_bond(bond):
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
        emol.AddBond(e[0], e[1], transform_bond(e[2]))
    mol = emol.GetMol()
    return mol


def transform_bond_float(bond):
    if bond == "single":
        return 1.0
    if bond == "double":
        return 2.0
    if bond == "triple":
        return 3.0
    return "error"
