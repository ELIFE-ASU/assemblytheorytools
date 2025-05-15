import copy
import json
import re

import numpy as np


def v_string_convert(v_string, input_type):
    """v_string_convert transforms a string "[0 1 2]" into a list of ints of strings [0, 1, 2]

    :param v_string: input string "[0 1 2]" or "[A B C]"
    :param input_type: use either a integer or string conversion
    :return: outputs a list [0, 1, 2] or ["A", "B", "C"]
    """
    indices = []
    if input_type == "int":
        f = int
    if input_type == "str":
        f = str
    for i, c in enumerate(v_string):
        if c == " ":
            indices.append(i)
    if indices:
        v = [f(v_string[1:indices[0]])]
        for i, j in enumerate(indices):
            if i == len(indices) - 1:
                break
            v.append(f(v_string[indices[i] + 1:indices[i + 1]]))
    else:
        indices = [0]
        v = []
    v.append(f(v_string[indices[-1] + 1:-1]))
    return v


def e_string_convert(e_string):
    """e_string_convert transforms a string "[[0 1] [1 2]]" into a list of ints [[0, 1],[1,2]]

    :param e_string: input string "[[0 1] [1 2]]"
    :return: outputs a list of ints [[0, 1],[1,2]]
    """
    export = np.fromstring(e_string.replace("[", "").replace("]", ""), dtype=int, sep=' ')
    e = export.reshape((int(len(export) / 2), 2)).tolist()
    return e


def repeated_sizes(repeated):
    """repeated_sizes takes a list of sorted equivalences [[[[0, 2], [0, 15]], [[1, 3], [1, 16]]],[[[2, 49], [31, 49]], [[3, 50], [37, 50]]] [[[63, 75], [67, 70], [70, 75]], [[49, 50], [50, 53], [52, 53]]]],and returns the set of  lenghts of the equivalence array.

    :param repeated: [[[[0, 2], [0, 15]], [[1, 3], [1, 16]]],[[[2, 49], [31, 49]], [[3, 50], [37, 50]]] [[[63, 75], [67, 70], [70, 75]], [[49, 50], [50, 53], [52, 53]]]]
    :return: len of each equivalence [2,3]
    """
    rep = list(set([len(rep[1]) for rep in repeated]))
    rep.sort()
    return rep


def equal(list_a, list_b):
    """equal takes two lists [A,B,C,D] and [A,C,B,D] and test if they contain the same elements, setwise equality

    :param list_a: input list [A,B,C,D]
    :param list_b: input list [A,C,B,D]
    :return: outputs if the two lists are equal or not
    """
    if set(tuple(row) for row in list_a) == set(tuple(row) for row in list_b):
        return True
    else:
        return False


def check_edge_in_lista(edges, list_a):
    """check_edge_in_lista takes a list of lists [[A,B,C,D], [A,C,B,D]] and a list [A,C,B,D] and checks if the list is contained in the list of lists

    :param list_a: input list of lists [[A,B,C,D], [A,C,B,D]]
    :param edges: input list [A,C,B,D]
    :return: outputs if the list is contained in the list of lists
    """
    for l in list_a:
        if equal(l, edges):
            return True
    return False


def index_set(lists, list_b):
    """index_set takes a list of list of lists [[[1,2],[2,3]],[[3,4],[5,6]]] and a list of lists [[2,3],[1,2]] and returns the first index where the list and the element of the list of list
        have a set equality

    :param lists: input list of list of lists [[[1,2],[2,3]],[[3,4],[5,6]]]
    :param list_b: input list [[2,3],[1,2]]
    :return: outputs the first index for which the list  of lists appears in the list of list of lists in set equality
    """
    for i, lista in enumerate(lists):
        if set(tuple(row) for row in list_b) == set(tuple(row) for row in lista):
            return i + 1


def select_length(e):
    """select_length takes a dictionary and return the entry for the 'len' entry

    :param e: dictionary of arrays lengths and indexes
    :return: entry for e['len']
    """
    return e["len"]


def equivalence(pieces, equivalences):
    """equivalence transforms a list of pieces of the remnant graph that can have a label not found in the original graph, to thier equivalent set of pieces all from the original graph

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
            self, v, e, v_l, e_l, remnant_e, equivalences, duplicates, ifstring=False
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
        atoms_list_indx = []
        atoms_pre = []
        full_atoms_list = []
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
        self.atoms_list_indx = atoms_list_indx

    def consistent_join(
            self, pieces_mod, steps_mod, repeated_mo1_cp, step, digraph, indexes
    ):
        """final takes a set of graph pieces[[[18 25],[25 17]],[[12 18],[14 18],[18 17]],[[14 26]]] and "intelligently" join one pair of edges at a time depending
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
                        # temporary fix for strings
                        #                        if (idx%2==0 and ed==0 and self.ifstring):
                        #                            continue
                        step = step + 1
                        if self.ifstring:
                            steps_mod.append(np.sort(pic + pici, axis=0).tolist())
                        else:
                            steps_mod.append(pic + pici)
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
                        if not (len(pici) == 1):
                            if pici in left_sort:
                                digraph.append(
                                    [
                                        "step{}".format(indexes[left_sort.index(pici)]),
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
                                        "step{}".format(steps_mod.index(pici) + 1),
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
                                    {self.v_l[pici[0][0]], self.v_l[pici[0][1]]},
                                    self.e_l[self.e.index(pici[0])],
                                ]
                            )
                            digraph.append(
                                ["atom{}".format(atom1), "step{}".format(step)]
                            )
                        pieces_mod.remove(pic)
                        pieces_mod.remove(pici)
                        if self.ifstring:
                            pieces_mod.insert(0, np.sort(pic + pici, axis=0).tolist())
                        else:
                            pieces_mod.insert(0, pic + pici)
                        return pieces_mod, steps_mod, step, digraph
        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(
            self, pieces_mod, steps_mod, sorted_repeated_mod1, step, digraph
    ):
        """repeated_construction takes list of sorted equivalences [[[[0,2],[0,15]],[[1,3],[1,16]]],[[[2,49],[31,49]],[[3,50],[37,50]]][[[63,75],[67,70],[70,75]],[[49,50],[50,53],[52,53]]]]
        and if the right side of the equivalence is on pices_mod, it adds the left side to pices_mod and captures the index for the entry of the equivalences list.
        If right side is not on the pieces_mod, it constructs it from the known pieces and/or equivalences(final function) and it adds the final pice the pieces_mod,
        and the index of the equivalence in the steps_mod.
        """
        # print(pieces_mod)
        # print(sorted_repeated_mod1)
        # print(steps_mod)
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
                # print(repeated_sizes(sorted_repeated_mod1))
                # print(len(repeat[0]))
                # print(repeat[1])
                # print(pieces_mod)
                # print(steps_mod)
                if check_edge_in_lista(repeat[1], pieces_mod) or check_edge_in_lista(repeat[1], steps_mod):
                    pieces_mod.append(repeat[0])
                    sorted_repeated_mod1.remove(repeat)
                    step_ind[j] = 0
                    if index_set(steps_mod, repeat[1]) != None:
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
                    if indices == []:
                        continue
                    cum = [pieces_mod[i] for i in indices]
                    # print(cum)
                    for idx in indices:
                        pieces_mod.remove(pieces_mod_cp[idx])
                    # We consistently join the remant pieces in such a way that the constructed molecule never has disconnected pieces
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
        ## Construct Remant Graph
        # The Remnant Graph are usually are usually disjoing pieces
        step = 0
        # The digraph contains the information of the assembly path
        digraph = []
        # The steps contains all the the new assembled pieces at each step
        steps = []
        # We construct each piece of the remnant graph one edge at a time
        pieces = [[edge] for edge in self.remnant_e]

        # change to equivalences
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
        # We consistently join the remant pieces in such a way that the constructed molecule never has disconnected pieces
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


def parse_str_log(file_name):
    """parse_str_log takes a file from the AssemblyGo String log output and outputs all relevant information for its pathway reconstruction

    :param file_name: input file location e.g. "/your_file_location"
    :return string: outputs the original string e.g. "aaaaaaaabbbbaaaaaaa"
    :return rem_pre_str: outputs the remant string list: ['aa','bb','a']
    :return rem_pre_id: outputs the remant string correspoinding original list location [0,16,18]
    :return repeated: outputs the duplicated strings [[[0, 6], [12, 18]],[[6, 8], [12, 14]],[[8, 10], [10, 12]],[[12, 14], [14, 16]],[[14, 16], [16, 18]]]
    :return runtime: outputs runtime
    """
    f = open(file_name, "r")
    counter = 0
    repeated = []
    for x in f:
        if x[-16:-1] == "ORIGINAL String":
            counter = -1
            continue
        if counter == -1:
            counter = -2
            continue
        if counter == -2:
            string = x[18:-2]
            counter = -3
            continue
        if x[0:14] == "Remnant String":
            counter = 1
            continue
        if counter == 1:
            rem_pre_str = v_string_convert(x[17:-1], "str")
            counter = 2
            continue
        if counter == 2:
            rem_pre_id = v_string_convert(x[19:-1], "int")
            counter = 3
            continue
        if x[0:18] == "Duplicated Strings":
            counter = 6
            continue
        if counter == 6 and x == "\n":
            counter = 7
            continue
        if counter == 6:
            repeated.append(e_string_convert(x[0:-1]))
        s = re.sub(r'[0-9]', " ", x)
        s = re.sub(r'\W+', " ", s).strip()
        if s == "Time":
            runtime = float(x[x.find("Time") + 7:])

    return string, rem_pre_str, rem_pre_id, repeated, runtime


def transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre):
    """transform_string_mol takes the pathway reconstruction information of an String and convert it to the path reconstruction information for a molecule that is a linear chain

    The string "aaaaaaaabbbbaaaaaaa" will be converted to the following molecule chain:
            
        p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p
          a   a   a   a   a   a   a   a   b   b   b   b   a   a   a   a   a   a   a

    :param string: outputs the original string e.g. "aaaaaaaabbbbaaaaaaa"
    :param rem_pre_str: outputs the remant string list: ['aa','bb','a']
    :param rem_pre_id: outputs the remant string correspoinding original list location [0,16,18]
    :param repeated: outputs the duplicated strings [[[0, 6], [12, 18]],[[6, 8], [12, 14]],[[8, 10], [10, 12]],[[12, 14], [14, 16]],[[14, 16], [16, 18]]]
    :return v: vertices list [0, 1, 2, 3, 4, ... , 17, 18, 19]
    :return e: edges list [[0, 1],[1, 2],[2, 3],[3, 4],...,[17, 18],[18, 19]]
    :return v_l: vertex labels ['p','p','p','p',...,,'p','p','p']
    :return e_l: edges labels['a','a','a','a','a','a','a','a','b','b','b','b','a','a','a','a','a','a','a']
    :return remnant_e: remanant edges [[10, 11], [11, 12], [16, 17], [17, 18], [18, 19]]
    :return equivalences: compatibility of equivalences to molecule construction [[1,1]]
    :return duplicates: duplicates edges [[[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],[[12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]]],[[[6, 7], [7, 8]], [[12, 13], [13, 14]]],[[[14, 15], [15, 16]], [[16, 17], [17, 18]]]]
    """
    v = list(range(len(string) + 1))
    e = [[i, i + 1] for i in v[:-1]]
    v_l = ["p" for _ in v]
    e_l = [string[i] for i, _ in enumerate(e)]
    remnant_e_pre = [[[item + j, item + j + 1] for j, _ in enumerate(rem_pre_str[i])] for i, item in
                     enumerate(rem_pre_id)]
    remnant_e = []
    for rem in remnant_e_pre:
        remnant_e = remnant_e + rem
    duplicates = [[[[index, index + 1] for index in range(pair[0], pair[1])] for pair in duplicate] for duplicate in
                  dup_pre]
    equivalences = [[1, 1]]
    return v, e, v_l, e_l, remnant_e, equivalences, duplicates


def generate_string_pathway(filename):
    """generate_string_pathway takes a file from the AssemblyGo String log output and outputs an assemblypath construction object 

    :param file_name: input file location e.g. "/your_file_location"
    :return construction_object: construction object that contains the pathway information of the string assembly
    """
    string, rem_pre_str, rem_pre_id, dup_pre, runtime = parse_str_log(filename)
    v, e, v_l, e_l, remnant_e, equivalences, duplicates = transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre)
    construction_object = assemblyConstruction(v, e, v_l, e_l, remnant_e, equivalences, duplicates, ifstring=True)
    try:
        construction_object.generate_pathway()
        steps_str = []
        for step in construction_object.steps:
            step_str = ""
            for edge in step:
                step_str = step_str + e_l[e.index(edge)]
            steps_str.append(step_str)
        construction_object.steps_str = steps_str
        pathway_success = True
    except ValueError as e:
        pathway_success = False
        print("error")
    except IndexError as e:
        pathway_success = False
        print("error")

    return construction_object, pathway_success


def generate_string_pathway_ian(file):
    f = open(file)
    data = json.load(f)
    string = data["file_graph"][0]['Fragments'][0]
    rem_pre_str = data["remnant"][0]['Fragments'][0].split(', ')
    rem_pre_id = data["remnant"][0]['Positions']
    dup_pre = [[[dup["Right"][0], dup["Right"][0] + dup["Right"][1]], [dup["Left"][0], dup["Left"][0] + dup["Left"][1]]]
               for dup in data["duplicates"]]
    v, e, v_l, e_l, remnant_e, equivalences, duplicates = transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre)
    construction_object = assemblyConstruction(v, e, v_l, e_l, remnant_e, equivalences, duplicates, ifstring=True)

    try:
        construction_object.generate_pathway()
        steps_str = []
        for step in construction_object.steps:
            step_str = ""
            for edge in step:
                step_str = step_str + e_l[e.index(edge)]
            steps_str.append(step_str)
        construction_object.steps_str = steps_str
        pathway_success = True
    except ValueError as e:
        pathway_success = False
        print("error")
    except IndexError as e:
        pathway_success = False
        print("error")

    return construction_object, pathway_success


def get_graph_string_explicit(construction_object):
    """get_graph_string takes an assemblypath construction object for strings and outputs the vertices and edges of the DAG(Directed Acylcic Graph) from the pathway 

    :param construction_object: construction object that contains the pathway information of the string assembly
    :return vertices,construction_object.digraph: Vertex and Edges explicit string information of the DAG(Directed Acylcic Graph) from the pathway 
    """
    vertex_dict = {}
    for i, a in enumerate(construction_object.atoms_list):
        if a[1] == "C":
            continue
        ato = "{}".format(a[1])
        vertex_dict["atom{}".format(i)] = ato

    # print("*Steps List*")
    for i, step_mol in enumerate(construction_object.steps_str):
        ato = ''.join(step_mol)
        vertex_dict["step{}".format(i + 1)] = ato
    vertices = []
    for i, _ in enumerate(construction_object.atoms_list):
        vertices.append(vertex_dict["atom{}".format(i)])
    for i, _ in enumerate(construction_object.steps):
        vertices.append(vertex_dict["step{}".format(i + 1)])
    digraph = []
    for edges in construction_object.digraph:
        digraph.append([vertex_dict[edges[0]], vertex_dict[edges[1]]])
    return vertices, digraph


def get_graph_string(construction_object):
    """get_graph_string takes an assemblypath construction object for strings and outputs the vertices and edges of the DAG(Directed Acylcic Graph) from the pathway 

    :param construction_object: construction object that contains the pathway information of the string assembly
    :return vertices,construction_object.digraph: Vertex and Edges infromation of the DAG(Directed Acylcic Graph) from the pathway 
    """
    vertices = []
    for i, _ in enumerate(construction_object.atoms_list):
        vertices.append("atom{}".format(i))
    for i, _ in enumerate(construction_object.steps):
        vertices.append("step{}".format(i + 1))
    return vertices, construction_object.digraph
