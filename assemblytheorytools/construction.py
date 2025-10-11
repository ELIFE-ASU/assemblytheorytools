import copy
import json
import os

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from .tools_graph import bond_order_assout_to_int, bond_order_int_to_rdkit, canonicalize_node_labels
from .tools_mol import smi_remove_implicit_hydrogen


def transform_array(target_array, comp_array, source_val, target_val, new_val, pairs_list):
    """
    Transforms the target array by replacing specific values based on the comparison array and pairs list.

    This function iterates over the comparison array and updates the target array by replacing elements
    that match the target value and source value with a new value, according to the pairs list.

    Parameters
    ----------
    target_array : list of list
        List of lists, where each sublist represents an edge in the target array.
    comp_array : list of list
        List of lists, where each sublist represents an edge in the comparison array.
    source_val : int
        The source value to be replaced.
    target_val : int
        The target value to be replaced.
    new_val : int
        The new value to replace the source and target values.
    pairs_list : list of list
        List of pairs that determine valid replacements.

    Returns
    -------
    list of list
        The modified target array with updated values.
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

    Parameters
    ----------
    repeated : list of tuple
        List of tuples, where each tuple contains two elements.

    Returns
    -------
    list
        Sorted list of unique sizes of the second element in each tuple.
    """
    rep = sorted(set(len(rep[1]) for rep in repeated))
    return rep


def equal_list(list_a, list_b):
    """
    Compares two lists of lists and checks if they contain the same elements.

    This function converts each sublist in the input lists to a set of tuples and compares them.
    It returns True if both lists contain the same sets of tuples, otherwise False.

    Parameters
    ----------
    list_a : list of list
        First list of lists to compare.
    list_b : list of list
        Second list of lists to compare.

    Returns
    -------
    bool
        True if both lists contain the same sets of tuples, otherwise False.
    """
    set_a = {tuple(sorted(sublist)) for sublist in list_a}
    set_b = {tuple(sorted(sublist)) for sublist in list_b}
    return set_a == set_b


def check_edge_in_list(edges, list_in):
    """
    Checks if a given list of edges is present in any of the lists within a list of lists.

    This function iterates over each list in the input list of lists and uses the `equal_list` function
    to check if any of these lists contain the same elements as the given list of edges.

    Parameters
    ----------
    edges : list
        List of edges to check.
    list_in : list of list
        List of lists, where each sublist is a list of edges.

    Returns
    -------
    bool
        True if the given list of edges is present in any of the lists within the input 
        list of lists, otherwise False.
    """
    return any(equal_list(l, edges) for l in list_in)


def equivalence(remnant_pieces, equivalences):
    """
    Applies equivalence transformations to the remnant pieces based on the provided equivalences.

    This function creates a deep copy of the remnant pieces and iterates through each edge in each piece.
    If an edge's vertex matches any vertex in the equivalences list, it replaces the vertex with the corresponding
    equivalent vertex.

    Parameters
    ----------
    remnant_pieces : list of list
        List of lists, where each sublist represents a piece containing edges.
    equivalences : list of list
        List of pairs, where each pair represents an equivalence between two vertices.

    Returns
    -------
    list of list
        A deep copy of the remnant pieces with applied equivalence transformations.
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

    Parameters
    ----------
    edge_list : list
        List of edges to be transformed.
    repeated_equiv : list
        List of repeated equivalences to be fixed.
    equivalences : list
        List of equivalences to be applied.
    edge_pairs : list
        List of valid edge pairs for transformations.

    Returns
    -------
    tuple
        A tuple containing:
        - edge_list : list
            The updated edge list.
        - repeated_equiv : list
            The updated repeated equivalences.
        - equivalences : list
            The updated equivalences.
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
    It then checks if any of these sets match the set of the input list and returns the index (1-based) 
    of the matching list.

    Parameters
    ----------
    lists : list of list
        List of lists to search within.
    list_in : list
        List to find within the list of lists.

    Returns
    -------
    int or None
        1-based index of the matching list, or None if no match is found.
    """
    list_in_set = set(tuple(row) for row in list_in)
    for i, i_list in enumerate(lists):
        if set(tuple(row) for row in i_list) == list_in_set:
            return i + 1


def select_length(dict_array):
    """
    Takes a dictionary and returns the entry for the 'len' key.

    Parameters
    ----------
    dict_array : dict
        Dictionary containing arrays lengths and indexes.

    Returns
    -------
    int or float
        Entry for the 'len' key.
    """
    return dict_array["len"]


def tables_to_mol(tables):
    """
    Converts atom and bond information into an RDKit molecule object.

    This function takes a tuple containing atom and bond information, constructs an RDKit RWMol object,
    adds atoms and bonds to it.

    Parameters
    ----------
    tables : tuple
        A tuple containing two lists:
        - atoms_info (list): A list of tuples where each tuple contains an atom index and atom type.
        - bonds_info (list): A list of tuples where each tuple contains two atom indices and a bond type.

    Returns
    -------
    Chem.Mol
        An RDKit molecule object with the specified atoms and bonds.
    """
    atoms_info, bonds_info = tables
    edit_mol = RWMol()

    # Add atoms to the molecule
    for v in atoms_info:
        edit_mol.AddAtom(Chem.Atom(v[1]))

    # Add bonds to the molecule
    for e in bonds_info:
        edit_mol.AddBond(e[0], e[1], bond_order_int_to_rdkit(e[2]))

    mol = edit_mol.GetMol()
    return mol  # reset_mol_charge(mol) # mol  #


def tables_to_nx(tables):
    """
    Converts atom and bond information into a NetworkX graph object.

    This function takes a tuple containing atom and bond information, constructs a NetworkX graph object,
    adds nodes and edges to it, and assigns attributes to them.

    Parameters
    ----------
    tables : tuple
        A tuple containing two lists:
        - atoms_info (list): A list of tuples where each tuple contains an atom index and atom type.
        - bonds_info (list): A list of tuples where each tuple contains two atom indices and a bond type.

    Returns
    -------
    nx.Graph
        A NetworkX graph object with the specified nodes and edges.
    """
    atoms_info, bonds_info = tables
    graph = nx.Graph()

    # Add nodes with atom type attributes
    for i, v in enumerate(atoms_info):
        graph.add_node(i, color=v[1])

    # Add edges with bond type attributes
    for e in bonds_info:
        graph.add_edge(e[0], e[1], color=int(e[2]))

    return canonicalize_node_labels(graph)


class AssemblyConstruction:
    def __init__(self, data, if_string=False, vo_type="graph", input_graph=None):
        """
        Initialize the AssemblyConstruction object with pathway data.

        Parameters
        ----------
        data : dict
            The pathway data from assemblycpp containing graph information,
            remnants, duplicates, and other assembly metadata.
        if_string : bool, optional
            Whether to sort combined pieces during construction, by default False.
        vo_type : str, optional
            Type of virtual object representation to use ("graph", "mol", 
            "smiles", or "inchi"), by default "graph".
        input_graph : nx.Graph, optional
            Input graph to read edge colors from when AssemblyCpp fails to 
            output edge colors beyond index 5, by default None.

        Notes
        -----
        This constructor initializes all internal data structures needed for 
        pathway construction, including vertices, edges, vertex/edge labels,
        remnant edges, duplicates, and equivalences.
        """
        self.v = data["file_graph"][0]['Vertices']
        self.e = data["file_graph"][0]['Edges']
        self.v_l = data["file_graph"][0]['VertexColours']
        if input_graph is None:
            self.e_l = data["file_graph"][0]['EdgeColours']
        else:  # AssemblyCpp fails to print edge colors beyond index 5 to pathway json. This is a workaround to read the colors from the original input graph.
            self.e_l = [input_graph[u][v]['color'] for u, v in
                        self.e]  # CAUTION: This assumes assemblycpp does not permute vertex labels.
        self.remnant_e = data["remnant"][0]["Edges"] + data["removed_edges"]
        self.duplicates = [[dup["Right"], dup['Left']] for dup in data["duplicates"]]
        self.equivalences = [[1, 1]]

        self.remnant_e, self.duplicates, self.equivalences = fix_repeated_equiv(
            self.remnant_e, self.duplicates, self.equivalences, self.e
        )
        self.if_string = if_string
        self.vo_type = vo_type

        # Construct the atoms list
        self.atoms = []
        self.full_atoms_list = []
        self.atoms_list = []
        self.atoms_list_index = []

        for i, bond in enumerate(self.e):
            atom_list = [[self.v_l[bond[0]], self.v_l[bond[1]]], self.e_l[i]]
            atom_list_index = [bond[0], bond[1]]
            atom_set = [{self.v_l[bond[0]], self.v_l[bond[1]]}, self.e_l[i]]
            if atom_set not in self.atoms:
                self.atoms.append(atom_set)
                self.atoms_list.append(atom_list)
                self.atoms_list_index.append(atom_list_index)
            self.full_atoms_list.append(atom_list)

    def consistent_join(self, pieces_mod, steps_mod, repeated_mo1_cp, step, digraph, indexes):
        """
        Attempt to merge overlapping pathway fragments into a consistent transformation step.

        This method scans the current list of disjoint molecular fragments (`pieces_mod`) 
        and looks for overlapping components (shared atoms or edges). When such overlaps 
        are detected, the method merges the fragments, appends the result as a new step, 
        and updates the digraph to reflect the transformation lineage.

        Parameters
        ----------
        pieces_mod : list
            Current disjoint graph fragments to be scanned for merging.
        steps_mod : list
            List of previously constructed transformation steps.
        repeated_mo1_cp : list
            Copy of repeated motifs used to resolve digraph parentage.
        step : int
            Current index in the step construction sequence.
        digraph : list
            Directed graph tracking transformation steps and dependencies.
        indexes : list
            Step indices associated with repeated motifs.

        Returns
        -------
        tuple
            A tuple containing:
            - pieces_mod : list
                Updated fragment list after possible merging.
            - steps_mod : list
                Updated list of transformation steps.
            - step : int
                Updated step count.
            - digraph : list
                Updated digraph with new transformation links.
        """
        left_sort = [rep[0] for rep in repeated_mo1_cp]
        right_sort = [rep[1] for rep in repeated_mo1_cp]

        def add_digraph_entry(piece, step):
            """
            Append an edge to the digraph indicating a dependency between a prior step or 
            virtual object (`piece`) and the current step.

            This helper function attempts to identify the origin of `piece` by checking:
                - If it's a known left-side repeated motif (`left_sort`)
                - If it's a right-side repeated motif (`right_sort`)
                - If it's already part of a previously constructed step (`steps_mod`)
    
            If no match is found, an "_error" label is used as a fallback source in the digraph.

            Parameters
            ----------
            piece : list
                A graph fragment or reaction step being traced as a source node.
            step : int
                The current step index being constructed as the target node.

            Returns
            -------
            None
                Modifies the `digraph` list in-place by appending a [source, target] entry.
            """
            if piece in left_sort:
                digraph.append(["step_{}".format(indexes[left_sort.index(piece)]), "step_{}".format(step)])
            elif piece in right_sort:
                digraph.append(["step_{}".format(indexes[right_sort.index(piece)]), "step_{}".format(step)])
            elif piece in steps_mod:
                digraph.append(["step_{}".format(steps_mod.index(piece) + 1), "step_{}".format(step)])
            else:
                digraph.append(["step_{}".format("_error"), "step_{}".format(step)])

        for pic in pieces_mod:
            for pic_i in pieces_mod:
                if pic == pic_i:
                    continue

                if any(ed in np.reshape(pic, -1) for ed in np.reshape(pic_i, -1)):
                    step += 1
                    combined = np.sort(pic + pic_i, axis=0).tolist() if self.if_string else pic + pic_i
                    steps_mod.append(combined)

                    if len(pic) > 1:
                        add_digraph_entry(pic, step)
                    else:
                        v_object1 = self.atoms.index(
                            [{self.v_l[pic[0][0]], self.v_l[pic[0][1]]}, self.e_l[self.e.index(pic[0])]])
                        digraph.append(["virtual_object_{}".format(v_object1), "step_{}".format(step)])

                    if len(pic_i) > 1:
                        add_digraph_entry(pic_i, step)
                    else:
                        v_object1 = self.atoms.index(
                            [{self.v_l[pic_i[0][0]], self.v_l[pic_i[0][1]]}, self.e_l[self.e.index(pic_i[0])]])
                        digraph.append(["virtual_object_{}".format(v_object1), "step_{}".format(step)])

                    pieces_mod.remove(pic)
                    pieces_mod.remove(pic_i)
                    pieces_mod.insert(0, combined)
                    return pieces_mod, steps_mod, step, digraph

        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(self, pieces_mod, steps_mod, sorted_repeated_mod1, step, digraph):
        """
        Construct the initial pathway by integrating repeated molecular fragments.

        This method iterates over repeated motifs (subgraphs) and attempts to either:
            - Append them directly to the current set of pathway pieces if no conflicts exist,
            - Or integrate them by merging overlapping pieces through consistent joining logic.

        The method also tracks which fragments have been added and associates them with 
        their corresponding indices for downstream pathway construction.

        Parameters
        ----------
        pieces_mod : list
            Current list of disjoint pathway fragments (edge groups).
        steps_mod : list
            List of transformation steps constructed so far.
        sorted_repeated_mod1 : list
            List of repeated motifs sorted by subgraph size.
        step : int
            Current step index in the construction process.
        digraph : list
            Current digraph representing the reaction assembly.

        Returns
        -------
        tuple
            A tuple containing:
            - pieces_mod : list
                Updated list of pathway fragments after integration.
            - steps_mod : list
                Updated list of pathway steps.
            - sorted_repeated_mod1_cp : list
                Deep copy of the original motif list for reference.
            - step : int
                Updated step counter.
            - digraph : list
                Updated digraph structure.
            - indexes : list
                Index mapping of added fragments to step references.
        """
        step_ind = [1] * len(sorted_repeated_mod1)
        indexes = [0] * len(sorted_repeated_mod1)
        sorted_repeated_mod1_cp = copy.deepcopy(sorted_repeated_mod1)
        left_sort = [rep[0] for rep in sorted_repeated_mod1_cp]

        while sorted_repeated_mod1:
            for j, repeat in enumerate(sorted_repeated_mod1_cp):
                if not step_ind[j] or repeated_sizes(sorted_repeated_mod1)[0] != len(repeat[0]):
                    continue
                if check_edge_in_list(repeat[1], pieces_mod) or check_edge_in_list(repeat[1], steps_mod):
                    pieces_mod.append(repeat[0])
                    sorted_repeated_mod1.remove(repeat)
                    step_ind[j] = 0
                    indexes[j] = index_set(steps_mod, repeat[1]) or indexes[index_set(left_sort, repeat[1]) - 1]
                else:
                    indices = [i for i, piece in enumerate(pieces_mod) if any(rep in piece for rep in repeat[1])]

                    if not indices:
                        continue

                    combined_pieces = [pieces_mod[i] for i in indices]
                    for idx in sorted(indices, reverse=True):
                        pieces_mod.pop(idx)

                    while len(combined_pieces) > 1:
                        combined_pieces, steps_mod, step, digraph = self.consistent_join(
                            combined_pieces, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes
                        )

                    pieces_mod.append(combined_pieces[0])
                    pieces_mod.append(repeat[0])
                    step_ind[j] = 0
                    indexes[j] = index_set(steps_mod, repeat[1])
                    sorted_repeated_mod1.remove(repeat)

        return pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes

    def generate_pathway(self):
        """
        Construct the reaction pathway by combining initial fragments and resolving overlaps.

        This method builds the overall pathway graph by:
            - Initializing edges as disjoint fragments (`pieces`)
            - Applying equivalence mappings to handle duplicate structures
            - Sorting and incorporating repeated motifs using a construction algorithm
            - Iteratively merging consistent fragments until convergence

        Intermediate steps and the resulting pathway are stored as attributes 
        on the instance for downstream use (e.g., visualization, logging, or graph generation).

        Parameters
        ----------
        None

        Returns
        -------
        None
            The results are stored in the following instance attributes:
            - self.steps : list
                Sequence of transformation steps forming the pathway.
            - self.digraph : list
                Final digraph structure representing stepwise assembly.
            - self.pieces_mod : list
                Remaining or modified substructures after construction.
        """
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

        while True:
            pieces_mod_cp = copy.deepcopy(pieces_mod)
            pieces_mod, steps_mod, step, digraph = self.consistent_join(
                pieces_mod, steps_mod, sorted_repeated_mod1_cp, step, digraph, indexes
            )
            if len(pieces_mod) == len(pieces_mod_cp):
                break

        self.steps = steps_mod
        self.digraph = digraph
        self.pieces_mod = pieces_mod

        return None

    def generate_vo(self):
        """
        Generate virtual objects (VOs) and transformation steps based on the specified VO type.

        This method processes the list of atoms and steps from a pathway to create their 
        corresponding molecular representations in one of several formats: NetworkX graph, 
        RDKit molecule, SMILES, or InChI. The generated data is stored as attributes on 
        the class instance for further use in graph construction or analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
            All generated data is stored in the instance attributes:
            - self.molecules_vo : list
                List of individual molecule representations per atom.
            - self.molecules_steps : list
                List of molecule representations per transformation step.
            - self.steps_indx_s : list
                Indexed and encoded transformation step data.
            - self.vs_atoms : list
                Vertex (atom) labels for each step.
    
        Raises
        ------
        ValueError
            If `self.vo_type` is not one of 'graph', 'mol', 'smiles', or 'inchi'.
        """
        # Generate the virtual objects
        molecules_vo = []
        for atom in self.atoms_list:
            if self.vo_type == "graph":
                mol = tables_to_nx(([(0, atom[0][0]), (1, atom[0][1])], [(0, 1, bond_order_assout_to_int(atom[1]))]))
            elif self.vo_type == "mol":
                mol = tables_to_mol(([(0, atom[0][0]), (1, atom[0][1])], [(0, 1, bond_order_assout_to_int(atom[1]))]))
            elif self.vo_type == "smiles":
                mol = tables_to_mol(([(0, atom[0][0]), (1, atom[0][1])], [(0, 1, bond_order_assout_to_int(atom[1]))]))
                mol = Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True)
                mol = smi_remove_implicit_hydrogen(mol)
            elif self.vo_type == "inchi":
                mol = tables_to_mol(([(0, atom[0][0]), (1, atom[0][1])], [(0, 1, bond_order_assout_to_int(atom[1]))]))
                mol = Chem.MolToInchi(mol)
            else:
                raise ValueError("Invalid vo_type. Choose from 'graph', 'mol', 'smiles', or 'inchi'.")

            molecules_vo.append(mol)

        # Generate the steps
        steps_index_s = []
        vs_atoms = []
        for step in self.steps:
            indices = list(set(np.reshape(step, -1)))
            steps_index_s.append(
                [[indices.index(edge[0]), indices.index(edge[1]), self.e_l[self.e.index(edge)]] for edge in step])
            vs_atoms.append([self.v_l[at] for at in indices])

        # Generate the molecules for each step
        molecules_steps = []
        for i, step in enumerate(steps_index_s):
            if self.vo_type == "graph":
                mol = tables_to_nx(([(i, at) for at in vs_atoms[i]],
                                    [(edge[0], edge[1], bond_order_assout_to_int(edge[2])) for edge in step]))
            elif self.vo_type == "mol":
                mol = tables_to_mol(([(i, at) for at in vs_atoms[i]],
                                     [(edge[0], edge[1], bond_order_assout_to_int(edge[2])) for edge in step]))
                mol = Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True)
                mol = smi_remove_implicit_hydrogen(mol)
            elif self.vo_type == "smiles":
                mol = tables_to_mol(([(i, at) for at in vs_atoms[i]],
                                     [(edge[0], edge[1], bond_order_assout_to_int(edge[2])) for edge in step]))
                mol = Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True)
                mol = smi_remove_implicit_hydrogen(mol)
            elif self.vo_type == "inchi":
                mol = tables_to_mol(([(i, at) for at in vs_atoms[i]],
                                     [(edge[0], edge[1], bond_order_assout_to_int(edge[2])) for edge in step]))
                mol = Chem.MolToInchi(mol)
            else:
                raise ValueError("Invalid vo_type. Choose from 'graph', 'mol', 'smiles', or 'inchi'.")
            molecules_steps.append(mol)

        self.molecules_vo = molecules_vo
        self.molecules_steps = molecules_steps
        self.steps_indx_s = steps_index_s
        self.vs_atoms = vs_atoms

        return None

    def get_assembly_digraph(self):
        """
        Creates a directed graph representation of the assembly pathway.

        Each node is connected according to self.digraph and contains attributes:
        - type: 'virtual_object' or 'step'
        - vo: The corresponding molecule representation from molecules_vo or molecules_steps
        - label: String representation for visualization

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            A tuple containing:
            - graph : nx.DiGraph
                A directed graph representing the assembly pathway.
            - unique_molecules : list
                List of unique virtual objects from the pathway.
        
        Raises
        ------
        ValueError
            If `self.vo_type` is not one of 'graph', 'mol', 'smiles', or 'inchi'.
        """
        self.generate_pathway()
        self.generate_vo()

        graph = nx.DiGraph()

        # Add all nodes with their corresponding molecule information
        for edge in self.digraph:
            source, target = edge

            # Handle virtual object nodes
            if source.startswith("virtual_object_"):
                vo_index = int(source.split("_")[-1])
                if vo_index < len(self.molecules_vo):
                    graph.add_node(source,
                                   type="virtual_object",
                                   vo=self.molecules_vo[vo_index])

            # Handle step nodes
            if source.startswith("step_"):
                if source.split("_")[-1].isdigit():
                    step_index = int(source.split("_")[-1]) - 1
                    if 0 <= step_index < len(self.molecules_steps):
                        graph.add_node(source,
                                       type="step",
                                       vo=self.molecules_steps[step_index])

            # Do the same for target nodes
            if target.startswith("virtual_object_"):
                vo_index = int(target.split("_")[-1])
                if vo_index < len(self.molecules_vo):
                    graph.add_node(target,
                                   type="virtual_object",
                                   vo=self.molecules_vo[vo_index])

            if target.startswith("step_"):
                if target.split("_")[-1].isdigit():
                    step_index = int(target.split("_")[-1]) - 1
                    if 0 <= step_index < len(self.molecules_steps):
                        graph.add_node(target,
                                       type="step",
                                       vo=self.molecules_steps[step_index])

        # Add all edges from digraph
        graph.add_edges_from(self.digraph)

        # Add the label attribute to the nodes
        for node in graph.nodes(data=True):
            if self.vo_type == "graph":
                graph.nodes[node[0]]["label"] = node[0]
            elif self.vo_type == "mol":
                mol = node[1]["vo"]
                mol = Chem.MolToSmiles(mol, allHsExplicit=True, isomericSmiles=True)
                graph.nodes[node[0]]["label"] = smi_remove_implicit_hydrogen(mol)

            elif self.vo_type == "smiles":
                graph.nodes[node[0]]["label"] = node[1]["vo"]
            elif self.vo_type == "inchi":
                graph.nodes[node[0]]["label"] = node[1]["vo"]
            else:
                raise ValueError("Invalid vo_type. Choose from 'graph', 'mol', 'smiles', or 'inchi'.")

        # Combine molecules_vo and molecules_steps into a single list and find the set of unique elements
        all_molecules = self.molecules_vo + self.molecules_steps
        unique_molecules = set(all_molecules)

        return graph, list(unique_molecules)

    def pathway_log_string(self) -> str:
        """
        Generate a formatted string summarizing the pathway construction.

        This method builds a multi-section string that summarizes the graph 
        structure, atoms involved, transformation steps, and the digraph 
        representation associated with the pathway.

        Sections included:
            - Graph metadata (`self.v`, `self.e`, `self.v_l`, `self.e_l`)
            - List of atoms (`self.atoms_list`)
            - Step descriptions (`self.steps`)
            - Digraph structure (`self.digraph`)

        Parameters
        ----------
        None

        Returns
        -------
        str
            A multi-line formatted string representing the internal state 
            of the pathway and its graph structure.
        """
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


def parse_pathway_file(file, vo_type="smiles", debug=False, log=False, input_graph=None):
    """
    Parse a pathway JSON file and construct an assembly graph.

    This function loads a pathway JSON file, constructs the corresponding
    assembly directed graph using the specified virtual object (VO) type,
    and optionally prints debug information or returns a pathway log.

    Parameters
    ----------
    file : str
        Path to the JSON pathway file.
    vo_type : str, optional
        Type of virtual object representation to use (e.g., "smiles", "graph", 
        "mol", "inchi"), by default "smiles".
    debug : bool, optional
        If True, prints debug information about each node, by default False.
    log : bool, optional
        If True, returns an additional string describing the pathway log, 
        by default False.
    input_graph : nx.Graph, optional
        Input graph to read edge colors from. If None, edge colors are read 
        from the pathway file, by default None.

    Returns
    -------
    tuple
        If log is False:
            (graph, vo_list) where:
            - graph : nx.DiGraph
                The constructed assembly directed graph.
            - vo_list : list
                List of virtual objects used in the graph.
        
        If log is True:
            (graph, vo_list, log_string) where:
            - log_string : str
                A summary log string of the pathway steps.
    """
    # # Load the pathway file
    with open(file) as f:
        data = json.load(f)

    # Make the construction object
    if input_graph == None:
        construction_object = AssemblyConstruction(data, vo_type=vo_type)
    else:  # If the input is a general graph, we need to pass the input graph to read the edge colors (AssemblyCpp drops color output after 5; bug is only in output.)
        construction_object = AssemblyConstruction(data, vo_type=vo_type, input_graph=input_graph)
    graph, vo_list = construction_object.get_assembly_digraph()

    if debug:
        # Loop over the nodes and print the type and smiles
        for node in graph.nodes(data=True):
            print(f"Node: {node[0]}, Type: {node[1]['type']}, VO: {node[1]['vo']}", flush=True)
    if log:
        log_string = construction_object.pathway_log_string()
        return graph, vo_list, log_string
    else:
        return graph, vo_list


def get_level(G: nx.DiGraph, node: str) -> int | None:
    """
    Returns the level of a node in a graph.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges represent assembly steps.
    node : str
        The node for which to determine the assembly depth.
    Returns
    -------
    int
        The assembly depth of the node, or None if the depth cannot be determined.
    Raises
    ------
    ValueError
        If the node is not present in the graph.
    """
    preds = [
        edge[0] for edge in list(G.in_edges(node))
    ]
    if len(preds) > 0:
        return max([G.nodes[pred]["level"] for pred in preds]) + 1
    elif len(preds) == 0:
        return 0
    no_pred = [pred for pred in preds if "level" not in G.nodes[pred]]
    try:
        return get_level(G, no_pred[0])
    except Exception:
        return None


def assign_levels(G: nx.DiGraph, inplace: bool = True) -> None | nx.DiGraph:
    """
    Assigns assembly depth to nodes in a graph.
    For consistency, assembly depth is referred to as "level" in this context.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges represent assembly steps.
    inplace : bool, optional
        If True, modifies the graph in place. If False, returns a modified copy, 
        by default True.

    Returns
    -------
    None or nx.DiGraph
        If inplace is True, modifies the graph in place and returns None.
        If inplace is False, returns a new graph with updated node attributes.
    
    Raises
    ------
    TypeError
        If the input graph G is not a directed graph (DiGraph).
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph G must be a directed graph (DiGraph).")
    if not inplace:
        G = G.copy()

    for node in G.nodes:
        if not list(G.predecessors(node)):
            G.nodes[node].update({"level": 0})
        else:
            _ = get_level(G, node)
            G.nodes[node].update({"level": get_level(G, node)})

    if not inplace:
        return G
    return None


def immediate_predecessors(data, interval):
    """
    Extract immediate predecessors in the pathway for a given interval.
    
    For example, if we have the abracadabra data, and we want the whole interval,
    then it will return ["abra", "c", "a", "d", "abra"].
    
    Parameters
    ----------
    data : dict
        The pathway data from assemblycpp (JSON format).
    interval : tuple
        A tuple of the form (start, length) indicating the interval.
    
    Returns
    -------
    list
        A list of strings representing the immediate predecessors in the pathway.
    """
    output = []

    c_idx = interval[0]
    while c_idx < sum(interval):
        parent = ""
        for dup in data["duplicates"]:
            if dup["Left"][1] < interval[1]:  # Make sure the duplicate can fit in the interval
                if c_idx in range(dup["Left"][0], sum(dup["Left"])):  # If duplicate contains c_idx
                    if dup["Left"][1] > len(parent):  # If this duplicate is larger than the current parent
                        if dup["Left"][0] >= interval[0] and sum(dup["Left"]) <= sum(
                                interval):  # If this duplicate is within the interval
                            parent = data["file_graph"][0]["Fragments"][0][dup["Left"][0]:sum(dup["Left"])]
                elif c_idx in range(dup["Right"][0], sum(dup["Right"])):  # now check the right copy
                    if dup["Right"][1] > len(parent):
                        if dup["Right"][0] >= interval[0] and sum(dup["Right"]) <= sum(interval):
                            parent = data["file_graph"][0]["Fragments"][0][dup["Right"][0]:sum(dup["Right"])]
        if parent == "":
            output.append(data["file_graph"][0]["Fragments"][0][c_idx])
            c_idx += 1
        else:
            output.append(parent)
            c_idx += len(parent)

    return output


def build_str(interval, data, path):
    """
    Builds the string from the pathway data and adds it to the path.
    
    Parameters
    ----------
    interval : tuple
        A tuple of the form (start, end) indicating the interval to build.
    data : dict
        The pathway data from assemblycpp (JSON format).
    path : nx.DiGraph
        The current pathway graph.
    
    Returns
    -------
    nx.DiGraph
        Updated pathway with the string added.
    """

    ledger = immediate_predecessors(data, interval)
    c_idx = interval[0]
    for sub_str in ledger:
        if sub_str not in path.nodes:
            path = build_str([c_idx, c_idx + len(sub_str)], data,
                             path)  # Recursively build the duplicate strings if not already in the path
        c_idx += len(sub_str)

    str_in_progress = ledger[0]
    for idx in range(1, len(ledger)):  # Builds string from left to right
        str_in_progress_new = str_in_progress + ledger[idx]
        if str_in_progress_new not in path.nodes:  # Only relevant if the path is not minimum
            path.add_node(str_in_progress_new)
        if (str_in_progress, str_in_progress_new) not in path.edges:  # Only relevant if the path is not minimum
            path.add_edge(str_in_progress, str_in_progress_new)
        if (ledger[idx], str_in_progress_new) not in path.edges:  # Only relevant if the path is not minimum
            path.add_edge(ledger[idx], str_in_progress_new)
        str_in_progress = str_in_progress_new
    return path


def parse_string_pathway_file(file_path_pathway):
    """
    Parses a pathway file and returns the pathway as a list of virtual objects.
    
    Parameters
    ----------
    file_path_pathway : str
        Path to the pathway file.
    
    Returns
    -------
    tuple
        A tuple containing:
        - VOs : dict
            Dictionary of virtual objects in calculated pathway.
        - path : nx.DiGraph
            NetworkX directed graph representing the pathway.
    
    Raises
    ------
    FileNotFoundError
        If the pathway file is not found at the specified path.
    """
    if not os.path.isfile(file_path_pathway):
        raise FileNotFoundError(f"Pathway file not found: {file_path_pathway}")
    else:
        # Load the pathway file
        with open(file_path_pathway) as f:
            data = json.load(f)

    VOs = dict()
    VOs["file_string"] = data["file_graph"][0]["Fragments"][0]
    VOs["remnant"] = data["remnant"][0]["Fragments"]
    dups = [data["duplicates"][i]["Left"] for i in range(len(data["duplicates"]))]
    VOs["duplicates"] = [VOs["file_string"][dup[0]:sum(dup)] for dup in dups]

    path = nx.DiGraph()

    # We will build the string from left to right, constructing duplicates as needed
    for char in list(set(VOs["file_string"])):
        path.add_node(char)  # Add units

    path = build_str([0, len(VOs["file_string"])], data, path)  # Build the string from the pathway data

    return VOs, path


def molstr_to_str(molstr, edge_color_dict=None):
    """
    Takes a mol string and translates it into the corresponding string.

    Parameters
    ----------
    molstr : nx.Graph
        The molecular graph to translate.
    edge_color_dict : dict, optional
        Dictionary mapping edges to colors. If None, the function assumes 
        the mol string is directed, by default None.

    Returns
    -------
    str
        The translated string.
    """

    out_str = ""
    if edge_color_dict is None:  # Directed
        for n_idx, node in enumerate(molstr.nodes(data=True)):  # Loop over nodes in molstr with odd indices
            # print(f"Node {n_idx}: {node}")
            if n_idx % 2 == 1:
                out_str += node[1]['color']
    else:  # Undirected
        # Prep the edge_color_dict
        edge_color_dict = {v: k for k, v in edge_color_dict.items()}
        if "1" in edge_color_dict.keys():
            edge_color_dict["single"] = edge_color_dict["1"]
        if "2" in edge_color_dict.keys():
            edge_color_dict["double"] = edge_color_dict["2"]
        if "3" in edge_color_dict.keys():
            edge_color_dict["triple"] = edge_color_dict["3"]
        if "4" in edge_color_dict.keys():
            edge_color_dict["quadruple"] = edge_color_dict["4"]
        edge_color_dict["0"] = "!"

        for u, v, data in molstr.edges(data=True):
            out_str += edge_color_dict[str(data.get('color'))]

    return out_str
