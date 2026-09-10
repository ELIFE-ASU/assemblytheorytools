"""
Parsing and construction of assembly pathways.

This module reads the pathway files emitted by the assembly calculators and
turns them into NetworkX directed graphs. It assigns hierarchical levels to
virtual objects, converts between molecule-string and graph representations, and
provides the :class:`AssemblyConstruction` class for generating pathways and
assembly digraphs from a target structure.
"""

import copy
import io
import json
import os
import re
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pydot
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from .tools_graph import (
    bond_order_assout_to_int,
    bond_order_int_to_rdkit,
    canonicalize_node_labels,
    mol_to_nx,
    nx_to_inchi,
    nx_to_mol,
    nx_to_smi,
    set_graph_layer,
)
from .tools_mol import smi_remove_implicit_hydrogen

# The virtual object representations every vo_type argument is checked against
_VO_TYPES = ("graph", "mol", "smiles", "inchi")
# Raised wherever an unrecognised virtual object representation is requested
_VO_TYPE_ERROR = "Invalid vo_type. Choose from 'graph', 'mol', 'smiles', or 'inchi'."


def transform_array(
    target_array: List[List[int]],
    comp_array: List[List[int]],
    source_val: int,
    target_val: int,
    new_val: int,
    pairs_list: List[List[int]],
) -> List[List[int]]:
    """
    Replace matching edge endpoints in the target array in place.

    An endpoint matching ``target_val`` in ``comp_array`` is replaced with
    ``new_val`` when substituting ``source_val`` produces an edge in
    ``pairs_list``. The other endpoint comes from the comparison edge.

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


def repeated_sizes(repeated: List[Tuple[Any, Any]]) -> List[int]:
    """
    Return the sorted unique sizes found in a repeated list.

    The size measured is that of the second element of each tuple.

    Parameters
    ----------
    repeated : list of tuple
        List of tuples, where each tuple contains two elements.

    Returns
    -------
    list
        Sorted list of unique sizes of the second element in each tuple.
    """
    return sorted({len(rep[1]) for rep in repeated})


def equal_list(list_a: List[List[Any]], list_b: List[List[Any]]) -> bool:
    """
    Compare edge lists, ignoring edge direction, order and duplicates.

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


def check_edge_in_list(edges: List[Any], list_in: List[List[Any]]) -> bool:
    """
    Check whether any candidate matches the edges according to ``equal_list``.

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
    return any(equal_list(candidate, edges) for candidate in list_in)


def equivalence(
    remnant_pieces: List[List[Any]], equivalences: List[List[int]]
) -> List[List[Any]]:
    """
    Apply vertex equivalences to a deep copy of the remnant pieces.

    Each pair maps its second vertex to its first. The first mapping for a
    vertex takes precedence, and replacements are applied only once.

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
    vertices = np.array(equivalences)[:, 1].tolist()
    replacements = {}
    for pair, vertex in zip(equivalences, vertices):
        replacements.setdefault(vertex, pair[0])

    for piece in pieces_copy:
        for edge in piece:
            for endpoint in (0, 1):
                if edge[endpoint] in replacements:
                    edge[endpoint] = replacements[edge[endpoint]]

    return pieces_copy


def fix_repeated_equiv(
    edge_list: List[Any],
    repeated_equiv: List[Any],
    equivalences: List[List[int]],
    edge_pairs: List[List[int]],
) -> Tuple[List[Any], List[Any], List[List[int]]]:
    """
    Resolve repeated vertex mappings, updating edge lists in place.

    Ambiguous targets receive a fresh vertex or reuse an existing equivalent
    vertex. ``edge_pairs`` determines which edge endpoints are transformed.

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
    edge_list : list
        The updated edge list.
    repeated_equiv : list
        The updated repeated equivalences.
    equivalences : list
        The updated equivalences.
    """
    equivalences = np.unique(equivalences, axis=0).tolist()
    if not equivalences:
        return edge_list, repeated_equiv, equivalences

    equiv_np = np.array(equivalences)
    # Keep NumPy's tie ordering: it determines which equivalent vertex is split.
    sorted_eq = equiv_np[equiv_np[:, 0].argsort()]
    pairs = sorted_eq.tolist()
    source_counts = Counter(pair[0] for pair in pairs)
    target_counts = Counter(pair[1] for pair in pairs)
    repeated_targets = [pair for pair in pairs if target_counts[pair[1]] > 1]
    repeated_sources = [pair for pair in pairs if source_counts[pair[0]] > 1]
    if not repeated_targets:
        return edge_list, repeated_equiv, equivalences

    repeated_indices = [i for i, pair in enumerate(pairs) if target_counts[pair[1]] > 1]
    remaining = np.delete(sorted_eq, repeated_indices, axis=0)
    repeated_mod = [equivalence(rep, remaining) for rep in repeated_equiv]
    transformed_edges = equivalence([edge_list], remaining)[0]

    def replace_vertex(source: int, target: int, replacement: int) -> None:
        transform_array(
            edge_list, transformed_edges, source, target, replacement, edge_pairs
        )
        for repeats, transformed_repeats in zip(repeated_equiv, repeated_mod):
            for piece, transformed_piece in zip(repeats, transformed_repeats):
                transform_array(
                    piece, transformed_piece, source, target, replacement, edge_pairs
                )

    shared_vertices = sorted(
        {vertex for pair in repeated_targets for vertex in pair}
        & {vertex for pair in repeated_sources for vertex in pair}
    )
    if not repeated_sources or not shared_vertices:
        sorted_repeats = np.array(repeated_targets)
        sorted_repeats = sorted_repeats[sorted_repeats[:, 1].argsort()]
        largest_target = equiv_np[:, 1].max()
        additional = []
        for i in range(len(sorted_repeats) // 2):
            source, target = sorted_repeats[2 * i]
            replacement = largest_target + 1 + i
            additional.extend(
                [[source, replacement], sorted_repeats[2 * i + 1].tolist()]
            )
            replace_vertex(source, target, replacement)
        equivalences = remaining.tolist() + additional
    else:
        source, replacement = next(
            pair
            for pair in reversed(repeated_sources)
            if pair[0] == shared_vertices[0] and pair[1] != shared_vertices[1]
        )
        remove_index = repeated_targets.index(shared_vertices)
        equivalences = (
            remaining.tolist()
            + repeated_targets[:remove_index]
            + repeated_targets[remove_index + 1 :]
        )
        replace_vertex(source, shared_vertices[1], replacement)

        if any(
            count > 1 for count in Counter(pair[1] for pair in equivalences).values()
        ):
            return fix_repeated_equiv(
                edge_list, repeated_equiv, equivalences, edge_pairs
            )

    return edge_list, repeated_equiv, equivalences


def index_set(lists: List[List[Any]], list_in: List[Any]) -> Optional[int]:
    """
    Return the first matching list's 1-based index, or None.

    Row order and duplicate rows are ignored; values within each row retain
    their order.

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
    list_in_set = {tuple(row) for row in list_in}
    for i, i_list in enumerate(lists):
        if {tuple(row) for row in i_list} == list_in_set:
            return i + 1
    return None


def select_length(dict_array: Dict[str, Any]) -> Union[int, float]:
    """
    Return the dictionary's ``'len'`` entry for use as a sorting key.

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


def tables_to_mol(
    tables: Tuple[List[Tuple[int, str]], List[Tuple[int, int, int]]],
) -> Chem.Mol:
    """
    Build an RDKit molecule from atom and bond tables.

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
    molecule = RWMol()

    for atom in atoms_info:
        molecule.AddAtom(Chem.Atom(atom[1]))

    for bond in bonds_info:
        molecule.AddBond(bond[0], bond[1], bond_order_int_to_rdkit(bond[2]))

    return molecule.GetMol()


def tables_to_nx(
    tables: Tuple[List[Tuple[int, str]], List[Tuple[int, int, int]]],
) -> nx.Graph:
    """
    Build a canonical NetworkX graph from atom and bond tables.

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
    graph.add_nodes_from((i, {"color": atom[1]}) for i, atom in enumerate(atoms_info))
    graph.add_edges_from(
        (bond[0], bond[1], {"color": int(bond[2])}) for bond in bonds_info
    )
    return canonicalize_node_labels(graph)


def _tables_to_vo(
    tables: Tuple[List[Tuple[int, str]], List[Tuple[int, int, int]]], vo_type: str
) -> Any:
    """Render atom and bond tables as a graph, molecule, SMILES or InChI."""
    if vo_type == "graph":
        return tables_to_nx(tables)
    if vo_type not in _VO_TYPES:
        raise ValueError(_VO_TYPE_ERROR)

    molecule = tables_to_mol(tables)
    if vo_type == "mol":
        return molecule
    if vo_type == "smiles":
        smiles = Chem.MolToSmiles(molecule, allHsExplicit=True, isomericSmiles=True)
        return smi_remove_implicit_hydrogen(smiles)
    return Chem.MolToInchi(molecule)


class AssemblyConstruction:
    """
    Construction of assembly pathways and digraphs from pathway data.

    Wraps the pathway data emitted by the assembly calculator and rebuilds
    the corresponding assembly pathway: it generates the virtual objects,
    joins them consistently, and assembles the result into a NetworkX
    directed graph suitable for plotting and further analysis.

    Attributes
    ----------
    v : list
        Vertices of the target graph.
    e : list of list of int
        Edges of the target graph, as pairs of vertex indices.
    v_l : list
        Vertex colours (atom types) of the target graph.
    e_l : list
        Edge colours (bond orders) of the target graph.
    remnant_e : list
        Remnant edges, i.e. those left over once the duplicated subgraphs
        have been accounted for.
    duplicates : list of list
        Duplicated subgraph pairs reported by the calculator, as ``[right,
        left]`` edge lists.
    equivalences : list of list
        Equivalence mappings between duplicated subgraphs.
    if_string : bool
        Whether combined pieces are sorted during construction.
    vo_type : str
        Virtual object representation used for the output.
    atoms : list
        Unique ``[{atom types}, bond order]`` records, one per distinct
        bond.
    atoms_list : list
        The same bonds as ``atoms``, but with the atom types kept ordered.
    atoms_list_index : list
        Vertex index pairs matching the entries of ``atoms_list``.
    full_atoms_list : list
        One ``[[atom types], bond order]`` record per edge, without
        deduplication.
    steps : list
        Pathway steps, populated by :meth:`generate_pathway`.
    digraph : list of list of str
        Edge list of the assembly digraph, populated by
        :meth:`generate_pathway`.
    pieces_mod : list
        Remaining pathway fragments, populated by :meth:`generate_pathway`.
    molecules_vo : list
        Virtual objects of the pathway, populated by :meth:`generate_vo`.
    molecules_steps : list
        Steps associated with each virtual object, populated by
        :meth:`generate_vo`.
    steps_indx_s : list
        Step indices of the virtual objects, populated by
        :meth:`generate_vo`.
    vs_atoms : list
        Atom records of the virtual objects, populated by
        :meth:`generate_vo`.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        if_string: bool = False,
        vo_type: str = "graph",
        input_graph: Optional[nx.Graph] = None,
    ) -> None:
        """Initialise the target graph, repeated fragments and bond records.

        Parameters
        ----------
        data : dict
            Pathway data from AssemblyCpp, including graph information,
            remnants and duplicates.
        if_string : bool, optional
            Whether to sort combined pieces during construction. Default is
            False.
        vo_type : str, optional
            Virtual object representation: "graph", "mol", "smiles" or
            "inchi". Default is "graph".
        input_graph : nx.Graph, optional
            Original target graph, used to recover edge colours omitted by
            AssemblyCpp beyond index 5. Default is None.
        """
        graph_data = data["file_graph"][0]
        self.v = graph_data["Vertices"]
        self.e = graph_data["Edges"]
        self.v_l = graph_data["VertexColours"]
        # Recover omitted colours from the input, assuming AssemblyCpp has
        # preserved its vertex labels.
        self.e_l = (
            graph_data["EdgeColours"]
            if input_graph is None
            else [input_graph[u][v]["color"] for u, v in self.e]
        )
        self.remnant_e = data["remnant"][0]["Edges"] + data["removed_edges"]
        self.duplicates = [[dup["Right"], dup["Left"]] for dup in data["duplicates"]]
        self.equivalences = [[1, 1]]
        self.remnant_e, self.duplicates, self.equivalences = fix_repeated_equiv(
            self.remnant_e, self.duplicates, self.equivalences, self.e
        )
        self.if_string = if_string
        self.vo_type = vo_type

        self.atoms = []
        self.full_atoms_list = []
        self.atoms_list = []
        self.atoms_list_index = []
        for i, (u, v) in enumerate(self.e):
            atom_types = [self.v_l[u], self.v_l[v]]
            atom = [atom_types, self.e_l[i]]
            bond_type = [set(atom_types), self.e_l[i]]
            if bond_type not in self.atoms:
                self.atoms.append(bond_type)
                self.atoms_list.append(atom)
                self.atoms_list_index.append([u, v])
            self.full_atoms_list.append(atom)

    def _virtual_object_index(self, edge: List[int]) -> int:
        """Return the index of the bond type represented by *edge*."""
        atom_types = {self.v_l[edge[0]], self.v_l[edge[1]]}
        bond_order = self.e_l[self.e.index(edge)]
        return self.atoms.index([atom_types, bond_order])

    def consistent_join(
        self,
        pieces_mod: List[List[Any]],
        steps_mod: List[List[Any]],
        repeated_mo1_cp: List[Any],
        step: int,
        digraph: List[List[str]],
        indexes: List[int],
    ) -> Tuple[List[List[Any]], List[List[Any]], int, List[List[str]]]:
        """Join the first pair of fragments sharing a vertex and record its sources.

        Parameters
        ----------
        pieces_mod : list
            Current graph fragments, updated in place when a pair is joined.
        steps_mod : list
            Constructed steps, extended in place with the joined fragment.
        repeated_mo1_cp : list
            Repeated motif pairs used to resolve each fragment's source.
        step : int
            Current step count.
        digraph : list
            Dependency edges, extended in place for the new step.
        indexes : list
            Step indices associated with repeated motifs.

        Returns
        -------
        tuple
            ``(pieces_mod, steps_mod, step, digraph)`` after the first join,
            or unchanged if no fragments share a vertex.
        """
        left_motifs = [repeat[0] for repeat in repeated_mo1_cp]
        right_motifs = [repeat[1] for repeat in repeated_mo1_cp]

        def source_name(piece: List[Any]) -> str:
            """Resolve a fragment to a virtual object or an earlier step."""
            if len(piece) <= 1:
                return f"virtual_object_{self._virtual_object_index(piece[0])}"
            if piece in left_motifs:
                return f"step_{indexes[left_motifs.index(piece)]}"
            if piece in right_motifs:
                return f"step_{indexes[right_motifs.index(piece)]}"
            if piece in steps_mod:
                return f"step_{steps_mod.index(piece) + 1}"
            return "step__error"

        for left in pieces_mod:
            vertices = {vertex for edge in left for vertex in edge}
            for right in pieces_mod:
                if left == right or not any(
                    vertex in vertices for edge in right for vertex in edge
                ):
                    continue

                step += 1
                combined = (
                    np.sort(left + right, axis=0).tolist()
                    if self.if_string
                    else left + right
                )
                steps_mod.append(combined)
                for piece in (left, right):
                    digraph.append([source_name(piece), f"step_{step}"])

                pieces_mod.remove(left)
                pieces_mod.remove(right)
                pieces_mod.insert(0, combined)
                return pieces_mod, steps_mod, step, digraph

        return pieces_mod, steps_mod, step, digraph

    def repeated_construction(
        self,
        pieces_mod: List[List[Any]],
        steps_mod: List[List[Any]],
        sorted_repeated_mod1: List[Any],
        step: int,
        digraph: List[List[str]],
    ) -> Tuple[
        List[List[Any]], List[List[Any]], List[Any], int, List[List[str]], List[int]
    ]:
        """Build repeated fragments in size order and record their step indices.

        Parameters
        ----------
        pieces_mod : list
            Current pathway fragments, updated in place.
        steps_mod : list
            Constructed steps, extended in place as fragments are joined.
        sorted_repeated_mod1 : list
            Repeated motif pairs sorted by size, consumed in place.
        step : int
            Current step count.
        digraph : list
            Dependency edges, extended in place as fragments are joined.

        Returns
        -------
        tuple
            Updated fragments, steps, a deep copy of the original motif list,
            step count, dependency edges and motif step indices.
        """
        pending = [True] * len(sorted_repeated_mod1)
        indexes = [0] * len(sorted_repeated_mod1)
        repeats = copy.deepcopy(sorted_repeated_mod1)
        left_motifs = [repeat[0] for repeat in repeats]

        while sorted_repeated_mod1:
            for j, (left, right) in enumerate(repeats):
                if not pending[j] or min(
                    len(repeat[1]) for repeat in sorted_repeated_mod1
                ) != len(left):
                    continue
                if check_edge_in_list(right, pieces_mod) or check_edge_in_list(
                    right, steps_mod
                ):
                    indexes[j] = (
                        index_set(steps_mod, right)
                        or indexes[index_set(left_motifs, right) - 1]
                    )
                else:
                    indices = [
                        i
                        for i, piece in enumerate(pieces_mod)
                        if any(edge in piece for edge in right)
                    ]
                    if not indices:
                        continue

                    combined_pieces = [pieces_mod[i] for i in indices]
                    for index in reversed(indices):
                        pieces_mod.pop(index)
                    while len(combined_pieces) > 1:
                        combined_pieces, steps_mod, step, digraph = (
                            self.consistent_join(
                                combined_pieces,
                                steps_mod,
                                repeats,
                                step,
                                digraph,
                                indexes,
                            )
                        )
                    pieces_mod.append(combined_pieces[0])
                    indexes[j] = index_set(steps_mod, right)

                pieces_mod.append(left)
                sorted_repeated_mod1.remove(repeats[j])
                pending[j] = False

        return pieces_mod, steps_mod, repeats, step, digraph, indexes

    def generate_pathway(self) -> None:
        """Construct the pathway from remnant edges and repeated fragments.

        Equivalences are applied before repeated motifs are built in size
        order. Remaining fragments are then joined until no pair overlaps.

        Returns
        -------
        None
            Stores the constructed steps in ``self.steps``, dependency edges
            in ``self.digraph`` and remaining fragments in ``self.pieces_mod``.
        """
        pieces = [[edge] for edge in self.remnant_e]
        duplicates = self.duplicates
        if self.equivalences:
            pieces = equivalence(pieces, self.equivalences)
            duplicates = [
                equivalence(repeat, self.equivalences) for repeat in duplicates
            ]
        duplicates = sorted(duplicates, key=lambda repeat: len(repeat[0]))

        pieces, steps, repeats, step, digraph, indexes = self.repeated_construction(
            pieces, [], duplicates, 0, []
        )
        while True:
            piece_count = len(pieces)
            pieces, steps, step, digraph = self.consistent_join(
                pieces, steps, repeats, step, digraph, indexes
            )
            if len(pieces) == piece_count:
                break

        self.steps = steps
        self.digraph = digraph
        self.pieces_mod = pieces

    def generate_vo(self) -> None:
        """Render the pathway's bond types and steps as virtual objects.

        Returns
        -------
        None
            Stores bond-type objects in ``self.molecules_vo``, step objects
            in ``self.molecules_steps``, locally indexed step bonds in
            ``self.steps_indx_s`` and their atom labels in ``self.vs_atoms``.

        Raises
        ------
        ValueError
            If ``self.vo_type`` is not "graph", "mol", "smiles" or "inchi".

        Notes
        -----
        For ``vo_type="mol"``, bond-type objects are RDKit molecules but
        steps are rendered as SMILES, preserving the existing representation.
        """
        molecules_vo = [
            _tables_to_vo(
                (
                    list(enumerate(atom_types)),
                    [(0, 1, bond_order_assout_to_int(order))],
                ),
                self.vo_type,
            )
            for atom_types, order in self.atoms_list
        ]

        steps_index_s = []
        vs_atoms = []
        for step in self.steps:
            vertices = list(set(np.reshape(step, -1)))
            local_index = {vertex: i for i, vertex in enumerate(vertices)}
            steps_index_s.append(
                [
                    [
                        local_index[edge[0]],
                        local_index[edge[1]],
                        self.e_l[self.e.index(edge)],
                    ]
                    for edge in step
                ]
            )
            vs_atoms.append([self.v_l[vertex] for vertex in vertices])

        step_vo_type = "smiles" if self.vo_type == "mol" else self.vo_type
        molecules_steps = [
            _tables_to_vo(
                (
                    list(enumerate(atoms)),
                    [(u, v, bond_order_assout_to_int(order)) for u, v, order in bonds],
                ),
                step_vo_type,
            )
            for atoms, bonds in zip(vs_atoms, steps_index_s)
        ]
        self.molecules_vo = molecules_vo
        self.molecules_steps = molecules_steps
        self.steps_indx_s = steps_index_s
        self.vs_atoms = vs_atoms

    def _add_pathway_node(self, graph: nx.DiGraph, name: str) -> None:
        """Add a named node when its index has a generated molecule payload."""
        suffix = name.rsplit("_", 1)[-1]
        if name.startswith("virtual_object_"):
            index = int(suffix)
            if index < len(self.molecules_vo):
                graph.add_node(name, type="virtual_object", vo=self.molecules_vo[index])
        elif name.startswith("step_") and suffix.isdigit():
            index = int(suffix) - 1
            if 0 <= index < len(self.molecules_steps):
                graph.add_node(name, type="step", vo=self.molecules_steps[index])

    def get_assembly_digraph(self) -> Tuple[nx.DiGraph, List[Any]]:
        """Construct the assembly digraph and its unique virtual objects.

        Each node carries a ``type`` ("virtual_object" or "step"), a ``vo``
        molecule payload and a ``label`` for plotting.

        Returns
        -------
        graph : nx.DiGraph
            Directed assembly pathway.
        unique_molecules : list
            Unique virtual objects from both bond types and steps.

        Raises
        ------
        ValueError
            If ``self.vo_type`` is not "graph", "mol", "smiles" or "inchi".
        """
        self.generate_pathway()
        self.generate_vo()

        graph = nx.DiGraph()
        for source, target in self.digraph:
            self._add_pathway_node(graph, source)
            self._add_pathway_node(graph, target)
        graph.add_edges_from(self.digraph)

        for name, data in graph.nodes(data=True):
            if self.vo_type == "graph":
                data["label"] = name
            elif self.vo_type == "mol":
                smiles = Chem.MolToSmiles(
                    data["vo"], allHsExplicit=True, isomericSmiles=True
                )
                data["label"] = smi_remove_implicit_hydrogen(smiles)
            elif self.vo_type in ("smiles", "inchi"):
                data["label"] = data["vo"]
            else:
                raise ValueError(_VO_TYPE_ERROR)

        return graph, list(set(self.molecules_vo + self.molecules_steps))

    def pathway_log_string(self) -> str:
        """Return graph metadata, bond types, steps and dependencies as a log.

        Returns
        -------
        str
            The pathway's internal state, grouped under Graph, Atoms, Steps
            and Digraph headings and terminated by a newline.
        """
        lines = [
            "#####Graph#####",
            str(self.v),
            str(self.e),
            str(self.v_l),
            str(self.e_l),
            "#####Atoms#####",
        ]
        lines.extend(f"atom{i}={atom}" for i, atom in enumerate(self.atoms_list))
        lines.append("#####Steps#####")
        lines.extend(f"step{i}={step}" for i, step in enumerate(self.steps, start=1))
        lines.append("#####Digraph#####")
        lines.extend(str(edge) for edge in self.digraph)
        return "\n".join(lines) + "\n"


def parse_pathway_file(
    file: str,
    vo_type: str = "smiles",
    debug: bool = False,
    log: bool = False,
    input_graph: Optional[nx.Graph] = None,
) -> Union[Tuple[nx.DiGraph, List[Any]], Tuple[nx.DiGraph, List[Any], str]]:
    """
    Parse a pathway JSON file and construct an assembly graph.

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
        from the pathway file, by default None. AssemblyCpp drops colour
        output after index 5, so a general graph must supply its own colours.

    Returns
    -------
    graph : nx.DiGraph
        The constructed assembly directed graph.
    vo_list : list
        List of virtual objects used in the graph.
    log_string : str
        A summary log string of the pathway steps. Only returned when ``log``
        is True.

    Examples
    --------
    Re-read a pathway that was saved by an earlier calculation, without
    recomputing it:

    >>> import assemblytheorytools as att
    >>> pathway, vo_list = att.parse_pathway_file(  # doctest: +SKIP
    ...     "pathway.json", vo_type="smiles")
    >>> pathway.number_of_nodes() == len(vo_list)  # doctest: +SKIP
    True

    Pass ``log=True`` for a third return value summarising the steps, and
    ``vo_type="graph"`` to get the virtual objects as graphs rather than
    SMILES.
    """
    with open(file) as f:
        data = json.load(f)

    construction = AssemblyConstruction(data, vo_type=vo_type, input_graph=input_graph)
    graph, vo_list = construction.get_assembly_digraph()

    if debug:
        for node, attributes in graph.nodes(data=True):
            print(
                f"Node: {node}, Type: {attributes['type']}, VO: {attributes['vo']}",
                flush=True,
            )
    if log:
        return graph, vo_list, construction.pathway_log_string()
    return graph, vo_list


# Matches a petgraph BitSet label such as "{}", "{14}" or "{3, 4, 5}"
_BOND_SET_PATTERN = re.compile(r"^\{\s*(\d+(?:\s*,\s*\d+)*)?\s*\}$")

# Raised wherever a DOT string does not look like an assembly pathway
_DOT_PARSE_ERROR = "Could not parse the assembly pathway as DOT."


def _read_single_digraph(dot: str) -> "pydot.Dot":
    """Parse exactly one directed DOT graph, raising ValueError otherwise."""
    # pydot reports syntax errors by printing them rather than raising, so
    # capture that text and fold it into the exception message instead.
    report = io.StringIO()
    with redirect_stdout(report), redirect_stderr(report):
        graphs = pydot.graph_from_dot_data(dot)

    if not graphs:
        detail = report.getvalue().strip()
        raise ValueError(f"{_DOT_PARSE_ERROR} {detail}" if detail else _DOT_PARSE_ERROR)
    if len(graphs) != 1:
        raise ValueError(f"Expected a single DOT graph, found {len(graphs)}.")
    graph = graphs[0]
    if graph.get_type() != "digraph":
        raise ValueError(
            f"An assembly pathway must be a DOT 'digraph', found '{graph.get_type()}'."
        )
    return graph


def _dot_node_id(name: str) -> int:
    """Convert a possibly quoted DOT node name to an integer identifier."""
    text = str(name).strip().strip('"')
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(
            f"Assembly pathway node names must be integers, found {name!r}."
        ) from e


def _parse_bond_set(label: Optional[str], where: str) -> frozenset:
    """Parse a DOT bond-set label, identifying invalid nodes or edges by *where*."""
    if label is None:
        raise ValueError(f"Assembly pathway {where} has no 'label' attribute.")

    # pydot keeps the surrounding quotes on attribute values
    match = _BOND_SET_PATTERN.match(str(label).strip().strip('"').strip())
    if match is None:
        raise ValueError(
            f"Assembly pathway {where} has a malformed bond set "
            f"label {label!r}; expected something like '{{3, 4, 5}}'."
        )

    body = match.group(1)
    return frozenset() if body is None else frozenset(int(i) for i in body.split(","))


def _format_bond_set(bonds: frozenset) -> str:
    """Render bond indices in ascending order, e.g. ``"{3, 4, 5}"``."""
    return "{" + ", ".join(map(str, sorted(bonds))) + "}"


def _bonds_to_vo(mol: Chem.Mol, bonds: frozenset, vo_type: str) -> Any:
    """
    Build a virtual object from bond indices in the searched molecule.

    Return a graph, molecule, SMILES or InChI, leaving the fragment
    unsanitised because it may have open valences.
    """
    fragment = Chem.PathToSubmol(mol, sorted(bonds))
    if vo_type == "mol":
        return fragment
    if vo_type == "smiles":
        return Chem.MolToSmiles(fragment)
    if vo_type == "inchi":
        return Chem.MolToInchi(fragment)
    if vo_type == "graph":
        return mol_to_nx(fragment, add_hydrogens=False, sanitize=False)
    raise ValueError(_VO_TYPE_ERROR)


def _validate_pathway_dag(graph: nx.MultiDiGraph) -> None:
    """
    Check that a parsed pathway obeys the Rust backend's bond bookkeeping.

    Every joined node must be the disjoint union of the fragments joined into
    it, and every edge must carry as many bonds as its source fragment, since
    the edge names an isomorphic copy of that fragment inside the target.

    Raise ValueError if any node or edge breaks these rules.
    """
    for node, bonds in graph.nodes(data="bonds"):
        covered = set()
        for _, _, data in graph.in_edges(node, data=True):
            overlap = covered & data["bonds"]
            if overlap:
                raise ValueError(
                    f"Assembly pathway node {node} reuses bond(s) "
                    f"{sorted(overlap)} from more than one input."
                )
            covered |= data["bonds"]

        if graph.in_degree(node) and covered != set(bonds):
            raise ValueError(
                f"Assembly pathway node {node} covers bonds "
                f"{sorted(bonds)} but its inputs supply "
                f"{sorted(covered)}."
            )

    for source, target, data in graph.edges(data=True):
        expected = len(graph.nodes[source]["bonds"])
        if len(data["bonds"]) != expected:
            raise ValueError(
                f"Assembly pathway edge {source} -> {target} places "
                f"{len(data['bonds'])} bond(s), but its source "
                f"fragment has {expected}."
            )


def parse_pathway_dot(
    dot: str,
    mol: Optional[Chem.Mol] = None,
    vo_type: str = "smiles",
    strict: bool = True,
) -> nx.MultiDiGraph:
    """
    Parse a DOT assembly pathway from the Rust backend into a graph.

    The Rust ``assembly_theory`` backend reports each minimum assembly pathway
    as a DOT-formatted directed acyclic multigraph. Nodes are fragments,
    labelled by the indices of the bonds they contain; edges are joining
    operations, labelled by the bonds the source fragment occupies inside the
    fragment being built. This function turns that string into a NetworkX
    graph carrying the same node attributes as
    :meth:`AssemblyConstruction.get_assembly_digraph`, so the result works
    with :func:`assign_levels`,
    :func:`~assemblytheorytools.tools_graph.set_graph_layer` and
    :func:`~assemblytheorytools.tools_plotting.plot_pathway` unchanged.

    Parameters
    ----------
    dot : str
        The DOT-formatted pathway.
    mol : Chem.Mol, optional
        The molecule the pathway was computed for. Bond indices only mean
        anything against the exact molecule that was searched, so this must be
        the molecule parsed from the same mol block that was passed to the
        backend. If None, fragments are not built and each ``vo`` falls back
        to the node's bond-set label. Default is None.
    vo_type : str, optional
        Representation for the virtual objects: 'graph', 'mol', 'smiles' or
        'inchi'. Default is 'smiles'.
    strict : bool, optional
        If True, check that each node is the disjoint union of the fragments
        joined into it and that each edge places as many bonds as its source
        fragment holds. Default is True.

    Returns
    -------
    nx.MultiDiGraph
        The pathway. Nodes are integers carrying ``type`` (always
        'virtual_object'), ``bonds`` (a frozenset of bond indices), ``label``
        (those indices as a string) and ``vo``. Edges carry ``bonds`` and
        ``label``. A multigraph is used because a fragment joined to itself
        produces parallel edges.

    Raises
    ------
    ValueError
        If the string is not a single DOT digraph, if a node or edge label is
        not a set of bond indices, if a node name is not an integer, if a bond
        index is out of range for `mol`, if `vo_type` is not recognised, or if
        `strict` is True and the bond bookkeeping does not add up.

    Notes
    -----
    - Pathway reconstruction is only available from ``assembly_theory``
      releases that accept ``max_pathways``; see
      :func:`~assemblytheorytools.assembly.calculate_assembly_index_rust_search`,
      which calls this function for you with the right molecule.
    - Fragments are built with ``Chem.PathToSubmol`` and left unsanitised,
      since they generally have open valences.

    Examples
    --------
    Load a pathway that was written to a file, together with the mol block it
    was computed from:

    >>> from rdkit import Chem
    >>> import assemblytheorytools as att
    >>> mol = Chem.MolFromMolBlock(open("anthracene.mol").read())  # doctest: +SKIP
    >>> pathway = att.parse_pathway_dot(  # doctest: +SKIP
    ...     open("pathway.dot").read(), mol=mol)
    >>> sorted(d["vo"] for _, d in pathway.nodes(data=True))[:2]  # doctest: +SKIP
    ['cc', 'cc']

    Pass ``vo_type="graph"`` to get the fragments as NetworkX graphs, or omit
    `mol` to read the pathway's structure without building any chemistry.
    """
    if vo_type not in _VO_TYPES:
        raise ValueError(_VO_TYPE_ERROR)

    parsed = nx.nx_pydot.from_pydot(_read_single_digraph(dot))
    n_bonds = None if mol is None else mol.GetNumBonds()

    if mol is not None:
        # The backend searches the kekulised graph, and aromatic flags on a
        # fragment torn out of a ring cannot be sanitised, so work on a
        # kekulised copy rather than the caller's molecule
        mol = Chem.Mol(mol)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Chem.KekulizeException:
            pass

    graph = nx.MultiDiGraph()
    for name, data in parsed.nodes(data=True):
        bonds = _parse_bond_set(data.get("label"), f"node {name!r}")
        if n_bonds is not None and any(bond >= n_bonds for bond in bonds):
            raise ValueError(
                f"Assembly pathway node {name!r} names bond "
                f"{max(bonds)}, but the molecule has {n_bonds} bonds. "
                "The pathway must be parsed against the molecule it "
                "was computed from."
            )
        label = _format_bond_set(bonds)
        graph.add_node(
            _dot_node_id(name),
            type="virtual_object",
            bonds=bonds,
            label=label,
            vo=label if mol is None else _bonds_to_vo(mol, bonds, vo_type),
        )

    for source, target, data in parsed.edges(data=True):
        bonds = _parse_bond_set(data.get("label"), f"edge {source!r} -> {target!r}")
        graph.add_edge(
            _dot_node_id(source),
            _dot_node_id(target),
            bonds=bonds,
            label=_format_bond_set(bonds),
        )

    if strict:
        _validate_pathway_dag(graph)

    return graph


def get_level(G: nx.DiGraph, node: str) -> int | None:
    """
    Return the level of a node in a graph.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges represent assembly steps.
    node : str
        The node for which to determine the assembly depth.

    Returns
    -------
    int
        The assembly depth of the node: one more than the deepest predecessor,
        or 0 when the node has no predecessors.

    Raises
    ------
    KeyError
        If a predecessor has not yet been assigned a level.
    """
    return (
        max((G.nodes[source]["level"] for source, _ in G.in_edges(node)), default=-1)
        + 1
    )


def assign_levels(G: nx.DiGraph, inplace: bool = True) -> None | nx.DiGraph:
    """
    Assign assembly depth to the nodes of a graph.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph where nodes represent (sub-)objects and edges
        represent assembly steps.
    inplace : bool, optional
        If True, modifies the graph in place. If False, returns a modified
        copy, by default True.

    Returns
    -------
    None or nx.DiGraph
        If inplace is True, modifies the graph in place and returns None. If
        inplace is False, returns a new graph with updated node attributes.

    Raises
    ------
    TypeError
        If the input graph G is not a directed graph (DiGraph).

    Notes
    -----
    Nodes are visited in insertion order, so a predecessor must appear
    before its successors for the levels to resolve.

    The pathway returned by
    :func:`~assemblytheorytools.assembly.calculate_assembly_index` is
    *not* in topological order, so passing it directly raises
    ``KeyError: 'level'``. Rebuild it in topological order first, as in
    the second example below.

    Examples
    --------
    >>> import networkx as nx
    >>> import assemblytheorytools as att
    >>> graph = nx.DiGraph()
    >>> graph.add_nodes_from(["a", "b", "c", "ab", "abc"])
    >>> graph.add_edges_from(
    ...     [("a", "ab"), ("b", "ab"), ("ab", "abc"), ("c", "abc")])
    >>> att.assign_levels(graph)
    >>> {n: d["level"] for n, d in graph.nodes(data=True)}
    {'a': 0, 'b': 0, 'c': 0, 'ab': 1, 'abc': 2}

    For a calculated pathway, re-order the nodes first:

    >>> _, _, pathway = att.calculate_assembly_index(
    ...     att.smi_to_nx("CCO"), strip_hydrogen=True)
    >>> ordered = nx.DiGraph()
    >>> ordered.add_nodes_from(
    ...     (n, pathway.nodes[n]) for n in nx.topological_sort(pathway))
    >>> ordered.add_edges_from(pathway.edges(data=True))
    >>> att.assign_levels(ordered)
    >>> max(d["level"] for _, d in ordered.nodes(data=True)) >= 0
    True
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph G must be a directed graph (DiGraph).")
    if not inplace:
        G = G.copy()

    for node, data in G.nodes(data=True):
        data["level"] = get_level(G, node)

    return None if inplace else G


def immediate_predecessors(
    data: Dict[str, Any], interval: Tuple[int, int]
) -> List[str]:
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
    fragment = data["file_graph"][0]["Fragments"][0]
    end = sum(interval)

    c_idx = interval[0]
    while c_idx < end:
        parent = ""
        for dup in data["duplicates"]:
            left = dup["Left"]
            if left[1] >= interval[1]:  # The duplicate cannot fit in the interval
                continue
            if c_idx in range(left[0], sum(left)):
                candidate = left
            else:
                candidate = dup["Right"]
                if c_idx not in range(candidate[0], sum(candidate)):
                    continue

            # Prefer the longest contained copy; the left copy takes precedence.
            if (
                candidate[1] > len(parent)
                and candidate[0] >= interval[0]
                and sum(candidate) <= end
            ):
                parent = fragment[candidate[0] : sum(candidate)]

        output.append(parent or fragment[c_idx])
        c_idx += len(parent) or 1

    return output


def build_str(
    interval: Union[List[int], Tuple[int, int]], data: Dict[str, Any], path: nx.DiGraph
) -> nx.DiGraph:
    """
    Build the string from the pathway data and add it to the path.

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
            # Recursively build the duplicate strings if not already in the path
            path = build_str([c_idx, c_idx + len(sub_str)], data, path)
        c_idx += len(sub_str)

    # Builds string from left to right. The membership checks below are only
    # relevant when the path is not minimum.
    assembled = ledger[0]
    for part in ledger[1:]:
        combined = assembled + part
        if combined not in path.nodes:
            path.add_node(combined)
        for source in (assembled, part):
            if (source, combined) not in path.edges:
                path.add_edge(source, combined)
        assembled = combined
    return path


def parse_string_pathway_file(file_path_pathway: str) -> Tuple[List[str], nx.DiGraph]:
    """
    Parse a string pathway file into virtual objects and a directed graph.

    Parameters
    ----------
    file_path_pathway : str
        Path to the pathway file.

    Returns
    -------
    VOs : list
        List of virtual objects in the calculated pathway.
    path : nx.DiGraph
        NetworkX directed graph representing the pathway.

    Raises
    ------
    FileNotFoundError
        If the pathway file is not found at the specified path.
    """
    if not os.path.isfile(file_path_pathway):
        raise FileNotFoundError(f"Pathway file not found: {file_path_pathway}")

    with open(file_path_pathway) as f:
        data = json.load(f)

    file_string = data["file_graph"][0]["Fragments"][0]
    path = nx.DiGraph()
    path.add_nodes_from(set(file_string))

    path = build_str([0, len(file_string)], data, path)
    return list(path.nodes), path


def molstr_to_str(
    molstr: nx.Graph, edge_color_dict: Optional[Dict[str, str]] = None
) -> str:
    """
    Decode a molecular graph representation of a string.

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
    if edge_color_dict is None:  # Directed
        odd = int(
            molstr.nodes(data=True)[0]["color"] == "null"
        )  # True if encoding was respected
        # Select alternate nodes, even when a fragment breaks the encoding.
        return "".join(
            data["color"]
            for index, (_, data) in enumerate(molstr.nodes(data=True))
            if index % 2 == odd
        )

    colors = {value: key for key, value in edge_color_dict.items()}
    for digit, name in (
        ("1", "single"),
        ("2", "double"),
        ("3", "triple"),
        ("4", "quadruple"),
    ):
        if digit in colors:
            colors[name] = colors[digit]
    colors["0"] = "!"
    return "".join(
        colors[str(data.get("color"))] for _, _, data in molstr.edges(data=True)
    )


def convert_digraph_vo_to_target(
    graph: nx.DiGraph,
    target: str = "smi",
    add_hydrogens: bool = False,
    sanitize: bool = True,
) -> nx.DiGraph:
    """
    Convert the virtual objects of a directed graph to a target format.

    Update each node's ``'vo'`` attribute in place.

    Parameters
    ----------
    graph : nx.DiGraph
        A NetworkX directed graph where each node contains a 'vo' attribute
        representing the virtual object.
    target : str, optional
        The target format for the virtual object. Must be one of:
        - 'smi': Convert to SMILES format.
        - 'inchi': Convert to InChI format.
        - 'mol': Convert to RDKit Mol object.
        Default is 'smi'.
    add_hydrogens : bool, optional
        Whether to add hydrogens during the conversion process, by default
        False.
    sanitize : bool, optional
        Whether to sanitize the molecule during the conversion process, by
        default True.

    Returns
    -------
    nx.DiGraph
        The updated directed graph with the 'vo' attribute of each node
        converted to the specified target format.

    Raises
    ------
    ValueError
        If the specified target format is not one of 'smi', 'inchi', or
        'mol'.

    Notes
    -----
    - The conversion functions `nx_to_smi`, `nx_to_inchi`, and `nx_to_mol`
      are used to perform the conversions.
    - The `add_hydrogens` and `sanitize` parameters are passed to the conversion functions.
    """
    converters = {"smi": nx_to_smi, "inchi": nx_to_inchi, "mol": nx_to_mol}
    if target not in converters:
        raise ValueError("Target must be 'smi', 'inchi', or 'mol'")
    convert = converters[target]

    for _, data in graph.nodes(data=True):
        data["vo"] = convert(data["vo"], add_hydrogens=add_hydrogens, sanitize=sanitize)
    return graph


def get_vos_on_layer(
    digraph: nx.DiGraph,
    layer: Union[int, List[int], str],
    target: str = "smi",
    add_hydrogens: bool = False,
    sanitize: bool = True,
) -> Union[List, List[List]]:
    """
    Retrieve virtual objects (VOs) from specific layers in a directed graph.

    Convert virtual objects and assign layers on a copy of the graph, then
    collect the requested layers in graph order.

    Parameters
    ----------
    digraph : nx.DiGraph
        A directed graph where nodes contain virtual object (VO) representations.
    layer : int, list of int, or 'all'
        The layer number(s) from which to retrieve the VOs. If 'all', VOs from all layers are returned.
    target : str, optional
        The target format for the VOs. Must be one of:
        - 'smi': Convert to SMILES format.
        - 'inchi': Convert to InChI format.
        - 'mol': Convert to RDKit Mol object.
        Default is 'smi'.
    add_hydrogens : bool, optional
        Whether to add hydrogens during the conversion process, by default False.
    sanitize : bool, optional
        Whether to sanitize the molecule during the conversion process, by default True.

    Returns
    -------
    list
        A list of VOs if `layer` is an int. A list of lists of VOs if `layer` is a list of ints or 'all'.

    Raises
    ------
    ValueError
        If the specified target format is not one of 'smi', 'inchi', or 'mol'.
    """
    digraph = convert_digraph_vo_to_target(
        digraph.copy(), target=target, add_hydrogens=add_hydrogens, sanitize=sanitize
    )
    digraph = set_graph_layer(digraph)

    def vos_on(layer_id: int) -> List[Any]:
        """Collect virtual objects on one layer, preserving graph order."""
        return [
            data.get("vo")
            for _, data in digraph.nodes(data=True)
            if data.get("layer") == layer_id
        ]

    if isinstance(layer, int):
        return vos_on(layer)

    if layer == "all":
        max_layer = max(data.get("layer", 0) for _, data in digraph.nodes(data=True))
        layers = range(max_layer + 1)
    elif isinstance(layer, list):
        layers = layer
    else:
        layers = []

    return [vos_on(layer_id) for layer_id in layers]
