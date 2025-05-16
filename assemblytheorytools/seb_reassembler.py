import ast
import random
import signal
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles

from .assembly import add_assembly_to_path
from .construction import parse_pathway_file
from .tools_mol import safe_standardize_mol

bond_types = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


def compose_all(graphs, attribute="level", get_atomic_count=True):
    R: nx.MultiDiGraph = None
    updated_nodes_data, node_counts = accumulate_nodes_data(
        graphs, attribute=attribute
    )
    updated_edges_data = accumulate_edges_data(graphs)

    # add graph attributes, H attributes take precedent over G attributes
    for i, G in enumerate(graphs):
        G = nx.DiGraph(G)  # necessary to accumulate edge data properly
        if i == 0:
            # create new graph
            R = G.__class__()
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError("All graphs must be graphs or multigraphs.")
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes())
        R.add_edges_from(
            G.edges(keys=False, data=False)
            if G.is_multigraph()
            else G.edges(data=False)
        )
    nx.set_node_attributes(R, values=updated_nodes_data, name=attribute)
    nx.set_node_attributes(R, values=node_counts, name="count")
    nx.set_edge_attributes(R, values=updated_edges_data, name="count")

    node_usage = accumulate_node_usage(R, attribute=attribute)
    nx.set_node_attributes(R, values=node_usage, name="usage")

    if get_atomic_count:
        atomic_count = get_atomic_distribution(R)
        nx.set_node_attributes(R, values=atomic_count, name="atomic_count")

    if R is None:
        raise ValueError("cannot apply compose_all to an empty list")
    return R


def accumulate_edges_data(graphs):
    edge_counts = {}
    for graph in graphs:
        for edge in graph.edges():
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return edge_counts


def accumulate_nodes_data(graphs, attribute="level"):
    nodes_data = {}
    node_counts = {}

    for graph in graphs:
        for node, attributes in graph.nodes(data=True):
            if node not in nodes_data or attributes[attribute] < nodes_data[node]:
                nodes_data[node] = attributes[attribute]
                node_counts[node] = node_counts.get(node, 0) + 1
            else:
                node_counts[node] += 1

    return nodes_data, node_counts


def accumulate_node_usage(graph, attribute="usage"):
    return {
        node: [graph.nodes[edge[-1]][attribute] for edge in graph.out_edges(node)]
        for node in graph.nodes()
    }


def get_atomic_distribution(graph) -> dict:
    pt = Chem.rdchem.GetPeriodicTable()
    atomic_count = defaultdict(set)

    for node in graph.nodes:
        mol = MolFromSmiles(node)
        if mol is None:
            atomic_count[node] = None
            continue

        for atom in mol.GetAtoms():
            free_valence = pt.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()
            if free_valence > 0:
                atomic_count[node].add(atom.GetAtomicNum())

    return atomic_count


def destringyfy(string):
    return ast.literal_eval(string)


def combine_fragments(fragment1, fragment2, combinations) -> str:
    max_idx = fragment1.GetNumAtoms()

    f1_atoms = [fragment1.GetAtomWithIdx(atom1) for atom1, _ in combinations]
    f2_atoms = [fragment2.GetAtomWithIdx(atom2) for _, atom2 in combinations]
    f1_bond_partners = [atom.GetNeighbors() for atom in f2_atoms]
    f1_bond_orders = [[bond.GetBondType() for bond in atom.GetBonds()] for atom in f2_atoms]

    combined_mol = Chem.CombineMols(fragment1, fragment2)
    rw_mol = Chem.RWMol(combined_mol)
    Chem.RemoveAllHs(rw_mol)

    rw_mol.BeginBatchEdit()
    for atom_id, (partners, orders) in enumerate(zip(f1_bond_partners, f1_bond_orders)):
        rw_mol.RemoveAtom(f2_atoms[atom_id].GetIdx() + max_idx)
        for partner, order in zip(partners, orders):
            rw_mol.AddBond(f1_atoms[atom_id].GetIdx(), partner.GetIdx() + max_idx, order)
    rw_mol.CommitBatchEdit()

    try:
        mol = safe_standardize_mol(rw_mol.GetMol(), add_hydrogens=True)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception as e:
        print(f"Standardization failed: {e}", flush=True)
        return None


def valence_check(atom1, atom2):
    pt = Chem.rdchem.GetPeriodicTable()
    val1 = pt.GetDefaultValence(atom1[0]) - atom1[1]
    val2 = pt.GetDefaultValence(atom2[0]) - atom2[1]
    return val1 + val2 <= pt.GetDefaultValence(atom1[0])


def count_non_overlapping_sublists(lst):
    # Sort the sublists by their first element
    lst.sort(key=lambda x: x[0])

    # Initialize the count of non-overlapping sublists
    count = 1

    # Compare the end of the current sublist with the start of the next
    for i in range(1, len(lst)):
        if lst[i - 1][1] <= lst[i][0]:
            count += 1

    return count


def get_possible_combinations(idx_map1, idx_map2):
    possible_combinations = [
        [atom1, atom2]
        for atom1 in idx_map1
        for atom2 in idx_map2
        if idx_map1[atom1][0] == idx_map2[atom2][0] and valence_check(idx_map1[atom1], idx_map2[atom2])
    ]

    if possible_combinations:
        random.shuffle(possible_combinations)
        return possible_combinations
    return None


def get_allowed_pairs(combinations, k, max_iterations=1000):
    sampled_sublists = []
    counter = 0

    while len(sampled_sublists) < k and counter < max_iterations:
        sublist = random.choice(combinations)

        if all(sublist[0] != prev[0] and sublist[1] != prev[1] for prev in sampled_sublists):
            sampled_sublists.append(sublist)
        counter += 1

    return sampled_sublists


def select_max_overlaps(atoms1, atoms2) -> int:
    return min(atoms1, atoms2)


def get_atom_type_index_mapping(fragment):
    try:
        mol = Chem.RemoveHs(Chem.MolFromSmiles(fragment, sanitize=False), implicitOnly=False)
    except (TypeError, Chem.KekulizeException, Chem.AtomValenceException):
        return None, None

    if mol is None:
        return None, None

    pt = Chem.rdchem.GetPeriodicTable()
    atom_type_index_mapping = defaultdict(list)

    for atom in mol.GetAtoms():
        free_valence = pt.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()
        atom_type_index_mapping[atom.GetIdx()] = [atom.GetSymbol(), free_valence]

    return mol, atom_type_index_mapping


class ParsePathwayLog:
    def __init__(self, pathway_log: str):
        self.pathway_log = pathway_log
        self.atom_lines, self.building_block_lines, self.steps_lines, self.digraph_lines = self._parse_log()
        self.G = self._build_multidigraph()
        self._validate_graph()
        self._assign_levels()
        self.nodes_per_level = self._count_nodes_per_level()

    def _parse_log(self):
        atom_lines, building_block_lines, steps_lines, digraph_lines = [], [], {}, []
        blocks = {"#####Graph#####": "atom", "#####Atoms#####": "building_block",
                  "#####Steps#####": "steps", "#####Digraph#####": "digraph"}
        current_block = None

        for line in self.pathway_log.split("\n"):
            if line in blocks:
                current_block = blocks[line]
                continue
            if not line.strip():
                continue

            if current_block == "atom":
                atom_lines.append(destringyfy(line))
            elif current_block == "building_block":
                building_block_lines.append([line.split("=")[0], destringyfy(line.split("=")[-1])])
            elif current_block == "steps":
                steps_lines[line.split("=")[0].replace("step", "")] = destringyfy(line.split("=")[-1])
            elif current_block == "digraph":
                digraph_lines.append(destringyfy(line))

        return atom_lines, building_block_lines, steps_lines, digraph_lines

    def _build_multidigraph(self):
        graph = nx.MultiDiGraph()
        smiles_graph = self._build_basic_building_blocks()
        graph.add_nodes_from(smiles_graph.values())

        for i in range(0, len(self.digraph_lines), 2):
            step = self.digraph_lines[i][-1].replace("step_", "")
            smiles_graph[self.digraph_lines[i][-1]] = self._build_fragment_for_step(step)
            graph.add_node(smiles_graph[self.digraph_lines[i][-1]])
            graph.add_edge(smiles_graph[self.digraph_lines[i][0]], smiles_graph[self.digraph_lines[i][-1]])
            graph.add_edge(smiles_graph[self.digraph_lines[i + 1][0]], smiles_graph[self.digraph_lines[i][-1]])

        return graph

    def _build_basic_building_blocks(self):
        bb = {}
        for i, line in enumerate(self.building_block_lines):
            edmol = Chem.EditableMol(Chem.Mol())
            id1 = edmol.AddAtom(Chem.Atom(line[1][0][0]))
            id2 = edmol.AddAtom(Chem.Atom(line[1][0][-1]))
            edmol.AddBond(id1, id2, bond_types[line[1][-1]])
            bb[f"virtual_object_{i}"] = Chem.MolToSmiles(edmol.GetMol())
        return bb

    def _build_fragment_for_step(self, step):
        bonds_ids = self.steps_lines[step]
        atom_ids = list(set(atom for bond in bonds_ids for atom in bond))
        atoms = [self.atom_lines[-2][atom_id] for atom_id in atom_ids]
        bonds = {tuple(bond_id): self.atom_lines[-1][self.atom_lines[1].index(bond_id)] for bond_id in bonds_ids}

        edmol = Chem.EditableMol(Chem.Mol())
        id_mapping = {atom_id: i for i, atom_id in enumerate(atom_ids)}
        for atom in atoms:
            edmol.AddAtom(Chem.Atom(atom))
        for bond_id, bond_type in bonds.items():
            edmol.AddBond(id_mapping[bond_id[0]], id_mapping[bond_id[1]], bond_types[bond_type])
        return Chem.MolToSmiles(edmol.GetMol())

    def _validate_graph(self):
        for node in self.G.nodes:
            if len(list(self.G.in_edges(node))) > 2:
                raise ValueError("Node has more than 2 predecessors - invalid!")

    def _assign_levels(self):
        for node in self.G.nodes:
            if not list(self.G.predecessors(node)):
                self.G.nodes[node].update({"level": 0, "assembly_index": 1})
            else:
                self.G.nodes[node]["level"] = self._get_level(node)

    def _get_level(self, node):
        predecessors = [edge[0] for edge in self.G.in_edges(node)]
        if not predecessors:
            return 0
        if len(predecessors) == 2:
            return max(self.G.nodes[pred]["level"] for pred in predecessors) + 1
        return self._get_level(predecessors[0])

    def _count_nodes_per_level(self):
        levels = {}
        for node in self.G.nodes:
            level = self.G.nodes[node]["level"]
            levels[level] = levels.get(level, 0) + 1
        return levels

    def plot_layered_graph(self, show_molecules=False, save_fig=True):
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = plt.get_cmap("Blues")
        node_colors = [cmap(0.4) for _ in self.G.nodes]
        positions = self._get_node_positions()

        nx.draw(self.G, pos=positions, ax=ax, with_labels=False, node_size=100,
                node_color=node_colors, connectionstyle="arc3,rad=0.05", edge_color="grey", width=1)

        if show_molecules:
            for mol, pos in zip(sorted(self.G.nodes, key=lambda x: self.G.nodes[x]["level"]), positions.values()):
                img = Draw.MolToImage(Chem.MolFromSmiles(mol), size=(150, 150))
                imagebox = OffsetImage(img, zoom=0.3)
                ab = AnnotationBbox(imagebox, (pos[0], pos[1] + 0.6), frameon=True)
                ax.add_artist(ab)

        if save_fig:
            plt.savefig("data/images/pathway.svg", dpi=200)
        else:
            plt.show()

    def _get_node_positions(self):
        positions = {}
        sorted_nodes = sorted(self.G.nodes(data=True), key=lambda x: x[1]["level"])
        current_level, layer_pos = None, None

        for node, _ in sorted_nodes:
            level = self.G.nodes[node]["level"]
            if level != current_level:
                current_level = level
                layer_pos = np.linspace(-self.nodes_per_level[level], self.nodes_per_level[level],
                                        self.nodes_per_level[level])
            positions[node] = [level * 2, layer_pos[0]]
            layer_pos = np.delete(layer_pos, 0)

        return positions


class Molecule:
    def __init__(self,
                 smiles: str = "",
                 pathway: Optional[list[str]] = None,
                 assembly_index: Optional[int] = None,
                 G: Optional[nx.DiGraph] = None,
                 timeout: Optional[int] = 60):
        self.smiles = smiles
        self.pathway = pathway
        self.assembly_index = assembly_index
        self.G = G
        self.timeout = timeout
        self.pathway_log_string = None
        self.pathway_fragments = None
        self.pathwayLogObj = None
        self.assembly_output_path = None

        if G:
            self.smiles = list(G.nodes)[-1]

    def get_smiles(self) -> str:
        if not self.smiles:
            self.reconstruct_pathway()
            self.construct_layered_graph()
        return self.smiles

    def calc_pathway(self) -> None:
        mol = Chem.MolFromSmiles(self.smiles)
        Chem.MolToMolFile(mol, 'temp.mol')
        mol_file_path = Path("temp.mol")
        proc = subprocess.Popen(
            [add_assembly_to_path(), mol_file_path.parent / mol_file_path.stem],
            stdout=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)

        self.assembly_output_path = f"{mol_file_path.parent / mol_file_path.stem}Pathway"

    def reconstruct_pathway(self) -> None:
        self.calc_pathway()
        _, self.pathway_fragments, self.pathway_log_string = parse_pathway_file(
            self.assembly_output_path, vo_type="inchi", log=True
        )

    def construct_layered_graph(self):
        self.pathwayLogObj = ParsePathwayLog(self.pathway_log_string)
        self.G = self.pathwayLogObj.G
        self.smiles = list(self.G.nodes)[-1]

    def plot_layered_graph(self, show_molecule: bool = True):
        self.pathwayLogObj.plot_layered_graph(show_molecule)


class MoleculeSpace:
    def __init__(self, molecules: List['Molecule']):
        self.molecules = molecules
        self.molecule_smiles = [molecule.get_smiles() for molecule in molecules]
        self._remove_none()
        self.joined_smiles = ".".join(self.molecule_smiles)
        self.joined_assembly_graph = None
        self.joined_assembly_graph_minus_x = None
        self.max_assembly_index = None
        self.root_nodes = None
        self.leaf_nodes = None

    def _remove_none(self):
        self.molecules = [mol for mol in self.molecules if mol.get_smiles() is not None]
        self.molecule_smiles = [mol.get_smiles() for mol in self.molecules]

    def _set_root_nodes(self):
        self.root_nodes = [
            node for node in self.joined_assembly_graph.nodes
            if self.joined_assembly_graph.in_degree(node) == 0
        ]

    def _set_leaf_nodes(self):
        self.leaf_nodes = [
            node for node in self.molecule_smiles
            if node in self.joined_assembly_graph.nodes
        ]

    def construct_joined_graph(self):
        self.joined_assembly_graph = compose_all([mol.G for mol in self.molecules])
        self._set_root_nodes()
        self._set_leaf_nodes()

    def a_minus_x_assembly_pool(
            self, X: int = 1, get_graph: bool = True, remove_paths: bool = False
    ) -> Union[nx.MultiDiGraph, List[str]]:
        if self.joined_assembly_graph is None:
            print("Constructing joined assembly graph.", flush=True)
            self.construct_joined_graph()

        self.max_assembly_index = max(
            self.joined_assembly_graph.nodes.data("level"), key=lambda x: x[-1]
        )[-1]

        if X > self.max_assembly_index:
            raise ValueError(
                f"X must be less than or equal to the maximum assembly index {self.max_assembly_index}"
            )

        temp_graph = self.joined_assembly_graph.copy()
        to_remove = [
            node for node in self.leaf_nodes
            if temp_graph.nodes[node]["level"] > self.max_assembly_index - X
        ]

        removed_observed = 0
        if remove_paths:
            for node in to_remove:
                if not temp_graph.has_node(node):
                    print(f"Node {node} not in graph", flush=True)
                    continue
                try:
                    pw_nodes = self.molecules[self.molecule_smiles.index(node)].G.nodes
                except Exception as e:
                    print(f"Could not find {node}: {e}", flush=True)
                    temp_graph.remove_node(node)
                    continue

                for pw_node in pw_nodes:
                    if temp_graph.has_node(pw_node):
                        if temp_graph.nodes[pw_node]["count"] > 1:
                            temp_graph.nodes[pw_node]["count"] -= 1
                        else:
                            temp_graph.remove_node(pw_node)
                removed_observed += 1

        for node in list(temp_graph.nodes):
            if temp_graph.nodes[node]["level"] > self.max_assembly_index - X:
                temp_graph.remove_node(node)

        if not remove_paths:
            removed_observed += sum(1 for leaf in self.leaf_nodes if not temp_graph.has_node(leaf))

        self.joined_assembly_graph_minus_x = temp_graph
        return (temp_graph, removed_observed) if get_graph else (list(temp_graph.nodes), removed_observed)


class MoleculeGenerationAssemblyPool:
    def __init__(self, assembly_pool) -> None:
        self.diverged_assembly_graph = None
        self.original_a_minus_X = None
        self.bu_level_to_fragment = None
        self.level_to_fragment = None
        self.num_removed = None
        self.assembly_pool = assembly_pool
        self.assembled_molecules: dict[int, list] = defaultdict(list)
        self.original_a_minus_X: None

        self.diverged_assembly_graph: nx.MultiDiGraph
        self.level_to_fragment: dict[int, list[str]]
        self.sampling_weights: list[int]

    def set_assembly_pool(self, X=10, remove_pathways=False):
        """
        Modifies the assembly pool by removing a subset of molecules and
        updates node-level attributes.

        This function performs the following tasks:
        1. Calls `a_minus_x_assembly_pool()` on `self.assembly_pool` to
        remove molecules from "X" layers and update `self.num_removed`.
        2. Extracts the "level" attributes from nodes in
        `self.assembly_pool.joined_assembly_graph_minus_x` and
        populates `self.level_to_fragment`, mapping levels to nodes.
        3. Creates a backup copy of `self.level_to_fragment` as `bu_level_to_fragment`.
        4. Creates a deep copy of `self.assembly_pool.joined_assembly_graph_minus_x`
        and stores it in `self.original_a_minus_X` for future reference.

        Args:
            X : int, optional (default = 10) Number of steps back in the assembly depth path.
            remove_pathways : bool, optional (default=False). If True, removes fragments from the pathways of the observed molecules that are being removed

        Returns:
            None. This method updates the `self.num_removed`, `self.level_to_fragment`,
            `self.bu_level_to_fragment`, and `self.original_a_minus_X` attributes.
        """
        _, self.num_removed = self.assembly_pool.a_minus_x_assembly_pool(
            X=X, get_graph=False, remove_paths=remove_pathways
        )

        # Populate self.level_to_fragment
        self.level_to_fragment = defaultdict(list)
        for node, level in nx.get_node_attributes(
                self.assembly_pool.joined_assembly_graph_minus_x, "level"
        ).items():
            if (
                    self.assembly_pool.joined_assembly_graph_minus_x.nodes[node][
                        "atomic_count"
                    ]
                    is not None
            ):
                self.level_to_fragment[level].append(node)

        self.bu_level_to_fragment = self.level_to_fragment.copy()
        self.original_a_minus_X = deepcopy(
            self.assembly_pool.joined_assembly_graph_minus_x
        )

        return None

    def reset_level_to_fragment(self):
        """
        Restores `level_to_fragment` to its original state before modifications.

        This function resets `self.level_to_fragment` by copying the backup stored
        in `self.bu_level_to_fragment`. It effectively undoes any changes made
        after calling `set_assembly_pool()`, ensuring that the level-to-fragment
        mapping returns to its initial configuration.

        Returns:
            None. This method updates the `self.level_to_fragment` attribute.

        Notes:
        ------
        - Uses `.copy()` to ensure `self.bu_level_to_fragment` remains unchanged.
        - Helps revert changes if an operation modifies `level_to_fragment` incorrectly.
        - Intended as a reset mechanism after molecule removals or modifications in the assembly process.
        """
        self.level_to_fragment = self.bu_level_to_fragment.copy()
        return None

    def get_leaf_nodes(self):
        """
        Returns:
            list[str]: A list of leaf nodes in the diverged assembly graph.
        """
        return [
            node
            for node in self.diverged_assembly_graph
            if self.diverged_assembly_graph.out_degree(node) == 0
        ]

    def get_assembled_molecules(self):
        """
        Returns:
            list[str]: A list of assembled molecules in the assembly pool.
        """
        if len(self.assembled_molecules) == 0:
            return None
        return [value[-1][-1] for _, value in self.assembled_molecules.items() if value]

    def get_leaf_counts_per_level(self, min_level=1) -> dict[int, int]:
        """
        Retrieves the number of leaf nodes per level in the assembly pool.
        Used to sample how many assembly steps to take

        Args:
            min_level : int, optional (default=1) The minimum level to consider.

        Returns:
            dict[int, int]: The number of leaf nodes per level in the assembly pool.
        """
        leaf_counts: dict[int, int] = defaultdict(int)
        for node in self.assembly_pool.leaf_nodes:
            if (
                    self.assembly_pool.joined_assembly_graph.out_degree(node) == 0
                    and self.assembly_pool.joined_assembly_graph.nodes[node]["level"]
                    >= min_level
            ):
                leaf_counts[
                    self.assembly_pool.joined_assembly_graph.nodes[node]["level"]
                ] += 1

        for level in range(min_level, self.assembly_pool.max_assembly_index + 1):
            if level not in leaf_counts:
                leaf_counts[level] = int(1e-8)

        return leaf_counts

    def add_to_assembly_graph(self, parents, child) -> bool:
        """
        Adds a child node to the assembly graph.

        Args:
            parents: list[str] The parent nodes of the child node.
            child: str The child node to add.

        Returns:
            bool: True if the child node is successfully added, False otherwise.
            Updates the `self.assembly_pool.joined_assembly_graph_minus_x` attribute.
        """
        PeriodicTable = Chem.rdchem.GetPeriodicTable()
        atomic_count = []
        try:
            for atom in Chem.MolFromSmiles(child).GetAtoms():
                free_atom_valence = (
                        PeriodicTable.GetDefaultValence(atom.GetSymbol()) - atom.GetExplicitValence()
                )
                # Append atoms with free valence
                if free_atom_valence > 0:
                    atomic_count.append(atom.GetAtomicNum())


        except AttributeError:
            print(child, flush=True)
            return False

        for parent in parents:
            self.assembly_pool.joined_assembly_graph_minus_x.add_edge(parent, child)

        # set level of child
        child_level = (
                max(
                    self.assembly_pool.joined_assembly_graph_minus_x.nodes[parents[0]][
                        "level"
                    ],
                    self.assembly_pool.joined_assembly_graph_minus_x.nodes[parents[1]][
                        "level"
                    ],
                )
                + 1
        )
        self.level_to_fragment[child_level].append(child)

        self.assembly_pool.joined_assembly_graph_minus_x.nodes[child]["level"] = (
            child_level
        )
        self.assembly_pool.joined_assembly_graph_minus_x.nodes[child]["count"] = 1

        self.assembly_pool.joined_assembly_graph_minus_x.nodes[child][
            "atomic_count"
        ] = set(atomic_count)

        return True

    def construct_diverged_assembly_graph(self):
        """
        Constructs a diverged assembly graph from the assembly pool.

        Returns:
            None. Updates the `self.diverged_assembly_graph` attribute.
        """
        self.diverged_assembly_graph = deepcopy(self.assembly_pool.joined_assembly_graph_minus_x)
        return None

    def set_sw_layer(self, exponent: float = 2.0):
        """
        Computes and assigns sampling weights for each layer in the assembly graph.

        This function initializes `self.layer_sampling_weights` as a dictionary where
        each key represents a layer level in the `joined_assembly_graph`, and the value
        is computed as the count of nodes in that layer raised to the given `exponent`.
        This weighting influences how nodes are sampled during assembly.

        Args:
            exponent (float, optional): The exponent applied to the node count to compute
                                        sampling weights. Defaults to 2.0. This enforces the deviation from
                                        the original graph.

        Updates:
            self.layer_sampling_weights (dict): A dictionary where keys are layer levels
                                                and values are computed sampling weights.

        Returns:
            None
        """
        self.layer_sampling_weights: dict = defaultdict(int)
        for node in self.assembly_pool.joined_assembly_graph:
            self.layer_sampling_weights[
                self.assembly_pool.joined_assembly_graph.nodes[node]["level"]
            ] += (
                    self.assembly_pool.joined_assembly_graph.nodes[node]["count"]
                    ** exponent
            )

        return None

    def set_sw_n_steps(self, level: int = 0, exponent: float = 1.0):
        """
        Computes and assigns sampling weights for different levels based on leaf node counts.
        Used to decide how many assembly steps to take, and ensures that generated molecules
        are of same complexity as molecules in the assembly pool.

        This function retrieves the count of leaf nodes at each level starting from
        `min_level = level + 1` using `get_leaf_counts_per_level()`. It then ensures
        that the levels are continuous (i.e., no level is skipped) and calculates
        sampling weights using an exponentiation method.

        Args:
            level (int, optional): The minimum level from which to start counting leaf nodes.
                                Defaults to 0.
            exponent (float, optional): The exponent applied to the leaf node count to compute
                                        sampling weights. Defaults to 1.0.

        Updates:
            self.n_steps_sampling_weights (list): A list where indices represent levels and
                                                values are computed sampling weights.

        Returns:
            None
        """
        leaf_counts = self.get_leaf_counts_per_level(min_level=level + 1)
        keys = list(leaf_counts.keys())
        keys.sort()
        # make sure no level is skipped (is the case if no leaf node on that level is present)
        continous_keys = np.arange(0, max(keys) + 1, 1)

        self.n_steps_sampling_weights = list(
            leaf_counts[key] ** exponent if key in keys else 0 for key in continous_keys
        )
        return None

    def sample_layer(self, exponent=2.0, curr_depth=0, black_listed_layers=None) -> int:
        """
        Sample a layer based on the sampling weights.

        Args:
            exponent (float, optional): The exponent applied to the node count to compute
                                        sampling weights. Defaults to 2.0.
            curr_depth (int, optional): The current depth in the assembly process. Defaults to 0.
            black_listed_layers (list, optional): A list of layers to exclude from sampling. Defaults to [].

        Returns:
            int: The sampled layer level.
        """
        # sample layer this is done to ensure roughly same molecule size distribution as in the assembly pool
        if black_listed_layers is None:
            black_listed_layers = []
        if not hasattr(self, "layer_sampling_weights"):
            self.set_sw_layer(exponent=exponent)

        layer_ids, layer_weights = [], []
        for item in self.layer_sampling_weights.items():
            if item[0] <= curr_depth and item[0] not in black_listed_layers:
                layer_ids.append(item[0])
                layer_weights.append(item[1])

        return random.choices(
            layer_ids,
            weights=layer_weights,
            k=1,
        )[0]

    def wrs_from_layer(self,
                       base_mol,
                       inverse: bool = False,
                       layer: int = 0,
                       exponent: float = 1.0,
                       blacklist=None,
                       ) -> tuple[str, int]:
        """
        Performs weighted random sampling (WRS) of a fragment from a specified layer.
        The main idea is to ensure roughly the same molecule size distribution as in the assembly pool.

        This function selects a fragment from a given `layer` in the assembly graph
        based on weighted probabilities derived from node counts. The weights are
        exponentiated to control selection bias. Only fragments that can combine
        with `base_mol` (based on atomic compatibility) are considered.

        If `inverse` is `True`, the exponent is negated, effectively inverting
        the weighting bias.

        Args:
            base_mol (str): The base molecule with which the selected fragment
                            must be compatible.
            inverse (bool, optional): If `True`, inverts the exponent to modify
                                    the weighting distribution. Defaults to `False`.
            layer (int, optional): The layer from which fragments should be sampled.
                                Defaults to `0`.
            exponent (float, optional): The exponent applied to fragment counts
                                        to determine their selection weight. Defaults to `1.0`.
            blacklist (list, optional): A list of fragments to exclude from selection.
                                        Defaults to an empty list.

        Returns:
            tuple[str, int]: A tuple containing:
                - `fragment` (str): The randomly chosen fragment from the layer.
                - `num_available_fragments` (int): The number of compatible fragments available.

        Notes:
            - If no compatible fragments are found, the function returns `(None, 0)`.
            - The weighting function accounts for fragment availability and applies
            an exponentiation factor to control bias.
        """
        # sample node from layer
        if blacklist is None:
            blacklist = []
        if inverse:
            exponent = -exponent
        relevant_fragments = self.level_to_fragment[layer]

        # check if available fragments can be combined with base_mol based on available atoms
        layer_fragments = []
        fragment_weights = []
        for node in relevant_fragments:
            if (
                    self.assembly_pool.joined_assembly_graph_minus_x.nodes[node][
                        "atomic_count"
                    ]
                    & self.assembly_pool.joined_assembly_graph_minus_x.nodes[base_mol][
                "atomic_count"
            ]
            ):
                layer_fragments.append(node)
                # Why does this happen when we have set_sw_layer?
                fragment_weights.append(
                    self.assembly_pool.joined_assembly_graph_minus_x.nodes[node][
                        "count"
                    ]
                    ** exponent
                )

        if len(layer_fragments) == 0:
            return None, 0
        fragment = random.choices(layer_fragments, weights=fragment_weights, k=1)[0]

        return fragment, len(layer_fragments)

    def weighted_n_steps_sampler(self, n, min_level=1):
        """
        Sample how many construction steps make for a
        molecule weighted by the number of observed molecules
        in each level in the assembly pool

        Args:
            n : int The number of steps to sample from.
            min_level : int, optional The minimum level to consider. Defaults to 1.

        Returns:
            int: The sampled number of steps.
        """
        return random.choices(
            list(range(1, n + 1)),
            weights=self.n_steps_sampling_weights[min_level + 1:],
            k=1,
        )[0]

    def get_random_fragment_with_level(self, node_level: int) -> str:
        """
        Selects a random fragment from the specified level.

        This function retrieves a list of fragments associated with the given
        `node_level` and returns a randomly selected fragment from that list.

        Args:
            node_level (int): The level from which to select a fragment.

        Returns:
            str: A randomly chosen fragment from the specified level.

        Raises:
            KeyError: If `node_level` is not present in `self.level_to_fragment`.
            IndexError: If the list of fragments at the given level is empty.
        """
        return random.choice(self.level_to_fragment[node_level])

    def combine_fragments_layer(self,
                                fragment1,
                                fragment2,
                                assemble_object,
                                layer=1):
        """
        Combines two molecular fragments to create a new molecule.

        This function takes two molecular fragments, retrieves their molecular
        representations and atomic index mappings, and attempts to bond them
        together using the provided `assemble_object`. If either fragment cannot
        be processed, the function returns `None`.

        Args:
            fragment1 (str): The first fragment to combine, represented as a SMILES string.
            fragment2 (str): The second fragment to combine, represented as a SMILES string.
            assemble_object (object): An object responsible for bonding the molecules,
                                    expected to have a `create_bond` method.
            layer (int, optional): The layer at which the bonding process occurs.
                                Defaults to `1`.

        Returns:
            object | None: The newly formed molecule if bonding is successful,
                        or `None` if either fragment could not be processed.

        Notes:
            - The `assemble_object.create_bond()` method is assumed to handle the
            bonding process and return the resulting molecule.
            - The function calls `self.get_atomtype_index_mapping()` to retrieve
            molecular representations and atomic index mappings for bonding.
        """
        mol1, atomtype_index_mapping1 = get_atom_type_index_mapping(fragment1)
        mol2, atomtype_index_mapping2 = get_atom_type_index_mapping(fragment2)

        if mol1 is None or mol2 is None:
            return None

        return assemble_object.create_bond(
            mol1,
            mol2,
            atomtype_index_mapping1,
            atomtype_index_mapping2,
            layer=layer,
        )

    # This is a beast.
    def random_construct_n_molecules(self,
                                     n,
                                     steps,
                                     X=10,
                                     inverse: bool = False,
                                     exponent: float = 1.0,
                                     layer_exponent: float = 2.0,
                                     step_exponent: float = 1.0,
                                     remove_pathways: bool = False
                                     ):
        """
        Constructs `n` molecules through a stochastic fragment assembly process.

        This function generates molecules by iteratively combining molecular
        fragments, starting from a sampled fragment at a given level and proceeding
        through `steps` of assembly. The fragments are selected based on weighted
        random sampling influenced by various exponent parameters.

        Args:
            n (int): The number of molecules to construct.
            steps (int): The number of assembly steps per molecule.
            X (int, optional): The number of initial fragments used in the assembly pool. Defaults to `10`.
            inverse (bool, optional): If `True`, inverts the exponent used in fragment sampling. Defaults to `False`.
            exponent (float, optional): The exponent controlling fragment selection bias. Defaults to `1.0`.
            layer_exponent (float, optional): The exponent controlling layer sampling bias. Defaults to `2.0`.
            step_exponent (float, optional): The exponent controlling step weighting. Defaults to `1.0`.
            remove_pathways (bool, optional): If `True`, removes pathways from the assembly pool. Defaults to `False`.

        Returns:
            None

        Notes:
            - The function uses `_combine_fragments()` to iteratively assemble molecules.
            - Fragments are sampled from a level-based fragment pool and combined using `self.combine_fragments()`.
            - The number of steps for each molecule is determined by `self.weighted_n_steps_sampler()`.
            - If fragment assembly fails, a blacklist mechanism prevents retrying invalid combinations.
            - Assembled molecules are stored in `self.assembled_molecules` and integrated into the assembly graph.
            - The function resets `self.level_to_fragment()` after every molecule regeneration.
            - `self.construct_diverged_assembly_graph()` is called at the end to finalize the assembly graph.
            """

        def _combine_fragments(fragment1, layer, curr_depth=None):
            """
            Attempts to combine a fragment with another randomly selected fragment from the specified layer.

            Args:
                fragment1 (str): The base fragment for combination.
                layer (int): The layer from which to sample a second fragment.
                curr_depth (int, optional): The current depth in the assembly process.

            Returns:
                tuple[str | None, str | None, int]:
                    - The newly formed molecule (or None if unsuccessful).
                    - The second fragment used in the combination.
                    - The number of available fragments at the given layer.
            """
            fragment2, num_fragments = self.wrs_from_layer(
                base_mol=fragment1,
                inverse=inverse,
                exponent=exponent,
                blacklist=black_list,
                layer=layer,
            )
            if fragment2 is None:
                return None, None, 0
            return (
                self.combine_fragments_layer(fragment1, fragment2, assemble_object, layer=curr_depth),
                fragment2,
                num_fragments,
            )

        assemble_object = Assemble()
        self.set_assembly_pool(X=X, remove_pathways=remove_pathways)

        # Starting level
        level = max(nx.get_node_attributes(self.original_a_minus_X, "level").values())
        self.set_sw_n_steps(level=level, exponent=step_exponent)

        # only relevant, if cont. loss assembly space has depths without observed nodes
        if self.assembly_pool.max_assembly_index - level != steps:
            steps = self.assembly_pool.max_assembly_index - level
        if n is None:
            n = self.num_removed

        # start generation
        # starting_level = list(range(0, level + 1))
        assembled_molecules = 0
        i = 0

        # n * 2 > i is used to make sure that the loop does not run forever. This cutoff is arbitrary
        while assembled_molecules < n:  # and n * 2 > i:
            # Sample level to start from (0 - level) fragment and number of steps
            start_level = level  # random.sample(starting_level, 1)[0]
            fragment1 = self.get_random_fragment_with_level(node_level=start_level)
            n_steps = self.weighted_n_steps_sampler(
                steps + (level - start_level), min_level=start_level
            )
            construction_continues = True

            for j in range(n_steps):
                max_tries = 5_000
                num_fragments = 99  # just has to be larger 1
                black_list = []
                black_listed_layer = []

                while num_fragments >= 1 and construction_continues:
                    max_tries -= 1
                    if max_tries <= 0:
                        try:
                            del self.assembled_molecules[i]
                            self.assembled_molecules[i] = []
                        except KeyError:
                            pass
                        break

                    layer = self.sample_layer(
                        exponent=layer_exponent, curr_depth=start_level + j,  # black_listed_layers=black_listed_layer
                    )

                    molecule, fragment2, num_fragments = _combine_fragments(
                        fragment1, layer, curr_depth=start_level + j
                    )
                    if molecule is not None:
                        if Chem.MolFromSmiles(molecule) is None:
                            continue

                        # update fragment1 and graph
                        self.assembled_molecules[i].append(
                            [fragment1, fragment2, molecule]
                        )
                        self.add_to_assembly_graph([fragment1, fragment2], molecule)
                        fragment1 = molecule
                        break
                    elif num_fragments == 0:
                        black_listed_layer.append(layer)
                    else:
                        black_list.append(fragment2)
                        continue
            if self.assembled_molecules[i]:
                assembled_molecules += 1
            i += 1

            # because regenerating a molecule
            # is done independently of the other molecules
            self.reset_level_to_fragment()
        self.construct_diverged_assembly_graph()
        return None


class Assemble:
    BASE_WEIGHTS = [0.908, 0.075, 0.0137, 0.003]  # Why there are only 4, I do not know.

    """These weights were used to construct drug-like molecules from the JAS of natural products
    # The 10,000 molecules example
    # First layer 
    BASE_WEIGHTS = {
        1: {1: 1.0},
        2: {1: 0.9832214765100671, 2: 0.016778523489932886},
        3: {1: 0.9209621993127147, 2: 0.0782741504390989, 3: 0.0007636502481863307},
        4: {
            1: 0.8330969782916985,
            2: 0.16052143558416057,
            3: 0.0062724991818479325,
            4: 0.00010908694229300753,
        },
        5: {
            1: 0.7996980976722644,
            2: 0.1778307631610546,
            3: 0.02025833233271009,
            4: 0.0021098855858792047,
            5: 0.00010292124809166852,
        },
        6: {
            1: 0.7745189613300952,
            2: 0.18646700817052908,
            3: 0.0330262420482366,
            5: 0.0005211047420531526,
            4: 0.005417522884363908,
            6: 4.9160824721995534e-05,
        },
        7: {
            1: 0.7420528562969427,
            2: 0.20179511951880824,
            3: 0.045524667310748904,
            4: 0.008952223641973385,
            5: 0.0014491850282045689,
            6: 0.00021036556861034063,
            7: 1.5582634711877086e-05,
        },
        8: {
            1: 0.7102899947432977,
            2: 0.21908036913731674,
            3: 0.05607879212662812,
            4: 0.011688861631914023,
            5: 0.0025918462706617604,
            6: 0.00026283511477133344,
            7: 7.300975410314818e-06,
        },
        9: {
            3: 0.06084251634778397,
            1: 0.6904138411237588,
            2: 0.23067025914264955,
            6: 0.00041626301767982564,
            4: 0.014485953015257931,
            5: 0.003103051586340518,
            7: 6.811576652942601e-05,
        },
        10: {
            1: 0.6733680207112318,
            3: 0.06238668274541882,
            2: 0.24377018628302538,
            5: 0.003347866051626287,
            4: 0.01637671504347166,
            6: 0.0006662000455377247,
            7: 8.432911968831957e-05,
        },
        11: {
            1: 0.6580834710008012,
            2: 0.25724856314732564,
            3: 0.06225207490429226,
            6: 0.0006034286618722116,
            4: 0.01753900028687592,
            7: 6.924591201812265e-05,
            5: 0.004204216086814589,
        },
        12: {
            2: 0.26739028524034475,
            1: 0.6444412390465544,
            3: 0.06679648528121357,
            4: 0.01637156972341423,
            6: 0.000781315735698917,
            5: 0.004098902551897396,
            7: 0.00012020242087675646,
        },
        13: {
            1: 0.629783049231136,
            2: 0.2766122303015854,
            3: 0.07062820359995232,
            4: 0.018178567171295745,
            5: 0.003948623197043748,
            7: 0.0001639051138395518,
            6: 0.0006854213851472166,
        },
        14: {
            1: 0.6140666254704792,
            2: 0.2834110443233526,
            3: 0.07621294683819262,
            4: 0.020485740501469955,
            5: 0.004775012639739341,
            6: 0.0009737280677115517,
            7: 7.490215905473475e-05,
        },
        15: {
            3: 0.08426615421570982,
            1: 0.5937785251020898,
            2: 0.2941868844583233,
            4: 0.021859236127792458,
            5: 0.005020417967811675,
            7: 7.20634158059092e-05,
            6: 0.000816718712466971,
        },
        16: {
            2: 0.30645872715816,
            1: 0.5681789540012603,
            3: 0.0924700693131695,
            4: 0.026307498424700693,
            5: 0.006080655324511657,
            6: 0.0005040957781978576,
        },
        17: {
            1: 0.5399424176013063,
            2: 0.3177774912981823,
            4: 0.03162734734218555,
            3: 0.10304671049804477,
            6: 0.0008594387864724335,
            5: 0.006617678655837738,
            7: 8.594387864724336e-05,
            8: 4.297193932362168e-05,
        },
        18: {
            1: 0.5144782825761358,
            2: 0.32757114328507236,
            3: 0.1133924113829256,
            4: 0.03507239141288068,
            5: 0.00798801797304044,
            6: 0.0013729405891163256,
            7: 0.00012481278082875687,
        },
        19: {
            1: 0.4565837749694252,
            4: 0.045556461475743985,
            2: 0.3562984101100693,
            3: 0.12902568283734203,
            5: 0.010497350183448838,
            6: 0.0019364044027721159,
            7: 0.00010191602119853241,
        },
        20: {
            3: 0.17463113851574544,
            2: 0.3811935696983043,
            1: 0.3497027086544814,
            6: 0.003963884606914776,
            4: 0.07201057035895177,
            5: 0.01849812816560229,
        },
    }"""

    def __init__(self) -> None:
        pass

    def select_n_overlaps(self, f1_atoms, f2_atoms, p_combinations_copy, layer=None) -> int:
        """
        Selects the number of atoms that can overlap between two fragments.

        Args:
            f1_atoms : int The number of atoms in fragment 1.
            f2_atoms : int The number of atoms in fragment 2.
            p_combinations_copy : list[list[int, int]] A list of possible atom combinations.
            layer : int, optional The layer of the assembly graph. Defaults to None.

        Returns:
            combinations: list[list[int, int]] A list of atom combinations.
        """
        max_overlap = select_max_overlaps(f1_atoms, f2_atoms)

        # Why is this only relevant for the BASE weights?
        allowed_num_combinations = min(
            count_non_overlapping_sublists(p_combinations_copy),
            max_overlap,
        )

        # Is this saying if BASE_WEIGHTS are defined, it will use them?
        # Better if this is an argument in the function.
        if isinstance(self.BASE_WEIGHTS, dict):
            weights = [
                v
                for k, v in sorted(self.BASE_WEIGHTS.get(layer, self.BASE_WEIGHTS[20]).items())
                if k <= allowed_num_combinations
            ]
        else:
            weights = self.BASE_WEIGHTS[:allowed_num_combinations]

        num_combinations = random.choices(
            range(1, len(weights) + 1), weights=weights, k=1
        )[0]

        combinations = get_allowed_pairs(p_combinations_copy, k=num_combinations)
        return combinations

    def create_bond(
            self,
            fragment1,
            fragment2,
            atomtype_index_mapping1,
            atomtype_index_mapping2,
            layer=None
    ):
        """
        Attempts to create a bond between two molecular fragments.

        This function generates all possible atom pairings for bonding between `fragment1`
        and `fragment2`, selects a subset of these pairings, and attempts to combine the
        fragments. If a valid bond is formed, the new molecule is returned. If no valid
        bond can be created, the function returns `None`.

        Args:
            fragment1 (str): The first molecular fragment, represented as a SMILES string
            fragment2 (str): The second molecular fragment, represented as a SMILES string
            atomtype_index_mapping1 (dict): A mapping of atomic indices to atom types for `fragment1`.
            atomtype_index_mapping2 (dict): A mapping of atomic indices to atom types for `fragment2`.
            layer (int, optional): The layer at which bonding occurs. Defaults to `None`.

        Returns:
            str (SMILES) | None:
                - The newly formed molecule if bonding is successful as a SMILES string.
                - `None` if no valid bond can be created.

        Raises:
            RuntimeError: If the function exceeds the maximum number of bonding attempts (`5000`).

        Notes:
            - The function first determines **all possible atom combinations** for bonding.
            - If no valid combinations exist, it returns `None`.
            - It then iterates through possible combinations, attempting to create a bond.
            - A **fail-safe counter** prevents infinite loops and prints a warning after 5000 attempts.
            - The function calls `self.combine_fragments()` to form the final molecule.
        """

        # Get all possible combinations of atoms
        possible_combinations = get_possible_combinations(atomtype_index_mapping1, atomtype_index_mapping2)
        if possible_combinations is None:
            return None

        p_combinations_copy = possible_combinations[:]
        fail_safe = 0
        while len(p_combinations_copy) > 0 and fail_safe < 5000:
            fail_safe += 1
            if fail_safe == 5000:
                print("fail safe create bond", flush=True)
            f1 = deepcopy(fragment1)
            f2 = deepcopy(fragment2)

            # Select a random combination of atoms
            # These overlaps will not be tried again, if fails
            combinations = self.select_n_overlaps(
                f1.GetNumAtoms(), f2.GetNumAtoms(), p_combinations_copy, layer=layer
            )
            for c in combinations:
                p_combinations_copy.remove(c)
            smiles = combine_fragments(f1, f2, combinations)
            if smiles is None:
                continue
            return smiles
        return None
