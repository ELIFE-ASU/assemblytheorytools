from rdkit import Chem
from rdkit.Chem import Draw

import ast
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Optional
import subprocess
import json
import signal
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

from .seb_pathway_tools import parse_pathway_file_ian, compose_all, assemblyConstruction
from .tools_mol import safe_standardize_mol, standardize_mol


# This needs to change...
# From tools_graph.py?
bond_types = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


class ConstructionObject:

    def reconstruct_joint_assembly_space(self, assembly_out: dict) -> tuple:
    # It will be better to just make the assembly_out BE the CPP output, and not 
    # have CPP as a subdirectory within in.
        """
        Get the estimated joint assembly space of a molecule space.

        Args:
            assembly_out: dict: output of assembly CPP

        Returns:
            joint_assembly_space: list[str]; the joint assembly space of the molecule space
                                as a list of inchi strings
        """

        # Check that output is from CPP
        if "cpp_output" in assembly_out:
            _, object = parse_pathway_file_ian(assembly_out["cpp_output"])
        else:
            raise ValueError(
                "assembly_out should contain 'cpp_output'"
            )

        pathway_log_string = object.pathway_log_string()
        pathway_fragments = object.pathway_inchi_fragments()
        return pathway_fragments, pathway_log_string


class ParsePathwayLog:

    # Conversion to a graph
    global bond_types

    def __init__(self, pathway_log: str):
        self.pathway_log = pathway_log
        (
            self.atom_lines,
            self.buildingblock_lines,
            self.steps_lines,
            self.digraph_lines,
        ) = self.construct_pathway_from_log()

        # build graph and calc levels
        self.G = self.construct_multidigraph()
        self.assembly_out_OK = True
        # check if construction was successful; node should have maximal 2 predecessors.
        # Assembly related problems has duplicate nodes sometimes, that I cant fix
        for node in self.G.nodes:
            if len(list(self.G.in_edges(node))) > 2:
                raise ValueError("Node has more than 2 predecessors - invalid!")

        self.assign_levels()
        self.nodes_per_level = self._nodes_per_level()

    def construct_pathway_from_log(self):
        """
        Parse a pathway log from assembly calculator to a graph.

        Returns:
            atom_lines: list[list]; list of atom types
            buildingblock_lines: list[list]; list of building blocks
            steps_lines: dict; dictionary of steps
            digraph_lines: list[list]; list of digraph lines
        """

        # Parse pathway log
        # Atom log
        def destringyfy(string):
            return ast.literal_eval(string)

        atom_block: bool = False
        buildingblock_block: bool = False
        steps_block: bool = False
        digraph_block: bool = False

        atom_lines = []
        buildingblock_lines = []
        steps_lines = {}
        digraph_lines = []

        for i, line in enumerate(self.pathway_log.split("\n")):
            if line.startswith("#####Graph#####"):
                atom_block = True
                continue
            elif line.startswith("#####Atoms#####"):
                buildingblock_block = True
                atom_block = False
                continue
            elif line.startswith("#####Steps#####"):
                steps_block = True
                buildingblock_block = False
                continue
            elif line.startswith("#####Digraph#####"):
                digraph_block = True
                steps_block = False
                continue

            if atom_block:
                atom_lines.append(destringyfy(line))

            if buildingblock_block:
                buildingblock_lines.append(
                    [line.split("=")[0], destringyfy(line.split("=")[-1])]
                )

            if steps_block:
                steps_lines[line.split("=")[0].replace("step", "")] = (
                    destringyfy(line.split("=")[-1])
                )

            if digraph_block and line != "":
                digraph_lines.append(destringyfy(line))

        return atom_lines, buildingblock_lines, steps_lines, digraph_lines



    def construct_basic_bb(self):
        """
        Construct basic building blocks from the pathway log.

        Returns:
            bb: dict; dictionary of basic building blocks
        """
        bb = {}

        for i, line in enumerate(self.buildingblock_lines):
            edmol = Chem.EditableMol(Chem.Mol())
            id1 = edmol.AddAtom(Chem.Atom(line[1][0][0]))
            id2 = edmol.AddAtom(Chem.Atom(line[1][0][-1]))
            edmol.AddBond(id1, id2, bond_types[line[1][-1]])
            bb["atom" + str(i)] = Chem.MolToSmiles(
                edmol.GetMol()
            )  # key name to match pathway_log format
        return bb




    def construct_fragment_for_step(self, step: str | int):
        """
        Reconstruct a molecule from a step in the pathway log.

        Args:
            step: str; step number
        
        Returns:
            smiles: str; SMILES representation of the molecule
        """
        step = str(step)
        bonds_ids = self.steps_lines[step]
        atom_ids = list(set([atom for bond in bonds_ids for atom in bond]))

        # Get atoms types
        atoms = [self.atom_lines[-2][atom_id] for atom_id in atom_ids]
        bonds = {
            tuple(bond_id): self.atom_lines[-1][
                self.atom_lines[1].index(bond_id)
            ]
            for bond_id in bonds_ids
        }

        # construct molecule
        id_mappig = {
            atom_id: i for i, atom_id in enumerate(atom_ids)
        }  # map atom ids to new ids for bond assignment
        edmol = Chem.EditableMol(Chem.Mol())
        for atom in atoms:
            edmol.AddAtom(Chem.Atom(atom))
        for bond_id, bond_type in bonds.items():
            edmol.AddBond(
                id_mappig[bond_id[0]],
                id_mappig[bond_id[1]],
                bond_types[bond_type],
            )
        return Chem.MolToSmiles(edmol.GetMol())

    def construct_multidigraph(self):
        """
        Construct a multidigraph from the pathway log.

        Returns:
            G: nx.MultiDiGraph; multidigraph of the pathway log
        """
        G = nx.MultiDiGraph()
        smiles_graph = self.construct_basic_bb()
        G.add_nodes_from(smiles_graph.values())

        for i in range(len(self.digraph_lines)):
            # skip every second line
            if (i % 2) - 1 == 0:  # two lines per step
                continue

            smiles_graph[self.digraph_lines[i][-1]] = (
                self.construct_fragment_for_step(
                    self.digraph_lines[i][-1].replace("step", "")
                )
            )

            G.add_node(smiles_graph[self.digraph_lines[i][-1]])
            G.add_edge(
                smiles_graph[self.digraph_lines[i][0]],
                smiles_graph[self.digraph_lines[i][-1]],
            )  # first building block
            G.add_edge(
                smiles_graph[self.digraph_lines[i + 1][0]],
                smiles_graph[self.digraph_lines[i][-1]],
            )  # second building block]
        return G

    def get_level(self, node):
        """
        Returns the level of a node in a graph.

        Args:
            node: str; node in the graph
        
        Returns:
            level: int; level of the node
        """

        preds = [
            edge[0] for edge in list(self.G.in_edges(node))
        ]  # predecessors
        if len(preds) == 2:
            return max([self.G.nodes[pred]["level"] for pred in preds]) + 1
        elif len(preds) == 0:
            return 0
        no_pred = [pred for pred in preds if not "level" in self.G.nodes[pred]]
        try:
            return self.get_level(no_pred[0])
        except Exception:
            return None

    def assign_levels(self):
        """
        Assigns levels to nodes in a graph.

        Returns:    
            None. Updates the graph in place. 
        """

        for node in self.G.nodes:
            if not list(self.G.predecessors(node)):
                self.G.nodes[node].update({"level": 0})
                self.G.nodes[node].update({"assembly_index": 1})
            else:
                _ = self.get_level(node)
                self.G.nodes[node].update({"level": self.get_level(node)})

        return None

    def _nodes_per_level(self):
        """
        Returns a list of nodes in a graph at a given level.

        Returns:
            num_nodes_per_level: dict; number of nodes per level
        """
        num_nodes_per_level = {}

        for node in self.G.nodes:
            level = self.G.nodes[node]["level"]
            if level not in num_nodes_per_level:
                num_nodes_per_level[level] = 1
            else:
                num_nodes_per_level[level] += 1

        return num_nodes_per_level

    def plot_layered_graph(self, show_molecules=False, save_fig=True):
        """
        Plot the layered graph of the molecule.

        Args:
            show_molecules: bool; if True, show the molecules in the graph
            save_fig: bool; if True, save the figure
        
        Returns:
            None. Displays the graph.
        """

        fig, ax = plt.subplots()
        # fig size
        fig.set_size_inches(12, 7)
        cmap = plt.get_cmap("Blues")
        node_colors = [self.G.nodes[node]["level"] for node in self.G.nodes]

        cmap = [cmap(0.4) for i in np.linspace(0.3, 1, len(node_colors))]
        positions = self._get_node_positions()
        nx.draw(
            self.G,
            pos=positions,
            ax=ax,
            with_labels=False,
            node_size=100,
            node_color=cmap,
            connectionstyle="arc3,rad=0.05",
            edge_color="grey",
            width=1,
        )

        if show_molecules:
            for mol, pos in zip(
                    [
                        item[0]
                        for item in sorted(
                        self.G.nodes(data=True), key=lambda x: x[1]["level"]
                    )
                    ],
                    positions.values(),
            ):
                dim = 150
                Draw.MolToFile(
                    Chem.MolFromSmiles(mol),
                    "data/images/temp.png",
                    size=(dim, dim),
                    imageType="png",
                    dpi=300,
                )
                img = plt.imread("data/images/temp.png")
                imagebox = OffsetImage(img, zoom=dim / (fig.dpi * 5.5))
                ab = AnnotationBbox(
                    imagebox, (pos[0], pos[1] + 0.6), frameon=True
                )
                ax.add_artist(ab)

        # Add labels for layers
        layers = list(
            set([self.G.nodes[node]["level"] for node in self.G.nodes])
        )
        max_y = max([positions[node][1] for node in positions])
        """
        for layer in layers:
            ax.text(
                layer * 2,
                max_y + 1,
                f"{layer}",
                fontsize=8,
                ha="center",
                va="center",
            )

            ax.add_line(
                plt.Line2D(
                    (layer * 2 + 1, layer * 2 + 1),
                    (-max_y, max_y + 2),
                    linewidth=1,
                    color="black",
                    linestyle="-",
                    alpha=0.3,
                )
            )

        # heading
        ax.text(
            len(layers),
            max_y + 1.5,
            f"Pathway of {list(self.G.nodes)[-1]}",
            fontsize=10,
            ha="center",
            va="center",
        )"""

        fig.tight_layout()
        if save_fig:
            plt.savefig("data/images/pathway.svg", dpi=200)
        else:
            fig.show()

    def _get_node_positions(self):
        """
        Returns the position of a node in a graph.

        Returns:
            positions: dict; positions of nodes in the graph  
        """

        positions = {}

        # according to layer attribute
        # sort nodes in graph by level
        sorted_nodes = sorted(
            self.G.nodes(data=True), key=lambda x: x[1]["level"]
        )

        current_level: int | None = None
        curr_id: int
        layer_pos: np.ndarray

        for node in sorted_nodes:
            node, _ = node  # first is node name, second is irrelevant

            if self.G.nodes[node]["level"] == current_level:
                if curr_id == 0:
                    curr_id = -1
                else:
                    curr_id = 0

                positions[node] = [
                    self.G.nodes[node]["level"] * 2,
                    layer_pos[curr_id],
                ]
                layer_pos = np.delete(layer_pos, curr_id)

            else:
                current_level = self.G.nodes[node][
                    "level"
                ]  # G.nodes[node]["level"]

                layer_pos = np.linspace(
                    -self.nodes_per_level[current_level],
                    self.nodes_per_level[current_level],
                    self.nodes_per_level[current_level],
                )

                curr_id = 0
                positions[node] = [
                    self.G.nodes[node]["level"] * 2,
                    layer_pos[curr_id],
                ]
                layer_pos = np.delete(layer_pos, curr_id)
        return positions


class Molecule(ConstructionObject):
    """
    A class to represent a molecule.
    """

    def __init__(
            self,
            smiles: str = "",
            pathway: Optional[list[str]] = None,
            assembly_index: Optional[int] = None,
            assembly_output: Optional[dict] = None,
            G: Optional[nx.DiGraph] = None,
            timeout: Optional[int] = 60,
            assembly_version: str = "assemblyCpp",
    ):
        super().__init__()
        self.smiles: Optional[str] = smiles
        self.pathway: Optional[list[str]] = pathway
        self.assembly_index: Optional[int] = assembly_index
        self.assembly_output: Optional[dict] = assembly_output
        self.pathway_log_string: (
            str  # populates after calling reconstruct_pathway
        )
        self.pathway_fragments: list[str]
        self.pathwayLogObj: (
            ParsePathwayLog  # populates after calling construct_layered_graph
        )
        self.G: nx.DiGraph = (
            G  # populates after calling construct_layered_graph
        )
        self.timeout: Optional[int] = timeout
        self.assembly_version: str = assembly_version

        if G is not None:
            self.smiles = list(G.nodes)[-1]

    def __repr__(self):
        """
        Returns a string representation of the Molecule object.

        Returns:
            str: A formatted string representing the Molecule instance.
        """
        return f"Molecule(smiles={self.smiles}, assembly_index={self.assembly_index})"

    def __str__(self):
        """
        Returns a human-readable string representation of the Molecule object.

        This method provides a concise and user-friendly description of the 
        Molecule instance, primarily intended for display purposes.

        Returns:
            str: A formatted string representing the Molecule instance.
        """
        return f"Molecule(smiles={self.smiles}, assembly_index={self.assembly_index})"

    def get_smiles(self) -> str:
        """
        Return the smiles string of this molecule.

        Returns:
            str: The SMILES representation of the molecule.
        """
        if self.smiles == "":
            self.reconstruct_pathway()
            self.construct_layered_graph()
        return self.smiles

    def reconstruct_pathway(self) -> None:
        """
        Reconstructs the molecular assembly pathway.

        If `assembly_output` is None, it attempts to calculate the pathway using `calc_pathway()`. 
        Then, it reconstructs the pathway fragments and log string using `ConstructionObject`.

        This method updates:
            - `self.pathway_fragments`: The fragments forming the pathway.
            - `self.pathway_log_string`: A log of the reconstruction process.

        Raises:
            ValueError: If `self.assembly_output` is still None after attempting to calculate it.
        """

        if self.assembly_output is None:
            print("Warning: `assembly_output` is None. Attempting to calculate pathway...", flush = True)
            self.calc_pathway()
        (
            self.pathway_fragments,
            self.pathway_log_string,
        ) = ConstructionObject().reconstruct_joint_assembly_space(
            self.assembly_output
        )
        return None

    def calculate_assembly(
            self, mol_file_path, set_timeout=64
    ) -> None:
        """
        Calculates the assembly index and pathway of the given molecular string by
        converting a SMILES string to a mol file, then calculating the assembly index
        using assemblyCpp, which have to be installed in the system

        Args:
            mol_file_path: str; path to the mol file
            set_timeout: int; timeout for the assembly calculation
        This method updates:
            - `self.assembly_output`: The output of the assembly calculation.

        """
        executable_path_cpp = "/Users/elife/Desktop/vm_sebs_code2/assemblyCpp"

        if self.assembly_version == "assemblyCpp":
            proc = subprocess.Popen(
                [executable_path_cpp, mol_file_path.parent / mol_file_path.stem],
                stdout=subprocess.DEVNULL,
            )

            try:
                proc.wait(timeout=set_timeout)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)

            output_path = str(mol_file_path.parent / mol_file_path.stem) + "Pathway"

            # Read the file correctly and parse it as JSON
            with open(output_path, "r") as f:
                try:
                    self.assembly_output = {"cpp_output": json.load(f)}  # Parses JSON properly
                except json.JSONDecodeError:
                    # If it's not valid JSON, store as a string (fallback)
                    f.seek(0)  # Reset file pointer
                    self.assembly_output = {"cpp_output": f.read()}

        else:
            raise(ValueError("assembly_version must be 'assemblyCpp'"))

        return None


    def calc_pathway(self) -> None:
        """
        Calculates the assembly pathway of the molecule.

        Returns:
            None. Updates the `self.assembly_output` attribute.
        """
        if self.smiles is None:
            assert ValueError("smiles is None. Cannot calculate pathway.")

        try:
            mol = Chem.MolFromSmiles(self.smiles)
            mol.SetProp("_Name", self.smiles)
            print(
                Chem.MolToMolBlock(mol),
                file=open("".join(["temp", ".mol"]), "w+"),
                flush = True
            )
        except Exception:
            return None

        self.calculate_assembly(Path("temp.mol"), set_timeout=self.timeout)
        return None

    def construct_layered_graph(self):
        """
        Construct the layered graph of this molecule.

        Returns:
            None. This method updates the `self.G` attribute.
        """
        self.pathwayLogObj = ParsePathwayLog(self.pathway_log_string)
        if (
                not self.pathwayLogObj.assembly_out_OK
        ):  # assembly go output was faulty
            self.smiles = None
            return None

        self.G = self.pathwayLogObj.G
        self.smiles = list(self.G.nodes)[-1]
        return None

    def plot_layered_graph(self, show_molecule: bool = True):
        """
        Plot the layered graph of the molecule.

        Args:
            show_molecule: bool; if True, show the molecule in the graph
        
        Returns:
            None. Displays the graph
        """
        self.pathwayLogObj.plot_layered_graph(show_molecule)


class MoleculeSpace(ConstructionObject):
    """
    A class to represent a space of molecules.
    """

    def __init__(self, molecules: list[Molecule]):
        self.molecules: list[Molecule] = molecules
        self.molecule_smiles: list[str] = [
            molecule.get_smiles()
            for molecule in self.molecules
        ]
        self._remove_none()  # remove molecules that failed to layered pathway as indicated by None in self.molecule_smiles
        self.joined_smiles: str = ".".join(self.molecule_smiles)
        self.assembly_out: dict
        self.assembly_pool: list[str]  # list of inchi strings
        self.joint_assembly_log_string: list[str]
        self.MA_assembly_pool: defaultdict = defaultdict(
            str
        )  # list of inchi strings
        self.assembly_graph: dict
        self.joined_assembly_graph: nx.MultiDiGraph
        self.joined_assembly_graph_minus_x: nx.MultiDiGraph

        self.root_nodes: list[str | None] = (
            None  # list of root nodes in joined_assembly_graph
        )
        self.leaf_nodes: list[str | None] = (
            None  # list of leaf nodes in joined_assembly_graph
        )

    def __len__(self):
        """
        Returns the number of molecules in the collection.
        """
        return len(self.molecules)

    def __getitem__(self, idx):
        """
        Retrieves a molecule by its index.

        Args:
            idx (int): The index of the molecule to retrieve.

        Returns:
            object: The molecule at the specified index.
        """
        return self.molecules[idx]

    def __iter__(self):
        """
        Returns an iterator over the molecules in the collection.

        Returns:
            iterator: An iterator over the `molecules` list.
        """
        return iter(self.molecules)

    def _remove_none(self):
        """
        Remove None from self.molecules and self.molecule_smiles

        Returns:
            None. Updates the `self.molecules` and `self.molecule_smiles` attributes.
        """
        # first get none ids
        none_ids = []
        for idx, mol in enumerate(self.molecule_smiles):
            if mol is None:
                none_ids.append(idx)

        # remove none ids
        for idx in sorted(none_ids, reverse=True):
            del self.molecules[idx]
            del self.molecule_smiles[idx]
        return None

    def _set_root_nodes(self):
        """
        Sets the self.root_nodes of the joined_assembly_graph

        Returns:
            None. Updates the `self.root_nodes` attribute.
        """
        self.root_nodes = [
            node
            for node in self.joined_assembly_graph.nodes
            if self.joined_assembly_graph.in_degree(node) == 0
        ]
        return None

    def _set_leaf_nodes(self):
        """
        Sets the self.leaf_nodes of the joined_assembly_graph

        Returns:
            None. Updates the `self.leaf_nodes` attribute.
        """
        if self.molecule_smiles is None:
            self.molecule_smiles = [
                molecule.get_smiles()
                for molecule in self.molecules
            ]
        self.leaf_nodes = self.molecule_smiles
        return None

    def construct_joined_graph(self) -> None:
        """
        Construct the joined graph of this molecule space.
        Only one node of each molecule is included in the joined graph.
        Edges that are in multiple graphs are only included once.

        Returns:
            None. Updates the `self.joined_assembly_graph` attribute.
        """
        self.joined_assembly_graph = compose_all(
            [mol.G for mol in self.molecules]
        )
        self._set_root_nodes()
        self._set_leaf_nodes()
        return None

    def a_minus_x_assembly_pool(
            self, X: int = 1, get_graph: bool = True, remove_paths: bool = False
    ) -> nx.MultiDiGraph | list[str]:
        """
        Compute the A-X assembly pool by removing molecules from the joined assembly graph.
        
        The function removes molecules from `self.joined_assembly_graph` that are within `X`
        steps back in the assembly process. If `remove_paths=True`, associated pathway fragments 
        are also removed. Returns either a modified directed graph or a list of remaining 
        molecule nodes.

        Args:
            X : int, optional (default=1) The number of steps back in the assembly path to remove molecules.
            get_graph : bool, optional (default=True) If True, returns the updated graph.
            remove_paths : bool, optional (default=False) If True, also removes molecular pathway fragments corresponding to the removed nodes.
        
        
        Returns:
            nx.MultiDiGraph | list[str]
                - If `get_graph=True`: Returns the modified directed graph (`nx.MultiDiGraph`).
                - If `get_graph=False`: Returns a list of remaining nodes (molecule identifiers).
        
        Raises:
            ValueError. If `X` is greater than the maximum assembly index, meaning too many steps are requested.

        Notes
        -----
        - The **A-X assembly pool** represents molecules that remain **after removing** fragments that 
        are `X` steps back in the assembly process.
        - **Leaf nodes** (molecules with no descendants in the assembly process) are prioritized for removal.
        - **Pathway removal (`remove_paths=True`)** eliminates fragments from associated pathways if
        they are removed from the assembly pool.
        """
        if self.joined_assembly_graph is None:
            print("Constructing joined assembly graph.", flush = True)
            self.construct_joined_graph()

        self.max_assembly_index = max(
            self.joined_assembly_graph.nodes.data("level"), key=lambda x: x[-1]
        )[-1]

        if X > self.max_assembly_index:
            raise ValueError(
                f"X must be less than or equal to the maximum assembly index {self.max_assembly_index}"
            )

        temp_graph: nx.MultiDiGraph = self.joined_assembly_graph.copy()
        to_remove = [
            node
            for node in self.leaf_nodes
            if temp_graph.nodes[node]["level"] > self.max_assembly_index - X
        ]

        removed_observed = 0
        if remove_paths:
            for node in to_remove:
                # happens sometimes when fragment is twice in a single pathway. Count is already 0
                if not temp_graph.has_node(node):
                    print(f"Node {node} not in graph", flush = True)
                    continue
                try:
                    pw_nodes = self.molecules[
                        self.molecule_smiles.index(node)
                    ].G.nodes
                except Exception as e:
                    print(f"Could not find {node}", e, flush = True)
                    # used to be problem in an old JAS
                    temp_graph.remove_node(node)
                    continue

                for pw_node in pw_nodes:
                    # happens sometimes when fragment is twice in a single pathway. Count is already 0
                    if not temp_graph.has_node(pw_node):
                        continue
                    if temp_graph.nodes[pw_node]["count"] > 1:
                        temp_graph.nodes[pw_node]["count"] -= 1
                    else:
                        temp_graph.remove_node(pw_node)
                removed_observed += 1

        # second loop to remove nodes that are not leaf nodes but have to be removed
        for node in self.joined_assembly_graph.nodes:
            try:
                if (
                        temp_graph.nodes[node]["level"]
                        > self.max_assembly_index - X
                ):
                    temp_graph.remove_node(node)
            except Exception as e:
                ...

        if not remove_paths:
            for leaf in self.leaf_nodes:
                if not temp_graph.has_node(leaf):
                    removed_observed += 1

        self.joined_assembly_graph_minus_x = temp_graph

        if get_graph:
            return temp_graph, removed_observed
        return list(temp_graph.nodes), removed_observed


atom_valence = {
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Si": 4,
    "As": 5,
    "Se": 2,
    "B": 3,
}


class MoleculeGenerationAssemblyPool:
    """
    A class to represent the assembly pool of molecules.
    """

    def __init__(
            self,
            assembly_pool,
    ) -> None:
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
                leaf_counts[level] = 1e-8

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
        atomic_count = []
        try:
            for atom in Chem.MolFromSmiles(child).GetAtoms():
                free_atom_valence = (
                        atom_valence.get(atom.GetSymbol(), 0) - atom.GetExplicitValence()
                )
                # only relevant atoms
                if free_atom_valence > 0:
                    atomic_count.append(atom.GetAtomicNum())
        except AttributeError:
            print(child, flush = True)
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
        self.diverged_assembly_graph = deepcopy(
            self.assembly_pool.joined_assembly_graph_minus_x
        )
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
        # make sure no level is skipped
        # (is the case if no leaf node on that level is present)
        continous_keys = np.arange(0, max(keys) + 1, 1)

        self.n_steps_sampling_weights = list(
            leaf_counts[key] ** exponent if key in keys else 0 for key in continous_keys
        )
        return None

    def sample_layer(self, exponent=2.0, curr_depth=0, black_listed_layers=[]) -> int:
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
        # sample layer this is done to ensure roughly same molecule size
        # distribution as in the assembly pool
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

    def wrs_from_layer(
            self,
            base_mol,
            inverse: bool = False,
            layer: int = 0,
            exponent: float = 1.0,
            blacklist=[],
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

    def combine_fragments(
            self, fragment1, fragment2, assemble_object, layer=1
    ):
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
        mol1, atomtype_index_mapping1 = self.get_atomtype_index_mapping(fragment1)
        mol2, atomtype_index_mapping2 = self.get_atomtype_index_mapping(fragment2)

        if mol1 is None or mol2 is None:
            return None

        return assemble_object.create_bond(
            mol1,
            mol2,
            atomtype_index_mapping1,
            atomtype_index_mapping2,
            layer=layer,
        )

    def get_atomtype_index_mapping(self, fragment):
        """
        Retrieves the molecular representation and atomic index mapping for a given fragment.

        This function converts a SMILES string into an RDKit `Mol` object and creates 
        a mapping of atomic indices to their atomic symbol and free valence count.

        Args:
            fragment (str): The molecular fragment represented as a SMILES string.

        Returns:
            tuple[Chem.Mol, dict[int, list]] | tuple[None, None]: 
                - `mol` (Chem.Mol): The RDKit molecular object representation of the fragment.
                - `atomtype_index_mapping` (dict[int, list]): A dictionary where:
                    - Keys are atomic indices.
                    - Values are lists containing:
                        - The atomic symbol (str).
                        - The free valence count (int).
                - If the fragment cannot be processed, returns `(None, None)`.

        Raises:
            TypeError: If the input fragment is not a valid string.
            Chem.KekulizeException: If RDKit fails to kekulize the molecule.
            Chem.AtomValenceException: If RDKit detects an invalid atomic valence.

        Notes:
            - The function first attempts to convert the SMILES string to an RDKit `Mol` object.
            - Implicit hydrogens are removed to simplify atom valence calculations.
            - Free valence is determined based on predefined valence rules (`atom_valence` dictionary).
        """
        try:
            mol = Chem.MolFromSmiles(fragment, sanitize=False)
            mol = Chem.RemoveHs(mol, implicitOnly=False)
        except (TypeError, Chem.KekulizeException, Chem.AtomValenceException):
            return None, None
        if mol is None:
            return None, None

        atomtype_index_mapping = defaultdict(list)

        for atom in mol.GetAtoms():
            # check for free valence of atoms
            free_atom_valence = (
                    atom_valence.get(atom.GetSymbol(), 0) - atom.GetExplicitValence()
            )
            atomtype_index_mapping[atom.GetIdx()].append(atom.GetSymbol())
            atomtype_index_mapping[atom.GetIdx()].append(free_atom_valence)
        return mol, atomtype_index_mapping

    def random_construct_n_molecules(
            self,
            n,
            steps,
            X=10,
            inverse: bool = False,
            exponent: float = 1.0,
            layer_exponent: float = 2.0,
            step_exponent: float = 1.0,
            remove_pathways: bool = False,
            check_final_molecule: bool = True,
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
            check_final_molecule (bool, optional): If `True`, validates the final constructed molecules using RDKit. Defaults to `True`.

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
                self.combine_fragments(
                    fragment1, fragment2, assemble_object, layer=curr_depth
                ),
                fragment2,
                num_fragments,
            )

        assemble_object = Assemble()
        self.set_assembly_pool(X=X, remove_pathways=remove_pathways)
        # starting level
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
    BASE_WEIGHTS = [0.908, 0.075, 0.0137, 0.003]

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

    def __init__(
            self,
    ) -> None:
        pass

    def select_max_overlaps(self, f1_atoms, f2_atoms) -> int:
        """
        Returns the maximum number of atoms that can overlap between two fragments.

        Args:
            f1_atoms : int The number of atoms in fragment 1.
            f2_atoms : int The number of atoms in fragment 2.
        
        Returns:
            int: The maximum number of atoms that can overlap
        """
        return min(f1_atoms, f2_atoms)

    def select_n_overlaps(
            self, f1_atoms, f2_atoms, p_combinations_copy, layer=None
    ) -> int:
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
        max_overlap = self.select_max_overlaps(f1_atoms, f2_atoms)

        allowed_num_combinations = min(
            self.count_non_overlapping_sublists(p_combinations_copy),
            max_overlap,
        )

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

        combinations = self.get_allowed_pairs(p_combinations_copy, k=num_combinations)
        return combinations

    def count_non_overlapping_sublists(self, lst):
        """
        Counts the number of non-overlapping sublists in a list of lists.

        Args:
            lst : list[list] A list of lists.
        
        Returns:
            counts (int) The number of non-overlapping sublists.
        """
        # Sort the list of lists by the first element of each sublist
        sorted_lst = sorted(lst, key=lambda x: x[0])

        # Initialize a variable to keep track of the number of non-overlapping sublists
        count = 1

        # Loop through the sorted list of lists and compare the second element of each sublist
        # with the first element of the next sublist
        for i in range(len(sorted_lst) - 1):
            if sorted_lst[i][1] <= sorted_lst[i + 1][0]:
                count += 1

        # Return the counter for the number of non-overlapping sublists
        return count

    def get_allowed_pairs(self, combinations, k):
        """
        Returns a list of k allowed pairs of atoms from a list of possible combinations.

        Args:
            combinations : list[list[int, int]] A list of possible atom combinations.
            k : int The number of pairs to select.

        Returns:
            sampled_sublists : list[list[int, int]] A list of k allowed pairs of atoms.
        """
        sampled_sublists = []
        counter = 0

        # Loop until k sublists have been sampled or the counter exceeds a certain number of iterations
        while len(sampled_sublists) < k and counter < 1000:
            # Sample a sublist from the original list
            sublist = random.sample(combinations, 1)[0]

            # Check if the first entry of the sampled sublist overlaps with any previous sublists
            if all(sublist[0] != prev[0] for prev in sampled_sublists):
                # Check if the second entry of the sampled sublist overlaps with any previous sublists
                if all(sublist[1] != prev[1] for prev in sampled_sublists):
                    # Add the sampled sublist to the list of sampled sublists
                    sampled_sublists.append(sublist)
            counter += 1
        return sampled_sublists

    def _valence_check(self, atom1, atom2):
        """
        Checks if the valence of two atoms allows them to be combined.

        Args:
            atom1 : list[str, int] A list containing the atom type and free valence of atom 1.
            atom2 : list[str, int] A list containing the atom type and free valence of atom 2.
        
        Returns:
            bool: True if the atoms can be combined, False otherwise.
        """
        atom_type = atom1[0]
        if (
                atom_valence.get(atom1[0], 0)
                - atom1[1]
                + atom_valence.get(atom2[0], 0)
                - atom2[1]
        ) <= atom_valence.get(atom_type, 0):
            return True
        else:
            return False

    def get_possible_combinations(
            self, atomtype_index_mapping1, atomtype_index_mapping2
    ):
        """
        Returns all possible combinations of atoms that can be combined.
        Possible combinations are atoms with the same atom type and sum of valence
        does not exceed the allowed valence.

        Args:
            atomtype_index_mapping1 : dict A dictionary mapping atom index to atom type and free valence for fragment 1.
            atomtype_index_mapping2 : dict A dictionary mapping atom index to atom type and free valence for fragment 2.
        
        Returns:
            possible_combinations : list[list[int, int]] A list of possible atom combinations.
        """
        possible_combinations = []
        for atom1 in atomtype_index_mapping1:
            for atom2 in atomtype_index_mapping2:
                # check if atom types are the same
                # and if the valence check is passed
                if atomtype_index_mapping1[atom1][0] == atomtype_index_mapping2[atom2][
                    0
                ] and self._valence_check(
                    atomtype_index_mapping1[atom1],
                    atomtype_index_mapping2[atom2],
                ):
                    possible_combinations.append([atom1, atom2])

        if possible_combinations:
            random.shuffle(possible_combinations)
            return possible_combinations
        return None

    def combine_fragments(
            self, fragment1, fragment2, combinations
    ) -> Chem.rdchem.Mol:
        """
        Combines two molecular fragments by forming bonds between specified atom pairs.

        This function takes two RDKit `Mol` objects (`fragment1` and `fragment2`) and 
        attempts to merge them by connecting specified atom pairs. The resulting molecule 
        undergoes structural validation before being returned.

        Args:
            fragment1 (Chem.rdchem.Mol): The first molecular fragment.
            fragment2 (Chem.rdchem.Mol): The second molecular fragment.
            combinations (list[tuple[int, int]]): A list of atom index pairs, where each tuple
                                                represents a bond to be formed between an 
                                                atom in `fragment1` and an atom in `fragment2`.

        Returns:
            Chem.rdchem.Mol | None: 
                - The combined molecule if the bonding is successful and chemically valid.
                - `None` if the molecule fails RDKit compliance checks.

        Raises:
            IndexError: If an invalid atom index is accessed in either fragment.

        Notes:
            - The function starts by retrieving the atom objects based on the given indices.
            - Atom neighbors and bond orders from `fragment2` are stored before merging.
            - `fragment1` and `fragment2` are merged using `Chem.CombineMols()`.
            - The function modifies bonds using `Chem.RWMol()` to ensure proper connectivity.
            - `self.check_rdkit_compliance()` is used to validate the final structure.
        """
        max_idx = fragment1.GetNumAtoms()

        f1_atoms = []
        f2_atoms = []
        f1_atom_bond_parters = []
        f1_atom_bond_orders = []
        # try to combine atoms and check the resulting
        # molecule for chemical validity
        for atom1, atom2 in combinations:
            # choose a random combination of atoms
            f1_atoms.append(fragment1.GetAtomWithIdx(atom1))
            f2_atoms.append(fragment2.GetAtomWithIdx(atom2))

            # get atom2 bond partners and bond orders
            f1_atom_bond_parters.append(f2_atoms[-1].GetNeighbors())
            f1_atom_bond_orders.append(
                [bond.GetBondType() for bond in f2_atoms[-1].GetBonds()]
            )

        # Remove atom2 and add atom2 bond partners to atom1
        # Important that all sanity checks before are valid!!!!
        combine = Chem.CombineMols(fragment1, fragment2)
        rw_mol = Chem.RWMol(combine)
        Chem.RemoveAllHs(rw_mol)

        # combine mols
        rw_mol.BeginBatchEdit()
        for atom_id in range(len(f1_atoms)):
            rw_mol.RemoveAtom(f2_atoms[atom_id].GetIdx() + max_idx)
            for bond, order in zip(
                    f1_atom_bond_parters[atom_id], f1_atom_bond_orders[atom_id]
            ):
                rw_mol.AddBond(
                    f1_atoms[atom_id].GetIdx(),
                    bond.GetIdx() + max_idx,
                    order,
                )
        rw_mol.CommitBatchEdit()

        # Check if molecule is valid
        try:
            mol = rw_mol.GetMol() # Convert editable mol to mol object
            if mol is None:
                return None
        except Exception as _:
            print("Molecule not valid", flush = True)
            return None
        
        # Standardize molecule
        mol = safe_standardize_mol(mol, add_hydrogens=True)
        mol = Chem.RemoveHs(mol)
        return mol



    # def check_rdkit_compilance(self, rw_mol):
    #     """
    #     Checks if the molecule is rdkit compilant

    #     Args:
    #         rw_mol : Chem.RWMol The molecule to check.
        
    #     Returns:
    #         mol: (C)hem.Mol) The molecule if it is compilant, None otherwise.
    #     """
    #     try:
    #         mol = Chem.MolToSmiles(Chem.Mol(rw_mol))
    #         if mol is None:
    #             return None
    #     except Exception as _:
    #         print("Molecule  not valid", flush = True)
    #         return None
    #     return mol

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
            fragment1 (Chem.rdchem.Mol): The first molecular fragment.
            fragment2 (Chem.rdchem.Mol): The second molecular fragment.
            atomtype_index_mapping1 (dict): A mapping of atomic indices to atom types for `fragment1`.
            atomtype_index_mapping2 (dict): A mapping of atomic indices to atom types for `fragment2`.
            layer (int, optional): The layer at which bonding occurs. Defaults to `None`.

        Returns:
            Chem.rdchem.Mol | None:
                - The newly formed molecule if bonding is successful.
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

        # get all possible combinations of atoms
        possible_combinations = self.get_possible_combinations(
            atomtype_index_mapping1, atomtype_index_mapping2
        )
        if possible_combinations is None:
            return None

        p_combinations_copy = possible_combinations[:]  # .copy()
        fail_safe = 0
        while len(p_combinations_copy) > 0 and fail_safe < 5000:
            fail_safe += 1
            if fail_safe == 5000:
                print("fail safe create bond", flush = True)
            f1 = deepcopy(fragment1)
            f2 = deepcopy(fragment2)

            # select a random combination of atoms
            # these overlaps wont be tried again, if fails
            combinations = self.select_n_overlaps(
                f1.GetNumAtoms(), f2.GetNumAtoms(), p_combinations_copy, layer=layer
            )
            for c in combinations:
                p_combinations_copy.remove(c)
            mol = self.combine_fragments(f1, f2, combinations)
            if mol is None:
                continue
            return mol
        return None
