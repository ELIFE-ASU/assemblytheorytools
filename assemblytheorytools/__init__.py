from .assembly import (calculate_assembly_index,
                       calculate_string_assembly_index,
                       joint_correction,
                       load_assembly_output,
                       run_command,
                       calculate_assembly_semi_metric,
                       run_command_simple,
                       compile_assembly_code,
                       assembly_dry_run,
                       add_assembly_to_path)

from .complexity_scores import (molecular_weight,
                                bertz_complexity,
                                wiener_index,
                                balaban_index,
                                randic_index,
                                kirchhoff_index,
                                get_mol_descriptors,
                                tanimoto_similarity,
                                dice_morgan_similarity,
                                get_chirality)

from .find_other_paths import (all_shortest_paths)

from .tools_graph import (nx_to_mol,
                          mol_to_nx,
                          write_ass_graph_file,
                          is_graph_isomorphic,
                          scramble_node_indices,
                          remove_hydrogen_from_graph,
                          get_disconnected_subgraphs,
                          create_ionic_molecule,
                          get_bond_smiles,
                          graph_to_smiles,
                          graph_to_inchi,
                          write_graphml,
                          read_graphml,
                          longest_path_length)

from .tools_mol import (safe_standardize_mol,
                        standardize_mol,
                        smi_to_mol,
                        inchi_to_mol,
                        molfile_to_mol,
                        combine_mols,
                        write_v2k_mol_file,
                        split_mols)

from .tools_file import (file_list_all,
                         write_to_shared_file)

from .tools_cell import (read_cif_file,
                         atoms_to_mol_file,
                         atoms_to_nx)

from .pathway import (get_pathway_to_graph,
                      get_pathway_to_mol,
                      get_pathway_to_inchi,
                      get_pathway_to_smi,
                      get_mol_pathway_to_inchi,
                      get_mol_pathway_to_smi,
                      convert_pathway_dict_to_list)

from .plotting import (n_plot,
                       ax_plot,
                       plot_mol_graph,
                       plot_interactive_graph,
                       plot_graph,
                       plot_digraph,
                       plot_digraph_with_images,
                       plot_digraph_metro)

from .tools_string import (load_fasta,
                           prep_joint_string_ai,
                           get_unique_char)

from .tools_mp import (mp_calc,
                       mp_calc_star,
                       tp_calc)

from .construction import parse_pathway_file

__version__ = "1.3.0"

"""
This module provides various tools and functions for assembly theory analysis, molecular complexity scores, 
graph manipulation, molecular tools, pathway analysis, and plotting.

Submodules:
    - assembly: Functions for calculating assembly index, joint correction, loading assembly output, and running commands.
    - complexity_scores: Functions for calculating molecular weight, Bertz complexity, Wiener index, Balaban index, Randic index, and Kirchhoff index.
    - find_other_paths: Functions for finding all shortest paths in a graph.
    - graph_tools: Functions for converting between NetworkX graphs and RDKit molecules, writing graph files, and checking graph isomorphism.
    - mol_tools: Functions for standardizing molecules, converting between different molecular representations, combining and splitting molecules, and writing V2000 mol files.
    - pathway: Functions for converting pathways to graphs, molecules, and InChI strings.
    - plotting: Functions for plotting molecular graphs, interactive graphs, residue graphs, and multiple graphs in subplots.

Attributes:
    __version__ (str): The version of the module.
"""
