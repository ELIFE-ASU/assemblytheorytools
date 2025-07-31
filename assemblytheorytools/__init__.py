from .assembly import (calculate_assembly_index,
                       calculate_string_assembly_index,
                       joint_correction,
                       load_assembly_output,
                       run_command,
                       calculate_assembly_semi_metric,
                       run_command_simple,
                       compile_assembly_code,
                       assembly_dry_run,
                       add_assembly_to_path,
                       load_assembly_time,
                       calculate_assembly_upper_bound,
                       calculate_assembly_lower_bound,
                       regularise_ai,
                       calculate_assembly_parallel,
                       calculate_sum_assembly,
                       calculate_assembly_similarity,
                       calculate_jo_from_pathway,
                       calculate_jo)
from .complexity_scores import (count_unique_bonds,
                                molecular_weight,
                                bertz_complexity,
                                wiener_index,
                                balaban_index,
                                randic_index,
                                kirchhoff_index,
                                spacial_score,
                                get_mol_descriptors,
                                tanimoto_similarity,
                                dice_morgan_similarity,
                                get_chirality,
                                compression_zlib_smi,
                                compression_bz2_smi,
                                compression_lzma_smi,
                                compress_zlib_graph,
                                decompress_zlib_graph,
                                compression_zlib_graph,
                                compression_ratio_zlib_graph,
                                fcfp4,
                                bottcher,
                                proudfoot,
                                mc1,
                                mc2)
from .construction import parse_pathway_file, assign_levels
from .construction_string import (generate_string_pathway_ian,
                                  get_graph_string_explicit)
from .find_other_paths import (all_shortest_paths)
from .neighborhood_enumeration import (enumerate_neighborhood,
                                       enumerate_up,
                                       enumerate_down)
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
                       plot_pathway_mol,
                       plot_pathway_graph,
                       plot_pathway_atoms,
                       plot_digraph_metro,
                       plot_assembly_circle)
from .reassembler import (assemble,
                          origami,
                          get_num_atom,
                          degree_unsaturation,
                          get_unique_mols,
                          reassemble_old,
                          enumerate_sterioisomers,
                          enumerate_tautomers,
                          enumerate_heterocycles,
                          reassemble)
from .seb_reassembler import (MoleculeGenerationAssemblyPool, Molecule, MoleculeSpace, Assemble)
from .tools_atoms import (smiles_to_atoms,
                          mol_to_atoms,
                          atoms_to_mol,
                          atoms_to_smiles,
                          get_charge,
                          get_spin_multiplicity,
                          cp2k_calc_preset,
                          orca_calc_preset,
                          optimise_atoms,
                          calculate_ccsd_energy,
                          calculate_free_energy,
                          get_virtual_objects_energy)
from .tools_cell import (read_cif_file,
                         atoms_to_mol_file,
                         atoms_to_nx,
                         find_clusters)
from .tools_data import (sample_boostrapping, sample_kde_resampling, sample_importance_sampling)
from .tools_file import (file_list,
                         file_list_all,
                         filter_files,
                         write_to_shared_file,
                         remove_files,
                         wipe_dir,
                         list_subdirs)
from .tools_graph import (nx_to_mol,
                          mol_to_nx,
                          write_ass_graph_file,
                          is_graph_isomorphic,
                          scramble_node_indices,
                          remove_hydrogen_from_graph,
                          get_disconnected_subgraphs,
                          join_graphs,
                          create_ionic_molecule,
                          get_bond_smi,
                          nx_to_smi,
                          smi_to_nx,
                          nx_to_inchi,
                          inchi_to_nx,
                          write_graphml,
                          read_graphml,
                          longest_path_length,
                          relabel_digraph,
                          relabel_identifiers,
                          canonicalize_node_labels,
                          get_graph_charges)
from .tools_mol import (safe_standardize_mol,
                        standardize_mol,
                        get_free_valence,
                        reset_mol_charge,
                        smi_to_mol,
                        inchi_to_mol,
                        molfile_to_mol,
                        combine_mols,
                        write_v2k_mol_file,
                        split_mols,
                        get_element_set_from_mols,
                        standardise_smiles,
                        smi_remove_implicit_hydrogen)
from .tools_mp import (mp_calc,
                       mp_calc_star,
                       tp_calc)
from .tools_string import (load_fasta,
                           prep_joint_string_ai,
                           get_unique_char,
                           generate_random_strings)
from .tools_test import (check_elements,
                         print_graph_details,
                         water_graph,
                         phosphine_graph,
                         ph_2p_graph)

__version__ = "1.8.0"

"""
AssemblyTheoryTools
===================
A comprehensive Python package for assembly theory analysis and molecular graph manipulation.


This package provides a suite of tools and functions for applying assembly theory concepts,
calculating molecular complexity scores, manipulating molecular graphs, analyzing pathways,
and visualizing molecular structures. It integrates with RDKit and NetworkX to provide
powerful molecular and graph analysis capabilities.

Contributors:
------------
Louie Slocombe, Joey Fedrow, Estelle Janin, Gage Siebert, Keith Patarroyo, Ian Seet, Sebastian Pagel, Veronica Mierzejewski, Marina Fernandez-Ruz.

Key Features:
------------
- Assembly theory calculations: Calculate assembly indices, joint corrections, and
  assembly semi-metrics for molecules and strings.
- Molecular complexity measures: Calculate various descriptors like molecular weight,
  Bertz complexity, Wiener index, Balaban index, Randic index, and Kirchhoff index.
- Graph manipulation: Convert between NetworkX graphs and RDKit molecules, check graph
  isomorphism, and modify molecular graphs.
- Molecular tools: Standardize molecules, convert between different molecular
  representations (SMILES, InChI, mol files), and combine/split molecules.
- Pathway analysis: Convert pathways to graphs, molecules, and various string
  representations.
- Visualization: Plot molecular graphs, interactive network diagrams, directed graphs,
  and metro-style pathway visualizations.
- Parallel processing: Multiprocessing tools for efficient calculations on large datasets.
- File handling: Tools for reading/writing various file formats including CIF, GraphML,
  and mol files.
- Data sampling: Various sampling techniques including bootstrapping, KDE resampling,
  and importance sampling.

Submodules:
----------
- assembly: Assembly theory calculations, command execution, and utility functions
- complexity_scores: Molecular complexity metrics and similarity measures
- construction: Pathway construction, parsing, and assembly digraph generation
- construction_string: String pathway generation and manipulation utilities
- find_other_paths: Path-finding algorithms for graphs
- pathway: Pathway analysis and conversion utilities
- plotting: Visualization tools for graphs and molecules
- reassembler: Molecular reassembly and modification utilities
- tools_cell: Crystal structure handling utilities
- tools_data: Statistical sampling and data manipulation utilities
- tools_file: File handling utilities
- tools_graph: Graph manipulation and conversion utilities
- tools_mol: Molecular manipulation and conversion utilities
- tools_mp: Multiprocessing utilities
- tools_string: String processing utilities
- tools_test: Testing and debugging utilities
"""
