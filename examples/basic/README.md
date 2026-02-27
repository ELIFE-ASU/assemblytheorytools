# Assembly Theory Tools – Example Scripts

This repository contains a set of minimal examples showing how to use assemblytheorytools to:

- Compute assembly indices for molecules, strings, and graphs

- Compare objects via assembly similarity and more traditional metrics

- Run calculations in parallel

- Visualise assembly pathways as plots

## 1_simple_molecule_example.py

Converts a benzene SMILES string (c1ccccc1) into a molecule object.

Computes its assembly index with att.calculate_assembly_index(mol).

Prints the assembly index and the associated virtual object.

This is the simplest “hello world” for molecular assembly.

## 2_simple_string_example.py

Takes a single input string (e.g. "abracadabra"). Uses string assembly index calculator to compute the string’s assembly
index.

Prints:

- The assembly index
- The virtual object
- The assembly path

This demonstrates how to treat strings as objects for assembly analysis.

## 3_simple_arb_graph.py

Builds a small undirected networkx graph with node and edge "color" labels.
Draws the graph using matplotlib.
Calls att.calculate_assembly_index(graph) and prints the graph’s assembly index.
This shows how to compute assembly indices on arbitrary labelled graphs.

## 4_joint_pathway.py

Works with two amino acids (glycine and alanine).
Converts SMILES to both networkx graphs and RDKit molecules.
Computes assembly similarity between the two graphs with
att.calculate_assembly_similarity.

Calculates:

- Bertz complexity for each molecule

- Tanimoto similarity between the molecules

Builds a joint molecular object (two molecules separated by "."), then:

- Computes its assembly index

- Extracts its assembly pathway

Visualises the pathway:

- Optional metro-style diagram of the assembly digraph (Linux-only)

- Standard pathway plots (molecule and graph forms), saved as SVG files

This example showcases similarity metrics and joint assembly pathways.

## 5_joint_undirected_string_example.py

Analyses a set of strings, e.g. ["abracadabra", "abra"].

Uses att.calculate_string_assembly_index in undirected "mol" mode to obtain a joint assembly index and pathway shared
across the strings.
This illustrates joint assembly and reuse of substructures across multiple strings.

## 6_parallel_calculations.py

Defines a list of SMILES strings (e.g. glycine, alanine, ethane, with some duplicates).
Converts each SMILES to a graph with att.smi_to_nx.
Runs att.calculate_assembly_index_parallel over all graphs with strip_hydrogen=True.
This demonstrates parallel batch calculation of assembly indices.

## 7_pathway_vis.py

Uses a combined SMILES string for glycine and alanine joined by ".".
Computes the assembly index and pathway for the molecule representation and:
Optionally generates a metro-style digraph plot of the pathway (Linux-only)
Plots and saves a standard pathway visualisation as an SVG file
Converts the same SMILES to a networkx graph, recomputes the assembly index, and:
Plots the pathway with plot_type='graph'
This script focuses on visualising assembly pathways for both molecule and graph representations.

## 8_recursive_ma_example.py

Demonstrates the use of the Mass Spectrometry Molecular Assembly (MA) Estimator.

Creates a complex fragmentation tree.
Estimates the molecular assembly (MA) for a given precursor mass-to-charge ratio (m/z).
Prints the tree structure, tree depth, and the estimated MA.
This script shows how to use the `MAEstimator` for fragmentation tree analysis.
