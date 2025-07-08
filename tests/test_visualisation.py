import os
import shutil

import networkx as nx
import numpy as np
import pytest
from ase.io import read
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att

def test_plot_digraph_metro():
    """
    Test that a metro-style plot of an assembly pathway can be
    generated and saved successfully.

    Workflow:
    - Loads a saved pathway graph from disk using `parse_pathway_file()`.
    - Uses `plot_digraph_metro()` to generate a visual diagram.
    - Saves the plot to 'test.png' and 'test.svg'.
    - Verifies plot generation by removing the files afterward.

    Notes:
    - Expects 'data/pathway/tmpPathway' to be a valid pathway file
      (e.g., from a prior call to `att.save_pathway()`).
    - Passes if no exceptions occur during loading, plotting, or file cleanup.
    """
    print(flush=True)
    # Path to saved pathway file (should be a valid .json or .pkl file)
    pathway_str = "data/pathway/tmpPathway"

    # Load the assembly pathway graph
    digraph = att.parse_pathway_file(pathway_str)
    digraph, _ = digraph  # Unpack the graph and VO list (we only need the graph)

    # Plot the pathway as a metro-style graph and save to files
    att.plot_digraph_metro(digraph, filename="test")

    # Clean up generated files
    os.remove("test.png")
    os.remove("test.svg")

    pass


def test_circle_plot():
    """
    Test the plot_assembly_circle function from the att module.

    Steps:

    1. Define a set of example node names.
    2. Set visualization parameters.
    3. Call att.plot_assembly_circle with the specified parameters.
    4. Assert that the output file is created.

    The test will fail if the file is not generated.
    """

    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']

    labels = True
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.png'
    att.plot_assembly_circle(nodes,
                             labels=labels,
                             node_size=node_size,
                             arrow_size=arrow_size,
                             node_color=node_color,
                             edge_color=edge_color,
                             fig_size=fig_size,
                             filename=filename)

    assert os.path.isfile('circle_plot.png'), "Failed to generate the file."
    os.remove('circle_plot.png')

