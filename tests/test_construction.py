import networkx as nx
import pytest
from rdkit import Chem

import assemblytheorytools as att


def test_assign_levels():
    """
    Test the `assign_levels` function with a directed graph.

    This function performs the following steps:
    1. Creates a directed graph.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes and adds them to the graph.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes with their levels and add them to the graph
    nodes = {"CC": 0, "C=C": 0, "CO": 0, "CC=C": 1, "OCC=C": 2}
    graph.add_nodes_from(nodes)

    # Define edges and add them to the graph
    edges = [("CC", "CC=C"), ("C=C", "CC=C"), ("CO", "OCC=C"), ("CC=C", "OCC=C")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node]["level"] == level, \
            f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_linear_chain():
    """
    Test the `assign_levels` function with a linear chain graph.

    This function performs the following steps:
    1. Creates a directed graph representing a linear chain of nodes.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes to form a linear chain.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node][
                   "level"] == level, f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_empty_graph():
    """
    Test the `assign_levels` function with an empty graph.

    This function performs the following steps:
    1. Creates an empty directed graph.
    2. Calls the `assign_levels` function on the empty graph.
    3. Asserts that the graph remains empty after the function call.

    Asserts:
        - The graph has no nodes after calling `assign_levels`.
    """
    print(flush=True)
    # Create an empty directed graph
    graph = nx.DiGraph()
    # Assign levels to the empty graph
    att.assign_levels(graph)
    # Verify that the graph has no nodes
    assert len(graph.nodes) == 0, "Empty graph should have no nodes."


def test_convert_digraph_vo_to_target():
    """
    Test the conversion of a digraph's virtual objects to target representations.

    This function performs the following steps:
    1. Defines the SMILES string for diethyl phthalate.
    2. Converts the SMILES string to a NetworkX graph.
    3. Calculates the assembly pathway for the graph.
    4. Converts the virtual objects in the pathway to their target representations.
    5. Extracts the SMILES strings of the converted virtual objects.
    6. Asserts that the extracted SMILES strings match a reference list.

    Asserts:
        - The extracted SMILES strings match the reference list.

    Notes
    -----
    The final (whole-molecule) reference SMILES is a non-canonical RDKit
    serialization, not the assembly pathway's own choice of representation.
    Its exact ring-traversal direction can shift between RDKit versions even
    though the molecule is unchanged -- confirmed by canonicalizing both
    forms: ``Chem.MolToSmiles`` and ``Chem.MolToInchi`` agree that
    ``'CCOC(=O)C1=CC=CC=C1C(=O)OCC'`` (this reference) and
    ``'CCOC(=O)C1=C(C(=O)OCC)C=CC=C1'`` (an older RDKit's output) are the
    same molecule, diethyl phthalate.
    """
    # The conversion logic does not depend on PubChem. Keeping this structure
    # local makes the regression deterministic and leaves service access to the
    # explicitly marked integration tests in test_tools_data.py.
    smi = 'CCOC(=O)C1=CC=CC=C1C(=O)OCC'
    print(f"SMILES: {smi}", flush=True)
    graph = att.smi_to_nx(smi, sanitize=True, add_hydrogens=True)

    pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)[2]
    pathway = att.convert_digraph_vo_to_target(pathway)
    smis = []
    for node in pathway.nodes():
        smis.append(pathway.nodes[node]['vo'])
    print(smis, flush=True)
    ref_smi = ['CC',
               'CCO',
               'CO',
               'C=O',
               'CC(=O)O',
               'C=CC(=O)O',
               'C=C',
               'CC=CC(=O)O',
               'CC=CC(=O)OCC',
               'CC=C(C)C(=O)OCC',
               'C=CC=C(C)C(=O)OCC',
               'CCOC(=O)C1=CC=CC=C1C(=O)OCC']

    assert att.check_elements(smis, ref_smi)


def test_get_vos_on_layer():
    """
    Test the retrieval of virtual objects (VOs) from specific layers of an assembly pathway.

    This function performs the following steps:
    1. Defines a list of SMILES strings.
    2. Converts the SMILES strings to NetworkX graphs and combines them.
    3. Calculates the assembly pathway for the combined graph.
    4. Retrieves VOs from layer 0 and asserts the count.
    5. Retrieves VOs from a range of layers (0 and 1) and asserts the count.
    6. Retrieves all VOs from the pathway and asserts the count.

    Asserts:
        - The number of VOs on layer 0 is 3.
        - The number of VOs on layers 0 and 1 is 2.
        - The total number of VOs on all layers is 4.
    """
    print(flush=True)
    smis = ['CC(OC)C=C',
            'CC(OC)C',
            'CCC']
    graphs = [att.smi_to_nx(smi) for smi in smis]
    # combine the graphs into one graph
    combined = att.join_graphs(graphs)
    pathway = att.calculate_assembly_index(combined, strip_hydrogen=True)[-1]
    vos_layer_0 = att.get_vos_on_layer(pathway, 0)
    print("VOs on layer 0:", vos_layer_0, flush=True)
    assert len(vos_layer_0) == 3

    vos_layer_range = att.get_vos_on_layer(pathway, [0, 1])
    print("VOs on layers 0 and 1:", vos_layer_range, flush=True)
    assert len(vos_layer_range) == 2

    vos_layer_all = att.get_vos_on_layer(pathway, 'all')
    print("VOs on all layers:", vos_layer_all, flush=True)
    assert len(vos_layer_all) == 4


def test_parse_pathway_dot(data_dir):
    """
    Test parsing a Rust-backend assembly pathway from its DOT representation.

    This function performs the following steps:
    1. Loads the anthracene mol file and the DOT pathway computed from it.
    2. Parses the pathway into a graph with `parse_pathway_dot`.
    3. Inspects the node and edge attributes it produced.

    Asserts:
        - The pathway is a MultiDiGraph with 8 nodes and 12 edges.
        - Nodes are integers carrying type, bonds, label and vo attributes.
        - The virtual objects build up from single bonds to anthracene.
        - Edges carry the bond indices their source fragment occupies.
    """
    print(flush=True)
    mol = att.molfile_to_mol(str(data_dir / "mol_files" / "anthracene.mol"),
                             add_hydrogens=False)
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()

    pathway = att.parse_pathway_dot(dot, mol=mol)
    print("Pathway:", pathway, flush=True)

    assert isinstance(pathway, nx.MultiDiGraph)
    assert pathway.number_of_nodes() == 8
    assert pathway.number_of_edges() == 12
    assert all(isinstance(node, int) for node in pathway.nodes)
    assert all(data["type"] == "virtual_object" for _, data in pathway.nodes(data=True))

    # Fragments are kekulised, matching the graph the backend searches
    vos = [pathway.nodes[node]["vo"] for node in sorted(pathway.nodes)]
    print("Virtual objects:", vos, flush=True)
    assert vos == ['CC', 'C=C', 'C=CC', 'CC=CC', 'C=CC=CC', 'CC=CC=CC=CC',
                   'CC=CC1=CC=CC=C1', 'C1=CC=C2C=C3C=CC=CC3=CC2=C1']

    # The root node covers every bond, the elementary parts exactly one
    assert pathway.nodes[7]["bonds"] == frozenset(range(mol.GetNumBonds()))
    assert pathway.nodes[7]["label"] == "{" + ", ".join(str(i) for i in range(16)) + "}"
    assert pathway.nodes[0]["bonds"] == frozenset({14})

    assert pathway[0][2][0]["bonds"] == frozenset({14})
    assert pathway[2][3][0]["bonds"] == frozenset({14, 15})


def test_parse_pathway_dot_vo_types(data_dir):
    """
    Test the virtual object representations offered by `parse_pathway_dot`.

    This function performs the following steps:
    1. Loads the anthracene mol file and its DOT pathway.
    2. Parses the pathway once per supported vo_type.
    3. Parses it again without a molecule.

    Asserts:
        - 'mol', 'graph', 'smiles' and 'inchi' give the expected payload types.
        - Omitting the molecule falls back to the bond-set label.
        - The caller's molecule is not modified.
    """
    print(flush=True)
    mol = att.molfile_to_mol(str(data_dir / "mol_files" / "anthracene.mol"),
                             add_hydrogens=False)
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()
    before = Chem.MolToSmiles(mol)

    assert isinstance(att.parse_pathway_dot(dot, mol=mol, vo_type="mol").nodes[2]["vo"],
                      Chem.Mol)
    assert isinstance(att.parse_pathway_dot(dot, mol=mol, vo_type="graph").nodes[2]["vo"],
                      nx.Graph)
    assert att.parse_pathway_dot(dot, mol=mol, vo_type="smiles").nodes[2]["vo"] == "C=CC"
    assert att.parse_pathway_dot(dot, mol=mol, vo_type="inchi").nodes[7]["vo"].startswith(
        "InChI=1S/C14H10")

    # Without a molecule the pathway still carries its structure
    bare = att.parse_pathway_dot(dot)
    print("Bare virtual object:", bare.nodes[2], flush=True)
    assert bare.nodes[2]["vo"] == "{14, 15}"
    assert bare.nodes[2]["bonds"] == frozenset({14, 15})

    assert Chem.MolToSmiles(mol) == before, "parse_pathway_dot modified the caller's molecule"


def test_parse_pathway_dot_errors(data_dir):
    """
    Test that `parse_pathway_dot` rejects malformed input.

    This function performs the following steps:
    1. Builds a series of invalid DOT strings and arguments.
    2. Checks that each raises a ValueError.
    3. Checks that strict=False lets bookkeeping violations through.

    Asserts:
        - Non-DOT input, undirected graphs, malformed labels, missing labels,
          non-integer node names, out-of-range bonds, unknown vo_types and
          broken bond bookkeeping all raise ValueError.
        - strict=False parses a graph whose bookkeeping does not add up.
    """
    print(flush=True)
    mol = att.molfile_to_mol(str(data_dir / "mol_files" / "anthracene.mol"),
                             add_hydrogens=False)
    mismatched = 'digraph { 0 [ label = "{1}" ]\n1 [ label = "{2, 3}" ]\n' \
                 '0 -> 1 [ label = "{2}" ] }'

    cases = {
        "not DOT at all": dict(dot="hello world"),
        "undirected": dict(dot='graph { 0 [ label = "{1}" ] }'),
        "malformed label": dict(dot='digraph { 0 [ label = "nope" ] }'),
        "missing label": dict(dot="digraph { 0 }"),
        "non-integer node": dict(dot='digraph { a [ label = "{1}" ] }'),
        "bond out of range": dict(dot='digraph { 0 [ label = "{99}" ] }', mol=mol),
        "unknown vo_type": dict(dot=mismatched, vo_type="banana"),
        "inputs do not add up": dict(dot=mismatched),
        "edge too large": dict(dot='digraph { 0 [ label = "{1}" ]\n'
                                   '1 [ label = "{2, 3}" ]\n'
                                   '0 -> 1 [ label = "{2, 3}" ] }'),
    }
    for name, kwargs in cases.items():
        print("Checking:", name, flush=True)
        with pytest.raises(ValueError):
            att.parse_pathway_dot(**kwargs)

    relaxed = att.parse_pathway_dot(mismatched, strict=False)
    print("Relaxed pathway:", relaxed, flush=True)
    assert relaxed.number_of_nodes() == 2


def test_parse_pathway_dot_assign_levels(data_dir):
    """
    Test that a parsed Rust pathway works with the rest of the pathway tools.

    This function performs the following steps:
    1. Parses the anthracene DOT pathway.
    2. Re-inserts its nodes in topological order, as `assign_levels` requires.
    3. Assigns levels and reads the layers back.

    Asserts:
        - Every node is assigned a level.
        - The elementary parts sit at level 0 and the target at the deepest
          level.
    """
    print(flush=True)
    mol = att.molfile_to_mol(str(data_dir / "mol_files" / "anthracene.mol"),
                             add_hydrogens=False)
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()
    pathway = att.parse_pathway_dot(dot, mol=mol)

    ordered = nx.MultiDiGraph()
    ordered.add_nodes_from((n, pathway.nodes[n]) for n in nx.topological_sort(pathway))
    ordered.add_edges_from(pathway.edges(data=True))

    att.assign_levels(ordered)
    levels = {node: data["level"] for node, data in ordered.nodes(data=True)}
    print("Levels:", levels, flush=True)

    assert levels[0] == 0 and levels[1] == 0
    assert levels[7] == max(levels.values())
