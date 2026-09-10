"""Molecular assembly indices, bounds and reconstructed pathways."""

import json

import networkx as nx
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import assembly


def test_readme_example():
    smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    graph = att.smi_to_nx(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    virt_obj = [att.nx_to_smi(graph, add_hydrogens=False) for graph in virt_obj]

    fig, ax = att.plot_pathway(pathway, plot_type="graph")

    assert ai == 9
    assert pathway.number_of_nodes() > 0
    assert pathway.number_of_edges() > 0
    assert any(
        Chem.MolToSmiles(Chem.MolFromSmiles(vo))
        == Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        for vo in virt_obj
    )
    assert fig.axes == [ax]


@pytest.mark.parametrize("representation", ["mol", "graph"])
def test_acetylene_index_and_fragments(representation):
    mol = att.smi_to_mol("[H]C#C[H]")
    input_data = att.mol_to_nx(mol) if representation == "graph" else mol
    ai, fragments, _ = att.calculate_assembly_index(input_data)
    if representation == "graph":
        fragments = [att.nx_to_smi(graph, add_hydrogens=False) for graph in fragments]

    assert ai == 2
    assert set(fragments) == {"[H]C", "C#C", "[H]C#C", "[H]C#C[H]"}


@pytest.mark.parametrize(
    "smiles", ["c1ccccc1", "[BH-]1-[NH+]=[BH-]-[NH+]=[BH-]-[NH+]=1"]
)
def test_molecular_representations_have_equal_indices(tmp_path, smiles):
    mol = att.smi_to_mol(smiles)
    mol_file = tmp_path / "molecule.mol"
    att.write_v2k_mol_file(mol, str(mol_file))
    inputs = [att.mol_to_nx(mol), Chem.MolFromMolFile(str(mol_file)), mol]

    indices = [att.calculate_assembly_index(value)[0] for value in inputs]

    assert indices[0] >= 0
    assert indices == [indices[0]] * len(inputs)


def test_big_chungus(data_dir):
    mol_file = str(data_dir / "mol_files" / "big_chungus.mol")
    mol = att.molfile_to_mol(mol_file)
    graph = att.mol_to_nx(mol)
    ai_graph, _, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
    ai_mol_file, _, _ = att.calculate_assembly_index(
        Chem.MolFromMolFile(mol_file), strip_hydrogen=True
    )
    ai_mol, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    assert 0 <= ai_graph <= 8
    assert 0 <= ai_mol_file <= 8
    assert 0 <= ai_mol <= 8


@pytest.mark.slow
def test_taxol_file(data_dir):
    mol_file = str(data_dir / "mol_files" / "taxol.mol")
    ai, _, _ = att.calculate_assembly_index(
        Chem.MolFromMolFile(mol_file), timeout=15.0, strip_hydrogen=True
    )
    # actual value is 23, but for timeout this is ok
    assert 23 <= ai <= 24


def test_joint_ass():
    molecules = ["NCC(O)=O", "CC(N)C(O)=O"]
    mols = [att.smi_to_mol(smile) for smile in molecules]
    mol = att.combine_mols(mols)

    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    ref_out = ["CN", "CCN", "CC(N)C(=O)O", "CO", "C=O", "CC", "NCCO", "NCC(=O)O"]

    assert ai == 4
    assert set(virt_obj) == set(ref_out)


@pytest.mark.parametrize("representation", ["mol", "graph"])
def test_joint_index_with_explicit_hydrogens(representation):
    smiles = ["[H]C#C[H]", "CC", "C", "O", "N", "[NH4+]", "[SH-]", "[H][H]"]
    mols = [att.smi_to_mol(value) for value in smiles]
    combined = (
        nx.disjoint_union_all(att.mol_to_nx(mol) for mol in mols)
        if representation == "graph"
        else att.combine_mols(mols)
    )

    assert att.calculate_assembly_index(combined)[0] == 11


def test_jai_self():
    """This validates that JAI does not artificially increase when the
    same molecule is duplicated, ensuring internal deduplication and
    fragment reuse work as expected."""
    molecules = [
        "O=P(O)(O)OC[C@@H](O)[C@@H](O)c1c[nH]c2ccccc12",
        "O=P(O)(O)OC[C@@H](O)[C@@H](O)c1c[nH]c2ccccc12",
    ]
    mols = [att.smi_to_mol(smile) for smile in molecules]
    mol = att.combine_mols(mols)

    jai, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    ai, _, _ = att.calculate_assembly_index(mols[0], strip_hydrogen=True)
    assert jai == ai


def test_joint_index_is_independent_of_input_order():
    mols = [
        att.smi_to_mol(smiles)
        for smiles in ["N[C@@H](CCO)C(=O)O", "O=C(O)CC(C(=O)O)C(O)C(=O)O"]
    ]
    forward = att.calculate_assembly_index(att.combine_mols(mols), strip_hydrogen=True)[
        0
    ]
    reverse = att.calculate_assembly_index(
        att.combine_mols(mols[::-1]), strip_hydrogen=True
    )[0]

    assert forward == reverse >= 0


def test_semi_metric():
    molecules = ["NCC(O)=O", "CC(N)C(O)=O"]
    mols = [att.smi_to_mol(smile) for smile in molecules]
    graphs = [att.mol_to_nx(mol) for mol in mols]
    settings = {
        "strip_hydrogen": True,
        "timeout": 100.0,
    }
    distance = att.calculate_assembly_index_semi_metric(graphs[0], graphs[1], settings)
    assert distance == 1


def test_construction_pathway_smi():

    smi = "CC=O"  # Acetaldehyde
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)
    vo_list_ref = [
        "[H]C[H]",
        "[H]C",
        "C=O",
        "CC",
        "[H]C([H])([H])C",
        "[H]CC([H])([H])[H]",
        "[H]C([H])[H]",
        "[H]C(=O)C([H])([H])[H]",
    ]

    assert ai == 5
    assert len(virt_obj) == len(set(vo_list_ref))
    assert pathway.number_of_nodes() == 8
    assert pathway.number_of_edges() == 9

    assert set(virt_obj) == set(vo_list_ref)


def test_construction_pathway_joint():
    smi = "CC=O.OCC"
    with pytest.warns(UserWarning, match="Disconnected molecules"):
        mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)
    vo_list_ref = [
        "[H]O",
        "[H]OC([H])([H])C([H])([H])[H]",
        "CO",
        "[H]C(=O)C([H])([H])[H]",
        "[H]C",
        "[H]CO[H]",
        "[H]CC",
        "[H]C([H])C",
        "[H]CC([H])([H])[H]",
        "CC",
        "C=O",
        "[H]C([H])([H])C",
        "[H]CO",
    ]

    assert ai == 8
    assert pathway.number_of_nodes() == 13
    assert pathway.number_of_edges() == 16
    assert set(virt_obj) == set(vo_list_ref)


@pytest.fixture
def molecular_ensemble():
    smiles = [
        "[H]OC(=O)C([H])([H])N([H])[H]",
        "[H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]",
        "[H]OC(=O)C([H])([H])N([H])[H]",
        "[H]C([H])([H])C([H])([H])[H]",
        "[H]OC(=O)C([H])([H])N([H])[H]",
    ]
    return [att.smi_to_nx(smi) for smi in smiles]


def test_calculate_assembly_index_parallel(molecular_ensemble):
    graphs = molecular_ensemble
    settings = {"strip_hydrogen": True}
    ai = att.calculate_assembly_index_parallel(graphs, settings)[0]
    ref_list = [3, 4, 3, 0, 3]
    assert ai == ref_list


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_sum_of_assembly_indices(parallel):
    graphs = [att.smi_to_nx(smiles) for smiles in ["c1ccccc1", "c1ccccc1O"]]

    assert (
        att.calculate_sum_assembly_index(
            graphs, {"strip_hydrogen": True}, parallel=parallel
        )
        == 7
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("failed_index", [None, -1], ids=["missing", "timed-out"])
def test_sum_propagates_failed_indices(monkeypatch, parallel, failed_index):
    graphs = [nx.path_graph(3), nx.path_graph(4)]

    def calculate(graph, **settings):
        return (failed_index if graph is graphs[1] else 2), None, None

    monkeypatch.setattr(assembly, "calculate_assembly_index", calculate)
    monkeypatch.setattr(
        assembly, "mp_calc", lambda function, values: list(map(function, values))
    )

    assert att.calculate_sum_assembly_index(graphs, parallel=parallel) == -1


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.parametrize("enforce_exact", [False, True], ids=["bounded", "exact"])
@pytest.mark.parametrize(
    "smiles, expected",
    [
        (["c1ccccc1", "c1ccccc1O"], 0.75),
        ([att.test_mols["glycine"].smiles] * 2, 1.0),
        ([att.test_mols["n-icosane"].smiles] * 2, 1.0),
    ],
    ids=["shared-ring", "identical-amino-acids", "identical-chains"],
)
def test_assembly_similarity(smiles, expected, parallel, enforce_exact):
    graphs = [att.smi_to_nx(value) for value in smiles]
    settings = {"strip_hydrogen": True, "exact": False}

    similarity = att.calculate_assembly_index_similarity(
        graphs, settings=settings, parallel=parallel, enforce_exact_mode=enforce_exact
    )

    assert similarity == pytest.approx(expected)
    assert settings == {"strip_hydrogen": True, "exact": False}


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("enforce_exact", [False, True])
@pytest.mark.parametrize("failure_stage", ["individual", "joint"])
def test_similarity_preserves_failure_sentinel_and_settings(
    monkeypatch, parallel, enforce_exact, failure_stage
):
    calls = []
    graphs = [att.smi_to_nx("CC"), att.smi_to_nx("CCO")]
    settings = {"strip_hydrogen": True, "exact": False}

    def sum_indices(inputs, forwarded, *, parallel):
        assert inputs is graphs
        calls.append(("individual", forwarded, parallel))
        return -1 if failure_stage == "individual" else 1

    def joint_index(graph, **forwarded):
        calls.append(("joint", forwarded, None))
        return -1, None, None

    monkeypatch.setattr(assembly, "calculate_sum_assembly_index", sum_indices)
    monkeypatch.setattr(assembly, "calculate_assembly_index", joint_index)

    assert (
        att.calculate_assembly_index_similarity(
            graphs, settings, parallel=parallel, enforce_exact_mode=enforce_exact
        )
        == -1.0
    )
    expected_settings = {"strip_hydrogen": True, "exact": enforce_exact}
    expected_calls = [("individual", expected_settings, parallel)]
    if failure_stage == "joint":
        expected_calls.append(("joint", expected_settings, None))
    assert calls == expected_calls
    assert settings == {"strip_hydrogen": True, "exact": False}


def test_node_canonicalization():
    graph = nx.Graph()

    # Keep the sparse labels and their insertion order: this graph used to
    # produce the wrong index when the backend skipped canonicalization.
    graph.add_nodes_from([0, 1, 2, 6, 10, 11, 12, 15, 18], color="C")
    graph.add_nodes_from([3, 4, 5, 7, 8, 13, 14, 16, 19], color="O")
    bonds = [
        (18, 19, 2),
        (8, 18, 1),
        (6, 8, 1),
        (6, 7, 2),
        (0, 18, 1),
        (0, 6, 1),
        (0, 1, 1),
        (0, 10, 1),
        (1, 5, 1),
        (1, 2, 1),
        (2, 3, 2),
        (2, 4, 1),
        (4, 15, 1),
        (10, 15, 1),
        (15, 16, 2),
        (10, 11, 2),
        (11, 12, 1),
        (12, 13, 2),
        (12, 14, 1),
    ]
    graph.add_edges_from(
        (left, right, {"color": order}) for left, right, order in bonds
    )

    a, _, _ = att.calculate_assembly_index(graph)

    assert a == 8


@pytest.mark.parametrize(
    "bound",
    [
        att.calculate_assembly_index_upper_bound,
        att.calculate_assembly_index_lower_bound,
    ],
    ids=["upper", "lower"],
)
@pytest.mark.parametrize("representation", ["mol", "graph"])
@pytest.mark.parametrize("strip_hydrogen, expected", [(True, 0), (False, 2)])
def test_acetylene_bounds(bound, representation, strip_hydrogen, expected):
    mol = att.smi_to_mol("[H]C#C[H]")
    input_data = att.mol_to_nx(mol) if representation == "graph" else mol

    assert bound(input_data, strip_hydrogen=strip_hydrogen) == expected


def test_calculate_jo():
    smi = "C1=CC=CC=C1"  # Benzene
    graph = att.smi_to_nx(smi)
    jo = att.calculate_assembly_index_jo(graph)[0]
    assert jo == 6, f"Expected JO to be 6, but got {jo}"


def test_calculate_assembly(molecular_ensemble):
    graphs = molecular_ensemble
    n_i = [1, 2, 3, 4, 5]
    settings = {"strip_hydrogen": True}
    ass = att.calculate_assembly(graphs, n_i, settings)
    ref = 11.87409143815135
    assert ass == ref


def test_hydrogen_stripping_matches_manual_removal(data_dir):
    mol_file = str(data_dir / "mol_files" / "alanine.mol")
    mol = att.smi_to_mol("C[C@@H](C(=O)O)N")
    graph = att.mol_to_nx(mol)
    assert att.is_graph_isomorphic(graph, att.mol_to_nx(att.molfile_to_mol(mol_file)))

    assert att.calculate_assembly_index(att.remove_hydrogen_from_graph(graph))[0] == 4
    for input_data in [graph, mol, Chem.MolFromMolFile(mol_file)]:
        assert att.calculate_assembly_index(input_data, strip_hydrogen=True)[0] == 4


def test_eight_membered_ring_has_index_three():
    graph = nx.cycle_graph(8)
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1, "color")

    assert att.calculate_assembly_index(graph)[0] == 3


@pytest.mark.parametrize(
    "value, expected", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (9998, 16), (9999, 16)]
)
def test_integer_chain(value, expected):
    assert att.calculate_integer_chain(value) == expected


def test_pairwise_joint_pathway_contains_individual_assembly_spaces():
    graphs = [att.smi_to_nx(smiles) for smiles in ["CC(OC)C=C", "CC(OC)C", "CCC"]]
    settings = {"strip_hydrogen": True}
    pathways = att.calculate_assembly_index_parallel(graphs, settings=settings)[-1]
    pairwise = att.calculate_assembly_index_pairwise_joint(graphs, settings=settings)
    direct = att.calculate_assembly_index(att.join_graphs(graphs), **settings)[-1]
    composed = nx.compose_all(pathways)

    assert len(pathways) == len(graphs)
    assert all(
        nx.is_directed_acyclic_graph(pathway)
        for pathway in [*pathways, pairwise, direct]
    )
    assert composed.number_of_nodes() > 0
    assert pairwise.number_of_nodes() >= composed.number_of_nodes()
    # All routes must contain each observed molecule, although intermediates
    # can differ between equally short assembly pathways.
    for graph in graphs:
        expected = att.nx_to_smi(
            att.remove_hydrogen_from_graph(graph), add_hydrogens=False
        )
        for pathway in [pairwise, direct, composed]:
            fragments = {
                att.nx_to_smi(data["vo"], add_hydrogens=False)
                for _, data in pathway.nodes(data=True)
            }
            assert expected in fragments


@pytest.mark.parametrize(
    "edges, fragments, expected",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 0)], [[(0, 1), (1, 2)]], 3),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], [[(1, 2), (2, 3)]], 2),
        ([(0, 1), (1, 2), (2, 3)], [[(0, 1), (1, 2), (2, 3)]], 0),
    ],
    ids=["shared-endpoints", "split-remnant", "empty-remnant"],
)
def test_joining_correction_tracks_overlap_and_components(
    tmp_path, edges, fragments, expected
):
    pathway = tmp_path / "graphPathway"
    pathway.write_text(
        json.dumps(
            {
                "file_graph": [{"Edges": edges}],
                "duplicates": [{"Right": fragment} for fragment in fragments],
            }
        ),
        encoding="utf-8",
    )

    assert assembly._calculate_jo_from_pathway(str(pathway)) == expected


@pytest.mark.parametrize("output", ["valid", "invalid", "missing"])
@pytest.mark.parametrize("retention", [None, "save_dir", "debug", "return_log_file", "telemetry"])
def test_joining_calculation_cleans_output_and_preserves_settings(
    tmp_path, monkeypatch, output, retention
):
    folder = tmp_path / "ai_calc_example"
    folder.mkdir()
    if output != "missing":
        text = json.dumps({"file_graph": [{"Edges": [[0, 1], [1, 2]]}]})
        (folder / "graph_inPathway").write_text(
            text if output == "valid" else "invalid JSON", encoding="utf-8"
        )

    graph = nx.path_graph(3)
    virtual_objects, pathway = ["fragment"], nx.DiGraph()
    forwarded_settings = []

    def calculate(input_graph, **settings):
        assert input_graph is graph
        forwarded_settings.append(settings)
        return 1, virtual_objects, pathway, str(folder / "assembly_output.log")

    monkeypatch.setattr(assembly, "calculate_assembly_index", calculate)
    # Another run must never supply this calculation's pathway or be deleted.
    unrelated = tmp_path / "ai_calc_other"
    unrelated.mkdir()
    monkeypatch.chdir(tmp_path)
    settings = {"save_dir": False, "timeout": 0.5}
    if retention == "telemetry":
        settings["cpp_options"] = att.AssemblyCppOptions(telemetry=True)
    elif retention:
        settings[retention] = True
    original_settings = dict(settings)

    result = assembly.calculate_assembly_index_jo(graph, settings)

    assert settings == original_settings
    assert forwarded_settings == [{**original_settings, "return_log_file": True}]
    assert folder.exists() == (retention is not None)
    assert unrelated.is_dir()
    if output == "valid":
        assert result[0] == 1
        assert result[1] is virtual_objects
        assert result[2] is pathway
    else:
        assert result == (-1, None, None)


def test_joining_trivial_graph_needs_no_calculator_or_pathway(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(assembly, "add_assembly_to_path", lambda: pytest.fail("backend lookup"))
    graph = nx.empty_graph(2)
    nx.set_node_attributes(graph, "C", "color")
    assert att.calculate_assembly_index_jo(graph) == (0, None, None)
    assert not list(tmp_path.iterdir())
