"""Behavioral coverage for molecular scores and graph compression contracts."""

import bz2
import copy
import lzma
import math
import zlib

import networkx as nx
import pytest
from rdkit import Chem

from assemblytheorytools import complexity_scores as scores


@pytest.mark.parametrize(
    "smiles, explicit_hydrogens, randic, kirchhoff",
    [
        ("", False, 0, 0),
        ("C", False, 0, 0),
        ("CCC", False, math.sqrt(2), 4),
        ("C1CC1", False, 1.5, 2),
        ("C", True, 2, 16),
        ("CC", True, 3.25, 58),
        ("CC.C", False, 1, 1.5),
    ],
    ids=["empty", "isolated", "path", "cycle", "methane", "ethane", "disconnected"],
)
def test_graph_indices_on_small_topologies(
    smiles, explicit_hydrogens, randic, kirchhoff
):
    mol = Chem.MolFromSmiles(smiles)
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Trees have resistance equal to distance; the triangle has resistance 2/3
    # per pair. The disconnected case records the existing pseudoinverse result.
    assert scores.randic_index(mol) == pytest.approx(randic)
    assert scores.kirchhoff_index(mol) == pytest.approx(kirchhoff)


@pytest.mark.parametrize("codec", [zlib, bz2, lzma], ids=lambda codec: codec.__name__)
def test_smiles_compression_overhead_and_integrity_flags(codec, monkeypatch, capsys):
    compress = getattr(scores, f"compression_{codec.__name__}_smi")
    mol = Chem.MolFromSmiles("CCO")
    raw_size = compress(mol, add_hydrogens=False, rm_overhead=False)
    net_size = compress(mol, add_hydrogens=False)

    assert raw_size - net_size == len(codec.compress(b""))

    def fail_to_decompress(data):
        raise ValueError("corrupt compressed payload")

    monkeypatch.setattr(codec, "decompress", fail_to_decompress)
    assert compress(mol, add_hydrogens=False, check=False) == net_size
    with pytest.raises(ValueError, match="corrupt compressed payload"):
        compress(mol, add_hydrogens=False, check=True)
    assert "Decompression failed: corrupt compressed payload" in capsys.readouterr().out


@pytest.mark.parametrize("level", [0, 1, 9])
def test_graph_overhead_uses_default_level_even_for_uncompressed_payloads(level):
    graph = nx.path_graph(4)
    raw_size = scores.compression_zlib_graph(graph, level=level, rm_overhead=False)
    net_size = scores.compression_zlib_graph(graph, level=level, rm_overhead=True)

    assert raw_size == len(scores.compress_zlib_graph(graph, level=level))
    assert raw_size - net_size == len(scores.compress_zlib_graph(nx.Graph(), level=9))


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_graph_roundtrip_preserves_kind_and_json_attributes(graph_type):
    graph = graph_type(name="éthanol", provenance={"batches": [1, 2]})
    graph.add_node(1, color="C", labels=["carbon", "first"])
    graph.add_node("oxygen", color="O", charge=-1)
    graph.add_edge(1, "oxygen", color="single", weight=1.5)
    if graph.is_directed():
        graph.add_edge("oxygen", 1, color="reverse")
    if graph.is_multigraph():
        graph.add_edge(1, "oxygen", key="parallel", color="double")
    original = copy.deepcopy(graph)

    restored = scores.decompress_zlib_graph(scores.compress_zlib_graph(graph))

    assert type(restored) is graph_type
    assert nx.utils.graphs_equal(restored, original)
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize(
    "measure", [scores.compression_zlib_graph, scores.compression_ratio_zlib_graph]
)
def test_graph_hydrogen_filtering_preserves_input_and_other_attributes(measure):
    graph = nx.Graph(name="methanol", labels={"source": ["test"]})
    graph.add_nodes_from([(0, {"color": "C"}), (1, {"color": "O"})])
    graph.add_nodes_from((node, {"color": "H"}) for node in range(2, 6))
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], color="single")
    original = copy.deepcopy(graph)
    heavy_graph = graph.subgraph([0, 1]).copy()
    with_hydrogens = measure(graph)

    assert measure(graph, add_hydrogens=False) == measure(heavy_graph)
    assert measure(graph, add_hydrogens=True) == with_hydrogens
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize(
    "measure", [scores.compression_zlib_graph, scores.compression_ratio_zlib_graph]
)
def test_graph_integrity_check_can_be_disabled(measure, monkeypatch, capsys):
    graph = nx.path_graph(4)
    expected = measure(graph)

    def fail_to_decompress(data):
        raise ValueError("corrupt compressed graph")

    monkeypatch.setattr(scores, "decompress_zlib_graph", fail_to_decompress)
    assert measure(graph, check=False) == expected
    with pytest.raises(ValueError, match="corrupt compressed graph"):
        measure(graph, check=True)
    assert "Decompression failed: corrupt compressed graph" in capsys.readouterr().out


def test_descriptor_failure_uses_sentinel_and_continues(monkeypatch, capsys):
    sentinel = object()

    def fail(mol):
        raise ValueError("descriptor unavailable")

    monkeypatch.setattr(
        scores.Descriptors,
        "_descList",
        [
            ("before", lambda mol: mol.GetNumAtoms()),
            ("failed", fail),
            ("after", lambda mol: 7),
        ],
    )

    result = scores.get_mol_descriptors(Chem.MolFromSmiles("CCO"), missingval=sentinel)

    assert result == {"before": 3, "failed": sentinel, "after": 7}
    assert result["failed"] is sentinel
    assert "ValueError: descriptor unavailable" in capsys.readouterr().err


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("CC(=O)N", 2),
        ("O=C(C)N", 2),
        ("CC(=O)O", 2),
        ("O=C(C)O", 2),
        ("CC(=O)C", 4),
        ("O=C(C)C", 4),
        ("CC=N", 2),
        ("O=C=O", 0),
    ],
    ids=[
        "amide",
        "reversed-amide",
        "acid",
        "reversed-acid",
        "ketone",
        "reversed-ketone",
        "imine",
        "carbon-dioxide",
    ],
)
def test_mc2_carbonyl_exclusions_are_independent_of_bond_orientation(smiles, expected):
    assert scores.mc2(Chem.MolFromSmiles(smiles)) == expected


@pytest.mark.parametrize(
    "smiles, substituents, shared_atoms, depths",
    [
        ("C1CCC1", {1: [1, 2], 3: [3, 2]}, {1: 1, 3: 1, 2: 2}, {1: 2, 3: 2}),
        (
            "C1CCCCC1",
            {1: [1, 2, 3], 5: [5, 4, 3]},
            {1: 1, 5: 1, 2: 1, 4: 1, 3: 2},
            {1: 3, 5: 3},
        ),
        ("CCCC.CC", {1: [1, 2, 3]}, {1: 1, 2: 1, 3: 1}, {1: 3}),
    ],
    ids=["four-membered-ring", "six-membered-ring", "disconnected-fragment"],
)
def test_substituent_shells_preserve_order_and_shared_ring_atoms(
    smiles, substituents, shared_atoms, depths
):
    mol = Chem.MolFromSmiles(smiles)
    distances = Chem.GetDistanceMatrix(mol)
    if "." in smiles:
        assert distances[0, -1] > mol.GetNumAtoms()

    actual = scores._determine_atom_substituents(0, mol, distances)

    assert actual == (substituents, shared_atoms, depths)
    assert list(actual[0]) == list(substituents)


def test_chemical_non_equivalence_logs_and_returns_zero_above_four_substituents(capsys):
    mol = Chem.MolFromSmiles("P(F)(F)(F)(F)F")

    value = scores._get_chemical_non_equivs(mol.GetAtomWithIdx(0), mol)

    assert value == 0.0
    assert isinstance(value, float)
    output = capsys.readouterr().out
    assert "Error calculating chemical non-equivalence for atom 0" in output
    assert "IndexError" in output


@pytest.mark.parametrize(
    "text, expected",
    [("", 0.0), ("AAAA", 0.0), ("🧪é🧪é", 1.0)],
    ids=["empty", "repeated-character", "unicode"],
)
def test_shannon_entropy_boundaries(text, expected):
    entropy = scores.shannon_entropy(text)

    assert entropy == expected
    assert isinstance(entropy, float)
