"""Exercise current native algorithms and Unicode through the public API."""

import networkx as nx
import pytest

import assemblytheorytools as att


@pytest.mark.parametrize("text, expected", [("éééé", 2), ("🧬β🧬β", 2), ("e\u0301e\u0301", 2)])
@pytest.mark.parametrize("algorithm", ["full", "re-pair"])
def test_native_unicode_symbols_and_pathways(text, expected, algorithm):
    ai, objects, pathway = att.calculate_string_assembly_index(
        text, cpp_options=att.AssemblyCppOptions(algorithm=algorithm),
    )
    assert ai == expected
    assert text in objects
    assert nx.is_directed_acyclic_graph(pathway)
    assert len(pathway.nodes) - len(set(text)) == ai


@pytest.mark.parametrize("kind", ["graph", "molecule", "undirected-string", "string"])
@pytest.mark.parametrize("pathway_enabled", [False, True])
def test_native_re_pair_returns_bound_and_usable_pathway(kind, pathway_enabled, capsys):
    options = att.AssemblyCppOptions(algorithm="re-pair", pathway=pathway_enabled)
    if kind == "graph":
        value = nx.path_graph(9)
        nx.set_node_attributes(value, "C", "color")
        nx.set_edge_attributes(value, 1, "color")
        ai, objects, pathway = att.calculate_assembly_index(value, cpp_options=options)
        expected = 3
    elif kind == "molecule":
        ai, objects, pathway = att.calculate_assembly_index(
            att.smi_to_mol("CCCCCCCCC"), strip_hydrogen=True, cpp_options=options,
        )
        expected = 3
    else:
        ai, objects, pathway = att.calculate_string_assembly_index(
            "abababab", directed=kind == "string", cpp_options=options,
        )
        expected = 3
    assert ai == expected
    assert "minimum not proven" in capsys.readouterr().out
    if pathway_enabled:
        assert objects
        assert nx.is_directed_acyclic_graph(pathway)
        assert sum(data["cost"] for _, data in pathway.nodes(data=True)) == ai
        assert len(objects) == pathway.number_of_nodes()
        if kind != "graph":
            assert all(isinstance(value, str) for value in objects)
    else:
        assert objects is pathway is None


def test_native_graph_legacy_repair_selector_and_exact_contract():
    graph = att.smi_to_nx("CCCCCCCCC")
    options = att.AssemblyCppOptions(upper_bound="graph-repair")
    assert att.calculate_assembly_index(graph, strip_hydrogen=True, cpp_options=options)[0] == 3
    assert att.calculate_assembly_index(
        graph, strip_hydrogen=True, cpp_options=options, exact=True,
    )[0] == -1


@pytest.mark.parametrize("joint_corr, expected", [(True, 1), (False, 2)])
def test_native_re_pair_joint_pathway_cost_matches_corrected_bound(joint_corr, expected):
    graph = nx.disjoint_union_all([nx.path_graph(3), nx.path_graph(3), nx.empty_graph(1)])
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1, "color")
    ai, _, pathway = att.calculate_assembly_index(
        graph, joint_corr=joint_corr, cpp_options=att.AssemblyCppOptions(algorithm="re-pair"),
    )
    assert ai == expected
    assert pathway.graph["upper_bound"] == ai
    assert pathway.graph["trivial_upper_bound"] == (2 if joint_corr else 3)
    assert pathway.graph["compensate_disjoint"] is joint_corr
    assert sum(data["cost"] for _, data in pathway.nodes(data=True)) == ai


def test_native_re_pair_reversal_has_zero_cost_operations():
    text = "abcxcba"
    ai, objects, pathway = att.calculate_string_assembly_index(
        text, cpp_options=att.AssemblyCppOptions(algorithm="re-pair", accept_palindromes=True),
    )
    assert ai == 4
    assert text in objects
    assert sum(data["cost"] for _, data in pathway.nodes(data=True)) == ai
    reversals = [data for _, data in pathway.nodes(data=True) if data["operation"] == "reverse"]
    assert reversals and all(data["cost"] == 0 for data in reversals)
