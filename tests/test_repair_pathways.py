"""Replay native Re-Pair certificates, including orientations and attachment sites."""

import copy
import json

import networkx as nx
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools.construction import parse_pathway_file, parse_string_pathway_file


def write_certificate(tmp_path, data):
    source = tmp_path / "pathway.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    return source


def graph_certificate(*, disconnected=False, compensate=False):
    # Two uses of a two-bond chain, whose sole production reuses a bond twice.
    edges = ([[0, 1, 1], [1, 2, 1], [3, 4, 1], [4, 5, 1]] if disconnected
             else [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1]])
    return {
        "schema": "graph-repair-assembly-v1", "upper_bound": 2 - int(disconnected and compensate),
        "trivial_upper_bound": 3 - int(disconnected and compensate), "rule_count": 1,
        "remaining_fragments": 2, "components": 2 if disconnected else 1,
        "compensate_disjoint": compensate, "atoms": ["C"] * (6 if disconnected else 5),
        "edges": edges, "terminals": [{"id": 0, "edges": [0]}],
        "rules": [{"id": 1, "left": 0, "right": 0, "left_edges": [0],
                   "right_edges": [1], "edges": [0, 1]}],
        "residual": [{"symbol": 1, "edges": [0, 1]}, {"symbol": 1, "edges": [2, 3]}],
    }


def string_certificate():
    # Exported by native Re-Pair for abcxcba with reversal equivalence enabled.
    return {
        "schema": "string-repair-assembly-v1", "upper_bound": 4,
        "trivial_upper_bound": 6, "rule_count": 2, "remaining_fragments": 3,
        "accept_reversed": True, "length": 7, "input": "abcxcba",
        "terminals": [{"id": i, "code_point": ord(c)} for i, c in enumerate("abcx")],
        "rules": [
            {"id": 4, "left": 0, "right": 1, "left_reversed": False,
             "right_reversed": False, "length": 2},
            {"id": 5, "left": 4, "right": 2, "left_reversed": False,
             "right_reversed": False, "length": 3},
        ],
        "residual": [
            {"symbol": 5, "reversed": False, "offset": 0, "length": 3},
            {"symbol": 3, "reversed": False, "offset": 3, "length": 1},
            {"symbol": 5, "reversed": True, "offset": 4, "length": 3},
        ],
    }


def assert_bound(path, bound):
    assert nx.is_directed_acyclic_graph(path)
    assert path.graph["upper_bound"] == bound
    assert path.graph["minimum_proven"] is False
    assert sum(data["cost"] for _, data in path.nodes(data=True)) == max(0, bound)


@pytest.mark.parametrize("vo_type", ["graph", "smiles", "mol", "inchi"])
def test_graph_repair_reconstructs_bond_placements(tmp_path, vo_type):
    source = write_certificate(tmp_path, graph_certificate())
    path, objects, log = parse_pathway_file(source, vo_type=vo_type, log=True)

    assert_bound(path, 2)
    assert len(objects) == 3
    assert path.edges["virtual_object_0", "step_1"]["occurrences"] == [
        frozenset({0}), frozenset({1})]
    assert path.edges["step_1", "step_2"]["occurrences"] == [
        frozenset({0, 1}), frozenset({2, 3})]
    final = path.nodes["step_2"]["vo"]
    if vo_type == "graph":
        assert nx.is_isomorphic(final, nx.path_graph(5))
    elif vo_type == "mol":
        assert isinstance(final, Chem.Mol)
        assert final.GetNumBonds() == 4
    elif vo_type == "smiles":
        assert Chem.MolFromSmiles(final).GetNumBonds() == 4
    else:
        assert final.startswith("InChI=")
    assert "minimum not proven" in log


@pytest.mark.parametrize("compensate", [False, True])
def test_graph_repair_disjoint_cost_and_isolated_atoms(tmp_path, compensate):
    certificate = graph_certificate(disconnected=True, compensate=compensate)
    certificate["atoms"].append("He")
    path, _ = parse_pathway_file(write_certificate(tmp_path, certificate), vo_type="graph")

    assert_bound(path, 1 if compensate else 2)
    assert path.nodes["step_2"]["operation"] == "combine_components"
    assert nx.number_connected_components(path.nodes["step_2"]["vo"]) == 2
    assert path.graph["target"].nodes[6]["color"] == "He"


def test_graph_repair_keeps_high_integer_colors_without_input_graph(tmp_path):
    certificate = graph_certificate()
    for edge in certificate["edges"]:
        edge[2] = 17
    path, _ = parse_pathway_file(write_certificate(tmp_path, certificate), vo_type="graph")
    assert {d["color"] for _, _, d in path.nodes["step_2"]["vo"].edges(data=True)} == {17}


def test_graph_repair_empty_bond_set(tmp_path):
    certificate = graph_certificate()
    certificate.update(atoms=["He"], edges=[], terminals=[], rules=[], residual=[],
                       upper_bound=0, trivial_upper_bound=0, components=0,
                       rule_count=0, remaining_fragments=0)
    path, objects = parse_pathway_file(write_certificate(tmp_path, certificate), vo_type="graph")
    assert_bound(path, 0)
    assert not path and objects == []
    assert path.graph["target"].nodes[0]["color"] == "He"


@pytest.mark.parametrize("topology", ["cycle", "branched", "disconnected"])
def test_native_graph_repair_reconstructs_heterogeneous_graphs(topology):
    target = {"cycle": nx.cycle_graph(8), "branched": nx.balanced_tree(2, 3),
              "disconnected": nx.disjoint_union(nx.cycle_graph(6), nx.path_graph(5))}[topology]
    nx.set_node_attributes(target, {node: "C" if node % 2 else "N" for node in target}, "color")
    nx.set_edge_attributes(target, {edge: 1 + i % 2 for i, edge in enumerate(target.edges)}, "color")
    bound, objects, path = att.calculate_assembly_index(
        target, joint_corr=False, cpp_options=att.AssemblyCppOptions(algorithm="re-pair"),
    )
    assert_bound(path, bound)
    final = max(objects, key=lambda graph: graph.number_of_edges())
    assert nx.is_isomorphic(
        final, target,
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
    )


@pytest.mark.parametrize("mutation", ["cost", "placement", "partition", "cycle", "label", "color"])
def test_graph_repair_rejects_invalid_constructions(tmp_path, mutation):
    certificate = graph_certificate()
    if mutation == "cost":
        certificate["upper_bound"] = 1
    elif mutation == "placement":
        certificate["residual"][1]["edges"] = [0, 1]
    elif mutation == "partition":
        certificate["rules"][0]["right_edges"] = [0]
    elif mutation == "cycle":
        certificate["rules"][0]["left"] = 1
    elif mutation == "label":
        certificate["atoms"][-1] = "N"
    else:
        certificate["edges"][-1][-1] = 2
    with pytest.raises(ValueError, match="Re-Pair pathway certificate"):
        parse_pathway_file(write_certificate(tmp_path, certificate), vo_type="graph")


def test_string_repair_uses_certificate_reversal_mode(tmp_path):
    source = write_certificate(tmp_path, string_certificate())
    objects, path = parse_string_pathway_file(source)

    assert_bound(path, 4)
    assert objects == list(path)
    assert path.edges["abc", "cba"] == {"operation": "reverse", "cost": 0}
    assert path.nodes["abcxcba"]["vo"] == "abcxcba"
    for _, attributes in path.nodes(data=True):
        if attributes["operation"] == "concatenate":
            assert "".join(path.nodes[p]["vo"] for p in attributes["operands"]) == attributes["vo"]


def test_string_repair_reversed_rule_operands_and_unicode(tmp_path):
    certificate = string_certificate()
    certificate["input"] = "λ🧬x!x🧬λ"
    certificate["terminals"] = [{"id": i, "code_point": ord(c)} for i, c in enumerate("λ🧬x!")]
    certificate["rules"][0].update(left=1, right=0)
    certificate["rules"][1]["left_reversed"] = True
    source = write_certificate(tmp_path, certificate)
    objects, path = parse_string_pathway_file(source)

    assert_bound(path, 4)
    assert certificate["input"] in objects
    assert path.nodes["λ🧬"]["operation"] == "reverse"
    assert path.nodes["x🧬λ"]["operation"] == "reverse"


@pytest.mark.parametrize("string", ["", "λ"])
def test_string_repair_zero_operations(tmp_path, string):
    data = {
        "schema": "string-repair-assembly-v1", "input": string, "length": len(string),
        "upper_bound": len(string) - 1, "trivial_upper_bound": len(string) - 1,
        "rule_count": 0, "remaining_fragments": len(string), "accept_reversed": False,
        "terminals": [{"id": 0, "code_point": ord(string)}] if string else [], "rules": [],
        "residual": [{"symbol": 0, "reversed": False, "offset": 0, "length": 1}] if string else [],
    }
    objects, path = parse_string_pathway_file(write_certificate(tmp_path, data))
    assert objects == list(string)
    assert_bound(path, len(string) - 1)


def test_string_repair_retains_separate_paid_constructions(tmp_path):
    # A valid (though less compressed) grammar can build the same expanded
    # value twice. Preserve the certificate's cost and the DAG in that case.
    data = {
        "schema": "string-repair-assembly-v1", "input": "abab", "length": 4,
        "upper_bound": 3, "trivial_upper_bound": 3, "rule_count": 2,
        "remaining_fragments": 2, "accept_reversed": False,
        "terminals": [{"id": 0, "code_point": 97}, {"id": 1, "code_point": 98}],
        "rules": [{"id": i, "left": 0, "right": 1, "left_reversed": False,
                   "right_reversed": False, "length": 2} for i in (2, 3)],
        "residual": [{"symbol": 2, "reversed": False, "offset": 0, "length": 2},
                     {"symbol": 3, "reversed": False, "offset": 2, "length": 2}],
    }
    objects, path = parse_string_pathway_file(write_certificate(tmp_path, data))
    assert_bound(path, 3)
    assert objects == ["a", "b", "ab", "abab"]
    assert sum(d["vo"] == "ab" for _, d in path.nodes(data=True)) == 2


@pytest.mark.parametrize("location,replacement", [
    (("upper_bound",), 1), (("length",), 8), (("accept_reversed",), False),
    (("terminals", 0, "code_point"), 0xD800), (("rules", 0, "id"), 0),
    (("rules", 0, "left"), 4), (("rules", 0, "length"), 9),
    (("residual", 0, "offset"), 1), (("residual", 0, "length"), 1),
    (("residual", 2, "reversed"), False),
])
def test_string_repair_rejects_invalid_constructions(tmp_path, location, replacement):
    certificate = copy.deepcopy(string_certificate())
    parent = certificate
    for key in location[:-1]:
        parent = parent[key]
    parent[location[-1]] = replacement
    with pytest.raises(ValueError, match="Re-Pair pathway certificate"):
        parse_string_pathway_file(write_certificate(tmp_path, certificate))
