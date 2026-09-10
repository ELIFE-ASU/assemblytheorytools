"""Regression coverage for pathway-derived ensemble and joining metrics."""

import json

import networkx as nx
import pytest

import assemblytheorytools.assembly as assembly


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
    pathway.write_text(json.dumps({
        "file_graph": [{"Edges": edges}],
        "duplicates": [{"Right": fragment} for fragment in fragments],
    }), encoding="utf-8")

    assert assembly._calculate_jo_from_pathway(str(pathway)) == expected


@pytest.mark.parametrize("output", ["valid", "invalid", "missing"])
def test_joining_calculation_cleans_output_and_preserves_settings(
    tmp_path, monkeypatch, output
):
    folder = tmp_path / "ai_calc_example"
    folder.mkdir()
    if output != "missing":
        text = json.dumps({"file_graph": [{"Edges": [[0, 1], [1, 2]]}]})
        (folder / "graphPathway").write_text(
            text if output == "valid" else "invalid JSON", encoding="utf-8"
        )

    graph = nx.path_graph(3)
    virtual_objects, pathway = ["fragment"], nx.DiGraph()
    forwarded_settings = []

    def calculate(input_graph, **settings):
        assert input_graph is graph
        forwarded_settings.append(settings)
        return 1, virtual_objects, pathway

    monkeypatch.setattr(assembly, "calculate_assembly_index", calculate)
    monkeypatch.setattr(assembly, "_get_most_recent_calc", lambda: str(folder))
    settings = {"save_dir": False, "timeout": 0.5}

    result = assembly.calculate_assembly_index_jo(graph, settings)

    assert settings == {"save_dir": False, "timeout": 0.5}
    assert forwarded_settings == [{"save_dir": True, "timeout": 0.5}]
    assert not folder.exists()
    if output == "valid":
        assert result[0] == 1
        assert result[1] is virtual_objects
        assert result[2] is pathway
    else:
        assert result == (-1, None, None)


def test_exploration_infers_observed_objects_before_relabelled_paths_merge():
    first = nx.DiGraph([(0, 1)])
    second = nx.DiGraph([(0, 1), (1, 2)])
    nx.set_node_attributes(first, {0: "a", 1: "ab"}, "vo")
    nx.set_node_attributes(second, {0: "a", 1: "ab", 2: "abb"}, "vo")

    assert assembly.exploration_ratio([first, second], node_key="vo") == 2 / 3
    assert assembly.exploration_ratio(
        [first, second], observed=["ab", "ab", "unobserved"], node_key="vo"
    ) == 1 / 3
    assert list(first) == [0, 1]
    assert list(second) == [0, 1, 2]
