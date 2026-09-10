"""Compatibility checks for the shared molecule and graph fixtures."""

import networkx as nx
import pytest

from assemblytheorytools import tools_test


@pytest.mark.parametrize(
    "input_list, reference_list, expected",
    [
        ([], [], False),
        ([1, 1], [1], True),
        ([1, 2], [1], False),
        ([[], {"a": 1}], [{"a": 1}, []], True),
    ],
)
def test_check_elements_keeps_membership_semantics(input_list, reference_list, expected):
    assert tools_test.check_elements(input_list, reference_list) is expected


@pytest.mark.parametrize(
    "factory, colors, bonds",
    [
        (tools_test.water_graph, ["O", "H", "H"], [(0, 1, 1), (0, 2, 1)]),
        (
            tools_test.phosphine_graph,
            ["P", "H", "H", "H"],
            [(0, 1, 1), (0, 2, 1), (0, 3, 1)],
        ),
        (tools_test.ph_2p_graph, ["P", "H"], [(0, 1, 1)]),
        (tools_test.co2_graph, ["C", "O", "O"], [(0, 1, 2), (0, 2, 2)]),
    ],
)
def test_graph_fixtures_preserve_order_and_return_fresh_graphs(factory, colors, bonds):
    graph = factory()

    assert list(graph.nodes(data="color")) == list(enumerate(colors))
    assert list(graph.edges(data="color")) == bonds

    graph.nodes[0]["color"] = "changed"
    graph.edges[0, 1]["color"] = 99
    fresh_graph = factory()
    assert list(fresh_graph.nodes(data="color")) == list(enumerate(colors))
    assert list(fresh_graph.edges(data="color")) == bonds


def test_print_graph_details_keeps_node_and_edge_order(capsys):
    graph = nx.Graph()
    graph.add_node("bare")
    graph.add_node("oxygen", color="O")
    graph.add_edge("oxygen", "bare", color=2)

    tools_test.print_graph_details(graph)

    assert capsys.readouterr().out == (
        "{\n"
        "(bare, No color): [('bare', 'oxygen')], [2]\n"
        "(oxygen, O): [('oxygen', 'bare')], [2]\n"
        "}\n"
    )


def test_print_graph_details_requires_edge_colors():
    graph = nx.Graph([(0, 1)])

    with pytest.raises(KeyError, match="color"):
        tools_test.print_graph_details(graph)


@pytest.fixture
def molecule_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_test, "__file__", str(tmp_path / "tools_test.py"))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir / "test_molecule_data.csv"


def test_load_molecules_trims_fields_and_preserves_duplicate_key_order(molecule_csv):
    molecule_csv.write_text(
        "name,category,smiles,inchi,assembly_index,test_include\n"
        " Water , original , O , old , 2 , True \n"
        " ETHANE , alkanes , CC , reference , 0 , tRuE \n"
        " WATER , replacement , O , , , False \n"
    )

    molecules = tools_test._load_molecules()

    assert list(molecules) == ["water", "ethane"]
    assert molecules["water"] == tools_test.Molecule(
        "water", "replacement", "O", None, None, False
    )
    assert molecules["ethane"] == tools_test.Molecule(
        "ethane", "alkanes", "CC", "reference", 0, True
    )


def test_load_molecules_allows_missing_optional_columns(molecule_csv):
    molecule_csv.write_text("name,category,smiles\nwater,small,O\n")

    assert tools_test._load_molecules() == {
        "water": tools_test.Molecule("water", "small", "O", None, None, False)
    }


def test_load_molecules_reports_missing_bundled_data(molecule_csv):
    with pytest.raises(FileNotFoundError) as error:
        tools_test._load_molecules()

    assert str(error.value) == f"Data file not found: {molecule_csv}"


def test_load_molecules_rejects_invalid_assembly_index(molecule_csv):
    molecule_csv.write_text(
        "name,category,smiles,assembly_index\nwater,small,O,invalid\n"
    )

    with pytest.raises(ValueError, match="invalid literal"):
        tools_test._load_molecules()
