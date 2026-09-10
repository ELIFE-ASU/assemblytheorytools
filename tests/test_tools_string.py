import random
import string

import pytest

import assemblytheorytools as att
from assemblytheorytools import tools_string


def test_load_fasta_ignores_headers_and_joins_sequence_lines(tmp_path):
    fasta = tmp_path / "sequence.fasta"
    fasta.write_text(
        ">first sequence\n"
        "AC GT\n"
        "\n"
        ">second sequence\n"
        "TTAA\n"
    )

    assert att.load_fasta(fasta) == "AC GTTTAA"


def test_load_fasta_returns_empty_string_for_headers_only(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text(">first\n>second\n")

    assert att.load_fasta(fasta) == ""


def test_prep_joint_string_ai_uses_distinct_non_input_delimiters():
    inputs = ["abc0", "def1", "ghi2"]

    combined, delimiters = att.prep_joint_string_ai(inputs)

    assert len(delimiters) == len(inputs) - 1
    assert len(set(delimiters)) == len(delimiters)
    assert all(delimiter not in "".join(inputs) for delimiter in delimiters)
    assert combined == (
            inputs[0] + delimiters[0] + inputs[1] + delimiters[1] + inputs[2]
    )


def test_prep_joint_string_ai_single_input_needs_no_delimiter():
    assert att.prep_joint_string_ai(["abc"]) == ("abc", [])


def test_prep_joint_string_ai_rejects_an_empty_input_list():
    with pytest.raises(ValueError, match="cannot be empty"):
        att.prep_joint_string_ai([])


@pytest.mark.parametrize("inputs", [[""], ["abc", ""], ["", "abc"]])
def test_prep_joint_string_ai_rejects_empty_strings(inputs):
    with pytest.raises(ValueError, match="Empty string"):
        att.prep_joint_string_ai(inputs)


def test_get_unique_char_prefers_printable_ascii():
    delimiter = att.get_unique_char("0123456789")

    assert delimiter in string.printable
    assert delimiter not in "0123456789"
    assert delimiter != " "


def test_get_unique_char_falls_back_to_printable_unicode():
    all_allowed_ascii = "".join(char for char in string.printable if char != " ")

    delimiter = att.get_unique_char(all_allowed_ascii)

    assert ord(delimiter) >= 0x00A1
    assert delimiter.isprintable()
    assert delimiter not in all_allowed_ascii


def test_get_unique_char_retains_ascii_whitespace_candidates():
    assert att.get_unique_char(string.printable.rstrip()) == "\t"


def test_get_unique_char_reports_exhausted_delimiters():
    reserved = string.printable + "".join(map(chr, range(0x00A1, 0x2FFF)))

    with pytest.raises(ValueError, match="Ran out of delimiter symbols"):
        att.get_unique_char(reserved)


def test_get_undirected_string_molecule_preserves_symbol_pattern():
    graph, colour_map = att.get_undir_str_molecule("aba")

    assert colour_map == {"a": "1", "b": "2"}
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert list(graph) == [0, 1, 2, 3]
    assert list(graph.edges) == [(0, 1), (1, 2), (2, 3)]
    assert [graph.edges[i, i + 1]["color"] for i in range(3)] == [1, 2, 1]
    assert set(graph.nodes[node]["color"] for node in graph) == {"null"}


def test_get_directed_string_molecule_encodes_symbols_as_nodes():
    graph = att.get_dir_str_molecule("ab")

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 4
    assert list(graph) == [0, 1, 2, 3, 4]
    assert list(graph.edges) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert [graph.nodes[node]["color"] for node in range(5)] == [
        "null",
        "a",
        "null",
        "b",
        "null",
    ]
    assert [graph.edges[i, i + 1]["color"] for i in range(4)] == [1, 2, 1, 2]


@pytest.mark.parametrize("builder", [att.get_dir_str_molecule, att.get_undir_str_molecule])
def test_string_graphs_reject_empty_strings(builder):
    with pytest.raises(IndexError, match="string index out of range"):
        builder("")


def test_undirected_string_graph_debug_output_precedes_empty_input_error(capsys):
    with pytest.raises(IndexError):
        att.get_undir_str_molecule("", debug=True)

    assert capsys.readouterr().out == "Edge color dict:\n{}\n"


def test_generate_random_strings_honours_pool_and_length(monkeypatch):
    calls = []

    def fake_choices(population, *, k):
        calls.append((population, k))
        return ["x"] * k

    monkeypatch.setattr(tools_string.random, "choices", fake_choices)

    assert att.generate_random_strings(3, 4) == ["xxxx", "xxxx", "xxxx"]
    assert calls == [(string.ascii_lowercase, 4)] * 3


def test_generate_random_strings_handles_empty_dimensions():
    assert att.generate_random_strings(0, 5) == []
    assert att.generate_random_strings(2, 0) == ["", ""]


def test_generate_random_strings_preserves_seeded_output(monkeypatch):
    monkeypatch.setattr(tools_string, "random", random.Random(42))

    assert att.generate_random_strings(3, 8) == ["qahftrxc", "kafnafqo", "fpvausie"]
