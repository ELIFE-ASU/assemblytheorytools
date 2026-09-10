"""Directed, undirected and joint string assembly across backends."""

import random
import string
from itertools import islice, product
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


@pytest.mark.parametrize("mode", ["mol", "str"])
@pytest.mark.parametrize("directed", [False, True])
def test_abracadabra_index(mode, directed):
    assert (
        att.calculate_string_assembly_index(
            "abracadabra", mode=mode, directed=directed
        )[0]
        == 7
    )


@pytest.mark.parametrize(
    "strings, minimum, maximum",
    [("abracadabra", 7, 7), (["aaaa", "bbbb", "aa"], 4, 7)],
    ids=["single", "joint"],
)
def test_cfg_bounds_for_known_strings(strings, minimum, maximum):
    # CFG is a heuristic upper bound: it may miss sharing between strings.
    upper_bound = att.calculate_string_assembly_index(strings, mode="cfg")[0]
    assert minimum <= upper_bound <= maximum


@pytest.mark.parametrize(
    "settings",
    [{"directed": True}, {"directed": False}, {"mode": "cfg"}],
    ids=["directed", "undirected", "cfg"],
)
def test_single_character_pool_needs_no_joining(settings):
    assert att.calculate_string_assembly_index(["a"] * 95, **settings) == (
        0,
        None,
        None,
    )


@pytest.mark.parametrize("mode", ["mol", "str"])
def test_joint_strings_share_intermediates(mode):
    assert (
        att.calculate_string_assembly_index(["aaaa", "bbbb", "aa"], mode=mode)[0] == 4
    )


@pytest.mark.parametrize(
    "log, expected",
    [
        ("min AI found so far: 9\nmin AI found so far: 7\n", 7),
        ("", -1),
        ("No paths found\n", -1),
    ],
    ids=["latest-bound", "empty-log", "no-bound"],
)
def test_string_timeout_returns_a_bound_only_when_one_was_logged(
    tmp_path, monkeypatch, log, expected
):
    calculation_dir = tmp_path / "calculation"

    def timed_out(executable, input_file, log_file, timeout, debug, **kwargs):
        Path(log_file).write_text(log)
        return True

    calculation_dir.mkdir()
    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda **kwargs: str(calculation_dir))
    monkeypatch.setattr(assembly, "_run_assembler", timed_out)

    result = att.calculate_string_assembly_index("abab", dir_code="assembler")

    assert result == (expected, None, None)
    assert not calculation_dir.exists()


def test_cfg_handles_more_strings_than_delimiter_characters():
    pool = [
        "".join(chars)
        for chars in islice(product(string.ascii_lowercase, repeat=3), 95)
    ]

    ai = att.calculate_string_assembly_index(pool, mode="cfg")[0]

    assert isinstance(ai, int)
    assert ai >= 0


@pytest.mark.parametrize("directed, expected", [(True, 3), (False, 2)])
def test_reversing_fragments_reduces_abba_index(directed, expected):
    assert att.calculate_string_assembly_index("abba", directed=directed)[0] == expected


def test_string_graph_conversion():

    rng = random.Random(0)
    strings = ["".join(rng.choices(string.ascii_lowercase, k=20)) for _ in range(50)]

    for s in strings:
        assert s == att.molstr_to_str(att.get_dir_str_molecule(s))
        graph, edge_color_dict = att.get_undir_str_molecule(s)
        assert s == att.molstr_to_str(graph, edge_color_dict=edge_color_dict)


def test_undirected_pathway_repairs_cpp_edge_colors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ai, vo, path = att.calculate_string_assembly_index(
        "yydpetgtwy", mode="mol", directed=False, debug=True
    )
    assert ai >= 0
    assert vo
    assert nx.is_directed_acyclic_graph(path)
    assert path.number_of_edges() > 0
    debug_dirs = att.list_subdirs(tmp_path, target="ai_calc")
    assert len(debug_dirs) == 1
    att.safe_folder_remove(tmp_path / debug_dirs[0])
    assert att.list_subdirs(tmp_path, target="ai_calc") == []


def test_string_ensemble_assembly():
    input_strings = ["abab", "cdcdcdcd", "c"]
    input_ns = [10, 100, 40]
    nt = sum(input_ns)
    answer = (
        np.exp(2) * ((10 - 1) / nt)
        + np.exp(3) * ((100 - 1) / nt)
        + np.exp(0) * ((40 - 1) / nt)
    )
    assert answer == att.calculate_string_assembly(strings=input_strings, n_i=input_ns)


def test_directed_str_data():
    s_inpt = "abracadabra"
    ai_ref = 7
    ai, vo, path = att.calculate_string_assembly_index(
        s_inpt, directed=True, mode="str"
    )
    assert ai == ai_ref
    assert len(vo) == 12  # 12 = 7 steps + 5 units
    assert len(path.nodes()) == len(vo)
    assert len(path.edges()) == ai_ref * 2


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("return_log_file", [False, True])
@pytest.mark.parametrize("timeouts", [0, 1, 2])
def test_string_backend_timeout_and_file_lifecycle(
    tmp_path, monkeypatch, debug, return_log_file, timeouts
):
    """Both interrupt and kill are bounded, and returned logs remain readable."""
    calculation_dir = tmp_path / "calculation"
    calculation_dir.mkdir()
    events = []
    output = "min AI found so far: 9\nmin AI found so far: 7\n"

    class Process:
        attempts = 0
        returncode = 0

        def wait(self, timeout=None):
            self.attempts += 1
            if self.attempts <= timeouts:
                assert timeout in (1, 2)
                raise assembly.subprocess.TimeoutExpired("assembler", timeout)

        def send_signal(self, signal):
            events.append(signal)

        def kill(self):
            events.append("kill")

        def poll(self):
            return self.returncode

    def start_process(command, *, stdout, stderr, stdin, cwd):
        assert command[:2] == ["assembler", str(calculation_dir / "string_in")]
        assert "-runStrings=1" in command
        assert "-runTime=1000000" in command
        assert stdout is stderr
        assert cwd == str(calculation_dir)
        assert Path(command[1]).read_text() == "abab0baba"
        Path(command[1] + "Out").write_text("assembly index: 5\n")
        stdout.write(output)
        return Process()

    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda **kwargs: str(calculation_dir))
    monkeypatch.setattr(assembly.subprocess, "Popen", start_process)

    result = assembly.calculate_string_assembly_index(
        ["abab", "baba"], dir_code="assembler", timeout=1, debug=debug,
        return_log_file=return_log_file,
    )

    assert result[:3] == (5 if timeouts else 3, None, None)
    assert calculation_dir.exists() is (debug or return_log_file)
    if return_log_file:
        assert Path(result[3]).read_text() == output
    assert events == (
        [] if not timeouts else [assembly.signal.SIGINT] + (["kill"] if timeouts == 2 else [])
    )


@pytest.mark.parametrize("mode", ["mol", "str", "cfg"])
def test_undirected_strings_use_molecular_backend(monkeypatch, capsys, mode):
    graph = nx.Graph()
    pathway = nx.DiGraph()
    calls = []

    def calculate(input_graph, **settings):
        calls.append((input_graph, settings))
        return 5, [], pathway, "molecular.log"

    monkeypatch.setattr(
        assembly, "get_undir_str_molecule", lambda string, debug: (graph, {})
    )
    monkeypatch.setattr(assembly, "calculate_assembly_index", calculate)

    result = assembly.calculate_string_assembly_index(
        "abab", mode=mode, directed=False, return_log_file=True
    )

    assert result == (5, [], pathway, "molecular.log")
    assert calls == [
        (
            graph,
            {
                "dir_code": None,
                "timeout": 100.0,
                "debug": False,
                "joint_corr": False,
                "strip_hydrogen": False,
                "return_log_file": True,
            },
        )
    ]
    assert ("Switching to 'mol'" in capsys.readouterr().out) is (mode != "mol")


@pytest.mark.parametrize("return_log_file", [False, True])
def test_cfg_backend_preserves_three_field_result(monkeypatch, return_log_file):
    """CFG has no log field even when the external-backend option is requested."""
    virtual_objects = ["ab", "abab"]
    pathway = nx.DiGraph([("ab", "abab")])
    calls = []

    def repair(input_data, *, f_print):
        calls.append((input_data, f_print))
        return 2, virtual_objects, pathway

    monkeypatch.setattr(assembly.assemblycfg, "repair_with_pathways", repair)

    result = assembly.calculate_string_assembly_index(
        ["a", "abab"], mode="cfg", return_log_file=return_log_file
    )

    assert result == (2, virtual_objects, pathway)
    assert calls == [(["abab"], False)]


@pytest.mark.parametrize("input_data", ["a", [], ["", "a"]])
def test_trivial_strings_return_without_resolving_backend(input_data, monkeypatch):
    def unexpected_backend(*args, **kwargs):
        pytest.fail("Trivial strings should not resolve an external executable")

    monkeypatch.setattr(assembly, "add_assembly_to_path", unexpected_backend)

    assert assembly.calculate_string_assembly_index(input_data) == (0, None, None)
    assert assembly.calculate_string_assembly_index(
        input_data, return_log_file=True
    ) == (0, None, None, None)
