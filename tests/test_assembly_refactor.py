"""Contracts at the boundary between assembly wrappers and their backends."""

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from assemblytheorytools import assembly


@pytest.mark.parametrize(
    "exact, log_text, expected",
    [
        (False, "min AI found so far: 9\nmin AI found so far: 7\n", 6),
        (True, "min AI found so far: 7\n", -1),
        (False, "No minimum available\n", -1),
    ],
)
def test_molecular_timeout_uses_latest_bound_and_preserves_log(
    tmp_path, monkeypatch, exact, log_text, expected
):
    """Timed-out joint calculations correct bounds, but preserve failure sentinels."""
    calculation_dir = tmp_path / "calculation"
    clock = SimpleNamespace(now=0.0)
    commands = []

    def start_process(command, *, stdout, stderr):
        commands.append(command)
        assert stdout is stderr
        stdout.write(log_text)
        Path(command[1] + "Out").write_text("assembly index: 99\n")
        return SimpleNamespace(wait=lambda: setattr(clock, "now", 2.0))

    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1, "color")
    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda: str(calculation_dir))
    monkeypatch.setattr(assembly.time, "time", lambda: clock.now)
    monkeypatch.setattr(assembly.platform, "system", lambda: "Linux")
    monkeypatch.setattr(assembly.subprocess, "Popen", start_process)

    result = assembly.calculate_assembly_index(
        graph, dir_code="assembler", timeout=1, exact=exact, return_log_file=True
    )

    assert result[:3] == (expected, None, None)
    assert isinstance(result[0], int)
    assert Path(result[3]).read_text() == log_text
    assert commands[0][-1] == "-runTime=1000000"
    assert graph.number_of_nodes() == 4


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("timeouts", [0, 1, 2])
@pytest.mark.parametrize("mode", ["str", "mol"])
def test_string_backend_timeout_and_file_lifecycle(
    tmp_path, monkeypatch, debug, timeouts, mode
):
    """String runs retain debug files and recover bounds after interrupt or kill."""
    calculation_dir = tmp_path / "calculation"
    events = []
    output = b"min AI found so far: 9\nmin AI found so far: 7\ninvalid: \xff\n"

    class Process:
        attempts = 0

        def communicate(self, timeout=None):
            self.attempts += 1
            if self.attempts <= timeouts:
                raise assembly.subprocess.TimeoutExpired("assembler", timeout)
            return output, None

        def send_signal(self, signal):
            events.append(signal)

        def wait(self):
            pass

        def kill(self):
            events.append("kill")

    def start_process(command, *, stdout, stderr, cwd):
        assert command == ["assembler", str(calculation_dir / "string_in"), "-runStrings=1"]
        assert cwd == str(calculation_dir)
        assert Path(command[1]).read_text() == "abab0baba"
        Path(command[1] + "Out").write_text("assembly index: 5\n")
        return Process()

    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda: str(calculation_dir))
    monkeypatch.setattr(assembly.subprocess, "Popen", start_process)

    result = assembly.calculate_string_assembly_index(
        ["abab", "baba"],
        dir_code="assembler",
        timeout=1,
        debug=debug,
        mode=mode,
        return_log_file=True,
    )

    assert result[:3] == (5 if timeouts else 3, None, None)
    assert isinstance(result[0], int)
    assert result[3] == str(calculation_dir / "assembly_output.log")
    assert calculation_dir.exists() is debug
    if debug:
        assert Path(result[3]).read_text() == output.decode(errors="replace")
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
    assert calls == [(graph, {
        "dir_code": None,
        "timeout": 100.0,
        "debug": False,
        "joint_corr": False,
        "strip_hydrogen": False,
        "return_log_file": True,
    })]
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
