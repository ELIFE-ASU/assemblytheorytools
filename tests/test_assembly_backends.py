"""External calculator process, timeout and logging contracts."""

import os
import subprocess
from pathlib import Path

import networkx as nx
import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


_STATUS_OUT = "assembly index: 8\nstatus: runtime limit reached\ntime elapsed: 3\n"


@pytest.mark.parametrize(
    "exact, log_text, out_text, expected",
    [
        (False, "Best assembly index: 9 (1 ticks)\nBest assembly index: 7 (2 ticks)\n",
         "", 6),
        (False, "min AI found so far: 9\nmin AI found so far: 7\n",
         "", 6),
        (True, "Best assembly index: 7 (2 ticks)\n", "assembly index: 99\n", -1),
        (True, "min AI found so far: 7\n", "assembly index: 99\n", -1),
        (False, "No minimum available\n", "", -1),
        (False, "Best assembly index: 9 (1 ticks)\n", _STATUS_OUT, 7),
        (True, "Best assembly index: 9 (1 ticks)\n", _STATUS_OUT, -1),
    ],
    ids=["log-v5", "log-legacy", "log-v5-exact", "log-legacy-exact", "no-bound",
         "status-bound", "status-bound-exact"],
)
def test_molecular_timeout_uses_latest_bound_and_preserves_log(
    tmp_path, monkeypatch, exact, log_text, out_text, expected
):
    """Timed-out joint calculations correct bounds, but preserve failure sentinels.

    The bound comes from the assembler's own output file when available,
    and from its log when it was killed before writing a result. Both the current
    ``Best assembly index`` spelling and the legacy ``min AI found so far`` one
    are recognised, so an older executable on ``ASS_PATH`` still reports a bound.
    """
    calculation_dir = tmp_path / "calculation"
    commands = []

    class Process:
        returncode = 0
        attempts = 0

        def wait(self, timeout=None):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired("assembler", timeout)

        def send_signal(self, signal):
            pass

        def terminate(self):
            pass

        def poll(self):
            return self.returncode

    def start_process(command, *, stdout, stderr, stdin, cwd):
        commands.append(command)
        assert stdout is stderr
        stdout.write(log_text)
        Path(command[1] + "Out").write_text(out_text)
        return Process()

    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1, "color")
    calculation_dir.mkdir()
    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda: str(calculation_dir))
    monkeypatch.setattr(assembly.platform, "system", lambda: "Linux")
    monkeypatch.setattr(assembly.subprocess, "Popen", start_process)

    result = assembly.calculate_assembly_index(
        graph, dir_code="assembler", timeout=1, exact=exact, return_log_file=True
    )

    assert result[:3] == (expected, None, None)
    assert isinstance(result[0], int)
    assert Path(result[3]).read_text() == log_text
    assert not any("runTime" in argument for argument in commands[0])
    assert graph.number_of_nodes() == 4


def test_run_command(capfd):
    """Command output streams to stdout; the wrapper returns no value."""
    # run_command does not use a shell, and on Windows echo is a cmd builtin
    # rather than an executable, so there it has to be invoked through cmd.
    assert att.run_command("cmd /c echo hello" if os.name == "nt" else "echo hello") is None
    assert capfd.readouterr().out.replace("\r\n", "\n") == "hello\n"

    with pytest.raises(ValueError):
        att.run_command(None)


def test_molecular_debug_retains_calculation_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = att.smi_to_mol("[H]C#C[H]")
    ai, virt_obj, _ = att.calculate_assembly_index(mol, debug=True)
    dir_list = att.list_subdirs(tmp_path, target="ai_calc")
    ref_out = ["C#C", "[H]C", "[H]C#C", "[H]C#C[H]"]
    assert ai == 2
    assert set(virt_obj) == set(ref_out)
    assert len(dir_list) == 1
    debug_dir = tmp_path / dir_list[0]
    assert {path.name for path in debug_dir.iterdir()} >= {
        "graph_in",
        "graph_inOut",
        "graph_inPathway",
    }


@pytest.mark.parametrize("return_log_file", [False, True])
def test_molecular_log_option_controls_retention(
    tmp_path, monkeypatch, return_log_file
):
    calculation_dir = tmp_path / "calculation"
    calculation_dir.mkdir()
    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda: str(calculation_dir))
    mol = att.smi_to_mol("c1ccccc1")

    result = att.calculate_assembly_index(mol, return_log_file=return_log_file)

    assert isinstance(result[0], int)
    assert result[0] > 0
    assert len(result) == (4 if return_log_file else 3)
    if return_log_file:
        assert Path(result[3]).is_file()
        assert Path(result[3]).read_text()

    assert calculation_dir.exists() is return_log_file
