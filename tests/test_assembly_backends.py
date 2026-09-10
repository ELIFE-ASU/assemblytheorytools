"""External calculator process, timeout, logging and build contracts."""

import os
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import assemblytheorytools as att
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


def test_run_command(capfd):
    """Command output streams to stdout; the wrapper returns no value."""
    assert att.run_command("echo hello") is None
    assert capfd.readouterr().out == "hello\n"

    with pytest.raises(ValueError):
        att.run_command(None)


def test_compile_assembly_cpp_orchestration(tmp_path, monkeypatch):
    precompiled = tmp_path / "assemblytheorytools" / "precompiled"
    precompiled.mkdir(parents=True)
    subprocess_calls = []
    build_calls = []

    def fake_subprocess_run(command, *, shell, check):
        subprocess_calls.append((command, shell, check))
        executable = tmp_path / "assemblycpp-v5" / "build" / "bin" / "assembly"
        executable.parent.mkdir(parents=True)
        executable.write_text("compiled executable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(assembly.platform, "system", lambda: "Linux")
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(assembly.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(assembly, "run_command", build_calls.append)

    result = att.compile_assembly_cpp()

    executable = precompiled / "assembly"
    assert result is None
    assert subprocess_calls == [
        (
            "git clone https://github.com/LouieSlocombe/assemblycpp-v5.git",
            True,
            True,
        )
    ]
    assert build_calls == ["cmake -S . -B build", "cmake --build build"]
    assert executable.read_text() == "compiled executable"
    assert executable.stat().st_mode & 0o111
    assert not (tmp_path / "assemblycpp-v5").exists()
    assert os.getcwd() == str(tmp_path)


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
def test_molecular_log_option_preserves_calculation(
    tmp_path, monkeypatch, return_log_file
):
    calculation_dir = tmp_path / "calculation"
    monkeypatch.setattr(assembly.tempfile, "mkdtemp", lambda: str(calculation_dir))
    mol = att.smi_to_mol("c1ccccc1")

    result = att.calculate_assembly_index(mol, return_log_file=return_log_file)

    assert isinstance(result[0], int)
    assert result[0] > 0
    assert len(result) == (4 if return_log_file else 3)
    if return_log_file:
        assert Path(result[3]).is_file()
        assert Path(result[3]).read_text()
