"""Exercise the subprocess boundary with small real executables."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from assemblytheorytools import assembly


def _calculate(kind, **settings):
    if kind == "string":
        return assembly.calculate_string_assembly_index(["abab", "baba"], **settings)
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1, "color")
    return assembly.calculate_assembly_index(graph, **settings)


@pytest.fixture
def executable(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly.tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    def create(body):
        script = tmp_path / "fake calculator"
        script.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n" + body)
        script.chmod(0o755)
        return "./fake calculator"

    return create


@pytest.mark.parametrize("kind", ["graph", "string"])
@pytest.mark.parametrize("keep_log", [False, True])
def test_real_success_and_log_lifecycle(executable, tmp_path, kind, keep_log):
    calculator = executable(
        'Path(sys.argv[1] + "Out").write_text("assembly index: 5\\n")\n'
        'print("calculator output")\n'
    )
    result = _calculate(kind, dir_code=calculator, return_log_file=keep_log)
    assert result[:3] == (3 if kind == "string" else 5, None, None)
    directories = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(directories) == int(keep_log)
    if keep_log:
        assert Path(result[3]).read_text() == "calculator output\n"


@pytest.mark.parametrize("kind", ["graph", "string"])
@pytest.mark.parametrize("failure", ["exit", "missing-output", "invalid-output", "missing-executable"])
def test_failures_preserve_diagnostics_and_cleanup(executable, tmp_path, kind, failure):
    body = 'print("useful diagnostic", file=sys.stderr)\n'
    if failure == "exit":
        body += "sys.exit(2)\n"
    if failure == "invalid-output":
        body += 'Path(sys.argv[1] + "Out").write_text("not an index\\n")\n'
    calculator = executable(body)
    if failure == "missing-executable":
        calculator = "./missing-calculator"
    with pytest.raises(OSError) as error:
        _calculate(kind, dir_code=calculator)
    if failure != "missing-executable":
        assert "useful diagnostic" in str(error.value)
        assert ("status 2" if failure == "exit" else "no assembly index") in str(error.value)
    assert not any(p.is_dir() for p in tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["graph", "string"])
def test_internal_early_stop_uses_output_bound(executable, kind):
    calculator = executable(
        'Path(sys.argv[1] + "Out").write_text('
        '"assembly index: 7\\nstatus: enumeration limit reached\\n")\n'
        'print("Best assembly index: 9 (1 ticks)")\n'
    )
    assert _calculate(kind, dir_code=calculator)[0] == (5 if kind == "string" else 7)
    if kind == "graph":
        assert _calculate(kind, dir_code=calculator, exact=True)[0] == -1


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal/reaping contract")
def test_uncooperative_calculator_is_killed_and_reaped(executable, tmp_path):
    pid_file = tmp_path / "child.pid"
    calculator = executable(
        'import os, signal, time\n'
        'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
        f'Path({str(pid_file)!r}).write_text(str(os.getpid()))\n'
        'print("Best assembly index: 7 (1 ticks)", flush=True)\n'
        'time.sleep(60)\n'
    )
    assert _calculate("graph", dir_code=calculator, timeout=0.5) == (7, None, None)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert not any(p.is_dir() for p in tmp_path.iterdir())


def test_python_interrupt_reaps_child(monkeypatch, tmp_path):
    events = []

    def wait(timeout=None):
        if timeout is not None:
            raise KeyboardInterrupt
        events.append("wait")

    process = SimpleNamespace(wait=wait, poll=lambda: None,
                              kill=lambda: events.append("kill"))
    monkeypatch.setattr(assembly.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(KeyboardInterrupt):
        assembly._run_assembler("calculator", str(tmp_path / "input"),
                                str(tmp_path / "log"), 1, False)
    assert events == ["kill", "wait"]


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("kind", ["graph", "string"])
def test_invalid_timeout_fails_before_backend_lookup(monkeypatch, timeout, kind):
    monkeypatch.setattr(assembly, "add_assembly_to_path", lambda **kwargs: pytest.fail("backend lookup"))
    with pytest.raises(ValueError, match="finite, non-negative"):
        _calculate(kind, timeout=timeout)


@pytest.mark.parametrize("text", ["ab\ncd", "ab\rcd", "ééé"])
def test_cpp_string_input_requires_one_ascii_line(monkeypatch, text):
    monkeypatch.setattr(assembly, "add_assembly_to_path", lambda **kwargs: pytest.fail("backend lookup"))
    with pytest.raises(ValueError, match="single line of ASCII"):
        assembly.calculate_string_assembly_index(text)


@pytest.mark.parametrize("nodes", [0, 1, 3])
def test_edgeless_graph_requires_no_joins_or_calculator(monkeypatch, nodes):
    monkeypatch.setattr(assembly, "add_assembly_to_path", lambda **kwargs: pytest.fail("backend lookup"))
    graph = nx.empty_graph(nodes)
    nx.set_node_attributes(graph, "C", "color")
    assert assembly.calculate_assembly_index(graph, return_log_file=True) == (0, None, None, None)


def test_undirected_timeout_preserves_failure_sentinel(monkeypatch):
    monkeypatch.setattr(assembly, "calculate_assembly_index", lambda *args, **kwargs: (-1, None, None))
    assert assembly.calculate_string_assembly_index(["ab", "cd"], directed=False) == (-1, None, None)


@pytest.mark.parametrize("kind", ["graph", "mol"])
@pytest.mark.parametrize("components", [1, 2])
def test_isolated_atoms_do_not_reduce_joint_index(kind, components):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(".".join(["CCCC"] * components + ["[He]", "[Ne]"]))
    graph = assembly.mol_to_nx(molecule)
    value = graph if kind == "graph" else molecule
    assert assembly.joint_assembly_index_correction(value, 3) == 3 - (components - 1)
    assert assembly.calculate_assembly_index(value, strip_hydrogen=True)[0] == 2


def test_joint_correction_reads_legacy_pathway_without_rewriting(tmp_path):
    file = tmp_path / "Pathway"
    raw = '{"file_graph": [{"Edges": [[0,1], [1,2]], "EdgeColours": [single,]}]}'
    file.write_text(raw)
    assert assembly._calculate_jo_from_pathway(str(file)) == 1
    assert file.read_text() == raw


def test_joint_string_timeout_without_bound_preserves_sentinel(monkeypatch):
    def timeout(executable, input_file, log_file, *args, **kwargs):
        Path(log_file).write_bytes(b"diagnostic: \xff\n")
        return True

    monkeypatch.setattr(assembly, "_run_assembler", timeout)
    assert _calculate("string", dir_code="calculator") == (-1, None, None)
