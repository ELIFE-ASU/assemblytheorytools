"""Public option forwarding and real calculator behavior."""

import json
import re
import sys
from pathlib import Path

import networkx as nx
import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


@pytest.fixture
def recording_calculator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "calculator"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "Path('arguments.json').write_text(json.dumps(sys.argv[2:]))\n"
        "Path(sys.argv[1] + 'Out').write_text('assembly index: 5\\n')\n"
    )
    script.chmod(0o755)
    return script


def graph():
    result = nx.path_graph(4)
    nx.set_node_attributes(result, "C", "color")
    nx.set_edge_attributes(result, 1, "color")
    return result


@pytest.mark.parametrize("kind", ["graph", "string", "undirected-string"])
def test_public_entrypoints_forward_cpp_options(recording_calculator, kind):
    settings = dict(runtime_ticks=1234, pathway=False, memory_report=True)
    if kind == "string":
        settings["accept_palindromes"] = True
    else:
        settings.update(enum_max=456, parallel="auto", threads=2, verbose=True,
                        telemetry=True, write_intermediate_mas=True)
    options = att.AssemblyCppOptions(**settings)
    kwargs = dict(cpp_options=options, timeout=None, return_log_file=True,
                  dir_code=recording_calculator)
    if kind == "graph":
        result = att.calculate_assembly_index(graph(), **kwargs)
    else:
        result = att.calculate_string_assembly_index(
            "abab", directed=kind == "string", **kwargs)
    assert result[:3] == (5, None, None)
    arguments = json.loads((Path(result[3]).parent / "arguments.json").read_text())
    expected = {"-runTime=1234", "--pathway=0", "-memTest=1",
                "-removeHydrogens=0", "-compensateDisjoint=0"}
    if kind == "string":
        expected |= {"-runStrings=1", "-acceptPalindromes=1"}
    else:
        expected |= {"-enumMax=456", "--parallel=auto", "--threads=2", "--verbose=1",
                     "--telemetry=1", "-writeIntermediateMAs=1"}
    assert set(arguments) == expected
    assert len(arguments) == len(expected)


@pytest.mark.parametrize("kind", ["graph", "string"])
def test_wall_timeout_does_not_impose_cpu_budget(recording_calculator, kind):
    kwargs = dict(dir_code=recording_calculator, timeout=5, return_log_file=True)
    if kind == "graph":
        kwargs["cpp_options"] = att.AssemblyCppOptions(parallel="on", threads=2)
        result = att.calculate_assembly_index(graph(), **kwargs)
    else:
        result = att.calculate_string_assembly_index("abab", **kwargs)
    arguments = json.loads((Path(result[3]).parent / "arguments.json").read_text())
    assert not any("runTime" in arg or "runtime=" in arg for arg in arguments)


def test_public_api_covers_the_native_help_options():
    """Fail the weekly upstream check if a new CLI control needs exposing."""
    advertised = set(re.findall(r"^  --([\w-]+)=", att.get_assembly_cpp_help(), re.MULTILINE))
    exposed = {"runtime", "enum-max", "pathway", "accept-palindromes", "parallel", "threads",
               "verbose", "memory-report", "telemetry", "write-intermediate-mas"}
    wrapper_owned = {"run-strings", "remove-hydrogens", "compensate-disjoint"}
    assert advertised == (exposed | wrapper_owned) - ({"telemetry"} - advertised)


@pytest.mark.parametrize("kind", ["graph", "string"])
def test_native_pathway_can_be_disabled(kind):
    options = att.AssemblyCppOptions(pathway=False)
    result = (att.calculate_assembly_index(graph(), cpp_options=options) if kind == "graph"
              else att.calculate_string_assembly_index("abab", cpp_options=options))
    assert result == (2, None, None)


def test_native_runtime_and_enumeration_limits_are_bounds():
    for options in (att.AssemblyCppOptions(runtime_ticks=0), att.AssemblyCppOptions(enum_max=1)):
        assert att.calculate_assembly_index(graph(), cpp_options=options, exact=True)[0] == -1
        assert att.calculate_assembly_index(graph(), cpp_options=options)[0] >= 0


def test_native_intermediate_output_and_verbose_log_are_retained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = att.AssemblyCppOptions(verbose=True, write_intermediate_mas=True)
    result = att.calculate_assembly_index(graph(), cpp_options=options, return_log_file=True)
    directory = Path(result[3]).parent
    assert directory.is_dir()
    assert (directory / "graph_inIntermediateMAs").is_file()
    assert Path(result[3]).read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="VmPeak is Linux-specific")
def test_native_memory_report_is_retained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = att.calculate_string_assembly_index(
        "abab", cpp_options=att.AssemblyCppOptions(memory_report=True), return_log_file=True)
    assert (Path(result[3]).parent / "memUsage").is_file()


def test_native_reversal_matching_returns_zero_cost_reversal():
    result = att.calculate_string_assembly_index(
        "abcxcba", cpp_options=att.AssemblyCppOptions(accept_palindromes=True))
    assert result[0] == 4
    assert result[2] is not None
    assert nx.is_directed_acyclic_graph(result[2])
    assert any(data.get("operation") == "reverse" for _, _, data in result[2].edges(data=True))


def test_options_pass_through_parallel_batch_settings():
    options = att.AssemblyCppOptions(pathway=False, enum_max=100)
    assert att.calculate_assembly_index_parallel(
        [graph(), graph()], {"cpp_options": options}) == [[2, 2], [None, None], [None, None]]


@pytest.mark.parametrize("kind", ["graph", "string"])
def test_invalid_option_container_fails_before_resolving_calculator(monkeypatch, kind):
    monkeypatch.setattr(assembly, "add_assembly_to_path", lambda **kwargs: pytest.fail("backend lookup"))
    with pytest.raises(TypeError, match="AssemblyCppOptions"):
        if kind == "graph":
            att.calculate_assembly_index(graph(), cpp_options={"threads": 2})
        else:
            att.calculate_string_assembly_index("abab", cpp_options={"threads": 2})


def test_cfg_rejects_unused_cpp_options():
    with pytest.raises(ValueError, match="CFG"):
        att.calculate_string_assembly_index("abab", mode="cfg", cpp_options=att.AssemblyCppOptions())
