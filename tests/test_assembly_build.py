"""Cache, rebuild and installation contracts for the external C++ calculator."""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from assemblytheorytools import assembly


@pytest.fixture
def builder(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("ATT_ASSEMBLYCPP_REF", raising=False)
    monkeypatch.setattr(assembly, "_require_cmake", lambda: "cmake")
    monkeypatch.setattr(assembly, "_which_build_tool", lambda name: None)
    state = SimpleNamespace(refs=[], fail=None, name=assembly._ASSEMBLYCPP_EXECUTABLE_NAMES[0])

    def fetch(source, ref):
        state.refs.append(ref)
        (source / ".git").mkdir(parents=True, exist_ok=True)
        # Leave time for simultaneous first calculations to contend for the lock.
        time.sleep(0.05)

    def run(command, **kwargs):
        if "-B" in command:
            Path(command[command.index("-B") + 1]).mkdir(parents=True)
        if state.fail in command:
            raise subprocess.CalledProcessError(1, command)
        if "--install" in command and state.fail != "missing-executable":
            install = Path(command[command.index("--prefix") + 1])
            executable = install / "bin" / state.name
            executable.parent.mkdir(parents=True)
            executable.write_text(state.refs[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(assembly, "_fetch_assembly_cpp", fetch)
    monkeypatch.setattr(assembly.subprocess, "run", run)
    return state


@pytest.mark.parametrize("via_environment", [False, True])
def test_changing_ref_rebuilds_cached_executable(builder, monkeypatch, via_environment):
    assembly.build_assembly_cpp(ref="old")
    if via_environment:
        monkeypatch.setenv("ATT_ASSEMBLYCPP_REF", "new")
        path = assembly.build_assembly_cpp()
        assert assembly.build_assembly_cpp() == path
    else:
        path = assembly.build_assembly_cpp(ref="new")
        assert assembly.build_assembly_cpp(ref="new") == path
    assert Path(path).read_text() == "new"
    assert builder.refs == ["old", "new"]
    assert json.loads((Path(path).parents[1] / "build.json").read_text())["ref"] == "new"


def test_force_refreshes_a_cached_ref(builder):
    path = assembly.build_assembly_cpp(ref="main")
    assert assembly.build_assembly_cpp(ref="main", force=True) == path
    assert builder.refs == ["main", "main"]


def test_explicit_ref_does_not_reuse_an_unrecorded_cache(builder):
    path = Path(assembly.build_assembly_cpp())
    (path.parents[1] / "build.json").unlink()
    assert assembly.build_assembly_cpp(ref="main") == str(path)
    assert builder.refs == ["main", "main"]


@pytest.mark.parametrize("failure", ["--build", "--install", "missing-executable"])
def test_failed_rebuild_preserves_previous_executable_and_diagnostics(builder, failure):
    path = Path(assembly.build_assembly_cpp(ref="working"))
    builder.fail = failure
    error = FileNotFoundError if failure == "missing-executable" else OSError
    with pytest.raises(error, match="left.*under"):
        assembly.build_assembly_cpp(ref="broken")
    assert path.read_text() == "working"
    assert (path.parents[1] / "src").is_dir()
    assert (path.parents[1] / "build").is_dir()
    assert assembly.build_assembly_cpp(ref="working") == str(path)
    assert builder.refs == ["working", "broken"]


def test_first_calculations_share_a_single_build(builder):
    ready = Barrier(2)

    def first_calculation():
        ready.wait(timeout=5)
        return assembly.build_assembly_cpp()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: first_calculation(), range(2)))
    assert results[0] == results[1]
    assert Path(results[0]).read_text() == "main"
    assert builder.refs == ["main"]


@pytest.mark.parametrize("legacy", [False, True])
def test_builder_accepts_current_and_legacy_upstream_names(builder, legacy):
    builder.name = assembly._ASSEMBLYCPP_EXECUTABLE_NAMES[int(legacy)]
    path = assembly.build_assembly_cpp()
    assert Path(path).name == assembly._ASSEMBLYCPP_EXECUTABLE
    assert Path(path).read_text() == "main"


@pytest.mark.parametrize("name", assembly._ASSEMBLYCPP_EXECUTABLE_NAMES)
def test_resolution_accepts_current_and_legacy_path_names(tmp_path, monkeypatch, name):
    monkeypatch.setenv("ASS_PATH", "")
    expected = str(tmp_path / name)
    monkeypatch.setattr(assembly.shutil, "which", lambda candidate: expected if candidate == name else None)
    assert assembly.add_assembly_to_path() == expected


@pytest.mark.parametrize("response", ["unrecognized", "failure"])
def test_cmake_must_report_a_usable_version(monkeypatch, response):
    monkeypatch.setattr(assembly.shutil, "which", lambda name: name)
    monkeypatch.setattr(assembly, "_which_build_tool", lambda name: name)

    def version(command, **kwargs):
        if response == "failure":
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="unexpected tool output")

    monkeypatch.setattr(assembly.subprocess, "run", version)
    with pytest.raises(OSError, match="version"):
        assembly._require_cmake()
