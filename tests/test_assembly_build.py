"""Cache, rebuild and installation contracts for the external C++ calculator."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


@pytest.fixture
def builder(assemblycpp_cache, monkeypatch):
    """Exercise the whole builder with one fake Git/CMake toolchain."""
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(assembly, "_which_build_tool", lambda name: f"/usr/bin/{name}")
    state = SimpleNamespace(
        refs=[], calls=[], fail=None, name=assembly._ASSEMBLYCPP_EXECUTABLE_NAMES[0]
    )

    def run(command, **kwargs):
        state.calls.append(command)
        if command[1:] == ["--version"]:
            return SimpleNamespace(stdout="cmake version 3.31.0\n")
        if command[:2] == ["git", "clone"]:
            (Path(command[-1]) / ".git").mkdir(parents=True)
        if "fetch" in command:
            state.refs.append(command[-1])
            # Allow simultaneous first calculations to contend for the lock.
            time.sleep(0.05)
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


def test_build_tools_use_the_python_environment_before_system_path(tmp_path, monkeypatch):
    """An absolute Python path must still use its pip-installed CMake and Ninja."""
    monkeypatch.setattr(assembly.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/system/{name}")
    for name in ("cmake", "ninja"):
        tool = tmp_path / name
        tool.write_text("environment tool")
        tool.chmod(0o755)
        assert assembly._which_build_tool(name) == str(tool)

    def version(command, **kwargs):
        assert command == [str(tmp_path / "cmake"), "--version"]
        return SimpleNamespace(stdout="cmake version 3.31.0\n")

    monkeypatch.setattr(assembly.subprocess, "run", version)
    assert assembly._require_cmake() == str(tmp_path / "cmake")


@pytest.mark.parametrize("present", [False, True])
def test_build_tools_fall_back_when_python_environment_tool_is_unusable(
    tmp_path, monkeypatch, present
):
    monkeypatch.setattr(assembly.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/system/{name}")
    if present:
        (tmp_path / "cmake").write_text("not executable")
    assert assembly._which_build_tool("cmake") == "/system/cmake"


@pytest.fixture
def assemblycpp_cache(tmp_path, monkeypatch):
    """Redirect the AssemblyCpp cache into tmp_path and isolate its environment."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Empty overrides also register cleanup for values the resolver later sets.
    for name in ("ASS_PATH", "ASS_STR_PATH", "ATT_ASSEMBLYCPP_REF"):
        monkeypatch.setenv(name, "")
    return tmp_path / "assemblytheorytools" / "assemblycpp"


def test_build_assembly_cpp_orchestration(assemblycpp_cache, builder):
    """The builder clones parallelassemblycpp, configures it safely, and installs it."""
    calls = builder.calls

    result = att.build_assembly_cpp()

    source = str(assemblycpp_cache / "src")
    build = str(assemblycpp_cache / "build")
    assert result == str(assemblycpp_cache / "bin" / "AssemblyCpp")
    assert Path(result).stat().st_mode & 0o111

    clone = next(argv for argv in calls if argv[:2] == ["git", "clone"])
    assert "--no-checkout" in clone
    assert clone[-2:] == [
        "https://github.com/ELIFE-ASU/parallelassemblycpp.git",
        source,
    ]
    assert ["git", "-C", source, "fetch", "--quiet", "origin", "main"] in calls

    configure = next(argv for argv in calls if "-S" in argv)
    assert configure[:5] == ["/usr/bin/cmake", "-S", source, "-B", build]
    # A newer compiler than parallelassemblycpp tests against must not fail the
    # build, and its test executables are not wanted here.
    assert "-DPARALLELASSEMBLYCPP_STRICT_WARNINGS=OFF" in configure
    assert "-DASSEMBLYCPP_STRICT_WARNINGS=OFF" in configure
    assert "-DBUILD_TESTING=OFF" in configure
    # cmake must be told where ninja is: a pip-installed one is not on PATH.
    assert configure[-3:-1] == ["-G", "Ninja"]
    assert configure[-1] == "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja"

    assert ["/usr/bin/cmake", "--build", build, "--config", "Release", "--parallel"] in calls
    assert ["/usr/bin/cmake", "--install", build, "--config", "Release", "--prefix",
            str(assemblycpp_cache / "build" / "install")] in calls
    # The build tree is transient; the source checkout is kept for rebuilds.
    assert not Path(build).exists()


@pytest.mark.parametrize("cached", [False, True])
def test_fetch_assembly_cpp_checks_out_requested_ref(tmp_path, monkeypatch, cached):
    """Fresh clones and renamed cached origins both resolve the requested ref."""
    upstream = tmp_path / "upstream"
    source = tmp_path / "cached-source"

    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(message):
        git("-C", upstream, "add", "revision.txt")
        git("-C", upstream, "-c", "user.name=Test", "-c",
            "user.email=test@example.com", "-c", "commit.gpgsign=false",
            "commit", "-m", message)

    git("init", "--initial-branch=main", upstream)
    revision = upstream / "revision.txt"
    revision.write_text("initial")
    commit("Initial revision")
    if cached:
        git("clone", upstream, source)
        git("-C", source, "remote", "set-url", "origin", tmp_path / "removed-origin")
    git("-C", upstream, "checkout", "-b", "requested")
    revision.write_text("updated")
    commit("Updated revision")
    expected = git("-C", upstream, "rev-parse", "HEAD")
    git("-C", upstream, "checkout", "main")
    monkeypatch.setattr(assembly, "_ASSEMBLYCPP_REPOSITORY", str(upstream))

    assembly._fetch_assembly_cpp(source, "requested")

    assert git("-C", source, "remote", "get-url", "origin") == str(upstream)
    assert (source / "revision.txt").read_text() == "updated"
    assert git("-C", source, "rev-parse", "HEAD") == expected


def test_build_assembly_cpp_reports_missing_build_tools(assemblycpp_cache, monkeypatch):
    """A missing or outdated cmake fails with advice instead of a build error."""
    monkeypatch.setattr(assembly.sys, "executable", str(assemblycpp_cache / "python"))
    monkeypatch.setattr(assembly.shutil, "which",
                        lambda name: None if name == "cmake" else f"/usr/bin/{name}")
    with pytest.raises(OSError, match="cmake was not found"):
        att.build_assembly_cpp()

    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        assembly.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(stdout="cmake version 3.22.1\n"),
    )
    with pytest.raises(OSError, match="needs cmake 3.25 or newer"):
        att.build_assembly_cpp()


def test_add_assembly_to_path_precedence(assemblycpp_cache, monkeypatch):
    """ASS_PATH wins; ASS_STR_PATH applies to string mode only and never leaks."""
    monkeypatch.setattr(assembly.shutil, "which", lambda name: None)
    monkeypatch.setattr(assembly, "build_assembly_cpp", lambda: "/built/AssemblyCpp")

    assert att.add_assembly_to_path() == "/built/AssemblyCpp"
    assert os.environ["ASS_PATH"] == "/built/AssemblyCpp"

    monkeypatch.setenv("ASS_PATH", "/configured/AssemblyCpp")
    monkeypatch.setenv("ASS_STR_PATH", "/strings/AssemblyCpp")

    assert att.add_assembly_to_path() == "/configured/AssemblyCpp"
    assert att.add_assembly_to_path(str_mode=True) == "/strings/AssemblyCpp"
    assert os.environ["ASS_PATH"] == "/configured/AssemblyCpp"


def test_add_assembly_to_path_finds_an_executable_before_building(
    assemblycpp_cache, monkeypatch
):
    """PATH is searched, then the cache; the builder is the last resort."""
    def unreachable():
        raise AssertionError("an existing executable must not trigger a build")

    monkeypatch.setattr(assembly, "build_assembly_cpp", unreachable)
    monkeypatch.setattr(assembly.shutil, "which", lambda name: "/usr/bin/AssemblyCpp")
    assert att.add_assembly_to_path() == "/usr/bin/AssemblyCpp"

    monkeypatch.delenv("ASS_PATH")
    monkeypatch.setattr(assembly.shutil, "which", lambda name: None)
    cached = assemblycpp_cache / "bin" / "AssemblyCpp"
    cached.parent.mkdir(parents=True)
    cached.write_text("cached executable")
    cached.chmod(0o755)
    monkeypatch.setattr(assembly, "build_assembly_cpp", att.build_assembly_cpp)
    monkeypatch.setattr(assembly, "_require_cmake", unreachable)

    assert att.add_assembly_to_path() == str(cached)
