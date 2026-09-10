"""External calculator process, timeout, logging and build contracts."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


_STATUS_OUT = "assembly index: 8\nstatus: runtime limit reached\ntime elapsed: 3\n"


@pytest.mark.parametrize(
    "exact, log_text, out_text, expected",
    [
        (False, "Best assembly index: 9 (1 ticks)\nBest assembly index: 7 (2 ticks)\n",
         "assembly index: 99\n", 6),
        (False, "min AI found so far: 9\nmin AI found so far: 7\n",
         "assembly index: 99\n", 6),
        (True, "Best assembly index: 7 (2 ticks)\n", "assembly index: 99\n", -1),
        (True, "min AI found so far: 7\n", "assembly index: 99\n", -1),
        (False, "No minimum available\n", "assembly index: 99\n", -1),
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

    The bound comes from the assembler's own output file when it stopped itself
    and said so, and from its log when it was killed first. Both the current
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
    assert att.run_command("echo hello") is None
    assert capfd.readouterr().out == "hello\n"

    with pytest.raises(ValueError):
        att.run_command(None)


def _fake_cmake_run(calls, prefix):
    """Return a subprocess.run stand-in that records argv and fakes cmake's effects."""
    def run(command, **kwargs):
        argv = [str(part) for part in command]
        calls.append(argv)
        if argv[1:] == ["--version"]:
            return SimpleNamespace(stdout="cmake version 3.31.0\n")
        if "-B" in argv:
            Path(argv[argv.index("-B") + 1]).mkdir(parents=True)
        if "--install" in argv:
            install = Path(argv[argv.index("--prefix") + 1])
            executable = install / "bin" / assembly._ASSEMBLYCPP_EXECUTABLE_NAMES[0]
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("compiled executable")
        return SimpleNamespace(returncode=0)

    return run


def _clear_env(monkeypatch, *names):
    """Unset variables so monkeypatch restores them even when they start unset.

    ``add_assembly_to_path`` caches its result by assigning ``os.environ``
    directly, which monkeypatch cannot undo, and ``delenv`` records no undo
    entry for a variable that was never set. Seeding a value first gives it one,
    so a fake path cannot leak into the rest of the session.
    """
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def assemblycpp_cache(tmp_path, monkeypatch):
    """Redirect the AssemblyCpp cache into tmp_path and isolate its environment."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _clear_env(monkeypatch, "ASS_PATH", "ASS_STR_PATH", "ATT_ASSEMBLYCPP_REF")
    return tmp_path / "assemblytheorytools" / "assemblycpp"


def test_build_assembly_cpp_orchestration(assemblycpp_cache, monkeypatch):
    """The builder clones parallelassemblycpp, configures it safely, and installs it."""
    calls = []
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(assembly.subprocess, "run",
                        _fake_cmake_run(calls, assemblycpp_cache))

    result = att.build_assembly_cpp()

    source = str(assemblycpp_cache / "src")
    build = str(assemblycpp_cache / "build")
    assert result == str(assemblycpp_cache / "bin" / "AssemblyCpp")
    assert Path(result).stat().st_mode & 0o111

    clone = next(argv for argv in calls if argv[:2] == ["git", "clone"])
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


def test_build_assembly_cpp_honours_the_ref_override(assemblycpp_cache, monkeypatch):
    """ATT_ASSEMBLYCPP_REF selects the revision, and a cached build is reused."""
    calls = []
    monkeypatch.setenv("ATT_ASSEMBLYCPP_REF", "some-feature-branch")
    monkeypatch.setattr(assembly.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(assembly.subprocess, "run",
                        _fake_cmake_run(calls, assemblycpp_cache))

    built = att.build_assembly_cpp()

    assert ["git", "-C", str(assemblycpp_cache / "src"), "fetch", "--quiet",
            "origin", "some-feature-branch"] in calls
    assert att.build_assembly_cpp() == built
    fetches = len([argv for argv in calls if "fetch" in argv])
    assert fetches == 1


def test_fetch_assembly_cpp_updates_cached_checkout_origin(tmp_path, monkeypatch):
    """A cached checkout still rebuilds after its original remote disappears."""
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
    git("clone", upstream, source)
    git("-C", source, "remote", "set-url", "origin", tmp_path / "removed-origin")
    revision.write_text("updated")
    commit("Updated revision")
    monkeypatch.setattr(assembly, "_ASSEMBLYCPP_REPOSITORY", str(upstream))

    assembly._fetch_assembly_cpp(source, "main")

    assert git("-C", source, "remote", "get-url", "origin") == str(upstream)
    assert (source / "revision.txt").read_text() == "updated"
    assert git("-C", source, "rev-parse", "HEAD") == git("-C", upstream, "rev-parse", "HEAD")


def test_build_assembly_cpp_reports_missing_build_tools(assemblycpp_cache, monkeypatch):
    """A missing or outdated cmake fails with advice instead of a build error."""
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
