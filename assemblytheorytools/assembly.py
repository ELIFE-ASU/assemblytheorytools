"""
Assembly index calculation for molecules, strings and graphs.

This module wraps the external assembly calculators and exposes them through a
uniform interface. Three backends are supported: the ``AssemblyCpp``
executable from parallelassemblycpp, the ``assembly_theory`` Rust extension, and
``assemblycfg`` for context-free-grammar upper bounds. Helpers are provided for
locating and building the C++ executable, parsing its output, correcting joint
assembly indices, and deriving bounds, ratios and similarity measures.
"""

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from functools import cache, partial
from importlib.metadata import PackageNotFoundError, version
from math import ceil, isfinite
from pathlib import Path
from typing import (Union, List, Optional, Sequence, Tuple, Dict, Any,
                    NamedTuple, Callable, Hashable, Iterable)

import assembly_theory as at_rust
import assemblycfg
from filelock import FileLock
import networkx as nx
import numpy as np
from rdkit.Chem import AllChem as Chem

from ._cpp_options import AssemblyCppOptions
from .construction import (_VO_TYPES,
                           _VO_TYPE_ERROR,
                           parse_pathway_file,
                           parse_pathway_dot,
                           parse_string_pathway_file,
                           molstr_to_str,
                           convert_digraph_vo_to_target)
from .tools_file import _read_assembly_json
from .tools_graph import (write_ass_graph_file,
                          remove_hydrogen_from_graph,
                          nx_to_mol,
                          mol_to_nx,
                          nx_to_smi,
                          canonicalize_node_labels,
                          join_graphs)
from .tools_mp import mp_calc
from .tools_string import (prep_joint_string_ai,
                           get_dir_str_molecule,
                           get_undir_str_molecule)

# Patterns emitted by the C++ assembler, on its output file and log respectively
_AI_PATTERN = re.compile(r"assembly index:[ \t]*(\d+)[ \t]*$")
# parallelassemblycpp logs "Best assembly index: N (T clock ticks)"; the executables
# this package used to bundle logged "min AI found so far: N". Accept both, so an
# older binary on ASS_PATH still reports a bound after a timeout.
_MIN_AI_PATTERN = re.compile(r"(?:min AI found so far|Best assembly index):\s*(\d+)")
# Written to the output file when the search stopped before proving a minimum
_STATUS_PATTERN = re.compile(r"^status:[ \t]*(.+?)[ \t]*$", re.MULTILINE)

# The source of the C++ calculator. Tracking a branch rather than a pinned
# commit keeps ATT current with the calculator it drives; the weekly scheduled
# test run is what catches a breaking change there.
_ASSEMBLYCPP_REPOSITORY = "https://github.com/ELIFE-ASU/parallelassemblycpp.git"
_ASSEMBLYCPP_MINIMUM_CMAKE = (3, 25)
_ASSEMBLYCPP_EXECUTABLE = (
    "AssemblyCpp.exe" if platform.system() == "Windows" else "AssemblyCpp"
)
_ASSEMBLYCPP_EXECUTABLE_NAMES = (
    "Parallel" + _ASSEMBLYCPP_EXECUTABLE, _ASSEMBLYCPP_EXECUTABLE
)


def _read_assembly_output(file_path: str) -> tuple[int, Optional[str]]:
    """Read the index and any early-stop status in a single pass.

    The native string output includes the input before its result, so match
    the final numeric field rather than an ``assembly index:`` in that input.
    """
    ai, status = -1, None
    with open(file_path, encoding="utf-8") as output:
        for line in output:
            match = _AI_PATTERN.search(line)
            if match:
                if ai == -1:
                    ai = int(match.group(1))
                continue
            match = _STATUS_PATTERN.search(line)
            if match:
                status = match.group(1)
    return ai, status


def _scan_log_for_min_ai(log_file: str, debug: bool = False) -> int:
    """Recover the last logged bound without loading the whole log into memory."""
    bound = -1
    with open(log_file, encoding="utf-8", errors="replace") as log:
        for line in log:
            if debug:
                print(line, end="")
            match = _MIN_AI_PATTERN.search(line)
            if match:
                bound = int(match.group(1))
    return bound


def _count_edges(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """Count graph edges or RDKit molecule bonds."""
    return mol.number_of_edges() if isinstance(mol, nx.Graph) else mol.GetNumBonds()


def _prepare_bound_input(mol: Union[nx.Graph, Chem.Mol],
                         strip_hydrogen: bool) -> Union[nx.Graph, Chem.Mol]:
    """Validate a graph or molecule and optionally remove its hydrogens."""
    if isinstance(mol, nx.Graph):
        return remove_hydrogen_from_graph(mol) if strip_hydrogen else mol
    if isinstance(mol, Chem.Mol):
        return Chem.RemoveHs(mol) if strip_hydrogen else mol
    raise ValueError("Input not supported")


def load_assembly_output(file_path: str) -> int:
    """
    Load the assembly output from a file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the assembly output.

    Returns
    -------
    int
        The assembly index extracted from the file.

    Raises
    ------
    ValueError
        If the file contains no assembly index.
    """
    ai, _ = _read_assembly_output(file_path)
    if ai == -1:
        raise ValueError(f"No assembly index found in {file_path}")
    return ai


def run_command(command: str) -> None:
    """
    Run a command in a subprocess, streaming its output to the console.

    The subprocess inherits this process's stdout/stderr rather than having
    them captured, so output (e.g. from a long-running compile) is visible
    live rather than buffered until the command finishes.

    Parameters
    ----------
    command : str
        The command to run as a string.

    Returns
    -------
    None
        The command's output goes straight to the console; nothing is
        captured or returned.

    Raises
    ------
    ValueError
        If command is None.
    """
    if command is None:
        raise ValueError("Command must be provided")

    subprocess.run(command.split())


def _assemblycpp_cache_dir() -> Path:
    """
    Return the directory AssemblyCpp is built into and looked up from.

    Returns
    -------
    Path
        ``<cache>/assemblytheorytools/assemblycpp``, where ``<cache>`` is
        ``XDG_CACHE_HOME`` or ``~/.cache``.

    Notes
    -----
    The location sits outside the installed package on purpose:
    ``site-packages`` is often read only and is replaced on upgrade. Honouring
    ``XDG_CACHE_HOME`` also lets the test suite redirect the cache.
    """
    root = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(root).expanduser() / "assemblytheorytools" / "assemblycpp"


def add_assembly_to_path(str_mode: bool = False) -> str:
    """
    Return the path to the AssemblyCpp executable, building it if necessary.

    A single ParallelAssemblyCpp executable computes molecular, graph and string
    assembly indices; string mode is selected per call with ``-runStrings=1``
    rather than by a separate binary. ``ASS_STR_PATH`` is therefore only an
    override for pointing string calculations at a different build.

    Parameters
    ----------
    str_mode : bool, optional
        If True, honour ``ASS_STR_PATH`` before falling back to the shared
        executable. Default is False.

    Returns
    -------
    str
        Path to the AssemblyCpp executable.

    Raises
    ------
    OSError
        If no executable is found and the build tools needed to produce one are
        missing or the build fails.
    FileNotFoundError
        If the build reports success but installs nothing.

    Notes
    -----
    Resolution order is ``ASS_STR_PATH`` (only when *str_mode*), ``ASS_PATH``,
    ``ParallelAssemblyCpp`` (or the older ``AssemblyCpp``) on ``PATH``, the
    cached build under
    ``$XDG_CACHE_HOME/assemblytheorytools/assemblycpp``, and finally a fresh
    :func:`build_assembly_cpp`. A resolved path is cached in ``ASS_PATH``, so
    the search and any build happen once per process. A value taken from
    ``ASS_STR_PATH`` is never written to ``ASS_PATH``.
    """
    if str_mode and os.environ.get("ASS_STR_PATH"):
        return os.environ["ASS_STR_PATH"]
    if os.environ.get("ASS_PATH"):
        return os.environ["ASS_PATH"]

    executable = next((found for name in _ASSEMBLYCPP_EXECUTABLE_NAMES
                       if (found := shutil.which(name))), None)
    if executable is None:
        executable = build_assembly_cpp()

    os.environ["ASS_PATH"] = executable
    return executable


def get_assembly_cpp_help(dir_code: Optional[str] = None) -> str:
    """Return the selected calculator's ``--help``, including build-specific options.

    Locate or build the calculator when ``dir_code`` is omitted. A failed
    executable raises the corresponding subprocess error.
    """
    executable = dir_code if dir_code is not None else add_assembly_to_path()
    return subprocess.run([os.path.expanduser(os.fspath(executable)), "--help"],
                          check=True, capture_output=True, text=True).stdout


def _which_build_tool(name: str) -> Optional[str]:
    """
    Locate a build tool beside the running interpreter, then on ``PATH``.

    Parameters
    ----------
    name : str
        Executable to look for, such as ``cmake``.

    Returns
    -------
    Optional[str]
        Path to the executable, or None if it was not found.

    Notes
    -----
    cmake and ninja are dependencies of this package, so pip installs them into
    the same directory as the interpreter. That directory is not on ``PATH``
    unless the environment has been activated, which is easy to miss when a
    script is run through an absolute path to the interpreter. Prefer those
    declared dependencies over a potentially older system installation.
    """
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidate = Path(sys.executable).with_name(name + suffix)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(name)


def _ninja_can_find_a_compiler() -> bool:
    """
    Report whether the Ninja generator would find a compiler to drive.

    Returns
    -------
    bool
        True if configuring with Ninja is safe on this platform.

    Notes
    -----
    Ninja does no toolchain discovery of its own, so on Windows it only works
    from a Visual Studio developer environment, which an ordinary Python session
    is not. Leaving the generator unset there lets CMake select the Visual
    Studio generator, which locates MSVC itself; the build and install steps
    already pass ``--config Release`` for that multi-configuration generator.
    """
    return os.name != "nt" or shutil.which("cl") is not None


def _require_cmake() -> str:
    """
    Return the cmake command to build with, raising if the build tools are unusable.

    Returns
    -------
    str
        Absolute path to a cmake new enough to configure parallelassemblycpp.

    Raises
    ------
    OSError
        If ``git`` or ``cmake`` is missing, or cmake is older than
        :data:`_ASSEMBLYCPP_MINIMUM_CMAKE`.
    """
    if shutil.which("git") is None:
        raise OSError(
            "Cannot build AssemblyCpp: git was not found on PATH. Install git, "
            "or set ASS_PATH to an existing AssemblyCpp executable."
        )

    cmake = _which_build_tool("cmake")
    minimum = ".".join(str(part) for part in _ASSEMBLYCPP_MINIMUM_CMAKE)
    advice = (
        f'Install them with `pip install "cmake>={minimum}" ninja`, or create the '
        f"conda environment from the environment.yml in {_ASSEMBLYCPP_REPOSITORY}. "
        f"Alternatively set ASS_PATH to an existing AssemblyCpp executable."
    )
    if cmake is None:
        raise OSError(f"Cannot build AssemblyCpp: cmake was not found on PATH. {advice}")

    try:
        report = subprocess.run([cmake, "--version"], check=True,
                                capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise OSError(f"Cannot build AssemblyCpp: {cmake} --version failed. {advice}") from error
    found = re.search(r"cmake version (\d+)\.(\d+)", report)
    if found is None:
        raise OSError(f"Cannot build AssemblyCpp: {cmake} reported no CMake version. {advice}")
    if tuple(int(part) for part in found.groups()) < _ASSEMBLYCPP_MINIMUM_CMAKE:
        raise OSError(
            f"Cannot build AssemblyCpp: it needs cmake {minimum} or newer, but "
            f"{cmake} reports {'.'.join(found.groups())}. {advice}"
        )
    return cmake


def _fetch_assembly_cpp(source: Path, ref: str) -> None:
    """
    Clone or update the parallelassemblycpp checkout at *source* and check out *ref*.

    Parameters
    ----------
    source : Path
        Directory holding the checkout. Created if absent.
    ref : str
        Branch, tag or commit to check out.

    Returns
    -------
    None

    Notes
    -----
    Fetching the ref by name and checking out ``FETCH_HEAD`` handles branches,
    tags and bare commit hashes identically. The blobless clone skips its
    default checkout so only the requested revision's files are downloaded.
    """
    if not (source / ".git").is_dir():
        shutil.rmtree(source, ignore_errors=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
             _ASSEMBLYCPP_REPOSITORY, str(source)],
            check=True,
        )
    else:
        # Cached checkouts can predate an upstream repository rename. Use the
        # current URL on rebuild instead of depending on a hosting redirect.
        subprocess.run(
            ["git", "-C", str(source), "remote", "set-url", "origin",
             _ASSEMBLYCPP_REPOSITORY],
            check=True,
        )

    subprocess.run(["git", "-C", str(source), "fetch", "--quiet", "origin", ref],
                   check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach",
                    "FETCH_HEAD"], check=True)


def _cached_assembly_cpp(prefix: Path, ref: Optional[str]) -> Optional[str]:
    """Reuse an executable only when its recorded source matches the request.

    Older caches have no build record. They remain usable without an explicit
    ref, but cannot satisfy a request for a particular branch, tag or commit.
    """
    executable = prefix / "bin" / _ASSEMBLYCPP_EXECUTABLE
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return None
    try:
        record = json.loads((prefix / "build.json").read_text())
    except FileNotFoundError:
        return str(executable) if ref is None else None
    except (OSError, ValueError):
        return None
    if record == {"repository": _ASSEMBLYCPP_REPOSITORY, "ref": ref or "main"}:
        return str(executable)
    return None


def build_assembly_cpp(ref: Optional[str] = None, force: bool = False) -> str:
    """
    Build parallelassemblycpp and install its executable into the ATT cache.

    Clone or update `parallelassemblycpp <https://github.com/ELIFE-ASU/parallelassemblycpp>`_,
    configure and build it with CMake, and install the executable under
    ``$XDG_CACHE_HOME/assemblytheorytools/assemblycpp`` (``~/.cache`` by
    default). The cache retains the historical executable name ``AssemblyCpp``.

    Parameters
    ----------
    ref : str, optional
        Branch, tag or commit to build. Defaults to ``ATT_ASSEMBLYCPP_REF`` if
        set, otherwise ``main``. Changing the ref rebuilds a cached executable.
    force : bool, optional
        Rebuild even when the cache already holds the requested ref. Use this
        to fetch updates to a branch or tag. Default is False.

    Returns
    -------
    str
        Path to the installed executable.

    Raises
    ------
    OSError
        If the build tools are missing or too old, or a build step fails.
    FileNotFoundError
        If the build succeeds but installs no executable.

    Notes
    -----
    - This clones over the network and runs a compiler. Set ``ASS_PATH`` to use
      an executable you built yourself.
    - CMake is configured explicitly instead of using the upstream ``release``
      preset: warnings are not errors, and tests and telemetry are disabled.
    - parallelassemblycpp is licensed CC BY-NC 4.0, which is more restrictive than
      this package's MIT licence. The executable is built on demand rather than
      distributed with ATT.
    - A file lock serializes builds across processes. Installation is staged
      before replacing the cached executable, so a failed build preserves it.
    - On success the build tree is removed and the source checkout is kept.
      On failure both are left in place for inspection.
    """
    prefix = _assemblycpp_cache_dir()
    prefix.mkdir(parents=True, exist_ok=True)
    ref = ref or os.environ.get("ATT_ASSEMBLYCPP_REF") or None
    with FileLock(str(prefix / "build.lock")):
        cached = _cached_assembly_cpp(prefix, ref)
        if cached is not None and not force:
            return cached
        return _build_assembly_cpp(prefix, ref or "main")


def _build_assembly_cpp(prefix: Path, ref: str) -> str:
    """Build and publish one executable while the caller holds the cache lock."""
    cmake = _require_cmake()
    source = prefix / "src"
    build = prefix / "build"
    install = build / "install"
    executable = prefix / "bin" / _ASSEMBLYCPP_EXECUTABLE

    print(f"Building parallelassemblycpp ({ref}) in {build}", flush=True)
    start_time = time.monotonic()
    configure = [
        cmake, "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_BINDIR=bin",
        "-DBUILD_TESTING=OFF",
        "-DPARALLELASSEMBLYCPP_STRICT_WARNINGS=OFF",
        "-DPARALLELASSEMBLYCPP_BUILD_TELEMETRY=OFF",
        # Keep explicit older refs buildable across upstream's project rename.
        "-DASSEMBLYCPP_STRICT_WARNINGS=OFF",
        "-DASSEMBLYCPP_BUILD_TELEMETRY=OFF",
    ]
    # A pip-installed ninja can sit beside the interpreter, outside PATH.
    ninja = _which_build_tool("ninja")
    if ninja is not None and _ninja_can_find_a_compiler():
        configure += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]

    # A failed build can retain an incompatible CMake generator or revision.
    shutil.rmtree(build, ignore_errors=True)
    try:
        _fetch_assembly_cpp(source, ref)
        subprocess.run(configure, check=True)
        subprocess.run([cmake, "--build", str(build), "--config", "Release",
                        "--parallel"], check=True)
        subprocess.run([cmake, "--install", str(build), "--config", "Release",
                        "--prefix", str(install)], check=True)
    except subprocess.CalledProcessError as error:
        raise OSError(
            f"Building parallelassemblycpp failed: {' '.join(str(part) for part in error.cmd)} "
            f"exited with {error.returncode}. The source and build trees are left "
            f"under {prefix} for inspection; see {_ASSEMBLYCPP_REPOSITORY} for the "
            f"build instructions."
        ) from error

    installed = next((install / "bin" / name for name in _ASSEMBLYCPP_EXECUTABLE_NAMES
                      if (install / "bin" / name).is_file()), None)
    if installed is None:
        raise FileNotFoundError(
            f"parallelassemblycpp built successfully but installed no executable "
            f"under {install}. The source and build trees are left under {prefix} "
            "for inspection."
        )

    installed.chmod(0o755)
    record = build / "build.json"
    record.write_text(json.dumps({"repository": _ASSEMBLYCPP_REPOSITORY, "ref": ref}) + "\n")
    executable.parent.mkdir(parents=True, exist_ok=True)
    installed.replace(executable)
    record.replace(prefix / "build.json")
    shutil.rmtree(build, ignore_errors=True)
    print(f"Build time: {time.monotonic() - start_time:.2f} seconds", flush=True)
    return str(executable)


def joint_assembly_index_correction(mol: Union[nx.Graph, Chem.Mol], ass_index: int) -> int:
    """
    Correct the assembly index based on the joint assembly components.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule or graph.
    ass_index : int
        The original assembly index.

    Returns
    -------
    int
        The corrected assembly index.

    Raises
    ------
    ValueError
        If the input type is not supported.
    """
    if isinstance(mol, nx.Graph):
        num_components = sum(len(component) > 1 for component in nx.connected_components(mol))
    elif isinstance(mol, Chem.Mol):
        num_components = sum(len(fragment) > 1 for fragment in Chem.rdmolops.GetMolFrags(mol))
    else:
        raise ValueError("Input not supported")

    # The calculator assembles bonds; isolated atoms add no joining operations.
    return ass_index - max(0, num_components - 1)


def _validate_cpp_timeout(timeout: Optional[float]) -> None:
    """Reject unusable budgets before creating files or resolving a calculator."""
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be None or a finite, non-negative number of seconds")


def _cpp_options(options: Optional[AssemblyCppOptions]) -> AssemblyCppOptions:
    """Resolve the optional, typed C++ controls at the public API boundary."""
    if options is None:
        options = AssemblyCppOptions()
    if not isinstance(options, AssemblyCppOptions):
        raise TypeError("cpp_options must be an AssemblyCppOptions instance or None")
    return options


@contextmanager
def _calculation_directory(*, save: bool = False, debug: bool = False,
                           return_log_file: bool = False):
    """Own all run files, retaining them only when the caller requests them."""
    directory = Path(
        tempfile.mkdtemp(prefix="ai_calc_", dir=".")
        if save or debug else tempfile.mkdtemp()
    ).resolve()
    if save or debug:
        print(f"Calculation directory: {directory}", flush=True)
    try:
        yield directory
    finally:
        if not (save or debug or return_log_file):
            shutil.rmtree(directory)


def _calculator_error(message: str, log_file: str) -> OSError:
    """Include the end of the log even when temporary files will be removed."""
    with open(log_file, "rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - 4096))
        detail = log.read().decode(errors="replace").strip()
    return OSError(f"{message}. AssemblyCpp log: {log_file}" +
                   (f"\n{detail}" if detail else ""))


def _run_assembler(dir_code: str, file_path_in: str, log_file: str,
                   timeout: Optional[float], debug: bool, *,
                   arguments: Sequence[str]) -> bool:
    """Run either C++ mode with a bounded wait and a log streamed to disk.

    SIGINT gives the calculator two seconds to save its best result. A process
    that ignores the interrupt is killed and reaped before its files are read.
    """
    executable = os.path.expanduser(os.fspath(dir_code))
    if os.path.dirname(executable):
        executable = os.path.abspath(executable)
    command = [executable, file_path_in, *arguments]
    if debug:
        print(f"Calling: {command}", flush=True)

    timed_out = False
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            cwd=os.path.dirname(file_path_in),
        )
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                print("Warning: Assembly calculation timed out.", flush=True)
                # The package supports POSIX. On Windows, terminate instead:
                # Popen cannot deliver SIGINT to an ordinary child process.
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            # Also reap children if the caller interrupts Python or an error
            # occurs while requesting the calculator's cooperative shutdown.
            if process.poll() is None:
                process.kill()
                process.wait()

    if process.returncode and not timed_out:
        raise _calculator_error(
            f"AssemblyCpp exited with status {process.returncode}", log_file)
    return timed_out


def _read_calculation_index(file_path_out: str, log_file: str, timed_out: bool,
                            *, exact: bool = False, debug: bool = False) -> int:
    """Interpret completion and bounds identically for molecules and strings."""
    try:
        ai, status = _read_assembly_output(file_path_out)
    except FileNotFoundError:
        ai, status = -1, None
    if status is not None:
        print(f"Warning: the assembly search stopped early ({status}).", flush=True)
    if timed_out or status is not None:
        # Prefer a saved result even when the wall-clock deadline raced with
        # normal completion. The log is only a fallback for missing output.
        bound = ai
        if bound == -1 and os.path.isfile(log_file):
            bound = _scan_log_for_min_ai(log_file, debug=debug)
        if bound == -1:
            print("No assembly index found before the search stopped.", flush=True)
        elif exact:
            print(f"Discarding the inexact bound AI <= {bound}.", flush=True)
        else:
            print(f"Upper bound to AI found: AI <= {bound}", flush=True)
        return -1 if exact else bound

    if ai == -1:
        raise _calculator_error("AssemblyCpp produced no assembly index", log_file)
    return ai


def calculate_assembly_index(graph: Union[nx.Graph, Chem.Mol],
                             dir_code: Optional[str] = None,
                             timeout: Optional[float] = 100.0,
                             save_dir: bool = False,
                             debug: bool = False,
                             joint_corr: bool = True,
                             strip_hydrogen: bool = False,
                             return_log_file: bool = False,
                             canonicalize: bool = True,
                             exact: bool = False, *,
                             cpp_options: Optional[AssemblyCppOptions] = None) -> Union[Tuple[int, Any, Any], Tuple[int, Any, Any, Optional[str]]]:
    """
    Calculate the assembly index for a given graph or molecule.

    This function computes an (optionally joint) assembly index for the input
    molecular graph or RDKit molecule using the external assembly calculator.
    It manages temporary file creation, process execution, and output parsing,
    and supports both single and joint assembly index calculations.

    Parameters
    ----------
    graph : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.
    dir_code : str, optional
        Path to the assembly executable. If None, locate or build it via
        :func:`add_assembly_to_path`.
    timeout : float, optional
        Maximum wall-clock search time in seconds; must be finite and
        non-negative, or None for no wall-clock limit. This is independent of
        the C++ CPU-time budget in ``cpp_options.runtime_ticks``.
        A timed-out calculator gets up to two more seconds to save its result
        before it is killed. Default is 100.0.
    save_dir : bool, optional
        If True, save the temporary files and directories used for the calculation.
        Default is False.
    debug : bool, optional
        If True, print debug information and keep temporary files.
        Default is False.
    joint_corr : bool, optional
        If True, apply joint assembly index correction based on graph components.
        Default is True.
    strip_hydrogen : bool, optional
        If True, remove hydrogen atoms from the graph before calculation.
        Default is False.
    return_log_file : bool, optional
        If True, return the path to the log file produced by the external run as
        the fourth element of the returned tuple, retaining its directory.
        No log is produced for a trivial input. Default is False.
    canonicalize : bool, optional
        If True, canonicalize the node labels in the graph.
        Default is True.
    exact : bool, optional
        If True, require a proven minimum: return -1 rather than the best bound
        the calculator reached when its search stopped early. Default is False.
    cpp_options : AssemblyCppOptions, optional
        C++ search and output controls: parallelism, threads, enumeration and
        CPU-time limits, pathway output and diagnostics. Diagnostic output
        requests retain the calculation directory and print its location.
        Hydrogen stripping and joint correction remain controlled by the
        corresponding Python arguments so input and pathway labels agree.

    Returns
    -------
    tuple
        If return_log_file is False returns a 3-tuple: (ai, virt_obj, path) where
        ai is the (possibly joint) assembly index (int), virt_obj is a list
        or other representation of virtual objects (or None), and path is the
        pathway representation (or None). If return_log_file is True returns
        (ai, virt_obj, path, log_file) where log_file is the path to the
        assembler log produced in the temporary ai_calc_* folder.

    Raises
    ------
    ValueError
        If the input graph is not supported.
    OSError
        If there are issues with file system access, process execution, or
        if required external tools or compiled executables are not available.

    Notes
    -----
    - When the calculator times out, the best bound logged so far is returned
      instead, unless ``exact`` is True in which case ``-1`` is returned.
    - Temporary working directories named like ``ai_calc_<unique suffix>`` are created
      in the working directory when ``save_dir`` (or ``debug``) is True; otherwise
      a system temporary directory is used. Files are removed on completion
      or failure unless ``save_dir``, ``debug`` or ``return_log_file`` is True.
    - For reproducible behaviour consider using ``debug=True`` to preserve the
      temporary folder and log files.
    - ``strip_hydrogen=True`` strips a copy, so ``graph`` is left unchanged
      and stays safe to reuse for a later calculation.
    - The order of ``virt_obj`` is not stable between runs; compare virtual
      objects as a set rather than by position.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
    >>> ai, virt_obj, pathway = att.calculate_assembly_index(
    ...     graph, strip_hydrogen=True)
    >>> ai
    9
    >>> len(virt_obj)
    14

    Hydrogens change the answer, so strip them for anything compared
    against published molecular assembly indices. Stripping works on a
    copy, so one graph can serve both calls:

    >>> ethanol = att.smi_to_nx("CCO")
    >>> att.calculate_assembly_index(ethanol)[0]
    6
    >>> att.calculate_assembly_index(ethanol, strip_hydrogen=True)[0]
    1
    >>> ethanol.number_of_nodes()
    9
    """
    _validate_cpp_timeout(timeout)
    options = _cpp_options(cpp_options)
    arguments = options._arguments(str_mode=False)
    molecule_input = isinstance(graph, Chem.Mol)
    if molecule_input:
        graph = mol_to_nx(graph)
    elif not isinstance(graph, nx.Graph):
        raise ValueError("Input must be a NetworkX graph or RDKit molecule")
    if strip_hydrogen:
        graph = remove_hydrogen_from_graph(graph)
    if canonicalize:
        graph = canonicalize_node_labels(graph)

    has_edges = graph.number_of_edges() > 0
    with _calculation_directory(save=save_dir or (options._retain_files and has_edges), debug=debug,
                                return_log_file=return_log_file and has_edges) as directory:
        file_path_in = str(directory / "graph_in")
        file_path_out = file_path_in + "Out"
        file_path_pathway = file_path_in + "Pathway"
        log_file = str(directory / "assembly_output.log")
        # Validate and serialize before resolving a possibly missing executable.
        write_ass_graph_file(graph, file_name=file_path_in)
        if not has_edges:
            return (0, None, None, None) if return_log_file else (0, None, None)
        if dir_code is None:
            dir_code = add_assembly_to_path()
        timed_out = _run_assembler(dir_code, file_path_in, log_file, timeout, debug,
                                   arguments=arguments)
        ai = _read_calculation_index(file_path_out, log_file, timed_out,
                                     exact=exact, debug=debug)
        virtual_objects = pathway = None
        if options.pathway and os.path.isfile(file_path_pathway):
            try:
                pathway, virtual_objects = parse_pathway_file(
                    file_path_pathway, vo_type="graph", debug=debug, input_graph=graph)
                if molecule_input:
                    virtual_objects = [nx_to_smi(v, add_hydrogens=False)
                                       for v in virtual_objects]
                    pathway = convert_digraph_vo_to_target(pathway, target="smi")
            except Exception as error:
                print(f"Failed to load pathway data: {error}", flush=True)
                if debug:
                    traceback.print_exc()

        if joint_corr and ai > 0:
            ai = joint_assembly_index_correction(graph, ai)
        result = (ai, virtual_objects, pathway)
        if return_log_file:
            print(f"Log file printed to: {log_file}", flush=True)
            return (*result, log_file)
        return result


def _calculate_assembly_indices(graphs: List[Union[nx.Graph, Chem.Mol]],
                                settings: Optional[Dict[str, Any]],
                                parallel: bool) -> List[Any]:
    """Collect only the assembly indices from serial or parallel calculations."""
    settings = settings or {}
    if parallel:
        return calculate_assembly_index_parallel(graphs, settings)[0]
    return [calculate_assembly_index(graph, **settings)[0] for graph in graphs]


def calculate_assembly(graphs: List[Union[nx.Graph, Chem.Mol]],
                       n_i: List[float],
                       settings: Optional[Dict[str, Any]] = None,
                       parallel: bool = True) -> float:
    """
    Calculate the assembly index for a list of graphs.

    This function computes the assembly index for each graph in the input list
    (graphs) using the calculate_assembly_index function. It then regularizes
    the assembly indices to ensure non-negative values and computes the weighted
    sum of the exponential of the assembly indices.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, Chem.Mol]]
        A list of molecular graphs or RDKit molecules to analyze.
    n_i : List[float]
        A list of weights corresponding to each graph, used for weighted sum calculation.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_assembly_index.
        If None, an empty dictionary is used. Default is None.
    parallel : bool, optional
        If True, run calculations in parallel using multiple processes.
        Default is True.

    Returns
    -------
    float
        The overall assembly index for the combined system of graphs.

    Raises
    ------
    ValueError
        If the input graphs are not of the same type or if the list lengths do not match.

    See Also
    --------
    calculate_assembly_from_indices : The same equation, when the assembly
        indices are already known.
    count_copies : Collapse repeated objects into the unique objects and
        copy numbers this function expects.

    Notes
    -----
    The equation sums over *unique* objects. Repeated entries are not
    collapsed, so pass each object once with its copy number.
    """

    ai_list = _calculate_assembly_indices(graphs, settings, parallel)
    return calculate_assembly_from_indices(ai_list, n_i)


def calculate_string_assembly(strings: List[str],
                              n_i: List[float],
                              settings: Optional[Dict[str, Any]] = None) -> float:
    """
    Calculate the assembly index for a list of strings.

    This function computes the assembly index for each string in the input list
    (strings) using the calculate_string_assembly_index function. It then regularizes
    the assembly indices to ensure non-negative values and computes the weighted
    sum of the exponential of the assembly indices.

    Parameters
    ----------
    strings : List[str]
        A list of strings to analyze.
    n_i : List[float]
        A list of weights corresponding to each string, used for weighted sum calculation.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_string_assembly_index.
        If None, an empty dictionary is used. Default is None.

    Returns
    -------
    float
        The overall assembly index for the combined system of strings.

    Raises
    ------
    ValueError
        If the input strings are not of the same type or if the list lengths do not match.

    See Also
    --------
    calculate_assembly_from_indices : The same equation, when the assembly
        indices are already known.
    count_copies : Collapse repeated objects into the unique objects and
        copy numbers this function expects.

    Notes
    -----
    The equation sums over *unique* objects. Repeated entries are not
    collapsed, so pass each object once with its copy number.
    """
    settings = settings or {}

    ai_list = [calculate_string_assembly_index(string, **settings)[0]
               for string in strings]
    return calculate_assembly_from_indices(ai_list, n_i)


def calculate_assembly_from_indices(ai_list: Sequence[Optional[int]],
                                    n_i: Sequence[float]) -> float:
    """
    Combine known assembly indices and copy numbers into an assembly value.

    The indices are regularised to be non-negative and then combined through
    the assembly equation, the copy-number weighted sum of their
    exponentials ``exp(a_i) * (n_i - 1) / N_T``, where ``N_T`` is the total
    copy number ``sum(n_i)``.

    Use this when the indices are already in hand: composed from several
    parts, measured rather than computed, or loaded from an earlier run.
    :func:`calculate_assembly` and :func:`calculate_string_assembly` compute
    the indices themselves and are the right entry points otherwise.

    Parameters
    ----------
    ai_list : Sequence[Optional[int]]
        Assembly index of each unique object. ``None`` and negative values
        mark a failed calculation and are regularised to zero, so a timed-out
        object still contributes ``(n_i - 1) / N_T``. Any sized sequence is
        accepted, including a NumPy array or a DataFrame column.
    n_i : Sequence[float]
        Copy number of each object, in the same order as `ai_list`.

    Returns
    -------
    float
        The assembly of the ensemble.

    Raises
    ------
    ValueError
        If the two sequences have different lengths, if either is empty, or
        if the copy numbers sum to zero.

    See Also
    --------
    count_copies : Collapse repeated objects into the counted input this
        function expects.
    exploration_ratio : The companion ensemble measure, over the joint
        assembly space rather than over copy numbers.

    Notes
    -----
    The equation assumes one entry per *unique* object, and nothing here
    enforces that. Passing the same object twice counts it as two species.

    An object seen once contributes nothing, because its ``n_i - 1`` factor
    is zero. That is the point of the equation: a single complex object is
    weak evidence of anything, while many copies of one are not.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.calculate_assembly_from_indices([1, 9], [100.0, 100.0])
    4012.372093654902

    Ethanol has an assembly index of 1 and caffeine 9. Drop caffeine to a
    single copy and it stops contributing, so assembly collapses:

    >>> att.calculate_assembly_from_indices([1, 9], [100.0, 1.0])
    2.664454465519262
    """
    if len(ai_list) != len(n_i):
        raise ValueError(
            f"ai_list and n_i must be the same length, "
            f"got {len(ai_list)} and {len(n_i)}"
        )
    if len(ai_list) == 0:
        raise ValueError("ai_list and n_i must not be empty")

    n_t = sum(n_i)
    if n_t == 0:
        raise ValueError("The copy numbers must not sum to zero")

    indices = [regularise_assembly_index(ai) for ai in ai_list]
    return float(sum(np.exp(ai) * ((n - 1) / n_t)
                     for ai, n in zip(indices, n_i)))


def count_copies(objects: Iterable[Any],
                 key: Optional[Callable[[Any], Hashable]] = None
                 ) -> Tuple[List[Any], List[int]]:
    """
    Collapse repeated objects into unique objects and their copy numbers.

    The assembly equation is a sum over *unique* objects weighted by how
    often each was observed, but the calculators take a flat list. This turns
    one into the other: the two returned lists are aligned and can be passed
    straight to :func:`calculate_assembly`,
    :func:`calculate_string_assembly` or
    :func:`calculate_assembly_from_indices`.

    Parameters
    ----------
    objects : Iterable[Any]
        The observed objects, with repeats. Order is preserved.
    key : Optional[Callable[[Any], Hashable]], optional
        Maps an object to the hashable identity that decides whether two
        objects are the same. Defaults to the object itself, which suits
        strings and other hashables. Pass ``Chem.MolToInchi`` for RDKit
        molecules, whose objects compare by identity rather than structure.

    Returns
    -------
    Tuple[List[Any], List[int]]
        The first object seen for each distinct identity, in first-seen
        order, and the number of times each was seen.

    See Also
    --------
    assemblytheorytools.reassembler.get_unique_mols : Deduplicates RDKit
        molecules by InChI without returning the counts.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.count_copies(["abab", "cdcd", "abab", "abab"])
    (['abab', 'cdcd'], [3, 1])

    The result feeds the assembly equation directly. Here both strings have
    an assembly index of 1, and only the repeated one contributes:

    >>> strings, n_i = att.count_copies(["ab", "ab", "cd"])
    >>> att.calculate_assembly_from_indices([1, 1], n_i)
    0.9060939428196817
    """
    unique: List[Any] = []
    counts: List[int] = []
    seen: Dict[Hashable, int] = {}

    for obj in objects:
        identity = obj if key is None else key(obj)
        if identity not in seen:
            seen[identity] = len(unique)
            unique.append(obj)
            counts.append(0)
        counts[seen[identity]] += 1

    return unique, counts


def joint_assembly_space(pathways: Sequence[nx.DiGraph],
                         node_key: Optional[str] = None) -> nx.DiGraph:
    """
    Compose individual assembly pathways into a joint assembly space.

    The exact joint assembly space of a large ensemble is out of reach, so it
    is approximated by the union of the individual minimum pathways, sharing
    every intermediate that appears in more than one of them. This is the
    approximation described under :term:`joint assembly space`.

    Parameters
    ----------
    pathways : Sequence[nx.DiGraph]
        One minimum pathway per object, as returned third by
        :func:`calculate_assembly_index` or
        :func:`calculate_string_assembly_index`.
    node_key : Optional[str], optional
        Node attribute holding the object each node stands for. When given,
        nodes are relabelled by it before composing, so that nodes carrying
        the same object merge. Needed for pathways whose node identifiers are
        positional rather than the object itself, such as the ``step_N`` and
        ``virtual_object_N`` identifiers the molecular calculator produces;
        pass ``node_key="vo"`` for those. Leave it as ``None`` when the node
        identifiers already are the objects.

    Returns
    -------
    nx.DiGraph
        The union of the pathways. Its nodes are the observed objects
        together with the contingent intermediates needed to build them.

    Raises
    ------
    ValueError
        If `pathways` is empty.
    KeyError
        If `node_key` is given and some node lacks that attribute.

    See Also
    --------
    exploration_ratio : How much of this space the observed objects occupy.
    calculate_assembly_index_pairwise_joint : Builds the same union from
        every pair of a set of graphs rather than from precomputed pathways.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> paths = [att.calculate_string_assembly_index(s, mode="cfg")[2]
    ...          for s in ("gavhp", "gavhh")]
    >>> jas = att.joint_assembly_space(paths)
    >>> sorted(jas.nodes)[:5]
    ['a', 'g', 'ga', 'gav', 'gavh']
    """
    if len(pathways) == 0:
        raise ValueError("pathways must not be empty")

    if node_key is not None:
        pathways = [
            nx.relabel_nodes(
                path,
                {node: attrs[node_key] for node, attrs in path.nodes.items()},
                copy=True,
            )
            for path in pathways
        ]

    return nx.compose_all(pathways)


def exploration_ratio(pathways: Sequence[nx.DiGraph],
                      observed: Optional[Iterable[Hashable]] = None,
                      node_key: Optional[str] = None) -> float:
    """
    Measure how fully an ensemble samples its joint assembly space.

    The joint assembly space holds the observed objects together with the
    contingent ones, which were never observed but are needed to build the
    observed ones along a minimum path. The exploration ratio is the number
    of observed objects divided by the total.

    A ratio near one means the system realised almost everything its own
    construction implies, which is undirected exploration. A markedly lower
    ratio means it pushed deep along a few routes and left most of the
    reachable space unrealised, which is the signature of directed
    exploration and hence of :term:`selectivity`.

    Parameters
    ----------
    pathways : Sequence[nx.DiGraph]
        One minimum pathway per observed object.
    observed : Optional[Iterable[Hashable]], optional
        The observed objects, as they are identified in the pathways. When
        omitted, the target of each pathway is used: its nodes with no
        outgoing edge, taken per pathway rather than from the union, since a
        target of one pathway is often an intermediate of another.
    node_key : Optional[str], optional
        Passed to :func:`joint_assembly_space` to relabel pathway nodes by a
        node attribute before composing.

    Returns
    -------
    float
        The ratio, greater than zero and at most one.

    Raises
    ------
    ValueError
        If `pathways` is empty.

    See Also
    --------
    joint_assembly_space : The union this ratio is measured against.
    calculate_assembly_from_indices : The companion ensemble measure, which
        weights the same objects by copy number.

    Notes
    -----
    The ratio is a property of the ensemble, not of any one object, and it
    depends on the ensemble being generated recursively. Objects drawn
    independently share few intermediates, so their union is mostly
    contingent and the ratio is low for reasons that have nothing to do with
    selection.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> path = att.calculate_string_assembly_index("gavhp", mode="cfg")[2]
    >>> att.exploration_ratio([path])
    0.1111111111111111

    One observed object among nine nodes: building it required eight
    intermediates that were never themselves observed.
    """
    if len(pathways) == 0:
        raise ValueError("pathways must not be empty")

    if observed is None:
        observed = {
            node if node_key is None else path.nodes[node][node_key]
            for path in pathways
            for node in path.nodes
            if path.out_degree(node) == 0
        }

    space = joint_assembly_space(pathways, node_key=node_key)
    nodes = set(space.nodes)

    return len(nodes & set(observed)) / len(nodes)


def _calculate_string_assembly_molecular(
    string: str, delimiters: Sequence[str], *, dir_code: Optional[str],
    timeout: Optional[float], debug: bool, return_log_file: bool,
    save_dir: bool, cpp_options: Optional[AssemblyCppOptions],
) -> tuple:
    """Calculate string assembly using the molecular backend."""
    graph, edge_color_dict = get_undir_str_molecule(string, debug=debug)

    if debug:
        print("\nNode colors:", flush=True)
        for node, data in graph.nodes(data=True):
            print(f"Node {node}: {data.get('color', 'No color')}", flush=True)

        print("\nEdge colors:", flush=True)
        for u, v, data in graph.edges(data=True):
            print(f"Edge {u}-{v}: {data.get('color', 'No color')}", flush=True)

        print("Return log file:", return_log_file, flush=True)

    graph_result = calculate_assembly_index(
        graph, dir_code=dir_code, timeout=timeout, debug=debug,
        joint_corr=False, strip_hydrogen=False, return_log_file=return_log_file,
        save_dir=save_dir, cpp_options=cpp_options)
    graph_ai, graph_virtual_obj, graph_path = graph_result[:3]

    # Each delimiter adds two joins to the encoded joint input.
    ai = graph_ai - 2 * len(delimiters) if graph_ai >= 0 else graph_ai
    if graph_virtual_obj is None or graph_path is None:
        result = (ai, None, None)
        return (*result, graph_result[3]) if return_log_file else result

    if debug:
        print(f"Assembly Index: {ai}", flush=True)
        print("\n\nGraph Virtual Objects:\n", flush=True)
        print(f"Graph VOs type is : {type(graph_virtual_obj)}")
        for item in graph_virtual_obj:
            print(molstr_to_str(item, edge_color_dict=edge_color_dict), flush=True)
        print(f"\nGraph Path type is: {type(graph_path)}", flush=True)
        print(graph_path.edges(data=True), flush=True)

    virt_obj = [molstr_to_str(item, edge_color_dict=edge_color_dict) for item in graph_virtual_obj]
    for _, data in graph_path.nodes(data=True):
        data["vo"] = molstr_to_str(data["vo"], edge_color_dict=edge_color_dict)

    result = (ai, virt_obj, graph_path)
    return (*result, graph_result[3]) if return_log_file else result


def _calculate_string_assembly_cpp(
    string: str, delimiters: Sequence[str], *, dir_code: Optional[str],
    timeout: Optional[float], debug: bool, return_log_file: bool,
    save_dir: bool, options: AssemblyCppOptions, arguments: Sequence[str],
) -> tuple:
    """Calculate string assembly using the shared C++ execution lifecycle."""
    if not string.isascii() or "\n" in string or "\r" in string:
        raise ValueError("C++ string assembly requires a single line of ASCII text")
    with _calculation_directory(save=save_dir or options._retain_files, debug=debug,
                                return_log_file=return_log_file) as directory:
        file_path_in = str(directory / "string_in")
        Path(file_path_in).write_text(string, encoding="ascii")
        log_file = str(directory / "assembly_output.log")
        if dir_code is None:
            dir_code = add_assembly_to_path(str_mode=True)
        timed_out = _run_assembler(dir_code, file_path_in, log_file, timeout, debug,
                                   arguments=arguments)
        ai = _read_calculation_index(file_path_in + "Out", log_file, timed_out, debug=debug)
        if ai >= 0:
            ai -= 2 * len(delimiters)

        virt_obj = path = None
        # The CLI writes one pathway per input line; this API supplies one line.
        file_path_pathway = directory / "string_in_0_Pathway"
        if options.pathway and file_path_pathway.is_file():
            try:
                virt_obj, path = parse_string_pathway_file(
                    str(file_path_pathway), accept_palindromes=options.accept_palindromes)
            except Exception as error:
                print(f"Failed to load pathway data: {error}", flush=True)
                if debug:
                    traceback.print_exc()
        result = (ai, virt_obj, path)
        if return_log_file:
            print(f"Log file printed to: {log_file}", flush=True)
            return (*result, log_file)
        return result


def calculate_string_assembly_index(input_data: Union[str, List[str]],
                                    dir_code: Optional[str] = None,
                                    timeout: Optional[float] = 100.0,
                                    debug: bool = False,
                                    directed: bool = True,
                                    mode: str = "str",
                                    return_log_file: bool = False, *,
                                    save_dir: bool = False,
                                    cpp_options: Optional[AssemblyCppOptions] = None) -> Union[
    Tuple[int, Any, Any], Tuple[int, Any, Any, Optional[str]]]:
    """
    Calculate the assembly index for a string or a list of strings.

    This function computes an (optionally joint) assembly index for textual inputs by
    mapping strings to molecular graphs or by using a dedicated string-assembly
    executable. It supports three modes: ``'mol'`` (map to molecular graph and use
    molecular calculator), ``'str'`` (use string-assembly executable), and ``'cfg'``
    (use CFG/RePair upper bound). Joint calculations for multiple strings are
    handled by `prep_joint_string_ai` and corrected for delimiters and directedness.

    Parameters
    ----------
    input_data : Union[str, List[str]]
        A single string or a list of strings to analyse. Lists are treated as joint
        inputs (limited to 95 items for joint calculations).
    dir_code : str, optional
        Path to the assembly executable. If None, locate or build it via
        :func:`add_assembly_to_path`.
    timeout : float, optional
        Maximum wall-clock search time in seconds; must be finite and
        non-negative, or None for no wall-clock limit. This is independent of
        the C++ CPU-time budget in ``cpp_options.runtime_ticks``.
        A timed-out calculator gets up to two more seconds to save its result
        before it is killed. Default is 100.0.
    debug : bool, optional
        If True, retain a temporary directory and print debug output.
        Default is False.
    directed : bool, optional
        If True, treat strings as directed; affects encoding and post-processing.
        Default is True.
    mode : {'mol', 'str', 'cfg'}, optional
        Selects the calculation backend:
        - ``'mol'``: encode strings as molecular graphs and run molecular assembler.
        - ``'str'``: use the string-assembly executable.
        - ``'cfg'``: use CFG/RePair upper bound (fast, approximate).
        Default is ``'str'``.
    return_log_file : bool, optional
        If True, return the path to the log file produced by the external run as
        the fourth element of the returned tuple, retaining its directory.
        No log is produced for a trivial input. Default is False.
    save_dir : bool, optional
        Retain calculation files in an ``ai_calc_*`` directory. Default is False.
    cpp_options : AssemblyCppOptions, optional
        C++ search and output controls. In native string mode,
        ``accept_palindromes=True`` permits reuse of reversed fragments;
        the pathway marks reversal operations with zero cost. Graph-only
        controls are rejected in this mode. Options are forwarded to the
        graph calculator for undirected molecular encoding. CFG mode does
        not accept C++ options.

    Returns
    -------
    tuple
        If ``return_log_file`` is False returns a 3-tuple: ``(ai, virt_obj, path)`` where
        ``ai`` is the (possibly joint) assembly index (int), ``virt_obj`` is a list
        or other representation of virtual objects (or ``None``), and ``path`` is the
        pathway representation (or ``None``). If ``return_log_file`` is True returns
        ``(ai, virt_obj, path, log_file)`` where ``log_file`` is the path to the
        assembler log produced in the temporary ``ai_calc_*`` folder.

    Raises
    ------
    ValueError
        If ``input_data`` is neither a string nor a list of strings, or if an
        unsupported ``mode`` is provided, or if list length exceeds supported limit.
    OSError
        If required external tools or compiled executables are not available and
        automatic compilation fails.

    Notes
    -----
    - Joint inputs (lists) are encoded with delimiters; the final returned AI is
      corrected by subtracting delimiter and directedness offsets.
    - In 'str' mode the shared C++ executable runs in string mode. Its input
      must be a single line of ASCII text because it indexes bytes and treats
      newlines as separate calculations.
    - ``return_log_file=True`` also retains the directory so the returned log
      remains readable. Otherwise temporary files are removed on success or
      failure, unless ``debug`` is True.
    - In 'cfg' mode the function delegates to ``assemblycfg.repair_with_pathways`` and
      returns an upper bound; no external binary is invoked.
    - For reproducible behaviour consider using ``debug=True`` to preserve the
      temporary folder and log files.
    - Undirected calculations only run through the molecule calculator, so
      ``directed=False`` switches ``mode`` to 'mol' and warns if it was not
      already set.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> ai, virt_obj, pathway = att.calculate_string_assembly_index(
    ...     "abracadabra")
    >>> ai
    7

    Building the string character by character would take ten joins; the
    index is 7 because ``abra`` is reused once it exists.

    Pass a list for a joint index that shares substrings across the set:

    >>> att.calculate_string_assembly_index(
    ...     ["abracadabra", "abra"], directed=False, mode="mol")[0]
    7
    """

    if not directed:
        if mode in ("str", "cfg"):
            mode = "mol"  # Use the molecular assembly calculator for undirected strings
            print("Warning: only mode 'mol' is currently supported for undirected strings. Switching to 'mol'.",
                  flush=True)
    elif mode == "mol":
        mode = "str"  # Use the string assembly calculator for directed strings
        print("Warning: mode 'mol' is not currently supported for directed strings. Switching to 'str'.", flush=True)

    if mode not in ("str", "mol", "cfg"):
        raise ValueError("Mode must be either 'mol', 'str', or 'cfg'.")
    if mode == "cfg":
        if cpp_options is not None:
            raise ValueError("cpp_options do not apply to CFG mode")
    else:
        _validate_cpp_timeout(timeout)
        options = _cpp_options(cpp_options)
        if mode == "str":
            arguments = options._arguments(str_mode=True)
        else:
            # The graph API serializes these controls after string encoding.
            options._validate_mode(str_mode=False)

    if isinstance(input_data, str):
        string = input_data
        delimiters = []
        if len(string) <= 1:
            return (0, None, None) if not return_log_file else (0, None, None, None)

    elif isinstance(input_data, list):
        input_data = [s for s in input_data if len(s) > 1]  # Remove elements of the list that are single characters
        if not input_data:
            return (0, None, None) if not return_log_file else (0, None, None, None)

        if mode != "cfg":
            if len(input_data) > 95:
                raise ValueError(
                    "Input list contains more than 95 objects. Joint assembly index calculations are only supported for up to 95 objects except in cfg (RePair) approximation mode.")
            # Handle joint assembly case
            string, delimiters = prep_joint_string_ai(input_data)
    else:
        raise ValueError("Input must be either a single string or a list of strings")

    assert dir_code is None or isinstance(dir_code, (str, os.PathLike)), "Directory code must be a path"
    assert isinstance(debug, bool), "Debug must be a boolean"
    assert isinstance(directed, bool), "Directed must be a boolean"

    if mode == "cfg":
        ai, virt_obj, path = assemblycfg.repair_with_pathways(input_data, f_print=False)
        return ai, virt_obj, path
    if mode == "mol":
        return _calculate_string_assembly_molecular(
            string, delimiters, dir_code=dir_code, timeout=timeout,
            debug=debug, return_log_file=return_log_file,
            save_dir=save_dir, cpp_options=options)
    return _calculate_string_assembly_cpp(
        string, delimiters, dir_code=dir_code, timeout=timeout,
        debug=debug, return_log_file=return_log_file,
        save_dir=save_dir, options=options, arguments=arguments)


def regularise_assembly_index(ai: Optional[int]) -> int:
    """
    Regularise the assembly index to a non-negative integer.

    Parameters
    ----------
    ai : int or None
        Assembly index to regularise. Negative values or ``None`` are interpreted
        as missing/invalid and are mapped to ``0``.

    Returns
    -------
    int
        A non-negative assembly index. If ``ai`` is ``None`` or negative, ``0``
        is returned; otherwise the original ``ai`` is returned unchanged.

    Notes
    -----
    - The function is idempotent for non-negative integer inputs.
    """
    return 0 if ai is None or ai < 0 else ai


def calculate_assembly_index_parallel(graphs: List[Union[nx.Graph, Chem.Mol]],
                                      settings: Optional[Dict[str, Any]]) -> List[List[Any]]:
    """
    Calculate assembly indices for multiple graphs in parallel.

    This function runs :func:`calculate_assembly_index` over an iterable of graphs
    using the parallel worker `mp_calc` and returns the transposed results so
    callers receive a list per returned field (e.g. list of AIs, list of virtual
    objects, list of pathways).

    Parameters
    ----------
    graphs : iterable
        Iterable of molecular graphs (for example, a list of NetworkX graphs).
    settings : dict or None
        Keyword arguments forwarded to :func:`calculate_assembly_index`. If ``None``
        an empty dictionary is used.

    Returns
    -------
    list of list
        Transposed results of the parallel calculation. If
        ``calculate_assembly_index`` returns tuples like ``(ai, vo, path)`` for
        each graph, the return value will be:
        ``[ [ai_1, ai_2, ...], [vo_1, vo_2, ...], [path_1, path_2, ...] ]``.

    Raises
    ------
    ValueError
        If ``graphs`` is not iterable.

    Notes
    -----
    - The function relies on ``mp_calc`` to execute ``calculate_assembly_index``
      in parallel workers and expects ``mp_calc`` to return a sequence of per-item
      results (one tuple per graph).
    - An empty ``graphs`` iterable yields an empty list.
    - The three returned lists are aligned with the input order, so results
      stay matched to their inputs even though the workers finish out of
      order.

    Examples
    --------
    ``settings`` has no default; pass ``None`` to use the defaults of
    :func:`calculate_assembly_index`.

    >>> import assemblytheorytools as att
    >>> graphs = [att.smi_to_nx(s) for s in ["NCC(=O)O", "CC(N)C(=O)O"]]
    >>> ai, virt_obj, pathway = att.calculate_assembly_index_parallel(
    ...     graphs, dict(strip_hydrogen=True))
    >>> ai
    [3, 4]
    """
    if graphs is None or not hasattr(graphs, "__iter__"):
        raise ValueError("`graphs` must be an iterable of graph objects")

    settings = settings or {}

    results = mp_calc(partial(calculate_assembly_index, **settings), graphs)
    return [list(group) for group in zip(*results)]


def _get_most_recent_calc() -> str:
    """
    Return the absolute path of the latest ``ai_calc_`` entry in the current directory.

    Recency follows ``os.path.getctime``: creation time on some platforms and
    the last metadata change on others. Raise ``FileNotFoundError`` when no
    matching entry exists.
    """
    cwd = os.getcwd()
    assembly_folders = [os.path.join(cwd, folder) for folder in os.listdir(cwd)
                        if folder.startswith("ai_calc_")]
    if not assembly_folders:
        raise FileNotFoundError("No 'ai_calc_' folders found in the current working directory")
    return max(assembly_folders, key=os.path.getctime)


def load_assembly_time() -> float:
    """
    Load the time-to-completion recorded by the most recent assembly run.

    The function locates the most recent directory in the current working
    directory whose name starts with ``ai_calc_``, finds the most recent file
    in that directory whose name ends with ``Out``, reads the last line of the
    file, extracts the numeric time value, and returns it in seconds.

    Returns
    -------
    float
        Time to completion in seconds (the value read from the file is assumed
        to be in microseconds and is converted to seconds).

    Raises
    ------
    FileNotFoundError
        If no ``ai_calc_`` directory is found or if no ``Out`` file exists in the
        most recent calculation directory.
    ValueError
        If the time value cannot be parsed as a number from the last line of the
        selected file.
    OSError
        If removal of the assembly folder fails during cleanup.

    Notes
    -----
    - The function removes the identified ``ai_calc_`` directory after reading
      the time value.
    - The implementation expects the last line of the ``Out`` file to contain a
      colon-separated value whose final token is the numeric time (matching the
      historical behavior of the project). The numeric value is interpreted as
      microseconds and converted to seconds by multiplying with ``1e-6``.
    - Uses the private helper ``_get_most_recent_calc`` to find the latest
      calculation folder.

    Examples
    --------
    This reads the most recent ``ai_calc_`` folder in the working directory,
    so it only works after a calculation has been run with ``save_dir=True``
    (or ``debug=True``); otherwise it raises :exc:`FileNotFoundError`.

    >>> t = load_assembly_time()  # doctest: +SKIP
    >>> isinstance(t, float)  # doctest: +SKIP
    True
    """
    assembly_path = _get_most_recent_calc()
    if not os.path.isdir(assembly_path):
        raise FileNotFoundError(f"No assembly calculation folder found: {assembly_path}")

    out_files = [f for f in os.listdir(assembly_path) if f.endswith("Out")]
    if not out_files:
        raise FileNotFoundError(f"No '*Out' files found in {assembly_path}")

    out_files.sort(key=lambda name: os.path.getctime(os.path.join(assembly_path, name)))
    latest_file = os.path.join(assembly_path, out_files[-1])

    with open(latest_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            raise ValueError(f"File {latest_file} is empty")
        last_line = lines[-1].strip()

    try:
        time_to_completion = float(last_line.split(":")[-1].strip())
    except Exception as e:
        raise ValueError(f"Failed to parse time from '{latest_file}': {e}") from e

    shutil.rmtree(assembly_path)
    return time_to_completion * 1e-6


def calculate_assembly_index_semi_metric(graph1: Union[nx.Graph, Chem.Mol],
                                         graph2: Union[nx.Graph, Chem.Mol],
                                         settings: Optional[Dict[str, Any]] = None,
                                         parallel: bool = True,
                                         normalise: bool = False) -> float:
    """
    Calculate the semi-metric distance between two molecular graphs.

    The semi-metric distance is computed as twice the joint assembly index minus
    the sum of the individual assembly indices. This value represents the
    "additional cost" or "savings" when combining the two structures into a
    single assembly.

    Parameters
    ----------
    graph1 : Union[nx.Graph, Chem.Mol]
        The first molecular graph or RDKit molecule.
    graph2 : Union[nx.Graph, Chem.Mol]
        The second molecular graph or RDKit molecule.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_assembly_index.
        If None, an empty dictionary is used. Default is None.
    parallel : bool, optional
        If True, run calculations in parallel using multiple processes.
        Default is True.
    normalise : bool, optional
        If True, normalize the semi-metric distance by the sum of the assembly indices.
        Default is False.

    Returns
    -------
    float
        The computed semi-metric distance, which may be negative, zero, or positive.
        Returns 0.0 for isomorphic inputs and -1.0 if the joint calculation
        timed out without a result.

    Raises
    ------
    ValueError
        If the input graphs are not of the same type.
    OSError
        If there are issues with file system access, process execution, or
        if required external tools or compiled executables are not available.

    Notes
    -----
    - A negative semi-metric distance indicates that the combined assembly is
      "cheaper" than the sum of the individual assemblies, suggesting a
      synergistic effect.
    - This metric is useful for evaluating the potential efficiency or
      feasibility of synthesizing the combined structure.
    """

    settings = settings or {}

    if type(graph1) is not type(graph2):
        raise ValueError("Input graphs must be of the same type")

    if type(graph1) is Chem.Mol:
        graph1 = mol_to_nx(graph1)
        graph2 = mol_to_nx(graph2)

    if nx.is_isomorphic(graph1, graph2):
        print("Input graphs are isomorphic.", flush=True)
        return 0.0

    jai = calculate_assembly_index(join_graphs([graph1, graph2]), **settings)[0]
    if jai <= -1:
        print("No minimum JAI found before timeout.", flush=True)
        return -1.0

    sum_ai = calculate_sum_assembly_index([graph1, graph2], settings, parallel=parallel)

    semi_metric = 2.0 * jai - sum_ai
    return semi_metric / sum_ai if normalise else semi_metric


def calculate_assembly_index_upper_bound(mol: Union[nx.Graph, Chem.Mol],
                                         strip_hydrogen: bool = False) -> int:
    """
    Calculate the upper bound of the assembly index for a molecular graph or RDKit molecule.

    The upper bound is estimated based on the number of bonds/edges in the structure,
    providing a theoretical maximum for the assembly index.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.
    strip_hydrogen : bool, optional
        If True, remove hydrogen atoms from the graph before calculation.
        Default is False.

    Returns
    -------
    int
        The estimated upper bound of the assembly index.

    Raises
    ------
    ValueError
        If the input type is not supported.
    """
    mol = _prepare_bound_input(mol, strip_hydrogen)

    # Every bond beyond the first can at worst be added by its own joining operation
    return _count_edges(mol) - 1


def calculate_assembly_index_lower_bound(mol: Union[nx.Graph, Chem.Mol],
                                         strip_hydrogen: bool = False) -> int:
    """
    Calculate the lower bound of the assembly index for a molecular graph or RDKit molecule.

    The lower bound is estimated based on the number of bonds/edges in the structure,
    providing a theoretical minimum for the assembly index.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.
    strip_hydrogen : bool, optional
        If True, remove hydrogen atoms from the graph before calculation.
        Default is False.

    Returns
    -------
    int
        The estimated lower bound of the assembly index.

    Raises
    ------
    ValueError
        If the input type is not supported.
    """
    mol = _prepare_bound_input(mol, strip_hydrogen)

    # Use tabulated addition chains below 1000, then fall back on logarithms.
    n_bonds = _count_edges(mol)
    if n_bonds < 1000:
        return calculate_integer_chain(n_bonds)
    return int(np.log2(n_bonds))


def calculate_sum_assembly_index(graphs: List[Union[nx.Graph, Chem.Mol]],
                                 settings: Optional[Dict[str, Any]] = None,
                                 parallel: bool = True) -> int:
    """
    Calculate the sum of assembly indices for multiple graphs.

    This function computes the assembly index for each graph in the input list
    (graphs) using the calculate_assembly_index function. It then sums the
    individual assembly indices to provide a total assembly index for the
    combined system of graphs.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, Chem.Mol]]
        A list of molecular graphs or RDKit molecules to analyze.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_assembly_index.
        If None, an empty dictionary is used. Default is None.
    parallel : bool, optional
        If True, run calculations in parallel using multiple processes.
        Default is True.

    Returns
    -------
    int
        The total assembly index for the combined system of graphs, or ``-1`` if
        any individual calculation failed.

    Raises
    ------
    ValueError
        If ``graphs`` is not iterable.
    OSError
        If there are issues with file system access, process execution, or
        if required external tools or compiled executables are not available.
    """

    if graphs is None or not hasattr(graphs, "__iter__"):
        raise ValueError("`graphs` must be an iterable of graph objects")

    ai_list = _calculate_assembly_indices(graphs, settings, parallel)
    if any(ai is None or ai < 0 for ai in ai_list):
        return -1

    return int(sum(ai_list))


def calculate_assembly_index_similarity(graphs: List[Union[nx.Graph, Chem.Mol]],
                                        settings: Optional[Dict[str, Any]] = None,
                                        parallel: bool = True,
                                        enforce_exact_mode: bool = True) -> float:
    """
    Calculate the assembly index similarity for a set of graphs.

    This function computes the assembly index for the joint graph (combined
    from all input graphs) and compares it to the sum of the individual
    assembly indices. The score is ``(sum_ai / joint_ai) - 1``.

    Parameters
    ----------
    graphs : List[Union[nx.Graph, Chem.Mol]]
        A list of molecular graphs or RDKit molecules to analyze.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_assembly_index.
        If None, an empty dictionary is used. Default is None.
    parallel : bool, optional
        If True, run calculations in parallel using multiple processes.
        Default is True.
    enforce_exact_mode : bool, optional
        If True, enforce exact mode for assembly index calculation.
        Default is True.

    Returns
    -------
    float
        The calculated shared-assembly score. For two inputs it lies between
        0.0 and 1.0; for more inputs it can be as high as ``len(graphs) - 1``.
        Returns -1.0 if any underlying calculation failed.

    Raises
    ------
    ValueError
        If the input graphs are not of the same type.
    OSError
        If there are issues with file system access, process execution, or
        if required external tools or compiled executables are not available.

    Notes
    -----
    - Exact mode is enabled by default so a timeout upper bound is not treated
      as an exact similarity value. Set ``enforce_exact_mode=False`` only when
      that tradeoff is intentional.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.calculate_assembly_index_similarity(
    ...     [att.smi_to_nx("NCC(=O)O"), att.smi_to_nx("CC(N)C(=O)O")],
    ...     settings={"strip_hydrogen": True})
    0.75

    Glycine and alanine share most of their structure, so the joint
    assembly index (4) sits well below the sum of the separate indices
    (3 + 4 = 7). That gap is what this score reports.
    """

    settings = settings or {}

    if enforce_exact_mode:
        settings = {**settings, "exact": True}

    ai_sum = calculate_sum_assembly_index(graphs, settings, parallel=parallel)
    if ai_sum < 0:
        return -1.0

    joint_ai = calculate_assembly_index(join_graphs(graphs), **settings)[0]
    if joint_ai < 0:
        return -1.0

    return ai_sum / joint_ai - 1.0 if joint_ai != 0 else 0.0


def _calculate_jo_from_pathway(json_file: str) -> int:
    """
    Derive the joint assembly correction from a saved pathway file.

    Rebuild the first ``file_graph`` and remove duplicate fragments in order.
    Their overlap with the remnant and changes in connected components supply
    the joint-object correction to the raw assembly index.
    """
    data = _read_assembly_json(json_file)

    edges = [tuple(edge) for edge in data["file_graph"][0].get("Edges", [])]
    original_graph = nx.Graph()
    original_graph.add_edges_from(edges)
    component_count = nx.number_connected_components(original_graph)

    ma = original_graph.number_of_edges() - component_count
    jo_correction = 0
    remaining_edges = set(edges)

    for duplicate in data.get("duplicates", []):
        fragment = duplicate.get("Right", [])
        ma -= len(fragment) - 1
        fragment_atoms = {atom for edge in fragment for atom in edge}
        remaining_edges -= {tuple(edge) for edge in fragment}

        remnant_graph = nx.Graph()
        remnant_graph.add_edges_from(remaining_edges)
        remnant_components = nx.number_connected_components(remnant_graph)

        # Shared atoms add joins; newly disconnected components offset them.
        overlap_correction = max(0, len(fragment_atoms & set(remnant_graph)) - 1)
        component_correction = max(remnant_components - component_count, 0)
        jo_correction += overlap_correction - component_correction
        component_count = remnant_components

    return ma + jo_correction


def calculate_assembly_index_jo(mol: Union[nx.Graph, Chem.Mol],
                                settings: Optional[Dict[str, Any]] = None) -> Tuple[int, Any, Any]:
    """
    Calculate the joining-operation (JO) assembly index for a molecular graph or RDKit molecule.

    The JO assembly index is a metric that reflects the efficiency or feasibility of
    synthesizing a molecular structure. It is computed based on the assembly pathways
    and the molecular assembly numbers of the constituent parts.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings forwarded to calculate_assembly_index.
        If None, an empty dictionary is used. Default is None.

    Returns
    -------
    tuple
        A 3-tuple: (jo, virt_obj, path) where
        jo is the joining-operation assembly index (int), virt_obj is a list
        or other representation of virtual objects (or None), and path is the
        pathway representation (or None).

    Raises
    ------
    ValueError
        If the input graph is not supported.
    OSError
        If there are issues with file system access, process execution, or
        if required external tools or compiled executables are not available.

    Notes
    -----
    - The pathway is read from this calculation's returned log directory.
      Files are removed afterwards unless ``save_dir``, ``debug``,
      ``return_log_file`` or a diagnostic output option requests retention.
      The return value remains a 3-tuple, including when retaining a log.
    - If no valid pathway file is found, or if JO calculation fails, the function
      returns -1 for the JO index.
    """

    settings = settings or {}
    ai, vo, pathway, log_file = calculate_assembly_index(
        mol, **{**settings, "return_log_file": True})
    if log_file is None:  # Trivial inputs need no calculator or saved pathway.
        return ai, vo, pathway

    assembly_path = Path(log_file).parent
    keep_files = (any(settings.get(name) for name in ("save_dir", "debug", "return_log_file"))
                  or _cpp_options(settings.get("cpp_options"))._retain_files)
    try:
        pathway_file = assembly_path / "graph_inPathway"
        if ai < 0 or not pathway_file.is_file():
            print("No usable pathway found. Returning -1.", flush=True)
            return -1, None, None
        return _calculate_jo_from_pathway(str(pathway_file)), vo, pathway
    except Exception as e:
        print(f"Error calculating joint assembly index: {e}", flush=True)
        return -1, None, None
    finally:
        if not keep_files:
            shutil.rmtree(assembly_path)


def calculate_assembly_index_ratio(graph: Union[nx.Graph, Chem.Mol], settings: Dict[str, Any]) -> float:
    """
    Calculate the assembly ratio for a molecular graph.

    The assembly ratio is defined as:

        assembly_ratio = n_edges / AI

    where:
      - n_edges is the number of edges (bonds) in the input graph or RDKit molecule.
      - AI is the assembly index computed by `calculate_assembly_index`.

    Parameters
    ----------
    graph : Union[nx.Graph, Chem.Mol]
        Input molecular representation. For a NetworkX graph the number of edges is
        obtained via `graph.number_of_edges()`. For an RDKit `Chem.Mol` the number of
        bonds is obtained via `graph.GetNumBonds()`.
    settings : dict
        Dictionary of settings forwarded to `calculate_assembly_index` (e.g. `dir_code`,
        `timeout`, `debug`, `strip_hydrogen`, `exact`, ...).

    Returns
    -------
    float
        The assembly ratio (n_edges divided by AI). Special cases:
          - If the graph has zero edges the function returns 1.0 to avoid division by zero.
          - If AI < 0 a negative value is returned (this typically indicates a failure
            or incomplete calculation upstream).

    Raises
    ------
    ZeroDivisionError
        If the computed assembly index (AI) is zero, the division `n_edges / AI` will raise.
        Callers may wish to validate AI before calling if this is a concern.

    Notes
    -----
    - The function does not modify `graph`.
    - The returned value may be meaningless if `calculate_assembly_index` returned a
      non-positive AI; callers should check the AI return value when exact/robust
      behaviour is required.
    """
    n_edges = _count_edges(graph)
    if n_edges == 0:
        return 1.0

    ai, _, _ = calculate_assembly_index(graph, **settings)
    return n_edges / ai


def calculate_assembly_index_jo_ratio(graph: Union[nx.Graph, Chem.Mol], settings: Dict[str, Any]) -> float:
    """
    Calculate the joining-operation (JO) assembly ratio for a molecular graph.

    The JO assembly ratio is computed as:

        assembly_ratio = n_edges / JO

    where:
      - n_edges is the number of edges (bonds) in the input graph or molecule.
      - JO is the joining-operation index computed by `calculate_assembly_index_jo`.

    Parameters
    ----------
    graph : Union[nx.Graph, Chem.Mol]
        The molecular graph to evaluate. For a NetworkX graph, the number of edges
        is obtained via `graph.number_of_edges()`. For an RDKit molecule the number
        of bonds is obtained via `graph.GetNumBonds()`.
    settings : dict
        Settings forwarded to `calculate_assembly_index_jo`. Typical keys control
        execution of the underlying assembly calculation (e.g., `dir_code`,
        `timeout`, `debug`, `strip_hydrogen`, `exact`).

    Returns
    -------
    float
        The JO assembly ratio (number of edges divided by JO). If the graph has no
        edges, returns 1.0 to avoid division by zero.

    Raises
    ------
    ZeroDivisionError
        If the computed JO is zero.

    Notes
    -----
    - If `calculate_assembly_index_jo` fails it returns -1, in which case the ratio
      is negative; callers should check for this.
    - This function does not modify the input graph or molecule.
    """
    n_edges = _count_edges(graph)
    if n_edges == 0:
        return 1.0

    jo = calculate_assembly_index_jo(graph, settings=settings)[0]
    return n_edges / jo


class RustSearchResult(NamedTuple):
    """
    Result of a Rust-backed assembly index search.

    Attributes
    ----------
    index : int
        The molecule's assembly index, or the best upper bound found so far if
        the search timed out.
    num_matches : int
        The number of edge-disjoint isomorphic subgraph pairs in the molecule.
    states_searched : int or None
        The number of assembly states visited, or None if the search timed out
        before finishing.
    pathways : list of nx.MultiDiGraph
        The minimum assembly pathways that were reconstructed. Empty unless
        ``max_pathways`` was given.
    """
    index: int
    num_matches: int
    states_searched: Optional[int]
    pathways: List[nx.MultiDiGraph]


def _rust_version() -> str:
    """Return the Rust package version, or 'unknown' if metadata is missing."""
    try:
        return version("assembly-theory")
    except PackageNotFoundError:
        return "unknown"


@cache
def _rust_supports_pathways() -> bool:
    """
    Return whether ``index_search`` accepts ``max_pathways``.

    Unknown keywords are rejected before the mol block is read, so an empty
    block probes support without running a search. Release 0.6.1 omits accepted
    arguments from its ``__text_signature__``, so inspecting it is unreliable.
    """
    try:
        at_rust.index_search("", max_pathways=0)
    except TypeError:
        return False
    except Exception:
        pass
    return True


def _mol_to_molblock(mol: Union[nx.Graph, Chem.Mol]) -> str:
    """
    Convert a molecule to the V2000 mol block the Rust backend expects.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule, either a NetworkX graph or an RDKit molecule.

    Returns
    -------
    str
        The molecule as a V2000 mol block.

    Raises
    ------
    ValueError
        If the molecule is neither a NetworkX graph nor an RDKit molecule, or
        if it has more than 999 atoms or bonds, which RDKit can only write as
        V3000, a format the Rust backend rejects.

    Notes
    -----
    - Hydrogens are not added, because the Rust backend discards them when it
      loads the mol block.
    - Directed graphs are undirected first, since a molecular graph has no
      direction.
    """
    if isinstance(mol, nx.Graph):
        if mol.is_directed():
            mol = mol.to_undirected()
        mol = nx_to_mol(mol, add_hydrogens=False)

    if not isinstance(mol, Chem.Mol):
        raise ValueError("Expected a NetworkX graph or an RDKit molecule, got "
                         f"{type(mol).__name__}.")

    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    if n_atoms > 999 or n_bonds > 999:
        raise ValueError("The Rust backend only reads V2000 mol blocks, which hold at most "
                         f"999 atoms and 999 bonds; this molecule has {n_atoms} atoms and "
                         f"{n_bonds} bonds.")

    return Chem.MolToMolBlock(mol)


def _rust_error(error: OSError) -> ValueError:
    """Translate a backend mol block error into a contextual ValueError."""
    return ValueError(f"The Rust backend could not read this molecule: {error}")


def _call_rust(function: Callable[[str], Any],
               mol: Union[nx.Graph, Chem.Mol]) -> Any:
    """Convert a molecule and call the backend with consistent error handling."""
    try:
        return function(_mol_to_molblock(mol))
    except OSError as error:
        raise _rust_error(error) from error


# The backend counts joining operations in an unsigned 32-bit integer and
# underflows to its maximum when there are none to count: a bare atom has no
# bonds to join, and a one-bond molecule is already at its full depth. Both of
# those answers are 0, and no molecule small enough for a V2000 mol block can
# have a genuine index or depth anywhere near this value.
_RUST_UNDERFLOW = 2 ** 32 - 1


def _rust_count(value: int) -> int:
    """Correct the backend's underflow sentinel to the zero it stands for."""
    return 0 if value == _RUST_UNDERFLOW else value


def calculate_assembly_index_rust(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """
    Calculate the assembly index of a molecule using the Rust-based assembly theory library.

    This function computes the assembly index for a given molecular graph or RDKit molecule
    by converting the input to an RDKit `Chem.Mol` object (if necessary) and then passing
    it to the Rust-based `assembly_theory` library for calculation.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule, which can be either a NetworkX graph or an RDKit `Chem.Mol` object.

    Returns
    -------
    int
        The assembly index of the molecule as computed by the Rust-based library.

    Raises
    ------
    ValueError
        If the molecule is not a NetworkX graph or an RDKit molecule, if it is
        too large for a V2000 mol block, or if the Rust backend cannot read it.

    Notes
    -----
    - If the input is a NetworkX graph, it is first converted to an RDKit `Chem.Mol` object
      using the `nx_to_mol` function.
    - This backend returns the index only: no virtual objects and no
      pathway. Use :func:`calculate_assembly_index` when those are needed, or
      :func:`calculate_assembly_index_rust_search` for search statistics and
      pathways.
    - Hydrogens are always stripped, so only compare the result against
      ``calculate_assembly_index(..., strip_hydrogen=True)``.
    - A molecule with no bonds, such as a bare atom or a metal ion, has an
      index of 0. The backend underflows to 4294967295 in that case, which is
      corrected here.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.calculate_assembly_index_rust(
    ...     att.smi_to_nx("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
    9
    >>> att.calculate_assembly_index_rust(att.smi_to_nx("CCO"))
    1
    >>> att.calculate_assembly_index_rust(att.smi_to_mol("[Fe+2]"))
    0
    """
    return _rust_count(_call_rust(at_rust.index, mol))


def calculate_assembly_depth_rust(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """
    Calculate the assembly depth of a molecule using the Rust-based library.

    Assembly depth counts the joining operations along the longest branch of an
    assembly pathway, as if independent joins ran concurrently, so it is
    generally smaller than the assembly index. See Pagel et al. (2024).

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule, which can be either a NetworkX graph or an RDKit
        `Chem.Mol` object.

    Returns
    -------
    int
        The assembly depth of the molecule.

    Raises
    ------
    ValueError
        If the molecule is not a NetworkX graph or an RDKit molecule, if it is
        too large for a V2000 mol block, or if the Rust backend cannot read it.

    Notes
    -----
    - Unlike :func:`~assemblytheorytools.construction.assign_levels`, which
      reports the depth of one particular pathway, this is the molecule's
      minimum achievable assembly depth.
    - The depth search is far more expensive than the index search and takes no
      timeout, so it is only practical on small molecules. Benzene returns
      instantly and naphthalene takes around four minutes, while both of their
      indices come back in well under a second.
    - Hydrogens are stripped, as they are for every Rust-backed calculation.
    - A molecule needing no joining operations, such as one with a single bond,
      has a depth of 0. The backend underflows to 4294967295 in that case,
      which is corrected here.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.calculate_assembly_depth_rust(att.smi_to_nx("c1ccccc1"))
    3
    >>> att.calculate_assembly_depth_rust(att.smi_to_nx("CC"))
    0
    """
    return _rust_count(_call_rust(at_rust.depth, mol))


def get_molecule_info_rust(mol: Union[nx.Graph, Chem.Mol]) -> str:
    """
    Describe the graph the Rust backend builds for a molecule.

    This returns the backend's own view of the molecule as a DOT-formatted
    undirected graph, listing every atom and bond it will work with. It is the
    quickest way to check what the backend actually sees, for example that
    hydrogens have been dropped or that a ring has been kekulised.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule, which can be either a NetworkX graph or an RDKit
        `Chem.Mol` object.

    Returns
    -------
    str
        A DOT-formatted description of the molecule's atoms and bonds.

    Raises
    ------
    ValueError
        If the molecule is not a NetworkX graph or an RDKit molecule, if it is
        too large for a V2000 mol block, or if the Rust backend cannot read it.

    Notes
    -----
    - Atom and bond indices in this description match the indices of the mol
      block, and therefore of an RDKit molecule parsed from that same mol
      block. Pathway bond indices refer to the same numbering.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> info = att.get_molecule_info_rust(att.smi_to_nx("CCO"))
    >>> info.count('label = "Atom')
    3
    """
    return _call_rust(at_rust.mol_info, mol)


def calculate_assembly_index_rust_search(mol: Union[nx.Graph, Chem.Mol],
                                         timeout: Optional[float] = None,
                                         canonize: str = "tree-nauty",
                                         parallel: str = "depth-one",
                                         memoize: str = "canon-index",
                                         kernel: str = "none",
                                         bounds: Sequence[str] = ("int", "matchable-edges"),
                                         max_pathways: Optional[int] = None,
                                         vo_type: str = "smiles") -> RustSearchResult:
    """
    Run the Rust backend's assembly index search with explicit options.

    Where :func:`calculate_assembly_index_rust` returns only the index, this
    exposes the backend's search parameters and reports what the search did:
    how many duplicate subgraph pairs it found, how many states it visited,
    and, on backends that support it, the minimum assembly pathways themselves.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecule, which can be either a NetworkX graph or an RDKit
        `Chem.Mol` object.
    timeout : float, optional
        Seconds after which to stop searching and return the best index found
        so far. Note that the Rust backend takes whole milliseconds; the
        conversion is done here, rounding up, so that this argument matches
        :func:`calculate_assembly_index`. Default is None, meaning no limit.
    canonize : str, optional
        Canonisation mode: 'nauty', 'faulon', 'tree-nauty' or 'tree-faulon'.
        Default is 'tree-nauty'.
    parallel : str, optional
        Parallelisation mode: 'none', 'depth-one' or 'always'. Default is
        'depth-one'. Use 'none' to make `states_searched` deterministic.
    memoize : str, optional
        Memoisation mode: 'none' or 'canon-index'. Default is 'canon-index'.
        The backend's error message also lists 'frags-index', but rejects it.
    kernel : str, optional
        Kernelisation mode: 'none', 'once', 'depth-one' or 'always'. Default
        is 'none'.
    bounds : Sequence[str], optional
        Branch-and-bound strategies to apply, drawn from 'log', 'int',
        'vec-simple', 'vec-small-frags' and 'matchable-edges'. Pass an empty
        sequence for an exhaustive search, and a one-element sequence rather
        than a bare string for a single strategy. Default is
        ``("int", "matchable-edges")``.
    max_pathways : int, optional
        How many minimum assembly pathways to reconstruct: a positive integer
        for at most that many, 0 for all of them, or None to skip
        reconstruction entirely. Default is None.
    vo_type : str, optional
        Representation for the virtual objects in the reconstructed pathways:
        'graph', 'mol', 'smiles' or 'inchi'. Default is 'smiles'.

    Returns
    -------
    RustSearchResult
        A named 4-tuple of the assembly index, the number of matching subgraph
        pairs, the number of states searched (None if the search timed out),
        and the list of reconstructed pathways.

    Raises
    ------
    ValueError
        If the molecule is not a NetworkX graph or an RDKit molecule, if it is
        too large for a V2000 mol block, if the Rust backend cannot read it, if
        `timeout` is negative, if `bounds` is a bare string, if any of the mode
        strings is not recognised, or if reconstructed pathways cannot be read
        back against the molecule that was searched.
    NotImplementedError
        If `max_pathways` is given but the installed ``assembly-theory`` release
        does not support pathway reconstruction.

    Notes
    -----
    - Pathway reconstruction arrived in ``assembly-theory`` 0.7.0. Support is
      detected at call time rather than by version, so this function still
      works on 0.6.1 as long as `max_pathways` is left as None.
    - Each pathway is parsed by
      :func:`~assemblytheorytools.construction.parse_pathway_dot` against the
      molecule that was searched, so its bond indices always line up.
    - Hydrogens are stripped, as they are for every Rust-backed calculation.
    - `index` is corrected for the backend's underflow the same way
      :func:`calculate_assembly_index_rust` corrects it.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> result = att.calculate_assembly_index_rust_search(
    ...     att.smi_to_nx("c1ccccc1"), parallel="none")
    >>> result.index
    3

    The pathways are ordinary graphs, so they plot with
    :func:`~assemblytheorytools.tools_plotting.plot_pathway` like any other
    assembly pathway:

    >>> result = att.calculate_assembly_index_rust_search(  # doctest: +SKIP
    ...     att.smi_to_nx("c1ccccc1"), max_pathways=1)
    >>> fig, ax = att.plot_pathway(result.pathways[0])  # doctest: +SKIP
    """
    if vo_type not in _VO_TYPES:
        raise ValueError(_VO_TYPE_ERROR)

    # A bare string is a Sequence[str], so it would otherwise be split into
    # characters and rejected one letter at a time
    if isinstance(bounds, str):
        raise ValueError("bounds must be a sequence of strategy names, not a single "
                         f"string; pass [{bounds!r}] to apply only that one.")

    if timeout is not None and timeout < 0:
        raise ValueError(f"timeout must not be negative, got {timeout}.")

    mol_block = _mol_to_molblock(mol)

    # The backend takes whole milliseconds and reads 0 as 'give up immediately',
    # so round up rather than truncating a sub-millisecond timeout into that
    options = {
        "timeout": None if timeout is None else ceil(timeout * 1000),
        "canonize_str": canonize,
        "parallel_str": parallel,
        "memoize_str": memoize,
        "kernel_str": kernel,
        "bound_strs": list(bounds),
    }

    # max_pathways is only accepted by releases that can reconstruct pathways,
    # so leave it out entirely when no pathways were asked for
    if max_pathways is not None:
        if not _rust_supports_pathways():
            raise NotImplementedError(
                f"The installed assembly-theory release ({_rust_version()}) cannot reconstruct "
                "assembly pathways. Leave max_pathways as None, or upgrade to 0.7.0 or newer.")
        options["max_pathways"] = max_pathways

    try:
        result = at_rust.index_search(mol_block, **options)
    except OSError as error:
        raise _rust_error(error) from error

    # Older releases return a 3-tuple, without the list of pathway DOT strings
    dot_pathways = result[3] if len(result) > 3 else []

    pathways = []
    if dot_pathways:
        # Bond indices only mean anything against the molecule the backend read,
        # so the pathways are parsed against that same mol block
        parsed_mol = Chem.MolFromMolBlock(mol_block)
        if parsed_mol is None:
            raise ValueError("The reconstructed pathways cannot be read back: RDKit could not "
                             "re-parse the mol block that was searched, so the pathway bond "
                             "indices cannot be resolved to fragments.")
        pathways = [parse_pathway_dot(dot, mol=parsed_mol, vo_type=vo_type)
                    for dot in dot_pathways]

    return RustSearchResult(_rust_count(result[0]), result[1], result[2], pathways)


def calculate_integer_chain(n: int) -> int:
    """
    Read the shortest integer chain length l(n) from a precomputed data file.

    The function looks up a precomputed table stored in `data/integer_chain_9999.txt`
    shipped with the package and returns the smallest length of an addition chain for
    the integer *n*.

    Parameters
    ----------
    n : int
        Positive integer for which to obtain the shortest addition-chain length.
        Valid range is 1 to 9999 (inclusive).

    Returns
    -------
    int
        The shortest addition-chain length l(n). For ``n == 1`` the function returns ``0``.
        Returns ``-1`` if the table has no entry for *n*.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``n > 9999`` because the precomputed data only covers 1..9999.

    Notes
    -----
    The implementation expects the data file to have the chain length for *n* on the
    line with index ``n + 1`` (0-based enumeration of lines) and to store the length as
    the fourth whitespace-separated field on that line.
    See https://wwwhomes.uni-bielefeld.de/achim/addition_chain.html for larger n.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if n > 9999:
        raise ValueError("n must be less than or equal to 9999.")
    if n == 1:
        return 0

    data_path = os.path.join(os.path.dirname(__file__), "data", "integer_chain_9999.txt")
    with open(data_path) as file:
        for line_number, line in enumerate(file):
            if line_number == n + 1:
                return int(line.split()[3])
    return -1


def calculate_assembly_index_pairwise_joint(graphs: List[nx.Graph],
                                            settings: Optional[Dict[str, Any]] = None) -> nx.DiGraph:
    """
    Calculate the pairwise joint assembly index for a list of graphs.

    This function computes the joint assembly index for all unique pairs of graphs
    in the input list. It joins each pair of graphs, calculates their assembly index
    in parallel, and then composes the resulting pathways into a directed graph.

    Parameters
    ----------
    graphs : List[nx.Graph]
        A list of NetworkX graphs representing molecular structures or other entities.
    settings : Optional[Dict[str, Any]], optional
        A dictionary of settings to configure the `calculate_assembly_index_parallel` function.
        Defaults to an empty dictionary if not provided.

    Returns
    -------
    nx.DiGraph
        A directed graph composed of the pathways resulting from the pairwise joint
        assembly index calculations.

    Notes
    -----
    - The function uses `join_graphs` to combine each pair of graphs.
    - The `calculate_assembly_index_parallel` function is used to calculate the assembly
      index for the joined graphs in parallel.
    """
    settings = settings or {}

    joined_pairs = [
        join_graphs([graphs[i], graphs[j]])
        for i in range(len(graphs))
        for j in range(i + 1, len(graphs))
    ]

    pathways = calculate_assembly_index_parallel(joined_pairs, settings)[-1]
    return nx.compose_all(pathways)
