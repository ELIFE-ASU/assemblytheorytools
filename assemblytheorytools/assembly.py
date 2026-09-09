"""
Assembly index calculation for molecules, strings and graphs.

This module wraps the external assembly calculators and exposes them through a
uniform interface. Three backends are supported: the bundled C++ ``assembly``
executable, the ``assembly_theory`` Rust extension, and ``assemblycfg`` for
context-free-grammar upper bounds. Helpers are provided for locating and
compiling the C++ binary, parsing its output, correcting joint assembly indices,
and deriving bounds, ratios and similarity measures.
"""

import assembly_theory as at_rust
import assemblycfg
import json
import networkx as nx
import numpy as np
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from functools import cache, partial
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from typing import (Union, List, Optional, Sequence, Tuple, Dict, Any,
                    NamedTuple, Callable, Hashable, Iterable)

from .construction import (_VO_TYPES,
                           _VO_TYPE_ERROR,
                           parse_pathway_file,
                           parse_pathway_dot,
                           parse_string_pathway_file,
                           molstr_to_str,
                           convert_digraph_vo_to_target)
from .tools_file import prep_json, safe_folder_remove
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
_AI_PATTERN = re.compile(r"assembly index:\s*(\d+)")
_MIN_AI_PATTERN = re.compile(r"min AI found so far:\s*(\d+)")


def _read_ai_from_output(file_path: str) -> int:
    """
    Read the assembly index from the first line of an assembler output file.

    Parameters
    ----------
    file_path : str
        Path to the ``*Out`` file written by the assembly calculator.

    Returns
    -------
    int
        The assembly index, or ``-1`` if the first line does not report one.
    """
    with open(file_path, "r") as f:
        match = _AI_PATTERN.search(f.readline())
    return int(match.group(1)) if match else -1


def _scan_log_for_min_ai(log_file: str, debug: bool = False) -> int:
    """
    Find the best assembly index the assembler reported before it was stopped.

    The log is scanned in reverse for the last ``min AI found so far`` entry,
    which acts as an upper bound when a run times out.

    Parameters
    ----------
    log_file : str
        Path to the assembler log file.
    debug : bool, optional
        If True, print the log contents. Default is False.

    Returns
    -------
    int
        The last reported minimum assembly index, or ``-1`` if none was logged.
    """
    with open(log_file, "r") as log:
        log_lines = log.readlines()

    if debug:
        print(f"log_lines: {log_lines}")

    for line in reversed(log_lines):
        match = _MIN_AI_PATTERN.search(line)
        if match:
            return int(match.group(1))
    return -1


def _count_edges(mol: Union[nx.Graph, Chem.Mol]) -> int:
    """
    Count the edges of a graph or the bonds of an RDKit molecule.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.

    Returns
    -------
    int
        Number of edges (bonds) in the input.
    """
    return mol.number_of_edges() if isinstance(mol, nx.Graph) else mol.GetNumBonds()


def _prepare_bound_input(mol: Union[nx.Graph, Chem.Mol],
                         strip_hydrogen: bool) -> Union[nx.Graph, Chem.Mol]:
    """
    Validate a bound-calculation input and optionally remove its hydrogens.

    Parameters
    ----------
    mol : Union[nx.Graph, Chem.Mol]
        The input molecular graph or RDKit molecule.
    strip_hydrogen : bool
        If True, remove hydrogen atoms before returning.

    Returns
    -------
    Union[nx.Graph, Chem.Mol]
        The input, with hydrogens removed when requested.

    Raises
    ------
    ValueError
        If the input type is not supported.
    """
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
    """
    with open(file_path, "r") as f:
        return next(int(line.split(":")[-1].strip().strip('\n')) for line in f if "assembly index" in line)


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


def add_to_bashrc(export_line: str, file: str = ".bashrc") -> None:
    """
    Append an export line to the specified bash configuration file.

    Parameters
    ----------
    export_line : str
        The export line to add to the bash configuration file.
    file : str, optional
        The name of the bash configuration file, by default ".bashrc".

    Returns
    -------
    None
    """
    # Get the path to the file in the user's home directory
    file_path = os.path.expanduser(f"~/{file}")

    # Open the .bashrc file in append mode and write the export line to it
    with open(file_path, "a") as f:
        f.write(f"\nexport {export_line}\n")


def add_assembly_to_path(str_mode: bool = False) -> str:
    """
    Ensure the assembly executable path is available in the environment and return it.

    The function checks the environment for a path variable (`ASS_STR_PATH` when
    *str_mode* is True, otherwise `ASS_PATH`). If not present it looks for a
    precompiled executable in the package `precompiled` directory, attempts to
    compile the assembly code when necessary, and sets the environment variable.

    Parameters
    ----------
    str_mode : bool, optional
        If True, operate on the string-assembly executable variable
        ``ASS_STR_PATH``; otherwise operate on the molecular assembly variable
        ``ASS_PATH``. Default is False.

    Returns
    -------
    str
        Absolute path to the assembly executable stored in the chosen environment variable.

    Raises
    ------
    FileNotFoundError
        If the executable cannot be located or compiled successfully.

    Notes
    -----
    - The function mutates ``os.environ`` by setting the selected key.
    - The function searches for executables inside the package `precompiled`
      folder adjacent to the module file and may call ``compile_assembly_cpp()``
      to build a missing executable.
    """
    # Determine the environment variable key based on the mode
    key = "ASS_STR_PATH" if str_mode else "ASS_PATH"

    # Check if the environment variable is already set
    if not os.environ.get(key):
        # Default executable name for Linux systems
        exec_name = "asscpp_public_static_linux" if str_mode else "asscpp_combined_static_linux"
        full_att_path = os.path.join(os.path.dirname(__file__), "precompiled", exec_name)

        # Check if the precompiled executable exists
        if not os.path.isfile(full_att_path):
            # Fallback to the generic assembly executable name
            exec_name = "assembly"
            full_att_path = os.path.join(os.path.dirname(__file__), "precompiled", exec_name)

            # If the executable still doesn't exist, attempt to compile it
            if not os.path.isfile(full_att_path):
                print("Assembly code not found.", flush=True)
                compile_assembly_cpp()  # Compile the assembly executable
                full_att_path = os.path.join(os.path.dirname(__file__), "precompiled", "assembly")

                # Raise an error if the compiled executable cannot be found
                if not os.path.isfile(full_att_path):
                    raise FileNotFoundError(f"Failed to compile assembly code: {full_att_path}")

        # Set the environment variable to the executable path
        os.environ[key] = full_att_path

    # Return the path stored in the environment variable
    return os.environ[key]


def compile_assembly_cpp_script(assembly_tar_path: str = "assemblycpp-main",
                                boost_version: str = "1_86_0",
                                exe_name: str = "asscpp_v5") -> None:
    """
    Compile a packaged assembly C++ tarball into a local executable and install it for user use.

    This helper extracts a tarball containing the assembly C++ source (expected to
    contain a v5 combined source tree), downloads or locates Boost as required,
    compiles the main source into a standalone executable and installs the result
    into the current working directory.

    Parameters
    ----------
    assembly_tar_path : str, optional
        Base path (without ``.tar.gz``) to the packaged assembly source archive.
        Default is ``assemblycpp-main``, implying an archive named
        ``assemblycpp-main.tar.gz`` in the current working directory.
    boost_version : str, optional
        Boost release identifier to download when a system-provided Boost is not
        available, formatted like ``1_86_0``. Default is ``1_86_0``.
    exe_name : str, optional
        Base name for the produced executable file. Default is ``asscpp_v5``.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If a required external command (``wget``, ``g++``, ``clang++``, ...) fails.
    OSError
        On filesystem or permission errors, or if the host platform is unsupported.
    FileNotFoundError
        If the source archive is missing or the executable is absent after the build.

    Notes
    -----
    - On Linux the GNU toolchain (``tar``, ``wget``, ``g++``) is used and an
      ``export ASS_PATH=...`` line is appended to ``~/.bashrc`` and ``~/.profile``;
      inspect those files if unwanted modifications occur.
    - On macOS Boost is located through Homebrew (``brew --prefix boost``) and the
      source is compiled with ``clang++``.
    - The tarball is assumed to contain ``v5_combined_linux/main.cpp``.
    - Use this helper only in trusted environments: it downloads over the network
      and runs compilers.
    """
    print("compile_assembly_code", flush=True)

    # Detect operating system
    system = platform.system().lower()  # Returns 'linux', 'darwin' (macOS), etc.

    if system == "linux":
        uncompress = "tar -xvzf"
        remove = "rm -r"
        boost_code = f"boost_{boost_version}"
        exe_dir = os.path.abspath(os.path.expanduser(os.path.join(os.getcwd(), exe_name)))  # Path to executable

        # Uncompress the assembly code
        run_command(f"{uncompress} {assembly_tar_path}.tar.gz")

        # Get the Boost library
        subprocess.run(
            f"wget 'https://archives.boost.io/release/{boost_version.replace('_', '.')}/source/{boost_code}.tar.gz'",
            shell=True, check=True)

        # Unzip the Boost code
        run_command(f"{uncompress} {boost_code}.tar.gz")

        # Compile the assembly code
        t0 = time.time()
        run_command(f"g++ {assembly_tar_path}/v5_combined_linux/main.cpp -O3 -o {exe_dir} -I{boost_code}/")
        t1 = time.time()
        print(f"Compilation time: {t1 - t0:.2f} seconds", flush=True)

        # Set the permissions to allow execution
        os.chmod(exe_dir, 0o755)

        # Remove unnecessary files and folders
        run_command(f"{remove} {boost_code}.tar.gz")
        run_command(f"{remove} {boost_code}/")
        run_command(f"{remove} {assembly_tar_path}/")

        # Add the executable path to the user's shell configuration
        add_to_bashrc(f"ASS_PATH={exe_dir}", file=".bashrc")
        add_to_bashrc(f"ASS_PATH={exe_dir}", file=".profile")

        print("Done!", flush=True)

    elif system == "darwin":  # macOS
        print("Running on macOS: Using brew to install Boost and clang++ to compile.", flush=True)

        # Install Boost using Homebrew
        subprocess.run("brew install boost", shell=True, check=True)

        # Use 'brew --prefix' to find the base installation directory for Boost
        brew_prefix = subprocess.check_output("brew --prefix boost", shell=True, text=True).strip()

        # Define paths for compilation based on the Brew prefix
        boost_include = os.path.join(brew_prefix, "include")
        boost_lib = os.path.join(brew_prefix, "lib")
        exe_dir = os.path.abspath(os.path.expanduser(os.path.join(os.getcwd(), "assemblycpp3")))

        # Compile the assembly code with clang++
        t0 = time.time()
        subprocess.run(
            f"clang++ -std=c++17 {assembly_tar_path}/v5_combined_linux/main.cpp -O3 -o {exe_dir} "
            f"-I{boost_include} -L{boost_lib}",
            shell=True, check=True)
        t1 = time.time()
        print(f"Compilation time: {t1 - t0:.2f} seconds", flush=True)

        # Set the permissions to allow execution
        os.chmod(exe_dir, 0o755)
        print("Compilation on macOS completed successfully!", flush=True)

    else:
        raise OSError(f"Unsupported operating system: {system}")


def compile_assembly_cpp() -> None:
    """
    Compile the assemblycpp C++ project and install the produced executable.

    This function clones the `assemblycpp-v5` repository, configures and builds it
    using CMake (platform-specific adjustments applied), moves the resulting
    executable to `assemblytheorytools/precompiled/assembly`, sets executable
    permissions, and removes temporary build artifacts.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If required build tools (e.g. `git`, `cmake`) are missing or the host
        operating system is unsupported.

    Notes
    -----
    - Host platform detection is performed via `platform.system().lower()` and
      behaviour is adjusted for `linux`, `darwin` (macOS) and `windows`.
    - On macOS the function may attempt to install missing dependencies using
      Homebrew; on Linux it currently requires `git` and `cmake` to be present.
    - The function temporarily changes the working directory to the cloned
      repository and restores the original working directory on exit.
    - Build failures are caught, reported, and terminate the interpreter via
      ``exit()`` rather than propagating.
    """

    start_dir = os.getcwd()
    try:
        print(flush=True)
        system = platform.system().lower()
        print(f"Compiling assCPP. Detected operating system: {system}", flush=True)

        if system == "darwin":
            # check if brew is installed
            if shutil.which("brew") is None:
                print('Homebrew is not installed. Installing Homebrew...', flush=True)
                run_command(
                    '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')

            # check if git is installed
            if shutil.which("git") is None:
                print('Git is not installed. Installing Git...', flush=True)
                run_command('brew install git')

            # Check if cmake is installed
            if shutil.which("cmake") is None:
                print('CMake is not installed. Installing CMake...', flush=True)
                run_command('brew install cmake')

        if system == "linux":
            # check if git is installed
            if shutil.which("git") is None:
                raise OSError(
                    "Git is not installed. Please install Git to compile assemblycpp on Linux.\n sudo apt update \n sudo apt install git")

            # Check if cmake is installed
            if shutil.which("cmake") is None:
                raise OSError(
                    "CMake is not installed. Please install CMake to compile assemblycpp on Linux.\n sudo apt update \n sudo apt install cmake")

        subprocess.run(
            "git clone https://github.com/LouieSlocombe/assemblycpp-v5.git",
            shell=True, check=True)

        # Change to the assemblycpp directory
        assemblycpp_dir = os.path.join(start_dir, "assemblycpp-v5")
        os.chdir(assemblycpp_dir)
        run_command('cmake -S . -B build')

        # Compile the assembly code
        if system in ("linux", "darwin"):
            run_command('cmake --build build')
        elif system == "windows":
            # For Windows, we need to specify the generator
            run_command('cmake --build build --config Release')
        else:
            raise OSError(f"Unsupported operating system: {system}")

        # Move the compiled executable into the package's precompiled folder
        exe_name = "assembly"
        exe_path = os.path.join(assemblycpp_dir, "build", "bin", exe_name)
        end_path = os.path.join(start_dir, "assemblytheorytools", "precompiled", exe_name)
        shutil.move(exe_path, end_path)
        # Remove the assemblycpp directory
        shutil.rmtree(assemblycpp_dir)
        # make the executable executable
        os.chmod(end_path, 0o755)
        os.chdir(start_dir)
        print("Assembly code compiled successfully.", flush=True)
    except Exception as e:
        print(f"Failed to automatically compile the assembly code: {e}", flush=True)
        print("Please refer to the manual compilation instructions on the ATT GitHub page.", flush=True)
        os.chdir(start_dir)
        exit()


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
        # Get the number of connected components in the graph
        num_components = nx.number_connected_components(mol)
    elif isinstance(mol, Chem.Mol):
        # Get the number of components in the RDKit molecular object
        num_components = len(Chem.rdmolops.GetMolFrags(mol=Chem.Mol(mol)))
    else:
        raise ValueError("Input not supported")

    # Each additional component costs one joining operation
    return ass_index - max(0, num_components - 1)


def _convert_timeout_for_platform(seconds: float) -> int:
    """
    Convert a timeout expressed in seconds to platform-specific integer units.

    Behavior:
      - Windows (platform name contains ``"windows"``) -> milliseconds (seconds * 1_000)
      - Linux (``"linux"``) or macOS (``"darwin"``) -> microseconds (seconds * 1_000_000)
      - Other platforms -> integer seconds (``int(seconds)``)

    Parameters
    ----------
    seconds : float
        Timeout value in seconds. Expected to be a numeric value (non-negative
        when used as a timeout).

    Returns
    -------
    int
        The converted timeout suitable for passing to platform-specific APIs.
    """
    system = platform.system().lower()
    if "windows" in system:
        return int(seconds * 1_000)
    if "linux" in system or "darwin" in system:
        return int(seconds * 1_000_000)
    return int(seconds)


def calculate_assembly_index(graph: Union[nx.Graph, Chem.Mol],
                             dir_code: Optional[str] = None,
                             timeout: float = 100.0,
                             save_dir: bool = False,
                             debug: bool = False,
                             joint_corr: bool = True,
                             strip_hydrogen: bool = False,
                             return_log_file: bool = False,
                             canonicalize: bool = True,
                             exact: bool = False) -> Union[Tuple[int, Any, Any], Tuple[int, Any, Any, Optional[str]]]:
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
        Path to the assembly executable; if None, the bundled/compiled binary
        is located via add_assembly_to_path or equivalent.
    timeout : float, optional
        Maximum time in seconds to allow the external calculator to run.
        Default is 100.0.
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
        the fourth element of the returned tuple. Default is False.
    canonicalize : bool, optional
        If True, canonicalize the node labels in the graph.
        Default is True.
    exact : bool, optional
        If True, enforce exact mode for assembly index calculation.
        Default is False.

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
    - Temporary working directories named like ``ai_calc_<timestamp>`` are created
      in the working directory when ``save_dir`` (or ``debug``) is True; otherwise
      a system temporary directory is used.
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
    # Initialize variables
    ai = -1
    virtual_objects = None
    pathway = None
    timed_out = False

    # Get the assembly code directory
    if dir_code is None:
        dir_code = add_assembly_to_path()

    # Create working directory
    if debug:
        save_dir = True
    temp_dir = f"ai_calc_{datetime.now().strftime('%H_%M_%f')}" if save_dir else tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    # Check the input type and prepare input files
    in_type = type(graph)
    if in_type is Chem.Mol:
        graph = mol_to_nx(graph)

    if strip_hydrogen:
        graph = remove_hydrogen_from_graph(graph)
    if canonicalize:
        graph = canonicalize_node_labels(graph)
    file_path_in = os.path.join(temp_dir, "graph_in")
    write_ass_graph_file(graph, file_name=file_path_in)

    # Define output and log file paths
    file_path_out = file_path_in + "Out"
    file_path_pathway = file_path_in + "Pathway"
    log_file = os.path.join(temp_dir, "assembly_output.log")

    # Convert timeout flag from seconds to x miliseconds in windows and x microseconds in linux/macOS
    timeout_flag = _convert_timeout_for_platform(timeout)

    # Run the assembly code and log output
    try:
        with open(log_file, "w") as log:
            start_time = time.time()
            process = subprocess.Popen(
                [dir_code,
                 file_path_in,
                 '-memTest=0',
                 '-removeHydrogens=0',
                 '-compensateDisjoint=0',
                 f'-runTime={timeout_flag}'],
                stdout=log,
                stderr=log
            )
            process.wait()
            if time.time() - start_time > timeout:
                timed_out = True
                print("Warning: Assembly calculation timed out.", flush=True)

    except Exception as e:
        print(f"Error: {e}", flush=True)
        if debug:
            traceback.print_exc()

    if not timed_out:
        # The calculation finished properly, so we can read the output file
        ai = _read_ai_from_output(file_path_out)
    elif os.path.exists(log_file):
        # Fall back on the best bound the assembler logged before it was stopped
        try:
            last_ai = _scan_log_for_min_ai(log_file)
            if not exact:
                ai = last_ai
            if ai == -1:
                print("No minimum AI found before timeout.", flush=True)
            else:
                print(f"Upper Bound to AI Found: AI =< {ai}", flush=True)
        except Exception as e:
            print(f"Failed to read AI from log file: {e}", flush=True)

    # Process pathway output if available
    if os.path.isfile(file_path_pathway):
        try:
            prep_json(file_path_pathway)
            pathway, virtual_objects = parse_pathway_file(file_path_pathway,
                                                          vo_type='graph',
                                                          debug=debug,
                                                          input_graph=graph)

            if in_type is Chem.Mol:
                # Convert virtual objects back to SMILES if input was Mol
                virtual_objects = [nx_to_smi(v, add_hydrogens=False) for v in virtual_objects]
                # Convert pathway to SMILES representation
                pathway = convert_digraph_vo_to_target(pathway, target='smi')

        except Exception as e:
            print(f"Failed to load pathway data: {e}", flush=True)
            if debug:
                traceback.print_exc()

    # Apply joint correction if necessary
    if joint_corr and ai > 0:
        ai = joint_assembly_index_correction(graph, ai)

    # Print log file path if required
    if return_log_file:
        print(f"Log file printed to: {log_file}", flush=True)

    # Return based on flag
    return (ai, virtual_objects, pathway) if not return_log_file else (ai, virtual_objects, pathway, log_file)


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

    settings = settings or {}

    if parallel:
        ai_list = calculate_assembly_index_parallel(graphs, settings)[0]
    else:
        ai_list = [calculate_assembly_index(graph, **settings)[0] for graph in graphs]

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

    ai_list = [calculate_string_assembly_index(s, **settings)[0] for s in strings]

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

    n_t = sum(n_i)  # Total copy number of all objects
    if n_t == 0:
        raise ValueError("The copy numbers must not sum to zero")

    ai_list = [regularise_assembly_index(ai) for ai in ai_list]
    return float(sum(np.exp(ai) * ((n - 1) / n_t)
                     for ai, n in zip(ai_list, n_i)))


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
        if identity in seen:
            counts[seen[identity]] += 1
        else:
            seen[identity] = len(unique)
            unique.append(obj)
            counts.append(1)

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
                {n: path.nodes[n][node_key] for n in path.nodes},
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
        if node_key is None:
            observed = {n for path in pathways
                        for n in path.nodes if path.out_degree(n) == 0}
        else:
            observed = {path.nodes[n][node_key] for path in pathways
                        for n in path.nodes if path.out_degree(n) == 0}

    space = joint_assembly_space(pathways, node_key=node_key)
    nodes = set(space.nodes)

    return len(nodes & set(observed)) / len(nodes)


def calculate_string_assembly_index(input_data: Union[str, List[str]],
                                    dir_code: Optional[str] = None,
                                    timeout: float = 100.0,
                                    debug: bool = False,
                                    directed: bool = True,
                                    mode: str = "str",
                                    return_log_file: bool = False) -> Union[
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
        Path to the assembly executable; when ``None`` the bundled/compiled binary
        is located via ``add_assembly_to_path`` or equivalent.
    timeout : float, optional
        Maximum time in seconds to allow the external calculator to run. Default is
        100.0.
    debug : bool, optional
        If True, create a timestamped temporary directory and print debug output.
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
        the fourth element of the returned tuple. Default is False.

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
    - In 'str' mode the function expects the string-assembly binary (set via
      environment variable ``ASS_STR_PATH`` or found by ``add_assembly_to_path``).
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
        if mode in ["str", "cfg"]:
            mode = "mol"  # Use the molecular assembly calculator for undirected strings
            print("Warning: only mode 'mol' is currently supported for undirected strings. Switching to 'mol'.",
                  flush=True)
    elif mode == "mol":
        mode = "str"  # Use the string assembly calculator for directed strings
        print("Warning: mode 'mol' is not currently supported for directed strings. Switching to 'str'.", flush=True)

    log_file = None
    if isinstance(input_data, str):
        # Handle the case where input_data is a single string
        string = input_data
        delimiters = []
        if len(string) == 1:
            return (0, None, None) if not return_log_file else (0, None, None, None)

    elif isinstance(input_data, list):
        input_data = [s for s in input_data if len(s) > 1]  # Remove elements of the list that are single characters
        if len(input_data) == 0:
            return (0, None, None) if not return_log_file else (0, None, None, None)

        if mode != "cfg":
            if len(input_data) > 95:
                raise ValueError(
                    "Input list contains more than 95 objects. Joint assembly index calculations are only supported for up to 95 objects except in cfg (RePair) approximation mode.")
            # Handle joint assembly case
            string, delimiters = prep_joint_string_ai(input_data)
    else:
        raise ValueError("Input must be either a single string or a list of strings")

    # Check input types
    assert (dir_code is None) or isinstance(dir_code, str), "Directory code must be a string"
    assert isinstance(timeout, (int, float)), "Timeout must be an integer or float"
    assert isinstance(debug, bool), "Debug must be a boolean"
    assert isinstance(directed, bool), "Directed must be a boolean"

    if mode == "mol":  # Use the molecular assembly cpp calculator
        if directed:
            graph = get_dir_str_molecule(string)
            edge_color_dict = None
        else:
            graph, edge_color_dict = get_undir_str_molecule(string, debug=debug)

        if debug:
            # String-Molecular Graph Nodes colors
            print("\nNode colors:", flush=True)
            for node, data in graph.nodes(data=True):
                print(f"Node {node}: {data.get('color', 'No color')}", flush=True)

            # String-Molecular Graph Edge colors
            print("\nEdge colors:", flush=True)
            for u, v, data in graph.edges(data=True):
                print(f"Edge {u}-{v}: {data.get('color', 'No color')}", flush=True)

            print("Return log file:", return_log_file, flush=True)

        graph_result = calculate_assembly_index(graph,
                                                dir_code=dir_code,
                                                timeout=timeout,
                                                debug=debug,
                                                joint_corr=False,
                                                strip_hydrogen=False,
                                                return_log_file=return_log_file)
        graph_ai, graph_virtual_obj, graph_path = graph_result[:3]
        if return_log_file:
            log_file = graph_result[3]

        # Correct for joint assembly and directed encoding
        ai = graph_ai - 2 * len(delimiters)
        if directed:
            ai = ai - len(set(string))

        if debug:
            print(f"Assembly Index: {ai}", flush=True)
            print(f"\n\nGraph Virtual Objects:\n", flush=True)
            print(f"Graph VOs type is : {type(graph_virtual_obj)}")
            for item in graph_virtual_obj:
                print(molstr_to_str(item, edge_color_dict=edge_color_dict), flush=True)
            print(f"\nGraph Path type is: {type(graph_path)}", flush=True)
            print(graph_path.edges(data=True), flush=True)

        # Parse the virtual object and path
        virt_obj = [molstr_to_str(item, edge_color_dict=edge_color_dict) for item in graph_virtual_obj]
        for node in graph_path.nodes(data=True):
            node[1]["vo"] = molstr_to_str(node[1]["vo"], edge_color_dict=edge_color_dict)
        path = graph_path

        # Convert to (joint) assembly index of directed strings.
        return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, log_file)

    elif mode == "str":  # Use the string assembly cpp calculator

        # Initialize variables
        ai = -1
        virt_obj = None
        path = None
        timed_out = False  # Flag for timeout tracking

        # Get the assembly code directory
        if dir_code is None:
            dir_code = add_assembly_to_path(str_mode=True)

        # Create working directory
        temp_dir = os.path.abspath(tempfile.mkdtemp())
        os.makedirs(temp_dir, exist_ok=True)

        # Put string into a temporary text file
        file_path_in = os.path.join(temp_dir, "string_in")
        with open(file_path_in, "w") as f:
            f.write(string)

        # Define output and log file paths
        file_path_out = file_path_in + "Out"
        log_file = os.path.join(temp_dir, "assembly_output.log")

        if debug:
            print(f"Temporary directory created: {temp_dir}", flush=True)

        # Run the assembly code and log output
        try:
            with open(log_file, "w") as log:

                if debug:
                    print(f"Calling\n {dir_code} {file_path_in} {str(int(directed == 0))} 1", flush=True)

                # Start the process
                process = subprocess.Popen(
                    [dir_code, file_path_in, "-runStrings=1"],  # Ian purged the directed string code from asscpp_public
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=temp_dir
                )

                try:
                    # Wait for process to finish or timeout
                    stdout_data, _ = process.communicate(timeout=timeout)

                except subprocess.TimeoutExpired:
                    print("Warning: Assembly calculation timed out. Terminating...")

                    # Send SIGINT to simulate Ctrl+C
                    process.send_signal(signal.SIGINT)
                    process.wait()
                    try:
                        # Give it 2 seconds to exit gracefully
                        stdout_data, _ = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        print("Process did not terminate, killing it.")
                        process.kill()
                        stdout_data, _ = process.communicate()
                        if debug:
                            traceback.print_exc()

                    timed_out = True

                # Write whatever output we got to the log
                if stdout_data:
                    log.write(stdout_data.decode(errors="replace"))
                    log.flush()

        except Exception as e:
            print(f"Error: {e}")
            if debug:
                traceback.print_exc()

        if not timed_out:
            # The calculation finished properly, so we can read the output file
            if debug:
                print("Assembly calculation completed successfully.", flush=True)

            ai = _read_ai_from_output(file_path_out)

        elif os.path.exists(log_file):
            # Fall back on the best bound the assembler logged before it was stopped
            if debug:
                print(f"log_file: {log_file}")
            try:
                ai = _scan_log_for_min_ai(log_file, debug=debug)

                # Print appropriate messages based on timeout
                if ai == -1:
                    print("No assembly paths found before timeout.")
                else:
                    print(f"Upper Bound to AI Found: AI =< {ai - 2 * len(delimiters)}")

            except Exception as e:
                print(f"Failed to read AI from log file: {e}")

        ai -= 2 * len(delimiters)  # Convert to (joint) assembly index of strings

        # Process pathway output if available
        pathway_files = [f for f in os.listdir(temp_dir) if f.endswith("Pathway")]
        if pathway_files:
            file_path_pathway = os.path.join(temp_dir, pathway_files[0])
            if os.path.isfile(file_path_pathway):
                if debug:
                    print(f"Parsing pathway data from: {file_path_pathway}", flush=True)
                try:
                    virt_obj, path = parse_string_pathway_file(file_path_pathway)
                except Exception as e:
                    print(f"Failed to load pathway data: {e}", flush=True)
                    if debug:
                        traceback.print_exc()
        elif debug:
            print(f"No pathway file found in: {temp_dir}", flush=True)

        # Print log file path if required
        if return_log_file:
            print(f"Log file printed to: {log_file}", flush=True)

        # Remove temporary files
        if not debug:
            if os.path.exists(file_path_in):
                os.remove(file_path_in)
            shutil.rmtree(temp_dir)  # Clean up the temporary directory
        else:
            print(f"Temporary directory retained for debugging: {temp_dir}", flush=True)
            print(f'File path in: {file_path_in}', flush=True)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    print(f' - {file}', flush=True)

        # Return based on flag
        return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, log_file)

    elif mode == "cfg":  # Use the RePair upper bound
        if not directed:
            raise ValueError(
                "Current CFG code works natively for directed strings. Directed string assembly index is an upper bound to undirected string assembly index, so you may still use the directed calculator.")

        # Convert to (joint) assembly index of directed strings
        path_len, virt_obj, path = assemblycfg.repair_with_pathways(input_data, f_print=False)
        return path_len, virt_obj, path  # Note: there is no log file for CFG

    else:
        raise ValueError("Mode must be either 'mol', 'str', or 'cfg'.")


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
    if ai is None or ai < 0:
        return 0
    return ai


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
    # Validate input
    if graphs is None or not hasattr(graphs, "__iter__"):
        raise ValueError("`graphs` must be an iterable of graph objects")

    settings = settings or {}

    # Prepare the calculation function with provided settings and run in parallel
    calc_ai = partial(calculate_assembly_index, **settings)
    results = mp_calc(calc_ai, graphs)

    # If no results (e.g. empty input) return empty list
    if not results:
        return []

    # Transpose results: group values of the same field together
    return [list(group) for group in zip(*results)]


def _get_most_recent_calc() -> str:
    """
    Locate the most recent assembly calculation directory in the current working directory.

    The function scans the current working directory for folders whose names start
    with ``ai_calc_`` and returns the path to the most recently created one (as
    determined by the file system creation time).

    Returns
    -------
    str
        Absolute path to the most recent calculation folder.

    Raises
    ------
    FileNotFoundError
        If no folder starting with ``ai_calc_`` is present in the current working directory.

    Notes
    -----
    - The function uses :func:`os.path.getctime` to determine the "most recent"
      folder. On some platforms this value measures the creation time; on others
      it may reflect the last metadata change.
    - The function returns an absolute path built from the current working
      directory and the selected folder name.
    """
    assembly_folders = [folder for folder in os.listdir(os.getcwd()) if folder.startswith("ai_calc_")]
    if not assembly_folders:
        raise FileNotFoundError("No 'ai_calc_' folders found in the current working directory")
    assembly_folder = max(assembly_folders, key=lambda fn: os.path.getctime(os.path.join(os.getcwd(), fn)))
    return os.path.join(os.getcwd(), assembly_folder)


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
    # Locate the most recent assembly calculation folder
    assembly_path = _get_most_recent_calc()
    if not os.path.isdir(assembly_path):
        raise FileNotFoundError(f"No assembly calculation folder found: {assembly_path}")

    # Find files ending with "Out" and pick the most recent by creation time
    out_files = [f for f in os.listdir(assembly_path) if f.endswith("Out")]
    if not out_files:
        raise FileNotFoundError(f"No '*Out' files found in {assembly_path}")

    out_files = sorted(out_files, key=lambda fn: os.path.getctime(os.path.join(assembly_path, fn)))
    latest_file = os.path.join(assembly_path, out_files[-1])

    # Read the last line and extract the trailing numeric token
    with open(latest_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            raise ValueError(f"File {latest_file} is empty")
        last_line = lines[-1].strip()

    # Extract the value after the last colon and convert to float (assumed microseconds)
    try:
        time_to_completion = float(last_line.split(":")[-1].strip())
    except Exception as e:
        raise ValueError(f"Failed to parse time from '{latest_file}': {e}") from e

    # Cleanup the assembly folder and return time in seconds
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

    # Ensure both graphs are of the same type
    if type(graph1) is not type(graph2):
        raise ValueError("Input graphs must be of the same type")

    if type(graph1) is Chem.Mol:
        # Convert RDKit Mol to NetworkX graph
        graph1 = mol_to_nx(graph1)
        graph2 = mol_to_nx(graph2)

    # Check if the inputs are isomorphic
    # in which case the semi-metric distance is 0 and the user may not intend to compare these mols
    if nx.is_isomorphic(graph1, graph2):
        print("Input graphs are isomorphic.", flush=True)
        return 0.0

    # Calculate the joint assembly index
    jai = calculate_assembly_index(join_graphs([graph1, graph2]), **settings)[0]
    if jai <= -1:
        print("No minimum JAI found before timeout.", flush=True)
        return -1.0

    # Calculate the assembly index for each subgraph
    sum_ai = calculate_sum_assembly_index([graph1, graph2], settings, parallel=parallel)

    # Calculate the semi-metric distance
    semi_metric = 2.0 * jai - sum_ai
    if normalise:
        return semi_metric / sum_ai

    return semi_metric


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

    # The shortest addition chain gives the tightest bound, but is only tabulated
    # up to 9999; beyond that fall back on repeated doubling
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

    settings = settings or {}

    if parallel:
        # calculate_assembly_index_parallel returns transposed results; first element is ai list
        ai_list = calculate_assembly_index_parallel(graphs, settings)[0]
    else:
        ai_list = [calculate_assembly_index(graph, **settings)[0] for graph in graphs]

    # If any assembly index is invalid, return -1 to indicate failure
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
        # Copy rather than mutate the caller's dictionary
        settings = {**settings, "exact": True}

    # Calculate assembly index sum
    ai_sum = calculate_sum_assembly_index(graphs, settings, parallel=parallel)

    if ai_sum < 0:
        return -1.0

    # Join the graphs into a single joint graph
    joint_graphs = join_graphs(graphs)

    # Calculate the joint assembly index
    ai_jai = calculate_assembly_index(joint_graphs, **settings)[0]

    if ai_jai < 0:
        return -1.0

    # Compute the assembly similarity index
    return (ai_sum / ai_jai - 1.0) if ai_jai != 0 else 0.0


def _parse_graph_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise a list of graph entries from a pathway JSON file.

    Parameters
    ----------
    entries : list of dict
        Raw graph entries using the assembler's ``Vertices``/``Edges``/
        ``VertexColours``/``EdgeColours`` key names.

    Returns
    -------
    list of dict
        The same entries keyed by ``vertices``, ``edges``, ``vertex_colours``
        and ``edge_colours``, with missing fields defaulting to empty lists.
    """
    return [{'vertices': entry.get('Vertices', []),
             'edges': entry.get('Edges', []),
             'vertex_colours': entry.get('VertexColours', []),
             'edge_colours': entry.get('EdgeColours', [])}
            for entry in entries]


def _parse_pathway_file(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and parse a pathway JSON structure into a stable dictionary.

    The function accepts a parsed JSON-like mapping produced by the assembly tool
    and returns a normalized dictionary with consistent keys used by downstream
    functions.

    Parameters
    ----------
    data : dict
        Parsed JSON object describing a pathway.

    Returns
    -------
    dict
        A normalized pathway dictionary with the following keys:
        - ``file_graph`` : list of dict
            Each dict contains keys ``vertices``, ``edges``, ``vertex_colours``
            and ``edge_colours``.
        - ``remnant`` : list of dict
            Same structure as ``file_graph`` representing the remnant graphs.
        - ``duplicates`` : list of dict
            Each dict contains ``left_edges`` and ``right_edges`` describing
            fragment edges.
        - ``removed_edges`` : list
            Edges removed during the pathway processing.

    Notes
    -----
    - Missing fields default to empty lists, so a partial pathway file still
      yields a usable structure.
    - The returned structure is safe for direct use by functions such as the
      private helper ``_calculate_jo_from_pathway``.
    """
    return {
        'file_graph': _parse_graph_entries(data.get('file_graph', [])),
        'remnant': _parse_graph_entries(data.get('remnant', [])),
        'duplicates': [{'left_edges': dup.get('Left', []),
                        'right_edges': dup.get('Right', [])}
                       for dup in data.get('duplicates', [])],
        'removed_edges': data.get('removed_edges', []),
    }


def _calculate_jo_from_pathway(json_file: str) -> int:
    """
    Derive the joint assembly correction from a saved pathway file.

    The original graph is rebuilt from the first ``file_graph`` entry of the
    pathway JSON, and its connected component count is used to work out the
    joint-object offset that must be applied to the raw assembly index.

    Parameters
    ----------
    json_file : str
        Path to the JSON pathway file written by the assembly calculator.

    Returns
    -------
    int
        The joint-object correction implied by the pathway.

    See Also
    --------
    joint_assembly_index_correction : Apply the correction to an assembly index.
    """
    # Load JSON pathway data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize pathway structure (expects compatible format)
    data = _parse_pathway_file(data)

    # Build the original graph from the first file_graph entry
    edges = [tuple(edge) for edge in data["file_graph"][0]["edges"]]
    original_graph = nx.Graph()
    original_graph.add_edges_from(edges)
    original_cc = nx.number_connected_components(original_graph)

    # Initial metric: number of edges minus number of connected components
    ma = original_graph.number_of_edges() - original_cc

    jo_correction = 0
    # Maintain a mutable set of remaining edges as we remove fragments
    edge_set = set(edges)

    # Iterate over duplicate fragments (use their right-hand edges)
    for fragment in [dup["right_edges"] for dup in data["duplicates"]]:
        # Each fragment reduces ma by (size - 1)
        ma -= len(fragment) - 1

        # Atoms (nodes) involved in the fragment
        fragment_atoms = {atom for edge in fragment for atom in edge}

        # Remove the fragment edges from the global edge set
        edge_set -= {tuple(edge) for edge in fragment}

        # Reconstruct the remnant graph from remaining edges
        remnant_edges = list(edge_set)
        remnant_graph = nx.Graph()
        remnant_graph.add_edges_from(remnant_edges)

        # Number of connected components after removal
        remnant_cc = nx.number_connected_components(remnant_graph)

        # Any increase in components contributes to a correction (non-negative)
        delta_cc = max(remnant_cc - original_cc, 0)

        # Update original component count for next iteration
        original_cc = remnant_cc

        # Nodes remaining in the remnant
        remnant_atoms = {atom for edge in remnant_edges for atom in edge}

        # Overlap between fragment and remnant: contribute (overlap - 1) but not negative
        overlap_correction = max(0, len(fragment_atoms & remnant_atoms) - 1)

        # Aggregate correction: overlap correction minus any component-increase correction
        jo_correction += overlap_correction - delta_cc

    # Final JO is the baseline metric plus accumulated corrections
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
    - ``save_dir`` is forced on so that the pathway output survives long enough
      to be read back.
    - If no valid pathway file is found, or if JO calculation fails, the function
      returns -1 for the JO index.
    """

    # Copy so the save_dir override below does not leak back into the caller's dict
    settings = dict(settings or {})

    # Ensure pathway output is requested
    settings["save_dir"] = True

    # Run the assembly index calculation to produce pathway output in a temp folder
    _, vo, pathway = calculate_assembly_index(mol, **settings)

    # Locate the most recent assembly calculation folder
    assembly_path = _get_most_recent_calc()

    # Find a file that ends with "Pathway" in that folder
    pathway_files = [f for f in os.listdir(assembly_path) if f.endswith("Pathway")]
    # No pathway output available -> cannot compute JO
    if not pathway_files:
        print("No pathway file found. Returning -1.", flush=True)
        safe_folder_remove(assembly_path)
        return -1, None, None

    # Use the first pathway file found
    pathway_file = os.path.join(assembly_path, pathway_files[0])

    # Compute JO from the pathway file and handle errors
    try:
        jo = _calculate_jo_from_pathway(pathway_file)
    except Exception as e:
        print(f"Error calculating joint assembly index: {e}", flush=True)
        return -1, None, None
    finally:
        # Cleanup the temporary assembly folder
        safe_folder_remove(assembly_path)

    # Return the computed JO along with virtual objects and pathway from the AI run
    return jo, vo, pathway


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
    # Determine number of edges/bonds depending on the input type
    n_edges = _count_edges(graph)

    # If there are no edges, return 1.0 to avoid division by zero (by design)
    if n_edges == 0:
        return 1.0

    # Compute the assembly index (AI) using the existing function
    ai, _, _ = calculate_assembly_index(graph, **settings)

    # Note: this will raise ZeroDivisionError if ai == 0; if ai < 0 the result will be negative.
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
    # Determine number of edges (bonds) depending on input type
    n_edges = graph.number_of_edges() if isinstance(graph, nx.Graph) else graph.GetNumBonds()

    # Avoid division by zero when there are no edges (also skips the expensive
    # JO calculation below, which would otherwise run needlessly)
    if n_edges == 0:
        return 1.0

    # Compute the joining-operation index (JO) using existing function
    jo = calculate_assembly_index_jo(graph, settings=settings)[0]

    # Note: if jo is 0 this will raise ZeroDivisionError; if jo is -1 the result
    # will be negative which indicates a failure in JO calculation upstream.
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
    """
    Report the installed version of the Rust assembly-theory package.

    Returns
    -------
    str
        The version string, or 'unknown' if the package metadata is missing.
    """
    try:
        return version("assembly-theory")
    except PackageNotFoundError:
        return "unknown"


@cache
def _rust_supports_pathways() -> bool:
    """
    Report whether the installed Rust backend can reconstruct pathways.

    Returns
    -------
    bool
        True if ``index_search`` accepts ``max_pathways``.

    Notes
    -----
    - The backend rejects unknown keyword arguments before it looks at the mol
      block, so an empty mol block is enough to ask the question without
      running a search: a release that understands ``max_pathways`` complains
      about the molecule, and one that does not complains about the argument.
    - Release 0.6.1 reports a ``__text_signature__`` that omits arguments it
      does in fact accept, so the signature cannot be trusted for this.
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
        # A molecular graph carries no direction, so undirect anything that
        # does before nx_to_mol tries to add each bond twice
        mol = nx_to_mol(mol.to_undirected() if mol.is_directed() else mol,
                        add_hydrogens=False)

    if not isinstance(mol, Chem.Mol):
        raise ValueError("Expected a NetworkX graph or an RDKit molecule, got "
                         f"{type(mol).__name__}.")

    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    if n_atoms > 999 or n_bonds > 999:
        raise ValueError(f"The Rust backend only reads V2000 mol blocks, which hold at most "
                         f"999 atoms and 999 bonds; this molecule has {n_atoms} atoms and "
                         f"{n_bonds} bonds.")

    return Chem.MolToMolBlock(mol)


def _rust_error(error: OSError) -> ValueError:
    """
    Translate a mol block error from the Rust backend into a ValueError.

    Parameters
    ----------
    error : OSError
        The error raised by the Rust backend.

    Returns
    -------
    ValueError
        The equivalent error, with context on what the backend rejected.
    """
    return ValueError(f"The Rust backend could not read this molecule: {error}")


# The backend counts joining operations in an unsigned 32-bit integer and
# underflows to its maximum when there are none to count: a bare atom has no
# bonds to join, and a one-bond molecule is already at its full depth. Both of
# those answers are 0, and no molecule small enough for a V2000 mol block can
# have a genuine index or depth anywhere near this value.
_RUST_UNDERFLOW = 2 ** 32 - 1


def _rust_count(value: int) -> int:
    """
    Correct the backend's underflow sentinel to the zero it stands for.

    Parameters
    ----------
    value : int
        An assembly index or depth as reported by the Rust backend.

    Returns
    -------
    int
        The value, or 0 where the backend underflowed.
    """
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
    try:
        return _rust_count(at_rust.index(_mol_to_molblock(mol)))
    except OSError as e:
        raise _rust_error(e) from e


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
    try:
        return _rust_count(at_rust.depth(_mol_to_molblock(mol)))
    except OSError as e:
        raise _rust_error(e) from e


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
    try:
        return at_rust.mol_info(_mol_to_molblock(mol))
    except OSError as e:
        raise _rust_error(e) from e


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
        raise ValueError(f"bounds must be a sequence of strategy names, not a single "
                         f"string; pass [{bounds!r}] to apply only that one.")

    if timeout is not None and timeout < 0:
        raise ValueError(f"timeout must not be negative, got {timeout}.")

    mol_block = _mol_to_molblock(mol)

    # The backend takes whole milliseconds and reads 0 as 'give up immediately',
    # so round up rather than truncating a sub-millisecond timeout into that
    options = {"timeout": None if timeout is None else ceil(timeout * 1000),
               "canonize_str": canonize,
               "parallel_str": parallel,
               "memoize_str": memoize,
               "kernel_str": kernel,
               "bound_strs": list(bounds)}

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
    except OSError as e:
        raise _rust_error(e) from e

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
    elif n > 9999:
        raise ValueError("n must be less than or equal to 9999.")
    elif n == 1:
        return 0

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'integer_chain_9999.txt')
    with open(data_path, 'r') as file:
        for i, line in enumerate(file):
            if i == n + 1:
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
    # Use an empty dictionary if no settings are provided
    settings = settings or {}

    # Create a list of all unique pairs of graphs joined together
    pairwise_joined_graphs = [
        join_graphs([graphs[i], graphs[j]])
        for i in range(len(graphs))
        for j in range(i + 1, len(graphs))
    ]

    # Calculate the assembly index for each joined graph in parallel and extract pathways
    pathways = calculate_assembly_index_parallel(pairwise_joined_graphs, settings)[-1]

    # Compose the pathways into a single directed graph and return it
    return nx.compose_all(pathways)
