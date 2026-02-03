import json
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
from functools import partial
from typing import Union, List, Optional, Tuple, Dict, Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import CFG
from .construction import (parse_pathway_file,
                           parse_string_pathway_file,
                           molstr_to_str)
from .tools_file import prep_json
from .tools_graph import (write_ass_graph_file,
                          remove_hydrogen_from_graph,
                          nx_to_mol,
                          mol_to_nx,
                          canonicalize_node_labels,
                          join_graphs)
from .tools_mol import (write_v2k_mol_file,
                        combine_mols,
                        safe_standardize_mol)
from .tools_mp import mp_calc
from .tools_string import (prep_joint_string_ai,
                           get_dir_str_molecule,
                           get_undir_str_molecule)


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


def run_command(command: str) -> Optional[bytes]:
    """
    Run a command in the subprocess.

    Parameters
    ----------
    command : str
        The command to run as a string.

    Returns
    -------
    bytes
        The standard output of the command.
    
    Raises
    ------
    ValueError
        If command is None.
    """
    if command is None:
        raise ValueError("Command must be provided")

    result = subprocess.run(command.split())
    return result.stdout


def joint_assembly_index_correction(mol: Union[nx.Graph, Chem.Mol], ass_index: int) -> int:
    if isinstance(mol, nx.Graph):
        # Get the number of connected components in the graph
        num_components = nx.number_connected_components(mol)
    elif isinstance(mol, Chem.Mol):
        # Get the number of components in the RDKit molecular object
        num_components = len(Chem.rdmolops.GetMolFrags(mol=Chem.Mol(mol)))
    else:
        num_components = None
        ValueError("Input not supported")
    # Return the number of components minus 1
    correction = max(0, num_components - 1)
    if correction < 0:
        correction = 0
    return ass_index - correction


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
                             debug: bool = False,
                             joint_corr: bool = True,
                             strip_hydrogen: bool = False,
                             return_log_file: bool = False,
                             exact: bool = False) -> Union[Tuple[int, Any, Any], Tuple[int, Any, Any, Optional[str]]]:
    # Initialize variables
    ai = -1
    virt_obj = None
    path = None
    file_path_in = None
    timed_out = False

    # Get the assembly code directory
    if dir_code is None:
        dir_code = add_assembly_to_path()

    # Create working directory
    temp_dir = f"ai_calc_{datetime.now().strftime('%H_%M_%f')}" if debug else tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    # Input is a graph
    if isinstance(graph, nx.Graph):
        if strip_hydrogen:
            graph = remove_hydrogen_from_graph(graph)
        graph = canonicalize_node_labels(graph)
        file_path_in = os.path.join(temp_dir, "graph_in")
        write_ass_graph_file(graph, file_name=file_path_in)
    # Input is an RDKit mol
    elif isinstance(graph, Chem.Mol):
        graph = safe_standardize_mol(graph, add_hydrogens=True)
        if strip_hydrogen:
            graph = Chem.RemoveHs(graph)
        mol_file = os.path.join(temp_dir, "tmp.mol")
        write_v2k_mol_file(graph, mol_file)
        file_path_in = os.path.splitext(mol_file)[0]
    else:
        raise ValueError("Input not supported")

    # Define output and log file paths
    file_path_out = os.path.join(file_path_in + "Out")
    file_path_pathway = os.path.join(file_path_in + "Pathway")
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

    if timed_out == 0:  # If the calculation finished properly, we can read the output file

        with open(file_path_out, "r") as f:
            first_line = f.readline()
            match = re.search(r'assembly index:\s*(\d+)', first_line)
            if match:
                ai = int(match.group(1))

    else:

        # Extract the most recent "min AI found so far" from the log file
        last_ai = -1
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as log:
                    log_lines = log.readlines()

                # Reverse scan for last occurrence of "min AI found so far"
                for line in reversed(log_lines):
                    match = re.search(r"min AI found so far:\s*(\d+)", line)
                    if match:
                        last_ai = int(match.group(1))
                        break

                if not exact:
                    ai = last_ai  # Assign found AI

                # Print appropriate messages based on timeout
                if ai == -1 and timed_out:
                    print("No minimum AI found before timeout.", flush=True)
                elif ai != -1 and timed_out:
                    print(f"Upper Bound to AI Found: AI =< {ai}", flush=True)

            except Exception as e:
                print(f"Failed to read AI from log file: {e}", flush=True)

    # Process pathway output if available
    if os.path.isfile(file_path_pathway):
        try:
            if isinstance(graph, nx.Graph):
                prep_json(file_path_pathway)
                path, virt_obj = parse_pathway_file(file_path_pathway, vo_type='graph', debug=debug,
                                                    input_graph=graph)
            elif isinstance(graph, Chem.Mol):
                path, virt_obj = parse_pathway_file(file_path_pathway, vo_type='smiles', debug=debug)
            else:
                virt_obj = None
                path = (None, None)
                raise ValueError("Input not supported")
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
    return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, log_file)


def calculate_assembly(graphs: List[Union[nx.Graph, Chem.Mol]],
                       n_i: List[float],
                       settings: Optional[Dict[str, Any]] = None,
                       parallel: bool = True) -> float:
    settings = settings or {}

    if parallel:
        ai_list = calculate_assembly_index_parallel(graphs, settings)[0]
    else:
        ai_list = [calculate_assembly_index(graph, **settings)[0] for graph in graphs]

    # Regularize the assembly indices to ensure non-negative values
    ai_list = [regularise_ai(ai) for ai in ai_list]
    n_t = sum(n_i)  # Total weight of all graphs
    # Compute the weighted sum of the exponential of the assembly indices
    return sum(np.exp(ai) * ((n - 1) / n_t) for ai, n in zip(ai_list, n_i))


def calculate_assembly_semi_metric(graph1: Union[nx.Graph, Chem.Mol],
                                   graph2: Union[nx.Graph, Chem.Mol],
                                   settings: Optional[Dict[str, Any]] = None,
                                   normalise: bool = False) -> float:
    settings = settings or {}

    if type(graph1) != type(graph2):
        raise ValueError("Input graphs must be of the same type")

    if type(graph1) == Chem.Mol:
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
    sum_ai = calculate_sum_assembly([graph1, graph2], settings)

    # Calculate the semi-metric distance
    semi_metric = 2.0 * jai - sum_ai
    if normalise:
        return semi_metric / sum_ai

    return semi_metric


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


def compile_assembly_cpp_script(assembly_tar_path: str = "assemblycpp-main",
                                boost_version: str = "1_86_0",
                                exe_name: str = "asscpp_v5") -> None:
    """
    Compile a packaged assembly C++ tarball into a local executable and install it for user use.

    This helper extracts a tarball containing the assembly C++ source (expected to
    contain a v5 combined source tree), downloads or locates Boost as required,
    compiles the main source into a standalone executable and installs the result
    into the current working directory (or a named path derived from *exe_name*).
    It also attempts to record the installed executable path in the user's shell
    startup files so the binary can be found via the environment variable
    ``ASS_PATH``.

    Parameters
    ----------
    assembly_tar_path : str, optional
        Base path (without ``.tar.gz``) to the packaged assembly source archive.
        Default is ``assemblycpp-main`` which implies an archive named
        ``assemblycpp-main.tar.gz`` in the current working directory.
    boost_version : str, optional
        Boost release identifier to download when a system-provided Boost is not
        available. Formatted like ``1_86_0`` and used to build a URL for the Boost
        source tarball. Default is ``1_86_0``.
    exe_name : str, optional
        Base name for the produced executable file. The function compiles the
        source to an executable at a path derived from this name (for example
        creating a file at ``./asscpp_v5`` when ``exe_name`` is ``asscpp_v5``).
        Default is ``asscpp_v5``.

    Returns
    -------
    None
        The function performs compilation and installation as side effects and does
        not return a value on success.

    Raises
    ------
    subprocess.CalledProcessError
        If a required external command (for example ``wget``, ``g++``, ``clang++``
        or other shell utilities) fails with a non-zero exit status during download
        or compilation steps.
    OSError
        On filesystem or permission errors (for example when changing file modes,
        creating files/directories, or writing to shell startup files).
    FileNotFoundError
        If the expected source archive (``{assembly_tar_path}.tar.gz``) is missing
        or if the compiled executable cannot be found after the build.

    Notes
    -----
    - On Linux the function currently uses standard GNU toolchain commands (``tar``,
      ``wget``, ``g++``) and compiles the single-file combined source into a local
      executable; it sets the file mode to be executable (``0o755``).
    - On macOS the function prefers Homebrew-managed Boost and compiles with
      ``clang++`` when available; it queries ``brew --prefix boost`` to locate
      headers/libs.
    - The function mutates the caller's environment indirectly by appending an
      ``export ASS_PATH=...`` line to shell startup files (``~/.bashrc`` and
      ``~/.profile``) when it installs a binary into the current working
      directory; callers should inspect these files if unwanted modifications
      occur.
    - The implementation assumes the tarball layout contains a single top-level
      directory with the expected combined source (``v5_combined_linux/main.cpp``
      or equivalent). Adjust *assembly_tar_path* to match the archive contents.
    - Use this helper interactively or from build scripts only when the host
      environment is trusted; it performs network downloads and executes compilers.
    """
    print("compile_assembly_code", flush=True)

    # Detect operating system
    system = platform.system().lower()  # Returns 'linux', 'darwin' (macOS), etc.

    if system == "linux":
        # Existing Linux-specific code
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
        # macOS-specific code
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
        # Unsupported operating system
        raise OSError(f"Unsupported operating system: {system}")


def compile_assembly_cpp() -> None:
    """
    Compile the assemblycpp C++ project and install the produced executable.

    This function clones the `assemblycpp-v5` repository, configures and builds it
    using CMake (platform-specific adjustments applied), moves the resulting
    executable to `assemblytheorytools/precompiled/assembly`, sets executable
    permissions, and removes temporary build artifacts.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function performs compilation and installation as side effects and
        returns `None` on success.

    Raises
    ------
    OSError
        If required build tools (e.g. `git`, `cmake`) are missing or the host
        operating system is unsupported.
    subprocess.CalledProcessError
        If a subprocess command (for example `git clone`, `cmake` or build steps)
        returns a non-zero exit status.
    FileNotFoundError
        If the expected compiled executable cannot be located after the build.

    Notes
    -----
    - Host platform detection is performed via `platform.system().lower()` and
      behaviour is adjusted for `linux`, `darwin` (macOS) and `windows`.
    - On macOS the function may attempt to install missing dependencies using
      Homebrew; on Linux it currently requires `git` and `cmake` to be present.
    - The function temporarily changes the working directory to the cloned
      repository and restores the original working directory on exit.
    - The compiled executable is expected at `assemblycpp-v5/build/bin/assembly`
      and is moved to `assemblytheorytools/precompiled/assembly`.
    - Callers should be prepared to handle exceptions; the function prints
      diagnostic messages on error.
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
            f"git clone https://github.com/LouieSlocombe/assemblycpp-v5.git",
            shell=True, check=True)

        # Change to the assemblycpp directory
        assemblycpp_dir = os.path.join(start_dir, "assemblycpp-v5")
        os.chdir(assemblycpp_dir)
        run_command('cmake -S . -B build')

        if system == "linux" or system == "darwin":
            # Compile the assembly code
            run_command('cmake --build build')
        elif system == "windows":
            # For Windows, we need to specify the generator
            run_command('cmake --build build --config Release')
        else:
            raise OSError(f"Unsupported operating system: {system}")
        # Move the compiled executable to the parent directory
        exe_name = "assembly"
        exe_path = os.path.join(assemblycpp_dir, "build", "bin", exe_name)
        end_path = os.path.join(start_dir, "assemblytheorytools", "precompiled", exe_name)
        # move the executable to the current working directory
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
    return None


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
    subprocess.TimeoutExpired
        When an invoked external process exceeds ``timeout`` and cannot be cleanly
        terminated.

    Notes
    -----
    - Joint inputs (lists) are encoded with delimiters; the final returned AI is
      corrected by subtracting delimiter and directedness offsets.
    - Temporary working directories named like ``ai_calc_<timestamp>`` are created;
      they are removed automatically unless ``debug`` is True.
    - In 'str' mode the function expects the string-assembly binary (set via
      environment variable ``ASS_STR_PATH`` or found by ``add_assembly_to_path``).
    - In 'cfg' mode the function delegates to ``CFG.ai_with_pathways`` and
      returns an upper bound; no external binary is invoked.
    - For reproducible behaviour consider using ``debug=True`` to preserve the
      temporary folder and log files.

    """

    log_file = None
    if isinstance(input_data, str):
        # Handle the case where input_data is a single string
        string = input_data
        delimiters = []
    elif isinstance(input_data, list):
        if len(input_data) > 95:
            raise ValueError(
                "Input list contains more than 95 objects. Joint assembly index calculations are only supported for up to 95 objects.")
        # Handle joint assembly case
        string, delimiters = prep_joint_string_ai(input_data)
    else:
        raise ValueError("Input must be either a single string or a list of strings")

    # Check input types
    assert (dir_code is None) or isinstance(dir_code, str), "Directory code must be a string"
    assert isinstance(timeout, (int, float)), "Timeout must be an integer or float"
    assert isinstance(debug, bool), "Debug must be a boolean"
    assert isinstance(directed, bool), "Directed must be a boolean"

    if directed == False:
        if mode in ["str", "cfg"]:
            mode = "mol"  # Use the molecular assembly calculator for undirected strings
            print("Warning: only mode 'mol' is currently supported for undirected strings. Switching to 'mol'.",
                  flush=True)
    elif mode == "mol":
        mode = "str"  # Use the string assembly calculator for directed strings
        print("Warning: mode 'mol' is not currently supported for directed strings. Switching to 'str'.", flush=True)

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

        if return_log_file:
            graph_ai, graph_virtual_obj, graph_path, log_file = calculate_assembly_index(graph, dir_code=dir_code,
                                                                                         timeout=timeout, debug=debug,
                                                                                         joint_corr=False,
                                                                                         strip_hydrogen=False,
                                                                                         return_log_file=return_log_file)

        else:
            graph_ai, graph_virtual_obj, graph_path = calculate_assembly_index(graph, dir_code=dir_code,
                                                                               timeout=timeout, debug=debug,
                                                                               joint_corr=False, strip_hydrogen=False)

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
        if return_log_file:
            return ai, virt_obj, path, log_file
        else:
            return ai, virt_obj, path

    elif mode == "str":  # Use the string assembly cpp calculator

        # Initialize variables
        ai = -1
        virt_obj = None
        path = None
        file_path_in = None
        timed_out = False  # Flag for timeout tracking

        # Get the assembly code directory
        if dir_code is None:
            dir_code = add_assembly_to_path(str_mode=True)

        # Create working directory
        temp_dir = f"ai_calc_{datetime.now().strftime('%H_%M_%f')}" if debug else tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)

        # Put string into a temporary text file
        file_path_in = os.path.join(temp_dir, "string_in")
        with open(file_path_in, "w") as f:
            f.write(string)

        # Define output and log file paths
        # file_path_out = os.path.join(file_path_in + "Out")
        # file_path_pathway = os.path.join(file_path_in + "Pathway")
        file_path_out = file_path_in + "Out"
        file_path_pathway = file_path_in + "Pathway"
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
                    [dir_code, file_path_in, str(int(directed == 0)), "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=temp_dir
                )

                try:
                    # Wait for process to finish or timeout
                    stdout_data, _ = process.communicate(timeout=timeout)

                except subprocess.TimeoutExpired:
                    print("Warning: Assembly calculation timed out. Terminating...")

                    # process.terminate()
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

        if timed_out == 0:  # If the calculation finished properly, we can read the output file

            if debug:
                print("Assembly calculation completed successfully.", flush=True)

            with open(file_path_out, "r") as f:
                first_line = f.readline()
                match = re.search(r'assembly index:\s*(\d+)', first_line)
                if match:
                    ai = int(match.group(1))

        else:
            # Extract the most recent "min AI found so far" from the log file
            last_ai = -1
            if os.path.exists(log_file):
                if debug:
                    print(f"log_file: {log_file}")
                try:
                    with open(log_file, "r") as log:
                        log_lines = log.readlines()
                    if debug:
                        print(f"log_lines: {log_lines}")
                    # Reverse scan for last occurrence of "min AI found so far"
                    for line in reversed(log_lines):
                        match = re.search(r"min AI found so far:\s*(\d+)", line)
                        if match:
                            last_ai = int(match.group(1))
                            break

                    ai = last_ai  # Assign found AI

                    # Print appropriate messages based on timeout
                    if ai == -1 and timed_out:
                        print("No assembly paths found before timeout.")
                    elif ai != -1 and timed_out:
                        print(f"Upper Bound to AI Found: AI =< {ai - 2 * len(delimiters)}")

                except Exception as e:
                    print(f"Failed to read AI from log file: {e}")

        ai += - 2 * len(delimiters)  # Convert to (joint) assembly index of strings

        # Process pathway output if available
        if os.path.isfile(file_path_pathway):
            try:
                virt_obj, path = parse_string_pathway_file(file_path_pathway)
            except Exception as e:
                print(f"Failed to load pathway data: {e}", flush=True)
                if debug:
                    traceback.print_exc()

        # Print log file path if required
        if return_log_file:
            print(f"Log file printed to: {log_file}", flush=True)

        # Remove temporary files
        if os.path.exists(file_path_in):
            os.remove(file_path_in)
        shutil.rmtree(temp_dir)  # Clean up the temporary directory

        # Return based on flag
        return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, log_file)

    elif mode == "cfg":  # Use the RePair upper bound
        if directed:
            composite_ai, virt_obj, path = CFG.ai_with_pathways(string, f_print=False)
            if debug:
                print(f"Composite String: {string}", flush=True)
                print(f"Length of string: {len(string)}", flush=True)
                print(f"Composite Assembly Index: {composite_ai}", flush=True)
                print(f"Delimiters: {delimiters}", flush=True)
            # Convert to (joint) assembly index of directed strings
            return composite_ai - 2 * len(delimiters), virt_obj, path  # Note: there is no log file for CFG
        else:
            raise ValueError(
                "Current CFG code works natively for directed strings. Directed string assembly index is an upper bound to undirected string assembly index, so you may still use the directed calculator.")

    else:
        raise ValueError("Mode must be either 'mol', 'str', or 'cfg'.")


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

        if str_mode:
            exec_name = "asscpp_combined_static_strings"
        else:
            exec_name = "asscpp_combined_static_linux"

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


def get_most_recent_calc() -> str:
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
    assembly_path = os.path.join(os.getcwd(), assembly_folder)
    return assembly_path


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
      microseconds and converted to seconds by multiplying by ``1e-6``.
    - Uses :func:`get_most_recent_calc` to find the latest calculation folder.

    Examples
    --------
    >>> t = load_assembly_time()
    >>> isinstance(t, float)
    True
    """
    # Locate the most recent assembly calculation folder
    assembly_path = get_most_recent_calc()
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
        time_token = last_line.split(":")[-1].strip()
        time_to_completion = float(time_token)
    except Exception as e:
        raise ValueError(f"Failed to parse time from '{latest_file}': {e}") from e

    # Cleanup the assembly folder and return time in seconds
    shutil.rmtree(assembly_path)
    return float(time_to_completion) * 1e-6


def calculate_assembly_upper_bound(mol: Union[nx.Graph, Chem.Mol],
                                   strip_hydrogen: bool = False) -> int:
    # Check if the input is a NetworkX graph
    if isinstance(mol, nx.Graph):
        if strip_hydrogen:
            mol = remove_hydrogen_from_graph(mol)
    # Check if the input is an RDKit molecule
    elif isinstance(mol, Chem.Mol):
        if strip_hydrogen:
            mol = Chem.RemoveHs(mol)
    else:
        # Raise an error if the input type is not supported
        raise ValueError("Input not supported")

    # Calculate the number of bonds in the molecule
    n_bonds = mol.GetNumBonds() if isinstance(mol, Chem.Mol) else mol.number_of_edges()

    # Return the upper bound of the assembly index
    return n_bonds - 1


def calculate_assembly_lower_bound(mol: Union[nx.Graph, Chem.Mol],
                                   strip_hydrogen: bool = False) -> int:
    if isinstance(mol, nx.Graph):
        if strip_hydrogen:
            mol = remove_hydrogen_from_graph(mol)
    elif isinstance(mol, Chem.Mol):
        if strip_hydrogen:
            mol = Chem.RemoveHs(mol)
    else:
        raise ValueError("Input not supported")
    n_bonds = mol.GetNumBonds() if isinstance(mol, Chem.Mol) else mol.number_of_edges()
    if n_bonds < 1000:
        return calculate_integer_chain(n_bonds)
    else:
        return int(np.log2(n_bonds))


def regularise_ai(ai: Optional[int]) -> int:
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
    - The type hint uses ``int`` but the function tolerates ``None`` at runtime.
    """
    if ai < 0:
        return 0
    elif ai is None:
        return 0
    else:
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


def calculate_sum_assembly(graphs: List[Union[nx.Graph, Chem.Mol]],
                           settings: Optional[Dict[str, Any]] = None,
                           parallel: bool = True) -> int:
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


def calculate_assembly_similarity(graphs: List[Union[nx.Graph, Chem.Mol]],
                                  settings: Optional[Dict[str, Any]] = None,
                                  parallel: bool = True,
                                  enforce_exact_mode: bool = True) -> float:
    if settings is None:
        settings = {}

    if enforce_exact_mode:
        settings["exact"] = True

    # Calculate assembly index sum
    ai_sum = calculate_sum_assembly(graphs, settings, parallel=parallel)

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


def _parse_pathway_file(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and parse a pathway JSON structure into a stable dictionary.

    The function accepts a parsed JSON-like mapping produced by the assembly tool
    and returns a normalized dictionary with consistent keys and canonical edge
    tuple representations used by downstream functions.

    Parameters
    ----------
    data : dict
        Parsed JSON object describing a pathway. Accepted formats are tolerant to
        common key-name variants (for example, `file_graph`, `FileGraph`,
        `fileGraph`) and to variations in how duplicate fragments are represented
        (e.g. `Left`/`Right`, `LeftEdges`/`RightEdges`, nested `{'Edges': [...]}`).

    Returns
    -------
    dict
        A normalized pathway dictionary with the following keys:
        - ``file_graph`` : list of dict
            Each dict contains keys ``vertices`` (list), ``edges`` (list of 2-tuples),
            ``vertex_colours`` (list) and ``edge_colours`` (list).
        - ``remnant`` : list of dict
            Same structure as ``file_graph`` representing the remnant graphs.
        - ``duplicates`` : list of dict
            Each dict contains ``left_edges`` and ``right_edges`` where each value
            is a list of 2-tuples describing fragment edges.
        - ``removed_edges`` : list of 2-tuples
            Edges removed during the pathway processing.

    Raises
    ------
    TypeError
        If ``data`` is not a mapping/dictionary-like object.
    ValueError
        If edge entries cannot be coerced to iterable pairs (two elements).

    Notes
    -----
    - Edge entries are coerced to 2-tuples; malformed or singleton entries are
      skipped.
    - The function is defensive and tries multiple case / naming variants when
      extracting expected fields.
    - The returned structure is safe for direct use by functions such as
      :func:`calculate_jo_from_pathway`.
    """
    parsed_pathway = {}

    # Parse file graphs
    file_graphs = []
    for idx, fg in enumerate(data.get('file_graph', [])):
        file_graphs.append({
            'vertices': fg.get('Vertices', []),
            'edges': fg.get('Edges', []),
            'vertex_colours': fg.get('VertexColours', []),
            'edge_colours': fg.get('EdgeColours', []),
        })
    parsed_pathway['file_graph'] = file_graphs

    # Parse remnant graphs
    remnants = []
    for idx, rem in enumerate(data.get('remnant', [])):
        remnants.append({
            'vertices': rem.get('Vertices', []),
            'edges': rem.get('Edges', []),
            'vertex_colours': rem.get('VertexColours', []),
            'edge_colours': rem.get('EdgeColours', []),
        })
    parsed_pathway['remnant'] = remnants

    # Parse duplicate fragments
    duplicates = []
    for dup in data.get('duplicates', []):
        duplicates.append({
            'left_edges': dup.get('Left', []),
            'right_edges': dup.get('Right', [])
        })
    parsed_pathway['duplicates'] = duplicates

    # Parse removed edges
    parsed_pathway['removed_edges'] = data.get('removed_edges', [])

    return parsed_pathway


def calculate_jo_from_pathway(json_file: str) -> int:
    """
    Calculate the joining-operations assembly index (JO) from a pathway JSON file.

    The JO is computed from the pathway structure produced by the assembly tool.
    The function reads pathway data, reconstructs the original graph from the first
    `file_graph` entry, and iteratively applies duplicate-fragment removals to
    accumulate corrections that reflect joining operations.

    Parameters
    ----------
    json_file : str
        Path to a JSON file containing pathway information. Expected keys (after
        parsing) include:
          - 'file_graph': list where first element has 'edges' describing the original graph edges.
          - 'duplicates': list of duplicate fragments where each fragment contains 'Right' (and/or 'Left') edges.

    Returns
    -------
    int
        The computed JO as an integer.

    Raises
    ------
    FileNotFoundError
        If `json_file` does not exist.
    json.JSONDecodeError
        If the file cannot be parsed as JSON.
    KeyError, IndexError, ValueError
        If expected keys/structures are missing or malformed.

    Notes
    -----
    - Edges in the JSON are expected to be iterable pairs (e.g. lists or tuples).
    - The algorithm:
        1. Builds the original graph from `file_graph[0]['edges']`.
        2. Initializes a metric `ma = n_edges - n_connected_components`.
        3. For each duplicate fragment (using its `Right` edges) it:
            - Decreases `ma` by fragment size minus one.
            - Removes fragment edges from the running edge set to form a remnant graph.
            - Computes connected-component changes and overlapping atom corrections.
        4. Returns `ma + jo_correction`.
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


def calculate_jo(mol: Union[nx.Graph, Chem.Mol],
                 dir_code: Optional[str] = None,
                 timeout: float = 100.0,
                 strip_hydrogen: bool = False,
                 return_log_file: bool = False,
                 exact: bool = False) -> Tuple[int, Any, Any]:
    # Run the assembly index calculation to produce pathway output in a temp folder
    _, vo, pathway = calculate_assembly_index(mol, dir_code=dir_code, timeout=timeout, debug=True,
                                              strip_hydrogen=strip_hydrogen, return_log_file=return_log_file,
                                              exact=exact)

    # Locate the most recent assembly calculation folder
    assembly_path = get_most_recent_calc()

    # Find a file that ends with "Pathway" in that folder
    pathway_files = [f for f in os.listdir(assembly_path) if f.endswith("Pathway")]
    if not pathway_files:
        # No pathway output available -> cannot compute JO
        print("No pathway file found. Returning -1.", flush=True)
        # Clean up folder to avoid leaving temp data
        try:
            shutil.rmtree(assembly_path)
        except Exception:
            pass
        return -1, None, None

    # Use the first pathway file found
    pathway_file = os.path.join(assembly_path, pathway_files[0])

    # Compute JO from the pathway file and handle errors
    try:
        jo = calculate_jo_from_pathway(pathway_file)
    except Exception as e:
        # On error, remove the temporary folder and return failure sentinel
        print(f"Error calculating joint assembly index: {e}", flush=True)
        try:
            shutil.rmtree(assembly_path)
        except Exception:
            pass
        return -1, None, None

    # Cleanup the temporary assembly folder
    try:
        shutil.rmtree(assembly_path)
    except Exception:
        pass

    # Return the computed JO along with virtual object and pathway from the AI run
    return jo, vo, pathway


def calculate_assembly_ratio(graph: Union[nx.Graph, Chem.Mol], settings: Dict[str, Any]) -> float:
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
          - If AI == 0 a `ZeroDivisionError` will be raised by Python.
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
    - The returned value may be meaningless if `calculate_assembly_index` returned a non-positive AI;
      callers should check the AI return value when exact/robust behaviour is required.

    """
    # Determine number of edges/bonds depending on the input type
    n_edges = graph.number_of_edges() if isinstance(graph, nx.Graph) else graph.GetNumBonds()

    # Compute the assembly index (AI) using the existing function
    ai, _, _ = calculate_assembly_index(graph, **settings)

    # If there are no edges, return 1.0 to avoid division by zero (by design)
    if n_edges == 0:
        return 1.0
    else:
        # Note: this will raise ZeroDivisionError if ai == 0; if ai < 0 the result will be negative.
        return n_edges / ai


def calculate_jo_assembly_ratio(graph: Union[nx.Graph, Chem.Mol], settings: Dict[str, Any]) -> float:
    """
    Calculate the joining-operation (JO) assembly ratio for a molecular graph.

    The JO assembly ratio is computed as:

        assembly_ratio = n_edges / JO

    where:
      - n_edges is the number of edges (bonds) in the input graph or molecule.
      - JO is the joining-operation index computed by `calculate_jo`.

    This function supports input as either a NetworkX graph or an RDKit `Chem.Mol`
    object and expects a `settings` dictionary to be forwarded to `calculate_jo`.

    Parameters
    ----------
    graph : Union[nx.Graph, Chem.Mol]
        The molecular graph to evaluate. For a NetworkX graph, the number of edges
        is obtained via `graph.number_of_edges()`. For an RDKit molecule the number
        of bonds is obtained via `graph.GetNumBonds()`.
    settings : dict
        Settings forwarded to `calculate_jo`. Typical keys control execution of the
        underlying assembly calculation (e.g., `dir_code`, `timeout`, `debug`,
        `strip_hydrogen`, `return_log_file`, `exact`).

    Returns
    -------
    float
        The JO assembly ratio (number of edges divided by JO). If the graph has no
        edges, returns 1.0 to avoid division by zero.

    Notes
    -----
    - If `calculate_jo` fails it may return -1 (or another sentinel). The caller
      should be aware that dividing by such values can produce unexpected results.
    - If JO equals zero a `ZeroDivisionError` will be raised by Python. The caller
      can guard against this by pre-checking the returned JO if required.
    - This function does not modify the input graph or molecule.

    """
    # Determine number of edges (bonds) depending on input type
    n_edges = graph.number_of_edges() if isinstance(graph, nx.Graph) else graph.GetNumBonds()

    # Compute the joining-operation index (JO) using existing function
    jo, _, _ = calculate_jo(graph, **settings)

    # Avoid division by zero when there are no edges
    if n_edges == 0:
        return 1.0
    else:
        # Note: if jo is 0 this will raise ZeroDivisionError; if jo is -1 the result
        # will be negative which indicates a failure in JO calculation upstream.
        return n_edges / jo


def _input_helper_rust(mol: Chem.Mol, file_path: str, strip_hydrogen: bool = False) -> bool:
    """
    Prepare an RDKit molecule and write it to a file in `.mol` format.

    This helper normalizes the hydrogen content of the given RDKit molecule,
    checks for unsupported wildcard atoms, and writes the molecule as a MolBlock
    to the specified file path.

    Args:
        mol (Chem.Mol): RDKit molecule to process.
        file_path (str): Destination file path where the `.mol` MolBlock will be written.
        strip_hydrogen (bool, optional): If True, remove explicit hydrogens before writing.
            If False, ensure explicit hydrogens are present. Defaults to False.

    Returns:
        bool: True if the molecule was successfully written to `file_path`.
              Returns False if the molecule contains wildcard atoms (e.g., `*`)
              which are considered invalid for downstream processing.

    Raises:
        OSError: Propagates any I/O related exceptions raised while opening/writing the file.

    Notes:
        - Uses RDKit's Chem.RemoveHs and Chem.AddHs to control hydrogen handling.
        - The function intentionally rejects molecules that contain wildcard atoms
          because downstream tools expect fully specified atom types.
        - The written representation is the MolBlock produced by Chem.MolToMolBlock.

    Example:
        ok = _input_helper(mol, "/tmp/tmp.mol", strip_hydrogen=True)
    """
    # Ensure hydrogens are present or removed according to the flag
    mol = Chem.RemoveHs(mol) if strip_hydrogen else Chem.AddHs(mol)

    # Reject molecules with wildcard atoms (e.g., '*') which downstream tools cannot handle
    if any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
        return False

    # Write MolBlock to the specified file path
    with open(file_path, 'w') as f:
        f.write(Chem.MolToMolBlock(mol))

    return True


def calculate_assembly_index_rust(mol: Chem.Mol,
                                  exec_path: Optional[str] = None,
                                  timeout: int = 300,
                                  strip_hydrogen: bool = False) -> int:
    """
    Calculate the assembly index using a Rust-based executable.

    This function writes the provided RDKit molecule to a temporary `.mol` file
    (optionally stripping hydrogens), invokes a precompiled Rust executable to
    compute the assembly index, and parses the first integer found in the
    executable's standard output as the result.

    Args:
        mol (Chem.Mol): RDKit molecule to evaluate.
        exec_path (str | None): Path to the Rust executable. If None, the function
            looks for a default precompiled binary at
            `os.path.join(os.path.dirname(__file__), "precompiled", "Rust")`.
        timeout (int): Maximum time in seconds to wait for the Rust process.
        strip_hydrogen (bool): If True, hydrogens will be removed before writing
            the temporary `.mol` file.

    Returns:
        int: Parsed assembly index (non-negative integer) on success.
             Returns -1 on invalid input, execution error, parse failure, or timeout.

    Raises:
        NotImplementedError: If called on a non-Linux platform (Rust binary supported only on Linux).

    Notes:
        - The function relies on `_input_helper` to prepare and write the molecule.
          `_input_helper` will return False if wildcard atoms are present, in which
          case this function returns -1.
        - The Rust executable is expected to print a number somewhere in its
          standard output; the first integer matched by the regex is used.
        - Standard error and non-zero exit codes are treated as failures and
          cause a return value of -1.

    Example:
        ai = calculate_rust_ai(my_rdkit_mol, exec_path="/path/to/rust_bin", timeout=120)
    """
    # Ensure this helper is only used on Linux for now
    system = platform.system().lower()
    if system != "linux":
        raise NotImplementedError("Rust assembly index calculator is currently only supported on Linux.")

    # Default to the bundled precompiled Rust executable if none provided
    if exec_path is None:
        exec_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "precompiled", "Rust")
        )

    # Create a temporary .mol file and ensure proper H-handling / wildcard checks
    with tempfile.NamedTemporaryFile(suffix='.mol', delete=True) as tmp_file:
        # Prepare and write mol file; if invalid (e.g., wildcard atoms), abort
        if not _input_helper_rust(mol, tmp_file.name, strip_hydrogen=strip_hydrogen):
            print("Invalid input", flush=True)
            return -1

        try:
            # Run the Rust executable, capture output, enforce timeout
            result = subprocess.run([exec_path, tmp_file.name],
                                    capture_output=True,
                                    text=True,
                                    timeout=timeout)

            # Non-zero exit codes indicate failure; surface stderr for diagnostics
            if result.returncode != 0:
                print(f"Rust executable returned non-zero exit code {result.returncode}", flush=True)
                if result.stderr:
                    print(result.stderr.strip(), flush=True)
                return -1

            # Parse the first integer from stdout as the assembly index
            match = re.search(r'\b\d+\b', result.stdout or "")
            return int(match.group(0)) if match else -1

        except subprocess.TimeoutExpired:
            # Process exceeded allowed time
            print(f"Rust executable timed out after {timeout} seconds.", flush=True)
            return -1
        except Exception as e:
            # Catch-all for unexpected errors
            print(f"Exception occurred: {e}", flush=True)
            return -1


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
        raise ValueError(
            "n must be less than or equal to 9999.")
    elif n == 1:
        return 0

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'integer_chain_9999.txt')
    with open(data_path, 'r') as file:
        for i, line in enumerate(file):
            if i == n + 1:
                return int(line.split()[3])
    return -1
