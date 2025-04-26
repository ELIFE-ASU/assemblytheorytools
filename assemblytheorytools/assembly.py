import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
import warnings
from datetime import datetime
from typing import Union, List

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import CFG
from .construction import parse_pathway_file
from .pathway import (get_pathway_to_graph,
                      get_pathway_to_mol,
                      get_pathway_to_inchi)
from .tools_graph import (write_ass_graph_file,
                          remove_hydrogen_from_graph,
                          nx_to_mol)
from .tools_mol import (write_v2k_mol_file,
                        combine_mols,
                        safe_standardize_mol)
from .tools_string import (prep_joint_string_ai,
                           get_dir_str_molecule,
                           get_undir_str_molecule)


def load_assembly_output(file_path):
    """
    Load the assembly output from a file.

    Args:
        file_path (str): Path to the file containing the assembly output.

    Returns:
        int: The assembly index extracted from the file.
    """
    with open(file_path, "r") as f:
        return next(int(line.split(":")[-1].strip().strip('\n')) for line in f if "assembly index" in line)


def run_command(command,
                output_file="output.out",
                error_file="error.err",
                timeout=10000.0,
                verbose=False):
    """
    Run a command in the subprocess with specified output and error files, and a timeout.

    Args:
        command (list): The command to run as a list of arguments.
        output_file (str): The file to write standard output to. Defaults to "output.out".
        error_file (str): The file to write standard error to. Defaults to "error.err".
        timeout (float): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        verbose (bool): If True, print the command execution result. Defaults to False.

    Returns:
        bool: True if the command executed successfully, False otherwise.
    """
    if command is None:
        raise ValueError("Command must be provided")

    try:
        # Open the output and error files using with statement
        with open(output_file, "w") as out, open(error_file, "w") as err:
            # Run the command with a timeout
            result = subprocess.run(command[0].split() + command[1].split(),
                                    stdout=out,
                                    stderr=err,
                                    timeout=timeout)
            if verbose:
                print("Command executed successfully:", result, flush=True)
                # Write to output file
                out.write(f"Command executed successfully: {result}")
        return True
    # If the command times out, catch the TimeoutExpired exception
    except subprocess.TimeoutExpired:
        print(f"Command timed out and was terminated, ran for {timeout} seconds.", flush=True)
        # Write to error file
        with open(error_file, "w") as err:
            err.write(f"Command timed out and was terminated, ran for {timeout} seconds.")
        return False
    # Catch all other exceptions
    except Exception as e:
        print(f"Failed to run command {command}", flush=True)
        print(e, flush=True)
        # Write to error file
        with open(error_file, "w") as err:
            err.write(f"Failed to run command {command}")
            err.write(str(e))
        return False


def run_command_simple(command):
    """
    Run a simple command in the subprocess.

    Args:
        command (str): The command to run as a string.

    Returns:
        bytes: The standard output of the command.
    """
    if command is None:
        raise ValueError("Command must be provided")

    result = subprocess.run(command.split())
    return result.stdout


def joint_correction(mol, ass_index):
    """
    Correct the assembly index based on the number of components in the molecule or chemical system.

    Args: mol (Union[nx.Graph, Chem.Mol, str]): The molecule or chemical system, which can be a NetworkX graph, an RDKit
    object, or a file path to a .mol file. ass_index (int): The initial assembly index.

    Returns:
        int: The corrected assembly index.
    """
    if isinstance(mol, nx.Graph):
        # Get the number of connected components in the graph
        num_components = nx.number_connected_components(mol)
    elif isinstance(mol, Chem.Mol):
        # Get the number of components in the RDKit molecular object
        num_components = len(Chem.rdmolops.GetMolFrags(mol=Chem.Mol(mol)))
    elif ".mol" in mol:
        # Get the number of components in the molecular object from the .mol file
        num_components = len(Chem.rdmolops.GetMolFrags(mol=Chem.MolFromMolFile(mol)))
    else:
        num_components = None
        ValueError("Input not supported")
    # Return the number of components minus 1
    correction = max(0, num_components - 1)
    if correction < 0:
        correction = 0
    return ass_index - correction


def calculate_assembly_index(mol,
                             dir_code=None,
                             timeout=100.0,
                             debug=False,
                             joint_corr=True,
                             strip_hydrogen=False,
                             return_log_file=False):
    """
    Calculate the assembly index for a given molecule.

    WARNING: It is the responsibility of the user to ensure the mol file has H or not!

    Args:
        mol (Union[nx.Graph, Chem.Mol, str]): The molecule (NetworkX graph, RDKit molecule, or .mol file path).
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): Maximum time in seconds before termination. Defaults to 100.0.
        debug (bool, optional): If True, creates a directory with a timestamp for debugging. Defaults to False.
        joint_corr (bool, optional): If True, corrects disjointed graphs. Defaults to True.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms before calculation. Defaults to False.
        return_log_file (bool, optional): If True, includes log file in return. Defaults to False.

    Returns:
        - (ai, virt_obj, path) if return_log_file=False
        - (ai, virt_obj, path, log_file) if return_log_file=True
    """

    # Initialize variables
    ai = -1
    virt_obj = None
    path = None
    file_path_in = None
    timed_out = False  # Flag for timeout tracking

    # Check if input is a string and not a .mol file
    if isinstance(mol, str) and not mol.endswith(".mol"):
        ai, virt_obj, path = CFG.ai_with_pathways(mol, f_print=False)
        return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, None)

    else:
        # Get the assembly code directory
        if dir_code is None:
            dir_code = add_assembly_to_path()

        # Create working directory
        temp_dir = f"ai_calc_{datetime.now().strftime('%H_%M_%f')}" if debug else tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)

        # Input is a graph
        if isinstance(mol, nx.Graph):
            if strip_hydrogen:
                mol = remove_hydrogen_from_graph(mol)
            file_path_in = os.path.join(temp_dir, "graph_in")
            write_ass_graph_file(mol, file_name=file_path_in)
        # Input is an RDKit mol
        elif isinstance(mol, Chem.Mol):
            mol = safe_standardize_mol(mol, add_hydrogens=True)
            if strip_hydrogen:
                mol = Chem.RemoveHs(mol)
            mol_file = os.path.join(temp_dir, "tmp.mol")
            write_v2k_mol_file(mol, mol_file)
            file_path_in = os.path.splitext(mol_file)[0]
        # Input is a mol file
        elif isinstance(mol, str) and mol.endswith(".mol"):
            if strip_hydrogen:
                mol_ob = Chem.MolFromMolFile(mol)
                mol_ob = safe_standardize_mol(mol_ob, add_hydrogens=True)
                mol_ob = Chem.RemoveHs(mol_ob)
                mol = os.path.join(temp_dir, "tmp.mol")
                Chem.MolToMolFile(mol_ob, mol)
            else:
                shutil.copy(mol, os.path.join(temp_dir, "tmp.mol"))
                mol = os.path.join(temp_dir, "tmp.mol")
            file_path_in = os.path.splitext(mol)[0]
        else:
            raise ValueError("Input not supported")

        # Define output and log file paths
        file_path_out = os.path.join(file_path_in + "Out")
        file_path_pathway = os.path.join(file_path_in + "Pathway")
        log_file = os.path.join(temp_dir, "assembly_output.log")

        # Run the assembly code and log output
        try:
            with open(log_file, "w") as log:
                start_time = time.time()
                process = subprocess.Popen(
                    [dir_code, file_path_in],
                    stdout=log,
                    stderr=log
                )
                while process.poll() is None:
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        print("Warning: Assembly calculation timed out. Terminating...", flush=True)
                        process.send_signal(
                            signal.SIGINT)  # This simulates Ctrl+C, getting the right output from assemblyCpp
                        process.wait()
                        time.sleep(0.5)
                        if process.poll() is None:
                            process.kill()
                        timed_out = True  # Mark timeout
                        break

        except Exception as e:
            print(f"Error: {e}", flush=True)

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
                if isinstance(mol, nx.Graph):
                    virt_obj = get_pathway_to_graph(file_path_pathway)
                    path = parse_pathway_file(file_path_pathway, vo_type='graph', debug=debug)
                elif isinstance(mol, Chem.Mol):
                    virt_obj = get_pathway_to_mol(file_path_pathway)
                    path = parse_pathway_file(file_path_pathway, vo_type='smiles', debug=debug)
                elif ".mol" in mol:
                    virt_obj = get_pathway_to_inchi(file_path_pathway)
                    path = parse_pathway_file(file_path_pathway, vo_type='smiles', debug=debug)
                else:
                    virt_obj = None
                    path = (None, None)
                    raise ValueError("Input not supported")
            except Exception as e:
                print(f"Failed to load pathway data: {e}", flush=True)

        # Apply joint correction if necessary
        if joint_corr:
            ai = joint_correction(mol, ai)

        # Print log file path if required
        if return_log_file:
            print(f"Log file printed to: {log_file}", flush=True)

        # Return based on flag
        return (ai, virt_obj, path) if not return_log_file else (ai, virt_obj, path, log_file)


def calculate_assembly_semi_metric(graph1,
                                   graph2,
                                   dir_code=None,
                                   timeout=100.0,
                                   debug=False,
                                   strip_hydrogen=False,
                                   normalise=False):
    """
    Calculate the assembly semi-metric distance between a pair of molecular graphs.

    Args:
        graph1 (nx.Graph): First input molecule as a NetworkX graph.
        graph2 (nx.Graph): Second input molecule as a NetworkX graph.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before calculation. Defaults to False.
        normalise (bool, optional): If True, normalizes the semi-metric distance. Defaults to False.

    Returns:
        int: The difference between the joint assembly index and the sum of the assembly indices of the disconnected subgraphs.
    """
    # Make input type checks
    assert isinstance(graph1, nx.Graph), "Input must be a NetworkX graph"
    assert isinstance(graph2, nx.Graph), "Input must be a NetworkX graph"
    assert (dir_code is None) or isinstance(dir_code, str), "Directory code must be a string"
    assert isinstance(timeout, (int, float)), "Timeout must be an integer or float"
    assert isinstance(debug, bool), "Debug must be a boolean"
    assert isinstance(strip_hydrogen, bool), "Strip hydrogen must be a boolean"
    assert isinstance(normalise, bool), "Normalise must be a boolean"

    # Ensure the inputs are connected graphs
    assert nx.is_connected(graph1), "Input graph must be connected"
    assert nx.is_connected(graph2), "Input graph must be connected"

    # Check if the inputs are isomorphic, in which case the semi-metric distance is 0 and the user may not intend to compare these mols
    if nx.is_isomorphic(graph1, graph2):
        warnings.warn("Input graphs are isomorphic.")
        return 0

    # Combine the graphs into a single molecular object with 2 disjoint components
    mols = [nx_to_mol(graph1), nx_to_mol(graph2)]
    combined_mol = combine_mols(mols)

    # Calculate the joint assembly index
    jai, _, _ = calculate_assembly_index(combined_mol,
                                         dir_code=dir_code,
                                         timeout=timeout,
                                         debug=debug,
                                         strip_hydrogen=strip_hydrogen)
    if debug:
        print(f"Joint Assembly Index: {jai}", flush=True)

    # Calculate the assembly index for each subgraph
    result = 0
    for subgraph in [graph1, graph2]:
        ai, _, _ = calculate_assembly_index(subgraph,
                                            dir_code=dir_code,
                                            timeout=timeout,
                                            debug=debug,
                                            strip_hydrogen=strip_hydrogen)
        if debug:
            print(f"Assembly Index: {ai}", flush=True)
        result += ai

    # Calculate the semi-metric distance
    semi_metric = 2 * jai - result
    if normalise:
        return semi_metric / result

    return semi_metric


def add_to_bashrc(export_line, file=".bashrc"):
    """
    Append an export line to the specified bash configuration file.

    Args:
        export_line (str): The export line to add to the bash configuration file.
        file (str, optional): The name of the bash configuration file. Defaults to ".bashrc".

    """
    # Get the path to the file in the user's home directory
    file_path = os.path.expanduser(f"~/{file}")

    # Open the .bashrc file in append mode and write the export line to it
    with open(file_path, "a") as f:
        f.write(f"\nexport {export_line}\n")


def compile_assembly_code(assembly_tar_path="assemblycpp-main", boost_version="1_86_0", exe_name="asscpp_v5"):
    """
    Compile the assembly code, adapting for Linux or macOS (UNIX-based systems).

    Args:
        assembly_tar_path (str): Path to the assembly .tar.gz file. Default is "assemblycpp-main".
        boost_version (str): Boost library version. Default is "1_86_0".
        exe_name (str): Name of the compiled executable. Default is "asscpp_v5".
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
        run_command_simple(f"{uncompress} {assembly_tar_path}.tar.gz")

        # Get the Boost library
        subprocess.run(
            f"wget 'https://archives.boost.io/release/{boost_version.replace('_', '.')}/source/{boost_code}.tar.gz'",
            shell=True, check=True)

        # Unzip the Boost code
        run_command_simple(f"{uncompress} {boost_code}.tar.gz")

        # Compile the assembly code
        t0 = time.time()
        run_command_simple(f"g++ {assembly_tar_path}/v5_combined_linux/main.cpp -O3 -o {exe_dir} -I{boost_code}/")
        t1 = time.time()
        print(f"Compilation time: {t1 - t0:.2f} seconds", flush=True)

        # Set the permissions to allow execution
        os.chmod(exe_dir, 0o755)

        # Remove unnecessary files and folders
        run_command_simple(f"{remove} {boost_code}.tar.gz")
        run_command_simple(f"{remove} {boost_code}/")
        run_command_simple(f"{remove} {assembly_tar_path}/")

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


def calculate_string_assembly_index(input_data: Union[str, List[str]],
                                    dir_code=None,
                                    timeout=100.0,
                                    debug=False,
                                    directed=False,
                                    mode="str",
                                    return_log_file=False):
    """
    Calculate the assembly index of a string or a set of strings. 
    This function uses the molecular assembly calculator by constructing molecular graphs which correspond to the
    strings.

    Args:
        input_data (Union[str, List[str]]): The input data, which can be a single string or a list of strings.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        directed (bool, optional): If True, treat strings as directed. Defaults to False, treating strings as
        undirected.
        mode ("mol"/"str"/"cfg",optional): "mol" uses the molecular assembly calculator, "str" uses the string assembly
        calculator, "cfg" uses the RePair upper bound.
    """
    log_file = None
    if isinstance(input_data, str):
        # Handle the case where input_data is a single string
        string = input_data
        delimiters = []
    elif isinstance(input_data, list):
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

        if return_log_file:
            graph_ai, graph_virtual_obj, graph_path, log_file = calculate_assembly_index(graph,
                                                                                         dir_code=dir_code,
                                                                                         timeout=timeout,
                                                                                         debug=debug,
                                                                                         joint_corr=False,
                                                                                         strip_hydrogen=False,
                                                                                         return_log_file=return_log_file)
        else:
            graph_ai, graph_virtual_obj, graph_path = calculate_assembly_index(graph,
                                                                               dir_code=dir_code,
                                                                               timeout=timeout,
                                                                               debug=debug,
                                                                               joint_corr=False,
                                                                               strip_hydrogen=False)

        # Correct for joint assembly and directed encoding
        ai = graph_ai - 2 * len(delimiters)
        if directed:
            ai = ai - len(set(string))

        if debug:
            print(f"Assembly Index: {ai}", flush=True)

        # Convert to (joint) assembly index of directed strings. Note: Virt obj and Path parsing still needs to be added
        if return_log_file:
            return ai, None, None, log_file
        else:
            return ai, None, None

    elif mode == "str":  # Use the string assembly cpp calculator
        # raise NotImplementedError("String assembly cpp calculator not yet supported.")

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
        file_path_out = os.path.join(file_path_in + "Out")
        file_path_pathway = os.path.join(file_path_in + "Pathway")
        log_file = os.path.join(temp_dir, "assembly_output.log")

        # Run the assembly code and log output
        try:
            with open(log_file, "w") as log:
                # Start the process
                process = subprocess.Popen(
                    [dir_code, file_path_in, str(int(directed == 0)), "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

                start_time = time.time()

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

                    timed_out = True

                # Write whatever output we got to the log
                if stdout_data:
                    log.write(stdout_data.decode(errors="replace"))
                    log.flush()

        except Exception as e:
            print(f"Error: {e}")

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

                ai += - 2 * len(delimiters)  # Convert to (joint) assembly index of strings

            except Exception as e:
                print(f"Failed to read AI from log file: {e}")

            # Print log file path if required
            if return_log_file:
                print(f"Log file printed to: {log_file}", flush=True)

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
            ValueError(
                "Current CFG code works natively for directed strings. Directed string assembly index is an upper bound to undirected string assembly index, so you may still use the directed calculator.")

    else:
        ValueError("Mode must be either 'mol', 'str', or 'cfg'.")


def assembly_dry_run(mol, temp_dir=None, strip_hydrogen=False):
    """
    Perform a dry run of the assembly process for a given molecule.

    Args:
        mol (Union[nx.Graph, Chem.Mol, str]): The molecule, which can be a NetworkX graph, an RDKit molecule, or a file
         path to a .mol file.
        temp_dir (str, optional): The temporary directory to use for file operations. Defaults to the current working
         directory.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before processing. Defaults
         to False.

    Raises:
        ValueError: If the input molecule type is not supported.
    """
    if temp_dir is None:
        temp_dir = os.getcwd()

    # Assuming a NetworkX graph
    if isinstance(mol, nx.Graph):
        # Check if we need to strip hydrogen
        if strip_hydrogen:
            mol = remove_hydrogen_from_graph(mol)
        # Make the input file path
        file_path_in = os.path.join(temp_dir, "graph_in")
        # Write the input graph file
        write_ass_graph_file(mol, file_name=file_path_in)
    # Check if the input is an RDKit molecule
    elif isinstance(mol, Chem.Mol):
        # Check if we need to strip hydrogen
        if strip_hydrogen:
            mol = Chem.RemoveHs(mol)

        # Write the mol file
        mol_file = os.path.join(temp_dir, "tmp.mol")
        # Write the input mol file
        write_v2k_mol_file(mol, mol_file)
    # Check if the input is a file path to a .mol file
    elif isinstance(mol, str) and mol.endswith(".mol"):
        # WARNING it is the responsibility of the user to ensure the mol file has H or not!
        if strip_hydrogen:
            # Load the mol file
            mol_ob = Chem.MolFromMolFile(mol, sanitize=False, removeHs=True)
            # Make a temp dir to prevent overwriting
            mol = os.path.join(temp_dir, "tmp.mol")
            # Make a new mol file
            Chem.MolToMolFile(mol_ob, mol)
        else:
            # Copy the mol file into the temp directory
            shutil.copy(mol, os.path.join(temp_dir, "tmp.mol"))
    else:
        raise ValueError("Input not supported")


def add_assembly_to_path(str_mode=False):
    """
    Adds the path to a precompiled assemblyCpp executable to the environment variable `ASS_PATH` or `ASS_STR_PATH` depending upon application.

    Args:
        str_mode (bool): If True, sets the path for the string mode executable. Defaults to False.

    Raises:
        NotImplementedError: If the operating system is MacOS or Windows.

    Returns:
        str: The path to the precompiled assemblyCpp executable.
    """
    if str_mode:
        key = "ASS_STR_PATH"
        exec_name = "asscpp_combined_static_strings"
    else:
        key = "ASS_PATH"
        exec_name = "asscpp_combined_static_linux"

    if not os.environ.get(key):
        full_att_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "precompiled", exec_name)
        )
        if platform.system() == "Linux":
            os.environ[key] = full_att_path
        else:
            raise NotImplementedError("Pre-compiled Assembly not implemented for MacOS or Windows.")

    return os.environ.get(key)


def load_assembly_time():
    """
    Load the assembly time from the most recent output file in the most recent "ai_calc_" folder.

    This function performs the following steps:
    1. Identifies the most recent folder starting with "ai_calc_" in the current working directory.
    2. Identifies the most recent file ending with "Out" in the identified folder.
    3. Reads the time to completion from the last line of the identified file.
    4. Removes the identified folder.

    Returns:
        float: The time to completion extracted from the file.
    """
    # Get the most recent folder starting with "ai_calc_"
    assembly_folders = [folder for folder in os.listdir(os.getcwd()) if folder.startswith("ai_calc_")]
    assembly_folder = max(assembly_folders, key=os.path.getctime)
    assembly_path = os.path.join(os.getcwd(), assembly_folder)

    # Get the most recent file ending with "Out"
    assembly_files = [file for file in os.listdir(assembly_path) if file.endswith("Out")]
    latest_file = os.path.join(assembly_path, assembly_files[-1])

    # Read the time to completion from the last line of the file
    with open(latest_file, "r") as f:
        time_to_completion = f.readlines()[-1].split(":")[-1].strip()

    # Remove the assembly folder
    shutil.rmtree(assembly_path)
    return float(time_to_completion)
