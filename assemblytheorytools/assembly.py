import shutil
import os
import subprocess
import tempfile
from datetime import datetime

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import CFG
from .graphtools import write_ass_graph_file, remove_hydrogen_from_graph, get_disconnected_subgraphs
from .moltools import write_v2k_mol_file
from .pathway import get_pathway_to_graph, get_pathway_to_mol, get_pathway_to_inchi


def load_assembly_output(file_path):
    """
    Load the assembly output from a file.

    Args:
        file_path (str): Path to the file containing the assembly output.

    Returns:
        int: The assembly index extracted from the file.
    """
    with open(file_path, "r") as f:
        return next(int(line.split(":")[-1]) for line in f if "assembly index" in line)


def run_command(command, output_file="output.out", error_file="error.err", timeout=10000.0, verbose=False):
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
    try:
        # Open the output and error files using with statement
        with open(output_file, "w") as out, open(error_file, "w") as err:
            # Run the command with a timeout
            result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout)
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


def joint_correction(mol, ass_index):
    """
    Correct the assembly index based on the number of fragments in the molecule.

    Args:
        mol (Union[nx.Graph, Chem.Mol, str]): The molecule, which can be a NetworkX graph, an RDKit molecule, or a file path to a .mol file.
        ass_index (int): The initial assembly index.

    Returns:
        int: The corrected assembly index.
    """
    if isinstance(mol, nx.Graph):
        # Get the number of connected components in the graph
        num_fragments = nx.number_connected_components(mol)
    elif isinstance(mol, Chem.Mol):
        # Get the number of fragments in the RDKit molecule
        num_fragments = len(Chem.rdmolops.GetMolFrags(mol=Chem.Mol(mol)))
    elif ".mol" in mol:
        # Get the number of fragments in the molecule from the .mol file
        num_fragments = len(Chem.rdmolops.GetMolFrags(mol=Chem.MolFromMolFile(mol)))
    else:
        num_fragments = None
        ValueError("Input not supported")
    # Return the number of fragments minus 1
    correction = max(0, num_fragments - 1)
    if correction < 0:
        correction = 0
    return ass_index - correction


def calculate_assembly_index(mol,
                             dir_code=None,
                             timeout=100.0,
                             debug=False,
                             joint_corr=True,
                             strip_hydrogen=False):
    """
    Calculate the assembly index for a given molecule.

    Args:
        mol (Union[nx.Graph, Chem.Mol, str]): The molecule, which can be a NetworkX graph, an RDKit molecule, or a file path to a .mol file.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        joint_corr (bool, optional): If True, corrects the joint assembly calculation to account for disjointed graphs. Defaults to True.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before calculation. Defaults to False.

    Returns:
        tuple: A tuple containing the corrected assembly index (int) and the pathway (varies based on input type).
    """
    file_path_in = None
    if isinstance(mol, str) and not mol.endswith(".mol"):
        ai, virt_obj, path = CFG.ai_with_pathways(mol, f_print=False)
        return ai, path
    else:
        # Get the assembly code directory
        if dir_code is None:
            dir_code = os.environ.get("ASS_PATH")

        # Make the directory
        if debug:
            # Define the directory name with the timestamp
            temp_dir = f"ai_calc_{datetime.now().strftime('%H_%M_%f')}"
            os.makedirs(temp_dir)
        else:
            temp_dir = tempfile.mkdtemp()

        # Check if the input is a rdkit mol
        if isinstance(mol, nx.Graph):
            # Check if we need to strip hydrogen
            if strip_hydrogen:
                mol = remove_hydrogen_from_graph(mol)
            # Make the in file
            file_path_in = os.path.join(temp_dir, f"graph_in")
            # Write the input graph file
            write_ass_graph_file(mol, file_name=file_path_in)

        elif isinstance(mol, Chem.Mol):
            # Check if we need to strip hydrogen
            if strip_hydrogen:
                mol = Chem.RemoveHs(mol)

            # Write the mol file
            mol_file = os.path.join(temp_dir, f"tmp.mol")
            # Write the input mol file
            write_v2k_mol_file(mol, mol_file)
            # Get the infile
            file_path_in = os.path.splitext(mol_file)[0]

        elif mol.endswith(".mol"):
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
                mol = os.path.join(temp_dir, "tmp.mol")

            # Get the infile
            file_path_in = os.path.splitext(mol)[0]
        else:
            ValueError("Input not supported")
        # Get the output file
        file_path_out = os.path.join(file_path_in + "Out")
        file_path_pathway = os.path.join(file_path_in + "Pathway")

        # Run the assembly code
        outcome = run_command([dir_code, file_path_in],
                              output_file=f"{file_path_in}.out",
                              error_file=f"{file_path_in}.err",
                              timeout=timeout)

        # Get the output
        if not outcome:
            # If the assembly code failed return -1
            return -1, None
        else:
            try:
                # Load the assembly output
                value = load_assembly_output(file_path_out)
                # Check the pathway file exits
                if os.path.isfile(file_path_pathway):
                    if isinstance(mol, nx.Graph):
                        path = get_pathway_to_graph(file_path_pathway)
                    elif isinstance(mol, Chem.Mol):
                        # Load the pathway data
                        path = get_pathway_to_mol(file_path_pathway)
                    elif ".mol" in mol:
                        # Load the pathway data
                        path = get_pathway_to_inchi(file_path_pathway)
                    else:
                        path = None
                        ValueError("Input not supported")
                else:
                    path = None
                if joint_corr:
                    return joint_correction(mol, value), path
                else:
                    return value, path
            except Exception as e:
                print(f"Failed to load assembly output: {file_path_out}, Error: {e}", flush=True)
                return -1, None


def calculate_assembly_semi_metric(graph, dir_code=None, timeout=100.0, debug=False, strip_hydrogen=False):
    """
    Calculate the assembly semi-metric for a given graph.

    Args:
        graph (nx.Graph): The input molecule as a NetworkX graph.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before calculation. Defaults to False.

    Returns:
        int: The difference between the joint assembly index and the sum of the assembly indices of the disconnected subgraphs.
    """
    # Here we have to assume it is in graph format to make sure we can split the system easily
    assert isinstance(graph, nx.Graph), "Input must be a NetworkX graph"

    # Calculate the joint assembly index
    jai, _ = calculate_assembly_index(graph, dir_code=dir_code, timeout=timeout, debug=debug,
                                      strip_hydrogen=strip_hydrogen)

    # Get the disconnected subgraphs
    subgraphs = get_disconnected_subgraphs(graph)

    # Calculate the assembly index for each subgraph
    result = 0
    for graph in subgraphs:
        ai, _ = calculate_assembly_index(graph, dir_code=dir_code, timeout=timeout, debug=debug,
                                         strip_hydrogen=strip_hydrogen)
        result += ai

    # Return the difference
    return jai - result


def compile_assembly_code():
    """
    # Get the assembly code
    git clone git@gitlab.com:croningroup/cheminformatics/assemblycpp.git
    or download the .tar.gz
    tar -xvzf assemblycpp-main.tar.gz
    rm assemblycpp-main.tar.gz

    # Update the system packages
    which g++ || sudo apt-get install g++ -y
    sudo apt-get update && sudo apt-get upgrade -y
    # Install g++
    which g++ || sudo apt-get install g++ -y

    # Get the boost code
    wget 'https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz'

    tar -xvzf boost_1_86_0.tar.gz
    rm boost_1_86_0.tar.gz

    cd /$HOME/assemblycpp-main/v5_boost_linux/
    g++ main.cpp -static -O3 -o asscpp_v5 -I/$HOME/boost_1_86_0/
    export ASS_PATH=$HOME/asscpp/v5_boost/asscpp_v5_boost_recursive

    """
    c_assembly_pull = "git clone git@gitlab.com:croningroup/cheminformatics/assemblycpp.git"
    c_assembly_extract = "tar -xvzf assemblycpp-main.tar.gz"
    c_assembly_remove = "rm assemblycpp-main.tar.gz"

    # Get the assembly code
    subprocess.run(c_assembly_pull)
    subprocess.run(c_assembly_extract)
    return None
