import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Union, List
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import CFG
from .graph_tools import write_ass_graph_file, remove_hydrogen_from_graph, nx_to_mol
from .mol_tools import write_v2k_mol_file, combine_mols
from .pathway import get_pathway_to_graph, get_pathway_to_mol, get_pathway_to_inchi
from .string_tools import prep_joint_string_ai, get_dir_str_molecule, get_undir_str_molecule


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

    WARNING it is the responsibility of the user to ensure the mol file has H or not!

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
    # Initialise the variables
    ai = -1
    virt_obj = None
    path = None
    file_path_in = None
    # Check if the input is a string and not a .mol file
    if isinstance(mol, str) and not mol.endswith(".mol"):
        ai, virt_obj, path = CFG.ai_with_pathways(mol, f_print=False)
        return ai, virt_obj, path
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

        # Assuming a NetworkX graph
        if isinstance(mol, nx.Graph):
            # Check if we need to strip hydrogen
            if strip_hydrogen:
                mol = remove_hydrogen_from_graph(mol)
            # Make the in file
            file_path_in = os.path.join(temp_dir, f"graph_in")
            # Write the input graph file
            write_ass_graph_file(mol, file_name=file_path_in)
        # Check if the input is a RDkit mol
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
        # Check if instance is a file path to a .mol file
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
            return ai, virt_obj, path
        else:
            try:
                # Load the assembly output
                ai = load_assembly_output(file_path_out)
                # Check the pathway file exits
                if os.path.isfile(file_path_pathway):
                    if isinstance(mol, nx.Graph):
                        virt_obj = get_pathway_to_graph(file_path_pathway)
                    elif isinstance(mol, Chem.Mol):
                        # Load the pathway data
                        virt_obj = get_pathway_to_mol(file_path_pathway)
                    elif ".mol" in mol:
                        # Load the pathway data
                        virt_obj = get_pathway_to_inchi(file_path_pathway)
                    else:
                        virt_obj = None
                        ValueError("Input not supported")
                else:
                    virt_obj = None
                if joint_corr:
                    return joint_correction(mol, ai), virt_obj, path
                else:
                    return ai, virt_obj, path
            except Exception as e:
                print(f"Failed to load assembly output: {file_path_out}, Error: {e}", flush=True)
                return ai, virt_obj, path


def calculate_assembly_semi_metric(graph1, graph2, dir_code=None, timeout=100.0, debug=False, strip_hydrogen=False):
    """
    Calculate the assembly semi-metric distance between a pair of molecular graphs. 

    Args:
        graph1 (nx.Graph): First input molecule as a NetworkX graph.
        graph2 (nx.Graph): Second input molecule as a NetworkX graph.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before calculation. Defaults to False.

    Returns:
        int: The difference between the joint assembly index and the sum of the assembly indices of the disconnected subgraphs.
    """
    # Here we have to assume it is in graph format to make sure we can split the system easily
    assert isinstance(graph1, nx.Graph), "Input must be a NetworkX graph"
    assert isinstance(graph2, nx.Graph), "Input must be a NetworkX graph"

    # # Get the disconnected subgraphs
    # subgraphs = get_disconnected_subgraphs(graph)
    # assert len(subgraphs) == 2, "Semimetric distance is between exactly two molecular graphs."

    # Combine the graphs into a single graph with 2 disjoint components
    mols = [nx_to_mol(graph1), nx_to_mol(graph2)]
    mol = combine_mols(mols)

    # Calculate the joint assembly index
    jai, _ = calculate_assembly_index(mol, dir_code=dir_code, timeout=timeout, debug=debug,
                                      strip_hydrogen=strip_hydrogen)
    if debug:
        print(f"Joint Assembly Index: {jai}", flush=True)
    # Calculate the assembly index for each subgraph
    result = 0
    for subgraph in [graph1, graph2]:
        ai, _ = calculate_assembly_index(subgraph, dir_code=dir_code, timeout=timeout, debug=debug,
                                         strip_hydrogen=strip_hydrogen)
        if debug:
            print(f"Assembly Index: {ai}", flush=True)
        result += ai

    # Calculate the semimetric distance
    return 2 * jai - result


def run_command_simple(command):
    """
    Run a simple command in the subprocess.

    Args:
        command (str): The command to run as a string.

    Returns:
        bytes: The standard output of the command.
    """
    result = subprocess.run(command.split())
    return result.stdout


def compile_assembly_code():
    """
    Set up the keys
    go to gitlab website
    click your profile icon
    go to ssh keys

    ssh-agent $(ssh-add rsa-key-20240430; git clone git@gitlab.com:croningroup/cheminformatics/assemblycpp.git)
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
    # Unzip the boost code
    tar -xvzf boost_1_86_0.tar.gz
    # Remove the boost zip file
    rm boost_1_86_0.tar.gz

    g++ main.cpp -O3 -o asscpp_v5 -I/boost_1_86_0/
    export ASS_PATH=$HOME/asscpp/v5_boost/asscpp_v5_boost_recursive

    # Remove the boost folder
    rm -r

    """
    boost_dir = os.path.abspath(os.path.expanduser(os.path.join(os.getcwd(), "/boost_1_86_0")))
    # Get the assembly code
    # run_command_simple("git clone git@gitlab.com:croningroup/cheminformatics/assemblycpp.git")

    # run_command_simple("tar -xvzf assemblycpp-main.tar.gz")
    # run_command_simple("rm assemblycpp-main.tar.gz")
    # run_command_simple("wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz")
    # run_command_simple("tar -xvzf boost_1_86_0.tar.gz")
    # run_command_simple("rm boost_1_86_0.tar.gz")
    # run_command_simple(f"g++ assemblycpp-main/v5_combined_linux/main.cpp -O3 -o asscpp_v5 -I {boost_dir}")
    # run_command_simple("rm -r")
    return None


def calculate_string_assembly_index(input_data: Union[str, List[str]], dir_code=None, timeout=100.0, debug=False, directed=False, mode="mol"):
    """
    Calculate the assembly index of a string or a set of strings. 
    This function uses the molecular assembly calculator by constructing molecular graphs which correspond to the strings.

    Args:
        input_data (Union[str, List[str]]): The input data, which can be a single string or a list of strings.
        dir_code (str, optional): The directory code for the assembly tool. Defaults to None.
        timeout (float, optional): The maximum time in seconds to allow the command to run. Defaults to 100.0 seconds.
        debug (bool, optional): If True, create a directory with a timestamp for debugging. Defaults to False.
        directed (bool, optional): If True, treat strings as directed. Defaults to False, treating strings as undirected.
        mode ("mol"/"str"/"cfg",optional): "mol" uses the molecular assembly calculator, "str" uses the string assembly calculator (not yet supported), "cfg" uses the RePair upper bound.
    """
    if isinstance(input_data, str):
        # Handle the case where input_data is a single string
        string = input_data
        delimiters = []
    elif isinstance(input_data, list):
        # Handle joint assembly case
        string, delimiters = prep_joint_string_ai(input_data)
    else:
        raise ValueError("Input must be either a single string or a list of strings")

    if mode == "mol": # Use the molecular assembly cpp calculator
        if directed:
            graph = get_dir_str_molecule(string, debug=debug)
        else:
            graph, edge_color_dict = get_undir_str_molecule(string, debug=debug)
        
        if debug:
            # String-Molecular Graph Nodes colors
            print("\nNode colors:")
            for node, data in graph.nodes(data=True):
                print(f"Node {node}: {data.get('color', 'No color')}")

            # String-Molecular Graph Edge colors
            print("\nEdge colors:")
            for u, v, data in graph.edges(data=True):
                print(f"Edge {u}-{v}: {data.get('color', 'No color')}")


        graph_ai, graph_path = calculate_assembly_index(graph, dir_code=dir_code, timeout=timeout, debug=debug, joint_corr=False, strip_hydrogen=False)

        if debug:
            print(f"Graph Assembly Index: {graph_ai}", flush=True)

        if directed:
            # Convert to (joint) assembly index of directed strings
            return graph_ai - len(set(string)) - 2 * len(delimiters), None # Path parsing still needs to be added
        else:
            # Convert to (joint) assembly index of undirected strings
            return graph_ai - 2 * len(delimiters), None # Path parsing still needs to be added
    
    elif mode == "str": # Use the string assembly cpp calculator
        raise NotImplementedError("String assembly cpp calculator not yet supported.")
    
    elif mode == "cfg": # Use the RePair upper bound
        composite_ai, virt_obj, path = CFG.ai_with_pathways(string, f_print=False)
        if directed:
            # Convert to (joint) assembly index of directed strings
            return composite_ai - 2 * len(delimiters), path
        else:
            ValueError("Current CFG code only works for directed strings. Directed string assembly index is an upper bound to undirected string assembly index, so you may still use the directed calculator.")
        


def assembly_dry_run(mol, temp_dir=None, strip_hydrogen=False):
    """
    Perform a dry run of the assembly process for a given molecule.

    Args:
        mol (Union[nx.Graph, Chem.Mol, str]): The molecule, which can be a NetworkX graph, an RDKit molecule, or a file path to a .mol file.
        temp_dir (str, optional): The temporary directory to use for file operations. Defaults to the current working directory.
        strip_hydrogen (bool, optional): If True, removes hydrogen atoms from the molecule before processing. Defaults to False.

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
