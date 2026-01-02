import fcntl
import json
import os
import re
from typing import List, Optional, Iterable, Match


def file_list(mypath: Optional[str] = None) -> List[str]:
    """
    Generate a list of all files in a specified directory.

    If no directory is specified, it defaults to the current working directory.

    Parameters
    ----------
    mypath : Optional[str], optional
        The path to the directory. Defaults to None, which means the current working directory.

    Returns
    -------
    List[str]
        A list of all files in the specified directory.
    """
    mypath = mypath or os.getcwd()
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def file_list_all(mypath: Optional[str] = None) -> List[str]:
    """
    Generate a list of all files in a specified directory and its subdirectories.
    
    If no directory is specified, it defaults to the current working directory.

    Parameters
    ----------
    mypath : Optional[str], optional
        The path to the directory. Defaults to None, which means the current working directory.

    Returns
    -------
    List[str]
        A list of all files in the specified directory and its subdirectories.
    """
    mypath = mypath or os.getcwd()  # If no path is provided, use the current working directory
    files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dirpath, dirnames, filenames in os.walk(mypath):
        for filename in filenames:
            # os.path.join joins one or more path parts intelligently
            files.append(os.path.expanduser(os.path.join(dirpath, filename)))
    return files


def filter_files(file_paths: Iterable[str], substring: str) -> List[str]:
    """
    Filter a list of file paths and return only those where the file name contains a given substring.

    Parameters
    ----------
    file_paths : Iterable[str]
        The iterable of file paths.
    substring : str
        The substring to look for in the file names.

    Returns
    -------
    List[str]
        A list of file paths where the file name contains the given substring.
    """
    return [file_path for file_path in file_paths if substring in os.path.basename(file_path)]


def write_to_shared_file(message: str, shared_file: str) -> None:
    """
    Write a message to a shared file with an exclusive lock.

    Parameters
    ----------
    message : str
        The message to write to the file.
    shared_file : str
        The path to the shared file.

    Returns
    -------
    None
        This function does not return a value.
    """
    with open(shared_file, 'a') as f:
        # Acquire an exclusive lock before writing
        fcntl.flock(f, fcntl.LOCK_EX)
        # Write the message to the file
        f.write(message)
        # Release the lock after writing
        fcntl.flock(f, fcntl.LOCK_UN)
    return None


def remove_files(target_dir: str, debug: bool = False) -> None:
    """
    Remove all files in the specified directory and its subdirectories.

    Parameters
    ----------
    target_dir : str
        The path to the target directory.
    debug : bool, optional
        If True, prints the name of each file being removed. Defaults to False.

    Returns
    -------
    None
        This function does not return a value.
    """
    files: List[str] = file_list_all(target_dir)
    for file_path in files:
        if debug:
            print(f"Removing file {file_path}", flush=True)
        os.remove(file_path)
    return None


def wipe_dir(temp_dir: str) -> None:
    """
    Remove all files in the specified directory and then remove the directory itself.

    Parameters
    ----------
    temp_dir : str
        The path to the directory to be wiped.

    Returns
    -------
    None
        This function does not return a value.
    """
    remove_files(temp_dir)
    os.rmdir(temp_dir)
    return None


def list_subdirs(directory: str, target: str = "ai_calc") -> List[str]:
    """
    List subdirectories in a given directory that start with a specific target string.

    Parameters
    ----------
    directory : str
        The path to the directory to search within.
    target : str, optional
        The prefix string that subdirectories must start with. Defaults to "ai_calc".

    Returns
    -------
    List[str]
        A list of subdirectory names that start with the target string.
    """
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(target)]


def prep_json(json_path: str) -> None:
    """
    Take JSON file with missing edge colors entries and fill them with "ERROR" placeholder.

    Parameters
    ----------
    json_path : str
        The path to the JSON file to be processed.

    Returns
    -------
    None
        This function modifies the JSON file in place and does not return a value.
    """
    # Read the file as raw text
    with open(json_path, 'r') as f:
        raw = f.read()

    # This regex matches "EdgeColours": [ ... ]
    pattern = r'"EdgeColours"\s*:\s*\[(.*?)\]'
    fixed_raw = re.sub(pattern, edge_colours_replacer, raw, flags=re.DOTALL)

    # Now parse the fixed text as JSON
    data = json.loads(fixed_raw)

    # Write the updated data back to the JSON file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    return None


def edge_colours_replacer(match: Match[str]) -> str:
    """
    Replace empty entries in EdgeColours list with "ERROR" placeholder.

    Parameters
    ----------
    match : re.Match
        A regular expression match object containing the EdgeColours list content.

    Returns
    -------
    str
        A string with the fixed EdgeColours list where empty entries are replaced with "ERROR".
    """
    items: str = match.group(1)
    fixed_items: List[str] = []
    for item in items.split(','):
        val = item.strip()
        if val == '':
            fixed_items.append('"ERROR"')
        elif '"' not in val:  # If the value is not already quoted, quote it
            fixed_items.append(f'"{val}"')
        else:
            fixed_items.append(val)
    return '"EdgeColours": [' + ', '.join(fixed_items) + ']'
