"""
Filesystem helpers.

This module provides directory listing and filtering, concurrency-safe appends to
a shared file using file locking, and guarded removal of files and directories.
"""

import glob
import json
import os
import re
from typing import Iterable, List, Match, Optional

from filelock import FileLock


def file_list(mypath: Optional[str] = None) -> List[str]:
    """
    List file names directly inside a directory, in filesystem order.

    Parameters
    ----------
    mypath : Optional[str], optional
        The directory to list. A missing or empty path uses the current
        working directory.

    Returns
    -------
    List[str]
        File names without the directory prefix.
    """
    mypath = mypath or os.getcwd()
    return [
        name
        for name in os.listdir(mypath)
        if os.path.isfile(os.path.join(mypath, name))
    ]


def file_list_all(mypath: Optional[str] = None) -> List[str]:
    """
    List file paths recursively, in filesystem traversal order.

    Parameters
    ----------
    mypath : Optional[str], optional
        The directory to walk. A missing or empty path uses the current
        working directory. Symlink directories encountered within the tree
        are not traversed.

    Returns
    -------
    List[str]
        Paths prefixed by the directory, with user-home prefixes expanded.
    """
    return [
        os.path.expanduser(os.path.join(root, name))
        for root, _, filenames in os.walk(mypath or os.getcwd())
        for name in filenames
    ]


def filter_files(file_paths: Iterable[str], substring: str) -> List[str]:
    """
    Keep paths whose file name contains the given substring.

    Parameters
    ----------
    file_paths : Iterable[str]
        The iterable of file paths.
    substring : str
        The substring to look for in the file names.

    Returns
    -------
    List[str]
        Matching paths in their original order, including duplicates.
    """
    return [path for path in file_paths if substring in os.path.basename(path)]


def write_to_shared_file(message: str, shared_file: str) -> None:
    """
    Append a message verbatim while holding an exclusive file lock.

    Parameters
    ----------
    message : str
        The message to write to the file.
    shared_file : str
        The path to the shared file.

    Returns
    -------
    None

    Notes
    -----
    The lock is taken on a sidecar ``<shared_file>.lock`` rather than on the
    shared file itself, because the POSIX and Windows APIs for locking a file
    in place have no common subset. Writers therefore only exclude each other
    if they all go through this function. The sidecar is left in place; that is
    what makes the lock visible to a process that arrives later.
    """
    with FileLock(f"{os.fspath(shared_file)}.lock"):
        with open(shared_file, "a") as stream:
            stream.write(message)
        # Closing flushes buffered writes before the lock is released.


def remove_files(target_dir: str, debug: bool = False) -> None:
    """
    Remove files recursively while preserving the directory structure.

    Parameters
    ----------
    target_dir : str
        The path to the target directory.
    debug : bool, optional
        If True, prints the name of each file being removed. Defaults to False.
    """
    for file_path in file_list_all(target_dir):
        if debug:
            print(f"Removing file {file_path}", flush=True)
        os.remove(file_path)


def wipe_dir(temp_dir: str) -> None:
    """
    Remove a directory and its contents via :func:`safe_folder_remove`.

    Parameters
    ----------
    temp_dir : str
        The path to the directory to be wiped.
    """
    safe_folder_remove(temp_dir)


def list_subdirs(directory: str, target: str = "ai_calc") -> List[str]:
    """
    List immediate subdirectory names starting with ``target``.

    Parameters
    ----------
    directory : str
        The path to the directory to search within.
    target : str, optional
        The prefix string that subdirectories must start with. Defaults to "ai_calc".

    Returns
    -------
    List[str]
        Matching subdirectory names in filesystem order.
    """
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name.startswith(target)
    ]


def prep_json(json_path: str) -> None:
    """
    Repair ``EdgeColours`` entries and rewrite the JSON file in place.

    Empty entries become ``"ERROR"`` and unquoted entries become strings.
    The repaired text is parsed before the original file is overwritten.

    Parameters
    ----------
    json_path : str
        The path to the JSON file to be processed.

    Raises
    ------
    json.JSONDecodeError
        If the repaired text is invalid JSON. The file remains unchanged.
    """
    data = _read_assembly_json(json_path, normalize_edge_colours=True)

    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=4)


def _read_assembly_json(
    json_path: str, *, normalize_edge_colours: bool = False
) -> dict:
    """Read pathway JSON, repairing legacy colours only if parsing fails.

    ``prep_json`` explicitly requests normalization even for valid JSON to
    retain its public conversion of numeric and empty colour entries.
    """
    with open(json_path, encoding="utf-8") as stream:
        raw = stream.read()
    if not normalize_edge_colours:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # Older executables emitted unquoted bond names and empty entries for
    # colours above five. Current output is valid JSON and needs no repair.
    repaired = re.sub(
        r'"EdgeColours"\s*:\s*\[(.*?)\]',
        _edge_colours_replacer,
        raw,
        flags=re.DOTALL,
    )
    return json.loads(repaired)


def _edge_colours_replacer(match: Match[str]) -> str:
    """Fill empty entries and quote bare values in an ``EdgeColours`` match."""
    fixed_items = []
    for item in match.group(1).split(","):
        value = item.strip() or "ERROR"
        fixed_items.append(value if '"' in value else f'"{value}"')
    return '"EdgeColours": [' + ", ".join(fixed_items) + "]"


def remove_file_pattern(pattern: str) -> None:
    """
    Remove files matching a glob pattern, ignoring individual removal errors.

    Parameters
    ----------
    pattern : str
        The glob pattern to match files, for example ``*.txt`` for all text
        files.
    """
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass


def safe_folder_remove(folder_path: str) -> None:
    """
    Remove a folder and its contents, ignoring missing or non-directory paths.

    Files and subdirectories are removed from the bottom up. Symlink
    directories encountered within the tree are not traversed, and removal
    errors are propagated.

    Parameters
    ----------
    folder_path : str
        The path to the folder to be removed.
    """
    if not os.path.isdir(folder_path):
        return

    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(folder_path)
