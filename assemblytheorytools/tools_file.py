import fcntl
import os
from typing import List, Optional


def file_list_all(mypath: Optional[str] = None) -> List[str]:
    """
    This function generates a list of all files in a specified directory and its subdirectories.
    If no directory is specified, it defaults to the current working directory.

    Parameters:
    mypath (Optional[str], optional): The path to the directory. Defaults to None, which means the current working directory.

    Returns:
    List[str]: A list of all files in the specified directory and its subdirectories.
    """
    mypath = mypath or os.getcwd()  # If no path is provided, use the current working directory
    files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dirpath, dirnames, filenames in os.walk(mypath):
        for filename in filenames:
            # os.path.join joins one or more path components intelligently
            files.append(os.path.expanduser(os.path.join(dirpath, filename)))
    return files


def write_to_shared_file(message: str, shared_file: str) -> None:
    """
    Write a message to a shared file with an exclusive lock.

    Args:
        message (str): The message to write to the file.
        shared_file (str): The path to the shared file.

    Returns:
        None
    """
    with open(shared_file, 'a') as f:
        # Acquire an exclusive lock before writing
        fcntl.flock(f, fcntl.LOCK_EX)
        # Write the message to the file
        f.write(message)
        # Release the lock after writing
        fcntl.flock(f, fcntl.LOCK_UN)
    return None
