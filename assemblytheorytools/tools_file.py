import fcntl
import os
from typing import List, Optional


def file_list(mypath=None):
    """
    This function generates a list of all files in a specified directory.
    If no directory is specified, it defaults to the current working directory.

    Parameters:
    mypath (str, optional): The path to the directory. Defaults to None, which means the current working directory.

    Returns:
    list: A list of all files in the specified directory.
    """
    mypath = mypath or os.getcwd()  # If no path is provided, use the current working directory
    return [f for f in os.listdir(mypath) if
            os.path.isfile(os.path.join(mypath, f))]  # Return a list of all files in the directory


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


def filter_files(file_paths, substring):
    """
    This function filters a list of file paths and returns only those where the file name contains a given substring.

    Parameters:
    file_paths (list): The list of file paths.
    substring (str): The substring to look for in the file names.

    Returns:
    list: A list of file paths where the file name contains the given substring.
    """
    return list(filter(lambda file_path: substring in os.path.basename(file_path), file_paths))


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


def remove_files(target_dir, debug=False):
        """
        This function removes all files in the specified directory and its subdirectories.

        Parameters:
        target_dir (str): The path to the target directory.
        debug (bool, optional): If True, prints the name of each file being removed. Defaults to False.

        Returns:
        None
        """
        # List all files in the directory
        list_files = file_list_all(target_dir)
        # Remove the files
        for file in list_files:
            if debug:
                print(f"Removing file {file}", flush=True)
            os.remove(file)
        return None


def wipe_dir(temp_dir):
        """
        This function removes all files in the specified directory and then removes the directory itself.

        Parameters:
        temp_dir (str): The path to the directory to be wiped.

        Returns:
        None
        """
        remove_files(temp_dir)
        os.rmdir(temp_dir)
        return None
