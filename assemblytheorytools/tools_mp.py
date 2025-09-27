import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Any, Tuple


def mp_calc(func: Callable[[Any], Any], arg: Iterable[Any], n: int = mp.cpu_count()) -> list[Any]:
    """
    Executes a function in parallel using a process pool.

    Parameters:
    func (Callable[[Any], Any]): The function to execute.
    arg (Iterable[Any]): An iterable of arguments to pass to the function.
    n (int, optional): The number of worker processes to use. Default is the number of CPU cores.

    Returns:
    list[Any]: A list of results from the function executions.
    """
    with mp.Pool(n) as pool:
        results = pool.map(func, arg)
    return results


def mp_calc_star(func: Callable[..., Any], args: Iterable[Tuple[Any, ...]], n: int = mp.cpu_count()) -> list[Any]:
    """
    Executes a function in parallel using a process pool with multiple arguments.

    Parameters:
    func (Callable[..., Any]): The function to execute.
    args (Iterable[Tuple[Any, ...]]): An iterable of argument tuples to pass to the function.
    n (int, optional): The number of worker processes to use. Default is the number of CPU cores.

    Returns:
    list[Any]: A list of results from the function executions.
    """
    with mp.Pool(n) as pool:
        results = pool.starmap(func, args)
    return results


def tp_calc(func: Callable[[Any], Any], arg: Iterable[Any], n: int = mp.cpu_count()) -> list[Any]:
    """
    Executes a function in parallel using a thread pool.

    Works best for I/O-bound tasks.

    Parameters:
    func (Callable[[Any], Any]): The function to execute.
    arg (Iterable[Any]): An iterable of arguments to pass to the function.
    n (int, optional): The number of worker threads to use. Default is the number of CPU cores.

    Returns:
    list[Any]: A list of results from the function executions.
    """
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(func, arg))
    return results
