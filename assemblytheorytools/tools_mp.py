import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Iterable, Any, Tuple


def mp_calc(func: Callable[[Any], Any],
            arg: Iterable[Any],
            n: int = mp.cpu_count(),
            **kwargs) -> list[Any]:
    """
    Executes a function in parallel using a process pool, supporting keyword arguments.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function to execute.
    arg : Iterable[Any]
        An iterable of arguments to pass to the function.
    n : int, optional
        The number of worker processes to use. Default is the number of CPU cores.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    list[Any]
        A list of results from the function executions.
    """
    if kwargs:
        func = partial(func, **kwargs)

    with mp.Pool(n) as pool:
        results = pool.map(func, arg)
    return results


def mp_calc_star(func: Callable[..., Any],
                 args: Iterable[Tuple[Any, ...]],
                 n: int = mp.cpu_count(),
                 **kwargs) -> list[Any]:
    """
    Executes a function in parallel using a process pool with multiple arguments, supporting keyword arguments.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to execute.
    args : Iterable[Tuple[Any, ...]]
        An iterable of argument tuples to pass to the function.
    n : int, optional
        The number of worker processes to use. Default is the number of CPU cores.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    list[Any]
        A list of results from the function executions.
    """
    if kwargs:
        func = partial(func, **kwargs)

    with mp.Pool(n) as pool:
        results = pool.starmap(func, args)
    return results


def tp_calc(func: Callable[[Any], Any],
            arg: Iterable[Any],
            n: int = mp.cpu_count(),
            **kwargs) -> list[Any]:
    """
    Executes a function in parallel using a thread pool, supporting keyword arguments.

    Works best for I/O-bound tasks.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function to execute.
    arg : Iterable[Any]
        An iterable of arguments to pass to the function.
    n : int, optional
        The number of worker threads to use. Default is the number of CPU cores.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    list[Any]
        A list of results from the function executions.
    """
    if kwargs:
        func = partial(func, **kwargs)

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(func, arg))
    return results


def mp_calc_chunked(
        func: Callable[[Any], Any],
        arg: Iterable[Any],
        n: int | None = None,
        chunksize: int | None = None,
        **kwargs
) -> list[Any]:
    """
    Executes a function in parallel using a process pool, with optional chunking and keyword arguments.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function to execute (on a single element).
    arg : Iterable[Any]
        An iterable of arguments to pass to the function.
    n : int, optional
        Number of worker processes (default: mp.cpu_count()).
    chunksize : int, optional
        How many items each worker gets per batch. If None, default is 1.
    **kwargs
        Keyword arguments to pass to `func`.

    Returns
    -------
    list[Any]
        A list of results from the function executions.
    """
    if kwargs:
        func = partial(func, **kwargs)

    if n is None:
        n = mp.cpu_count()

    with mp.Pool(n) as pool:
        results = pool.map(func, arg, chunksize=chunksize or 1)
    return results
