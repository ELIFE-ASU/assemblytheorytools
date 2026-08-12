"""
Parallel execution helpers.

This module provides thin wrappers around ``multiprocessing`` and
``concurrent.futures`` for mapping a function over an iterable, including
starmap-style calls for multi-argument functions, a thread-pool variant, and a
chunked variant for large workloads.
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Iterable, Any, Tuple


def _bind_kwargs(func: Callable[..., Any], kwargs: dict) -> Callable[..., Any]:
    """
    Partially apply *kwargs* to *func*, if any were given.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to bind keyword arguments to.
    kwargs : dict
        Keyword arguments to bind. If empty, *func* is returned unchanged.

    Returns
    -------
    Callable[..., Any]
        *func* itself, or a `partial` wrapping it with *kwargs* bound.
    """
    return partial(func, **kwargs) if kwargs else func


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
    func = _bind_kwargs(func, kwargs)
    with mp.Pool(n) as pool:
        return pool.map(func, arg)


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
    func = _bind_kwargs(func, kwargs)
    with mp.Pool(n) as pool:
        return pool.starmap(func, args)


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
    func = _bind_kwargs(func, kwargs)
    with ThreadPoolExecutor(max_workers=n) as executor:
        return list(executor.map(func, arg))


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
    func = _bind_kwargs(func, kwargs)
    with mp.Pool(n or mp.cpu_count()) as pool:
        return pool.map(func, arg, chunksize=chunksize or 1)
