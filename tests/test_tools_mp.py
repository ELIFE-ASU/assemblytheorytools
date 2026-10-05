"""Parallel mapping preserves result order across worker backends."""

import os
import threading

import pytest

import assemblytheorytools as att


def _transform(value, *, scale, offset):
    # Process workers need an importable, module-level callable.
    return value * scale + offset, os.getpid(), threading.get_ident()


def _add(a, b):
    return a + b


@pytest.mark.parametrize("mapper", [att.mp_calc, att.tp_calc, att.mp_calc_chunked])
def test_parallel_mapping_preserves_order_and_forwards_keywords_in_workers(mapper):
    # Exercise real pools without repeating the molecular calculator's tests.
    # A nonmonotone sequence with duplicates also spans several uneven chunks.
    values = [3, -2, 3, 0, 6, -1, 1]
    options = {"chunksize": 2} if mapper is att.mp_calc_chunked else {}
    results = mapper(_transform, values, n=2, scale=2, offset=1, **options)

    assert [value for value, _, _ in results] == [7, -3, 7, 1, 13, -1, 3]
    if mapper is att.tp_calc:
        assert all(pid == os.getpid() for _, pid, _ in results)
        assert all(thread != threading.get_ident() for _, _, thread in results)
    else:
        assert all(pid != os.getpid() for _, pid, _ in results)


def test_process_starmap_unpacks_arguments_in_order():
    assert att.mp_calc_star(_add, [(1, 2), (3, 4), (5, 6), (7, 8)], n=2) == [
        3,
        7,
        11,
        15,
    ]
