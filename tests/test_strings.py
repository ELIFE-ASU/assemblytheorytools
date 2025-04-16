import pytest

import assemblytheorytools as att


def test_undirected_str_ass():
    """
    Test the string assembly index calculation in mol mode for an undirected string.

    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index of the input string.
    3. Compares the calculated assembly index to the expected value.

    Asserts:
        - The calculated assembly index is equal to the expected value.
    """
    s_inpt = "abracadabra"
    ai_ref = 7
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=False, mode='mol')
    ai2, _, _ = att.calculate_string_assembly_index(s_inpt, directed=False, mode='str')
    assert ai == ai_ref
    assert ai2 == ai_ref


def test_directed_str_ass():
    """
    Test the string assembly index calculation in mol mode for a directed string.

    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index of the input string.
    3. Compares the calculated assembly index to the expected value.

    Asserts:
        - The calculated assembly index is equal to the expected value.
    """
    s_inpt = "abracadabra"
    ai_ref = 7
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode='mol')
    ai2, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode='str')
    assert ai == ai_ref
    assert ai2 == ai_ref


def test_cfg_str_ass():
    """
    Test the CFG upperbound to string assembly index for a directed string.

    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index upper bound for the input string.
    3. Compares the upper bound to the exact value.

    Asserts:
        - The calculated upper bound is <= the exact value.
    """
    s_inpt = "abracadabra"
    ai_ref = 7
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode="cfg")
    assert ai <= ai_ref


# @pytest.mark.skip  # broken
def test_directed_joint_str_ass():
    """
    Test the calculation of the assembly index for a set of strings.

    This function performs the following steps:
    1. Define a list of strings
    2. Calculate their assembly index
    3. Assert ai = 4


    Asserts:
        - The calculated assembly index is equal to 4.
    """
    s_inpt = ["aaaa", "bbbb", "aa"]
    ai_ref = 4
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode='mol')
    ai2, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode='str')
    assert ai == ai_ref
    assert ai2 == ai_ref


# @pytest.mark.skip  # broken
def test_joint_cfg_str_ass():
    """
    Test the calculation of the assembly index for a set of strings in CFG mode.

    This function performs the following steps:
    1. Defines a list of input strings.
    2. Calculates the assembly index for the input strings in directed CFG mode.
    3. Prints the virtual objects generated during the calculation.
    4. Asserts that the calculated assembly index is equal to the expected value.

    Asserts:
        - The calculated assembly index is equal to 4.
    """
    strs = ["aaaa", "bbbb", "aa"]
    ai_ref = 4
    ai, v_obj, path = att.calculate_string_assembly_index(strs, directed=True, mode="cfg")
    assert ai >= ai_ref


def test_string_early_exit():
    # I am trying to figure out how to get the early exit to work
    # Right now I either get exact or -1. I want to get the early exit upper bound.
    # s = ''.join(random.choices('abcd', k=50))
    s = "abacdbdacbcdadbccbadacdbadcbadcbadcbadcbbadcbdacbdcbdacbdcbdabcdabcdbcdabcdabcdabcdabcdbcdabcadbabc"
    l1, _, _ = att.calculate_string_assembly_index(s, directed=True, mode='str', timeout=2)
    print(f"Fast Upper Bound = {l1}", flush=True)
    l2, _, _ = att.calculate_string_assembly_index(s, directed=True, mode='str', timeout=20)
    print(f"Slow Upper Bound = {l2}", flush=True)
    assert l1 >= l2
