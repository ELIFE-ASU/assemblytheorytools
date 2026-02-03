import assemblytheorytools as att


def _get_ai(smi):
    """
    Calculate the assembly index (AI) for a given SMILES string.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string as input,
    converts it to a NetworkX graph representation, and calculates the assembly index with
    hydrogen atoms stripped.

    Parameters:
    -----------
    smi : str
        The SMILES string representing the molecular structure.

    Returns:
    --------
    int
        The calculated assembly index for the given SMILES string.
    """
    ai, _, _ = att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True)
    return ai


def test_parallel_processing():
    """
    Test the parallel processing of assembly index calculations for a list of SMILES strings.

    This function performs the following steps:
    1. Defines a list of SMILES strings representing various molecules.
    2. Calculates the assembly index for each SMILES string using multiprocessing.
    3. Asserts that the calculated assembly indices match the expected values.
    4. Repeats the calculation using thread-based parallel processing and chunked multiprocessing.
    5. Verifies that the results are consistent across all methods.

    Asserts:
        - The calculated assembly indices match the expected values for each method.

    Notes:
        - The `att.mp_calc`, `att.tp_calc`, and `att.mp_calc_chunked` functions are used for multiprocessing,
          thread-based processing, and chunked multiprocessing, respectively.
        - The `_get_ai` function is used to calculate the assembly index for a given SMILES string.
    """
    print(flush=True)
    smiles_list = [
        'C(C(=O)O)N',  # Glycine
        'C[C@@H](C(=O)O)N',  # Alanine
        'C([C@@H](C(=O)O)N)O',  # Serine
        'C1C[C@H](NC1)C(=O)O',  # Proline
        'CC(C)C(C(=O)O)N',  # Valine
        'CC(C)CC(C(=O)O)N',  # Leucine
        'CCC(C)CC(C(=O)O)N',  # Isoleucine
        'C1CCCCC1C(=O)O',  # Cyclohexane carboxylic acid
        'C1=CC=CC=C1C(=O)O',  # Benzoic acid
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    ]

    # Calculate assembly indices using multiprocessing
    results = att.mp_calc(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais

    # Calculate assembly indices using thread-based parallel processing
    results = att.tp_calc(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais

    # Calculate assembly indices using chunked multiprocessing
    results = att.mp_calc_chunked(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais


def _add(a, b):
    """
    Add two numbers.

    This helper function takes two numerical inputs and returns their sum.

    Parameters:
    -----------
    a : int or float
        The first number to add.
    b : int or float
        The second number to add.

    Returns:
    --------
    int or float
        The sum of the two input numbers.
    """
    return a + b


def test_mp_calc_star():
    """
    Test the `mp_calc_star` function for parallel execution with multiple arguments.

    This function performs the following steps:
    1. Defines a list of argument tuples to be passed to the `_add` function.
    2. Defines the expected results for the addition of each tuple.
    3. Calls the `mp_calc_star` function to perform parallel computation of `_add` on the arguments.
    4. Asserts that the results from `mp_calc_star` match the expected results.

    Asserts:
        - The results of the parallel computation are equal to the expected results.

    Notes:
        - The `_add` function is a helper function that adds two numbers.
        - The `mp_calc_star` function is used for multiprocessing with multiple arguments.
    """
    args = [(1, 2), (3, 4), (5, 6), (7, 8)]
    expected_results = [3, 7, 11, 15]
    results = att.mp_calc_star(_add, args)
    assert results == expected_results
