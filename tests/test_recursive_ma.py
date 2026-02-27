import numpy as np

import assemblytheorytools as att

np.random.seed(seed=0)


def create_simple_tree():
    """
    Create a simple fragmentation tree manually.

    This function constructs and returns a predefined fragmentation tree
    structure. The tree is represented as a nested dictionary, where the keys
    are m/z values (mass-to-charge ratios) and the values are subtrees.

    Returns
    -------
    dict
        A dictionary representing the fragmentation tree, with m/z values as
        keys and subtrees as values.
    """
    tree = {
        400.0: {
            300.0: {
                200.0: {},
                150.0: {},
                100.0: {}
            },
            250.0: {
                150.0: {},
                100.0: {},
                80.0: {}
            },
            200.0: {}
        }
    }
    return tree


def create_complex_tree():
    """
    Create a more complex fragmentation tree with deeper hierarchy.

    This function constructs and returns a predefined fragmentation tree
    structure with a deeper hierarchy compared to a simple tree. The tree
    is represented as a nested dictionary, where the keys are m/z values
    (mass-to-charge ratios) and the values are subtrees.

    Returns
    -------
    dict
        A dictionary representing the complex fragmentation tree, with m/z
        values as keys and subtrees as values.
    """
    tree = {
        500.0: {
            400.0: {
                300.0: {
                    200.0: {
                        100.0: {},
                        80.0: {}
                    },
                    150.0: {}
                },
                250.0: {
                    150.0: {},
                    100.0: {}
                }
            },
            350.0: {
                250.0: {},
                200.0: {
                    100.0: {}
                }
            },
            150.0: {}
        }
    }
    return tree


def test_unify_trees():
    """
    Test the `unify_trees` function from the `assemblytheorytools` (att) module.

    This function performs the following steps:
    1. Defines two sample fragmentation trees.
    2. Prints the structure of each tree.
    3. Unifies the two trees into a single tree and prints the unified structure.
    4. Asserts that the unified tree matches the expected structure.
    5. Calculates and prints the depth of the unified tree.
    6. Asserts that the depth of the unified tree matches the expected value.

    Assertions:
        - The unified tree structure should match the expected dictionary.
        - The depth of the unified tree should be 2.
    """
    print(flush=True)
    # Define two sample trees
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {400.0: {300.0: {}, 100.0: {}}}

    # Print the first tree
    print("\nTree 1:", flush=True)
    att.print_tree(tree1)

    # Print the second tree
    print("\nTree 2:", flush=True)
    att.print_tree(tree2)

    # Unify the two trees and print the unified tree
    print("\nUnified tree:", flush=True)
    unified = att.unify_trees([tree1, tree2])
    assert unified == {400.0: {100.0: {}, 200.0: {}, 300.0: {}}}
    att.print_tree(unified)

    # Calculate and print the depth of the unified tree
    print(f"\nDepth of unified tree: {att.tree_depth(unified)}", flush=True)
    assert att.tree_depth(unified) == 2


def test_meta_tree():
    """
    Test the `meta_tree` function from the `assemblytheorytools` (att) module.

    This function performs the following steps:
    1. Defines two sample fragmentation trees.
    2. Prints the structure of each tree.
    3. Combines the two trees into a meta tree with a specified parent m/z value.
    4. Prints the resulting meta tree structure.
    5. Asserts that the meta tree matches the expected structure.

    Assertions:
        - The meta tree structure should match the expected dictionary.
    """
    print(flush=True)
    # Define two new trees for joint MA calculations
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {500.0: {300.0: {}, 100.0: {}}}

    # Print the first tree
    print("\nTree 1:", flush=True)
    att.print_tree(tree1)

    # Print the second tree
    print("\nTree 2:", flush=True)
    att.print_tree(tree2)

    # Combine the trees into a meta tree and print the resulting structure
    print("\nMeta tree (combining multiple samples):", flush=True)
    meta = att.meta_tree([tree1, tree2], meta_parent_mz=1e6)
    att.print_tree(meta)

    # Assert that the meta tree matches the expected structure
    assert meta == {1000000.0: {400.0: {200.0: {}, 100.0: {}}, 500.0: {300.0: {}, 100.0: {}}}}


def test_ma_estimator_simple():
    """
    Test the MAEstimator with a simple fragmentation tree.

    This function performs the following steps:
    1. Creates a simple fragmentation tree.
    2. Prints the tree structure and its depth.
    3. Initializes an MAEstimator with specific parameters.
    4. Estimates the molecular assembly (MA) for the precursor m/z value.
    5. Prints the results, including the mean, standard deviation, and range of the MA estimates.
    6. Asserts that the calculated results match the expected values within a tolerance.

    Assertions:
        - The calculated MA mean, standard deviation, minimum, and maximum values
          should match the expected values within a tolerance of 0.01.
    """
    print(flush=True)
    # Create a simple fragmentation tree
    simple_tree = create_simple_tree()

    # Print the tree structure and its depth
    print("\nTree structure:", flush=True)
    att.print_tree(simple_tree)
    print(f"\nTree depth: {att.tree_depth(simple_tree)}", flush=True)

    # Initialize the MAEstimator with specific parameters
    estimator = att.MAEstimator(
        same_level=True,  # Consider fragments at the same level
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Define the precursor m/z value
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da", flush=True)

    # Estimate the molecular assembly (MA) for the precursor
    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0  # Disable progress tracking
    )

    # Print the results of the MA estimation
    print(f"\nResults:")
    print(f"  MA estimate (mean): {np.mean(ma_estimate):.2f}", flush=True)
    print(f"  MA estimate (std):  {np.std(ma_estimate):.2f}", flush=True)
    print(f"  MA range: [{np.min(ma_estimate):.2f}, {np.max(ma_estimate):.2f}]", flush=True)

    # Compare the results with the expected values
    results = np.array([np.mean(ma_estimate), np.std(ma_estimate), np.min(ma_estimate), np.max(ma_estimate)])
    expected = np.array([21.14, 1.92, 17.73, 25.38])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_complex():
    """
    Test the MAEstimator with a complex fragmentation tree.

    This function performs the following steps:
    1. Creates a complex fragmentation tree using the `create_complex_tree` function.
    2. Prints the tree structure and its depth.
    3. Initializes an MAEstimator with specific parameters:
       - `same_level=True`: Considers fragments at the same level.
       - `tol=0.5`: Sets the mass tolerance in Da.
       - `n_samples=20`: Specifies the number of Monte Carlo samples.
       - `min_chunk=20.0`: Sets the minimum fragment size to consider.
    4. Estimates the molecular assembly (MA) for a precursor m/z value of 400.0 Da.
    5. Prints the results, including the mean, standard deviation, and range of the MA estimates.
    6. Asserts that the calculated results match the expected values within a tolerance of 0.01.

    Assertions:
        - The calculated MA mean, standard deviation, minimum, and maximum values
          should match the expected values within a tolerance of 0.01.
    """
    print(flush=True)
    # Create a complex fragmentation tree
    simple_tree = create_complex_tree()

    # Print the tree structure and its depth
    print("\nTree structure:", flush=True)
    att.print_tree(simple_tree)
    print(f"\nTree depth: {att.tree_depth(simple_tree)}", flush=True)

    # Create an MAEstimator with specific parameters
    estimator = att.MAEstimator(
        same_level=True,  # Consider fragments at the same level
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Define the precursor m/z value
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da", flush=True)

    # Estimate the molecular assembly (MA) for the precursor
    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0  # Disable progress tracking
    )

    # Print the results of the MA estimation
    print(f"\nResults:")
    print(f"  MA estimate (mean): {np.mean(ma_estimate):.2f}", flush=True)
    print(f"  MA estimate (std):  {np.std(ma_estimate):.2f}", flush=True)
    print(f"  MA range: [{np.min(ma_estimate):.2f}, {np.max(ma_estimate):.2f}]", flush=True)

    # Compare the results with the expected values
    results = np.array([np.mean(ma_estimate), np.std(ma_estimate), np.min(ma_estimate), np.max(ma_estimate)])
    expected = np.array([25.83, 2.48, 19.26, 29.06])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_element():
    """
    Test the MAEstimator with elemental and non-elemental masses.

    This function performs the following steps:
    1. Initializes an MAEstimator with specific parameters:
       - `same_level=True`: Considers fragments at the same level.
       - `tol=0.5`: Sets the mass tolerance in Da.
       - `n_samples=20`: Specifies the number of Monte Carlo samples.
       - `min_chunk=20.0`: Sets the minimum fragment size to consider.
    2. Tests the MA estimation for Iron-56 (55.934939 Da), asserting that the
       MA estimate is approximately 0 for a pure element.
    3. Tests the MA estimation for Copper-63 (62.929599 Da), printing the
       result (expected to be approximately 0 for a pure element).
    4. Tests the MA estimation for a non-isotope mass (123.456 Da), asserting
       that the MA estimate is greater than 0 for a compound.

    Assertions:
        - The MA estimate for Iron-56 should be 0.0.
        - The MA estimate for the non-isotope mass should be greater than 0.0.
    """
    print(flush=True)
    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Iron isotope mass
    iron_tree = {55.934939: {}}
    print(f"\nTesting Iron-56 (55.934939 Da)", flush=True)
    ma_iron = estimator.estimate_ma(iron_tree, 55.934939, progress_levels=0)
    print(f"  MA estimate: {np.mean(ma_iron):.2f} (should be ~0 for pure element)", flush=True)
    assert np.mean(ma_iron) == 0.0

    # Copper isotope mass
    copper_tree = {62.929599: {}}
    print(f"\nTesting Copper-63 (62.929599 Da)", flush=True)
    ma_copper = estimator.estimate_ma(copper_tree, 62.929599, progress_levels=0)
    print(f"  MA estimate: {np.mean(ma_copper):.2f} (should be ~0 for pure element)", flush=True)

    # Non-isotope mass
    random_tree = {123.456: {}}
    print(f"\nTesting non-isotope mass (123.456 Da)", flush=True)
    ma_random = estimator.estimate_ma(random_tree, 123.456, progress_levels=0)
    print(f"  MA estimate: {np.mean(ma_random):.2f} (should be >0 for compound)", flush=True)
    assert np.mean(ma_random) > 0.0


def test_ma_estimator_detailed_tree():
    """
    Test the MAEstimator with a detailed fragmentation tree.

    This function performs the following steps:
    1. Initializes an MAEstimator with specific parameters:
       - `same_level=True`: Considers fragments at the same level.
       - `tol=0.5`: Sets the mass tolerance in Da.
       - `n_samples=20`: Specifies the number of Monte Carlo samples.
       - `min_chunk=20.0`: Sets the minimum fragment size to consider.
    2. Defines a detailed fragmentation tree structure.
    3. Prints the tree structure.
    4. Estimates the molecular assembly (MA) for the precursor m/z value of 300.0 Da.
    5. Prints the final MA estimate, including the mean and standard deviation.
    6. Asserts that the calculated results match the expected values within a tolerance of 0.01.

    Assertions:
        - The calculated MA mean and standard deviation should match the expected values
          within a tolerance of 0.01.
    """
    print(flush=True)
    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Define a detailed fragmentation tree
    detailed_tree = {
        300.0: {
            200.0: {
                100.0: {}
            },
            100.0: {}
        }
    }

    # Print the tree structure
    print("\nTree structure:", flush=True)
    att.print_tree(detailed_tree)

    # Estimate the molecular assembly (MA) for the precursor
    ma_estimate = estimator.estimate_ma(
        tree=detailed_tree,
        mw=300.0,
        progress_levels=3
    )

    # Print the final MA estimate
    print(f"\nFinal MA estimate: {np.mean(ma_estimate):.2f} ± {np.std(ma_estimate):.2f}", flush=True)

    # Compare the results with the expected values
    results = np.array([np.mean(ma_estimate), np.std(ma_estimate)])
    expected = np.array([17.48, 3.35])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_joint():
    """
    Test the MAEstimator with a meta tree combining multiple samples.

    This function performs the following steps:
    1. Initializes an MAEstimator with specific parameters:
       - `same_level=True`: Considers fragments at the same level.
       - `tol=0.5`: Sets the mass tolerance in Da.
       - `n_samples=20`: Specifies the number of Monte Carlo samples.
       - `min_chunk=20.0`: Sets the minimum fragment size to consider.
    2. Defines two sample fragmentation trees.
    3. Combines the two trees into a meta tree with a specified parent m/z value.
    4. Prints the structure of the meta tree.
    5. Estimates the molecular assembly (MA) for the meta tree.
    6. Prints the final MA estimate, including the mean and standard deviation.
    7. Asserts that the calculated results match the expected values within a tolerance of 0.01.

    Assertions:
        - The calculated MA mean and standard deviation should match the expected values
          within a tolerance of 0.01.
    """
    print(flush=True)
    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Define two sample trees
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {500.0: {300.0: {}, 100.0: {}}}

    # Combine the trees into a meta tree and print the structure
    print("\nMeta tree (combining multiple samples):")
    meta = att.meta_tree([tree1, tree2], meta_parent_mz=1e6)
    att.print_tree(meta)

    # Estimate the molecular assembly (MA) for the meta tree
    ma_estimate = estimator.estimate_ma(
        tree=meta,
        mw=1e6,
        progress_levels=3
    )

    # Print the final MA estimate
    print(f"\nFinal MA estimate: {np.mean(ma_estimate):.2f} ± {np.std(ma_estimate):.2f}", flush=True)

    # Compare the results with the expected values
    results = np.array([np.mean(ma_estimate), np.std(ma_estimate)])
    expected = np.array([66964.27, 5103.06])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_mock_data():
    """
    Test the MAEstimator with mock data.

    This function performs the following steps:
    1. Initializes an MAEstimator with specific parameters.
    2. Defines a mock fragmentation tree structure.
    3. Estimates the molecular assembly (MA) for the parent and child nodes.
    4. Prints the results for the parent and child nodes.
    5. Asserts that the parent's MA is less than or equal to the sum of the children's MAs plus a tolerance.

    Assertions:
        - The parent's MA should be less than or equal to the sum of the children's MAs plus 1.0.
    """
    print(flush=True)
    # Initialize the MAEstimator with same_level=True and a tolerance of 3e-3
    estimator = att.MAEstimator(same_level=True, tol=3e-3)

    # Define mock data representing a fragmentation tree
    mock_data = {
        371.4: {
            150.1: {72.3: None, 89.1: None},
            221.3: {72.3: None, 99.7: None}
        }
    }

    # Print a message indicating the start of the test
    print("Testing mock fragmentation tree...", flush=True)

    # Estimate the molecular assembly (MA) for the parent node
    parent_ma = np.mean(estimator.estimate_ma(mock_data, 371.4))

    # Estimate the molecular assembly (MA) for the first child node
    child1_ma = np.mean(estimator.estimate_ma({150.1: mock_data[371.4][150.1]}, 150.1))

    # Estimate the molecular assembly (MA) for the second child node
    child2_ma = np.mean(estimator.estimate_ma({221.3: mock_data[371.4][221.3]}, 221.3))

    # Print the estimated MAs for the parent and child nodes
    print(f"Parent MA: {parent_ma}", flush=True)
    print(f"Child1 MA: {child1_ma}", flush=True)
    print(f"Child2 MA: {child2_ma}", flush=True)

    # Assert that the parent's MA is less than or equal to the sum of the children's MAs plus 1.0
    assert parent_ma <= child1_ma + child2_ma + 1.0
