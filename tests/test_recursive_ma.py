import pickle
from pathlib import Path
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
    assert meta == {1000000.0: {400.0: {200.0: {}, 100.0: {}}, 500.0: {300.0: {}, 100.0: {}}}}


def test_ma_estimator_simple():
    print(flush=True)
    simple_tree = create_simple_tree()

    print("\nTree structure:", flush=True)
    att.print_tree(simple_tree)
    print(f"\nTree depth: {att.tree_depth(simple_tree)}", flush=True)

    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Estimate MA
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da", flush=True)

    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0
    )

    print(f"\nResults:")
    print(f"  MA estimate (mean): {np.mean(ma_estimate):.2f}", flush=True)
    print(f"  MA estimate (std):  {np.std(ma_estimate):.2f}", flush=True)
    print(f"  MA range: [{np.min(ma_estimate):.2f}, {np.max(ma_estimate):.2f}]", flush=True)

    results = np.array([np.mean(ma_estimate), np.std(ma_estimate), np.min(ma_estimate), np.max(ma_estimate)])
    expected = np.array([21.14, 1.92, 17.73, 25.38])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_complex():
    print(flush=True)
    simple_tree = create_complex_tree()

    print("\nTree structure:", flush=True)
    att.print_tree(simple_tree)
    print(f"\nTree depth: {att.tree_depth(simple_tree)}", flush=True)

    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Estimate MA
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da", flush=True)

    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0
    )

    print(f"\nResults:")
    print(f"  MA estimate (mean): {np.mean(ma_estimate):.2f}", flush=True)
    print(f"  MA estimate (std):  {np.std(ma_estimate):.2f}", flush=True)
    print(f"  MA range: [{np.min(ma_estimate):.2f}, {np.max(ma_estimate):.2f}]", flush=True)

    results = np.array([np.mean(ma_estimate), np.std(ma_estimate), np.min(ma_estimate), np.max(ma_estimate)])
    expected = np.array([25.83, 2.48, 19.26, 29.06])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_element():
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
    print(flush=True)
    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    detailed_tree = {
        300.0: {
            200.0: {
                100.0: {}
            },
            100.0: {}
        }
    }

    print("\nTree structure:", flush=True)
    att.print_tree(detailed_tree)

    ma_estimate = estimator.estimate_ma(
        tree=detailed_tree,
        mw=300.0,
        progress_levels=3
    )

    print(f"\nFinal MA estimate: {np.mean(ma_estimate):.2f} ± {np.std(ma_estimate):.2f}", flush=True)
    results = np.array([np.mean(ma_estimate), np.std(ma_estimate)])
    expected = np.array([17.48, 3.35])
    assert np.allclose(results, expected, atol=0.01)


def test_ma_estimator_joint():
    print(flush=True)
    # Create MA estimator
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {500.0: {300.0: {}, 100.0: {}}}

    print("\nMeta tree (combining multiple samples):")
    meta = att.meta_tree([tree1, tree2], meta_parent_mz=1e6)
    att.print_tree(meta)

    ma_estimate = estimator.estimate_ma(
        tree=meta,
        mw=1e6,
        progress_levels=3
    )

    print(f"\nFinal MA estimate: {np.mean(ma_estimate):.2f} ± {np.std(ma_estimate):.2f}", flush=True)
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
    child1_ma = np.mean(estimator.estimate_ma(mock_data[371.4], 150.1))

    # Estimate the molecular assembly (MA) for the second child node
    child2_ma = np.mean(estimator.estimate_ma(mock_data[371.4], 221.3))

    # Print the estimated MAs for the parent and child nodes
    print(f"Parent MA: {parent_ma}", flush=True)
    print(f"Child1 MA: {child1_ma}", flush=True)
    print(f"Child2 MA: {child2_ma}", flush=True)

    # Assert that the parent's MA is less than or equal to the sum of the children's MAs plus 1.0
    assert parent_ma <= child1_ma + child2_ma + 1.0
