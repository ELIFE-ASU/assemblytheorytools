import pickle
from pathlib import Path

import assemblytheorytools as att

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
    print("\nTree 1:",flush=True)
    att.print_tree(tree1)

    # Print the second tree
    print("\nTree 2:",flush=True)
    att.print_tree(tree2)

    # Combine the trees into a meta tree and print the resulting structure
    print("\nMeta tree (combining multiple samples):",flush=True)
    meta = att.meta_tree([tree1, tree2], meta_parent_mz=1e6)
    att.print_tree(meta)
    assert meta == {1000000.0: {400.0: {200.0: {}, 100.0: {}}, 500.0: {300.0: {}, 100.0: {}}}}


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
    parent_ma = estimator.estimate_ma(mock_data, 371.4)

    # Estimate the molecular assembly (MA) for the first child node
    child1_ma = estimator.estimate_ma(mock_data[371.4], 150.1)

    # Estimate the molecular assembly (MA) for the second child node
    child2_ma = estimator.estimate_ma(mock_data[371.4], 221.3)

    # Print the estimated MAs for the parent and child nodes
    print(f"Parent MA: {parent_ma}", flush=True)
    print(f"Child1 MA: {child1_ma}", flush=True)
    print(f"Child2 MA: {child2_ma}", flush=True)

    # Assert that the parent's MA is less than or equal to the sum of the children's MAs plus 1.0
    assert parent_ma <= child1_ma + child2_ma + 1.0


def test_ma_estimator_real_data():
    """
    Test the MAEstimator with real data loaded from pickle files.

    This function performs the following steps:
    1. Initializes an MAEstimator with specific parameters.
    2. Loads and processes pickle files containing test data.
    3. Renames columns in the data for consistency.
    4. Builds a fragmentation tree from the processed data.
    5. Estimates the molecular assembly (MA) for the tree.
    6. Prints and asserts the results to validate the estimator's behavior.

    Assertions:
        - The estimated MA should be equal to 11 (as an integer).
        - The molecular weight (MW) should be 304.1.
    """
    print(flush=True)
    # Initialize the MAEstimator with same_level=True and a tolerance of 3e-3
    estimator = att.MAEstimator(same_level=True, tol=3e-3)

    # Define the path to the directory containing the pickle files
    here = Path('tests/data/recursive_ma/')

    # Load and parse pickle files into a dictionary, using the level as the key
    pickle_files = {
        int(f.name.split("_")[1][2]): pickle.load(f.open("rb"))
        for f in sorted(here.glob("*.pkl"))
    }

    # Rename columns in the data for consistency
    for level, data in pickle_files.items():
        data.rename(columns={f"ms{level}_mz": "mz",
                             f"ms{level}_intensity": "intensity"},
                    inplace=True)

    # Build a fragmentation tree from the processed data
    test_data = att.build_tree(pickle_files, max_level=3)

    # Extract the molecular weight (MW) from the tree
    mw = next(iter(test_data.keys()))

    # Estimate the molecular assembly (MA) for the tree
    ma = estimator.estimate_ma(test_data, mw)

    # Print the molecular weight and molecular assembly
    print(mw, ma, flush=True)
    print(f"MW: {mw}", flush=True)
    print(f"MA: {ma}", flush=True)

    # Assert the expected values for MA and MW
    assert int(ma) == 11
    assert mw == 304.1
