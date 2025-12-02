"""
Test script for Mass Spectrometry Molecular Assembly (MA) Estimator

See:
Jirasek, M., (2024).
Investigating and quantifying molecular complexity using assembly theory and spectroscopy.
ACS Central Science, 10(5), 1054-1064.

This script demonstrates how to use the MA estimation code by:
1. Creating synthetic MS/MS data
2. Processing it through the pipeline
3. Building a fragmentation tree
4. Estimating molecular assembly complexity
"""

from assemblytheorytools.recursive_ma import tree_depth, MAEstimator, meta_tree, unify_trees


def create_simple_tree():
    """
    Create a fragmentation tree manually.
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
    """Create a more complex fragmentation tree with deeper hierarchy."""
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


def print_tree(tree, indent=0, max_depth=10):
    """Pretty print a fragmentation tree"""
    if indent > max_depth or not isinstance(tree, dict):
        return

    for mz, children in sorted(tree.items(), reverse=True):
        print("  " * indent + f"├─ m/z: {mz:.2f}")
        if children:
            print_tree(children, indent + 1, max_depth)


def test_tree_operations():
    """Test tree utility functions"""
    print("\n" + "=" * 70)
    print("Testing Tree Operations")
    print("=" * 70)

    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {400.0: {300.0: {}, 100.0: {}}}

    print("\nTree 1:")
    print_tree(tree1)

    print("\nTree 2:")
    print_tree(tree2)

    print("\nUnified tree:")
    unified = unify_trees([tree1, tree2])
    print_tree(unified)

    print(f"\nDepth of unified tree: {tree_depth(unified)}")

    print("\nUnify Tree for Joint MA calculations:")
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {500.0: {300.0: {}, 100.0: {}}}

    print("\nTree 1:")
    print_tree(tree1)

    print("\nTree 2:")
    print_tree(tree2)
    print("\nMeta tree (combining multiple samples):")
    meta = meta_tree([tree1, tree2], meta_parent_mz=1e6)
    print_tree(meta)


def main():
    print("=" * 70)
    print("MA Estimator Test Script")
    print("=" * 70)

    # Test 1: Simple tree
    print("\n[Test 1] Simple fragmentation tree")
    print("-" * 70)
    simple_tree = create_simple_tree()

    print("\nTree structure:")
    print_tree(simple_tree)
    print(f"\nTree depth: {tree_depth(simple_tree)}")

    # Create MA estimator
    estimator = MAEstimator(
        same_level=True,
        tol=0.5,  # Mass tolerance in Da
        n_samples=20,  # Number of Monte Carlo samples
        min_chunk=20.0  # Minimum fragment size to consider
    )

    # Estimate MA
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da")

    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0  # Set to 1 to see detailed calculations
    )

    print(f"\nResults:")
    print(f"  MA estimate (mean): {ma_estimate.mean():.2f}")
    print(f"  MA estimate (std):  {ma_estimate.std():.2f}")
    print(f"  MA range: [{ma_estimate.min():.2f}, {ma_estimate.max():.2f}]")

    # Test 2: Complex tree
    print("\n" + "=" * 70)
    print("[Test 2] Complex fragmentation tree")
    print("-" * 70)
    complex_tree = create_complex_tree()

    print("\nTree structure:")
    print_tree(complex_tree)
    print(f"\nTree depth: {tree_depth(complex_tree)}")

    precursor_mz = 500.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da")

    ma_estimate = estimator.estimate_ma(
        tree=complex_tree,
        mw=precursor_mz,
        progress_levels=0
    )

    print(f"\nResults:")
    print(f"  MA estimate (mean): {ma_estimate.mean():.2f}")
    print(f"  MA estimate (std):  {ma_estimate.std():.2f}")
    print(f"  MA range: [{ma_estimate.min():.2f}, {ma_estimate.max():.2f}]")

    # Test 3: Element detection (isotope masses should give MA=0)
    print("\n" + "=" * 70)
    print("[Test 3] Element isotope detection")
    print("-" * 70)

    # Iron isotope mass
    iron_tree = {55.934939: {}}
    print(f"\nTesting Iron-56 (55.934939 Da)")
    ma_iron = estimator.estimate_ma(iron_tree, 55.934939, progress_levels=0)
    print(f"  MA estimate: {ma_iron.mean():.2f} (should be ~0 for pure element)")

    # Copper isotope mass
    copper_tree = {62.929599: {}}
    print(f"\nTesting Copper-63 (62.929599 Da)")
    ma_copper = estimator.estimate_ma(copper_tree, 62.929599, progress_levels=0)
    print(f"  MA estimate: {ma_copper.mean():.2f} (should be ~0 for pure element)")

    # Non-isotope mass
    random_tree = {123.456: {}}
    print(f"\nTesting non-isotope mass (123.456 Da)")
    ma_random = estimator.estimate_ma(random_tree, 123.456, progress_levels=0)
    print(f"  MA estimate: {ma_random.mean():.2f} (should be >0 for compound)")

    # Test 4: Tree operations
    test_tree_operations()

    # Test 5: Detailed calculation (with progress)
    print("\n" + "=" * 70)
    print("[Test 4] Detailed MA calculation with intermediate steps")
    print("-" * 70)

    detailed_tree = {
        300.0: {
            200.0: {
                100.0: {}
            },
            100.0: {}
        }
    }

    print("\nTree structure:")
    print_tree(detailed_tree)

    ma_detailed = estimator.estimate_ma(
        tree=detailed_tree,
        mw=300.0,
        progress_levels=3
    )

    print(f"\nFinal MA estimate: {ma_detailed.mean():.2f} ± {ma_detailed.std():.2f}")

    # Test 6: Detailed Joint calculation (with progress)
    print("\n" + "=" * 70)
    print("[Test 4] Detailed Joint MA calculation with intermediate steps")
    print("-" * 70)
    tree1 = {400.0: {200.0: {}, 100.0: {}}}
    tree2 = {500.0: {300.0: {}, 100.0: {}}}

    print("\nMeta tree (combining multiple samples):")
    meta = meta_tree([tree1, tree2], meta_parent_mz=1e6)
    print_tree(meta)

    ma_detailed = estimator.estimate_ma(
        tree=meta,
        mw=1e6,
        progress_levels=3
    )

    print(f"\nFinal MA estimate: {ma_detailed.mean():.2f} ± {ma_detailed.std():.2f}")


if __name__ == "__main__":
    main()
