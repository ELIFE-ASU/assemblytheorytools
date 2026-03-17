"""
                Simple example script for Mass Spectrometry Molecular Assembly (MA) Estimator

                This script demonstrates the use of the `assemblytheorytools` library to create
                a complex fragmentation tree, estimate molecular assembly (MA) for a given precursor
                mass-to-charge ratio (m/z), and display the results.

                References:
                Jirasek, M., (2024).
                Investigating and quantifying molecular complexity using assembly theory and spectroscopy.
                ACS Central Science, 10(5), 1054-1064.
                """

import assemblytheorytools as att
import numpy as np


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


if __name__ == "__main__":
    # Print a blank line for better readability
    print(flush=True)

    # Create a complex fragmentation tree
    simple_tree = create_complex_tree()

    # Print the tree structure to the console
    print("\nTree structure:", flush=True)
    att.rma_print_tree(simple_tree)

    # Calculate and print the depth of the tree
    print(f"\nTree depth: {att.rma_tree_depth(simple_tree)}", flush=True)

    # Create an instance of the MAEstimator class
    # Parameters:
    # - same_level: Whether to consider fragments at the same level (True).
    # - tol: Mass tolerance in Daltons (0.5 Da).
    # - n_samples: Number of Monte Carlo samples for estimation (20 samples).
    # - min_chunk: Minimum fragment size to consider (20.0 Da).
    estimator = att.MAEstimator(
        same_level=True,
        tol=0.5,
        n_samples=20,
        min_chunk=20.0
    )

    # Define the precursor m/z value for which MA will be estimated
    precursor_mz = 400.0
    print(f"\nEstimating MA for precursor: {precursor_mz:.2f} Da", flush=True)

    # Estimate the molecular assembly (MA) for the given precursor m/z
    # Parameters:
    # - tree: The fragmentation tree to analyze.
    # - mw: The precursor m/z value.
    # - progress_levels: Level of progress reporting (0 for no progress).
    ma_estimate = estimator.estimate_ma(
        tree=simple_tree,
        mw=precursor_mz,
        progress_levels=0
    )

    # Print the results of the MA estimation
    print(f"\nResults:")
    print(f"  MA estimate (mean): {np.mean(ma_estimate):.2f}", flush=True)  # Mean MA estimate
    print(f"  MA estimate (std):  {np.std(ma_estimate):.2f}", flush=True)  # Standard deviation of MA estimate
    print(f"  MA range: [{np.min(ma_estimate):.2f}, {np.max(ma_estimate):.2f}]", flush=True)  # Range of MA estimates
