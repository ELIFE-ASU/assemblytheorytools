"""
Simple example script for Mass Spectrometry Molecular Assembly (MA) Estimator

See:
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
