import pickle
from pathlib import Path
import numpy as np
import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Initialize the MAEstimator with specific parameters:
    # - same_level=True: Indicates that the estimation considers the same level in the tree.
    # - tol=3e-3: Sets the tolerance for the estimation process.
    estimator = att.MAEstimator(same_level=True, tol=3e-3)

    # Define the path to the directory containing the pickle files
    here = Path('./recursive_ma')

    # Load and parse pickle files into a dictionary, using the level as the key.
    # Each pickle file is expected to have a name format that includes the level.
    pickle_files = {
        int(f.name.split("_")[1][2]): pickle.load(f.open("rb"))
        for f in sorted(here.glob("*.pkl"))
    }

    # Rename columns in the data for consistency.
    # The columns are renamed to "mz" (mass-to-charge ratio) and "intensity".
    for level, data in pickle_files.items():
        data.rename(columns={f"ms{level}_mz": "mz",
                             f"ms{level}_intensity": "intensity"},
                    inplace=True)

    # Build a fragmentation tree from the processed data.
    # The tree is constructed up to a maximum level of 3.
    test_data = att.build_tree(pickle_files, max_level=3)

    # Extract the molecular weight (MW) from the tree.
    # The MW is the first key in the tree structure.
    mw = next(iter(test_data.keys()))

    # Estimate the molecular assembly (MA) for the tree.
    # The MA is calculated as the mean of the estimated values.
    ma = np.mean(estimator.estimate_ma(test_data, mw))

    # Print the molecular weight (MW) and molecular assembly (MA) to the console.
    print(mw, ma, flush=True)
    print(f"MW: {mw}", flush=True)
    print(f"MA: {ma}", flush=True)
