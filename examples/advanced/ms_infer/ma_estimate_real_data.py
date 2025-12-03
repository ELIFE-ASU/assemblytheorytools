import pickle
from pathlib import Path
import numpy as np
import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)
    # Initialize the MAEstimator with same_level=True and a tolerance of 3e-3
    estimator = att.MAEstimator(same_level=True, tol=3e-3)

    # Define the path to the directory containing the pickle files
    here = Path('./recursive_ma')

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
    ma = np.mean(estimator.estimate_ma(test_data, mw))

    # Print the molecular weight and molecular assembly
    print(mw, ma, flush=True)
    print(f"MW: {mw}", flush=True)
    print(f"MA: {ma}", flush=True)
