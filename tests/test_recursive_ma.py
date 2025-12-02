import pickle
from pathlib import Path

import assemblytheorytools as att


def test_ma_estimator():
    estimator = att.MAEstimator(same_level=True, tol=3e-3)
    here = Path.cwd()
    pickle_files = {
        int(f.name.split("_")[1][2]): pickle.load(f.open("rb"))
        for f in sorted(here.glob("*.pkl"))
    }

    for level, data in pickle_files.items():
        data.rename(columns={f"ms{level}_mz": "mz",
                             f"ms{level}_intensity": "intensity"},
                    inplace=True)

    test_data = att.build_tree(pickle_files, max_level=3)
    mw = next(iter(test_data.keys()))
    ma = estimator.estimate_ma(test_data, mw)

    print(mw, ma)

    # test mock data
    mock_data = {
        371.4: {
            150.1: {72.3: None, 89.1: None},
            221.3: {72.3: None, 99.7: None}
        }
    }

    print("Testing mock fragmentation tree…")

    parent_ma = estimator.estimate_ma(mock_data, 371.4)
    child1_ma = estimator.estimate_ma(mock_data[371.4], 150.1)
    child2_ma = estimator.estimate_ma(mock_data[371.4], 221.3)

    print("Parent MA:", parent_ma)
    print("Child1 MA:", child1_ma)
    print("Child2 MA:", child2_ma)

    assert parent_ma <= child1_ma + child2_ma + 1.0
    print("Mock data test ✨PASSED✨")
