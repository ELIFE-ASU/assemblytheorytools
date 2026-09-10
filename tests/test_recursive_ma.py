"""Fragmentation tree processing and recursive molecular assembly estimates."""

import numpy as np
import pandas as pd
import pytest

from assemblytheorytools import recursive_ma as rma


@pytest.mark.parametrize(
    "tree, mass, expected",
    [
        (
            {
                400.0: {
                    300.0: {200.0: {}, 150.0: {}, 100.0: {}},
                    250.0: {150.0: {}, 100.0: {}, 80.0: {}},
                    200.0: {},
                }
            },
            400.0,
            [21.14, 1.92, 17.73, 25.38],
        ),
        (
            {
                500.0: {
                    400.0: {
                        300.0: {200.0: {100.0: {}, 80.0: {}}, 150.0: {}},
                        250.0: {150.0: {}, 100.0: {}},
                    },
                    350.0: {250.0: {}, 200.0: {100.0: {}}},
                    150.0: {},
                }
            },
            400.0,
            [25.83, 2.48, 19.26, 29.06],
        ),
        ({300.0: {200.0: {100.0: {}}, 100.0: {}}}, 300.0, [17.48, 3.35]),
        (
            {
                1_000_000.0: {
                    400.0: {200.0: {}, 100.0: {}},
                    500.0: {300.0: {}, 100.0: {}},
                }
            },
            1_000_000.0,
            [66964.27, 5103.06],
        ),
    ],
    ids=["simple-tree", "deep-tree", "shared-fragment", "meta-tree"],
)
def test_seeded_estimates_match_reference_statistics(tree, mass, expected):
    np.random.seed(0)
    estimator = rma.MAEstimator(same_level=True, tol=0.5, n_samples=20)

    samples = estimator.estimate_MA(tree, mass, progress_levels=0)

    assert samples.shape == (20,)
    statistics = [np.mean(samples), np.std(samples), np.min(samples), np.max(samples)]
    np.testing.assert_allclose(statistics[: len(expected)], expected, rtol=0, atol=0.01)


@pytest.mark.parametrize("mass", [55.934939, 62.929599], ids=["iron-56", "copper-63"])
@pytest.mark.parametrize("joint", [False, True], ids=["default", "joint"])
def test_isotope_leaves_have_zero_assembly(mass, joint):
    estimator = rma.MAEstimator(same_level=True, tol=0.5, n_samples=20)

    samples = estimator.estimate_MA({mass: {}}, mass, joint=joint, progress_levels=0)

    np.testing.assert_array_equal(samples, np.zeros(20))


def test_joint_leaf_returns_samples_from_the_default_prior():
    np.random.seed(0)
    estimator = rma.MAEstimator(same_level=True, tol=0.5, n_samples=20)
    leaf = {123.456: {}}

    joint_samples = estimator.estimate_MA(leaf, 123.456, joint=True)

    assert isinstance(joint_samples, np.ndarray)
    assert joint_samples.shape == (20,)
    assert np.mean(joint_samples) > 0
    np.testing.assert_array_equal(joint_samples, estimator.estimate_MA(leaf, 123.456))
    mean = rma.rma_estimate_ma(leaf, 123.456, joint=True, tol=0.5, n_samples=20)
    assert isinstance(mean, float)
    assert mean > 0

    parent_samples = estimator.estimate_MA(
        {400.0: {200.0: {}, 100.0: {}}}, 400.0, joint=True
    )
    assert parent_samples.shape == (20,)
    assert np.mean(parent_samples) > 0


def test_parent_estimate_is_bounded_by_joining_its_children():
    np.random.seed(0)
    estimator = rma.MAEstimator(same_level=True, tol=3e-3)
    children = {
        150.1: {72.3: None, 89.1: None},
        221.3: {72.3: None, 99.7: None},
    }

    parent_mean = np.mean(estimator.estimate_MA({371.4: children}, 371.4))
    child_means = [
        np.mean(estimator.estimate_MA({mass: fragments}, mass))
        for mass, fragments in children.items()
    ]

    assert parent_mean <= sum(child_means) + 1


def test_meta_tree_combines_samples_under_the_requested_mass():
    trees = [{400.0: {200.0: {}, 100.0: {}}}, {500.0: {300.0: {}, 100.0: {}}}]

    assert rma.rma_meta_tree(trees, meta_parent_mz=1e6) == {
        1_000_000.0: {
            400.0: {200.0: {}, 100.0: {}},
            500.0: {300.0: {}, 100.0: {}},
        }
    }


def test_tree_unification_preserves_unique_branches_and_merges_leaves():
    branch = {40.0: None}
    first = {400.0: {200.0: None, 100.0: branch}, 250.0: None}
    second = {400.0: {200.0: None, 150.0: {}}, 300.0: {}}

    assert rma.rma_unify_trees([]) == {}
    assert rma.rma_unify_trees([first]) is first
    unified = rma.rma_unify_trees([first, second])
    assert unified == {
        400.0: {200.0: {}, 100.0: branch, 150.0: {}},
        250.0: None,
        300.0: {},
    }
    assert unified[400.0][100.0] is branch
    assert first[400.0][200.0] is None
    assert rma.rma_meta_tree([first], meta_parent_mz=999.0)[999.0] is first


@pytest.mark.parametrize(
    ("tree", "depth"),
    [
        (None, 0),
        ([], 0),
        ({}, 0),
        ({100.0: None}, 1),
        ({300.0: {200.0: {100.0: {}}}, 400.0: None}, 3),
    ],
)
def test_tree_depth_accepts_terminal_values(tree, depth):
    assert rma.rma_tree_depth(tree) == depth


def test_print_tree_sorts_masses_and_includes_maximum_depth(capsys):
    tree = {100.0: None, 300.0: {200.0: {50.0: None}}}
    assert rma.rma_print_tree(tree, max_depth=1) is None
    assert capsys.readouterr().out == (
        "├─ m/z: 300.00\n  ├─ m/z: 200.00\n├─ m/z: 100.00\n"
    )


@pytest.fixture
def indexed_ms_data():
    return {
        1: pd.DataFrame({"mz": [400, 600]}, index=[10, 20]),
        2: pd.DataFrame(
            {"mz": [200, 200, 300], "parent_id": [10, 10, 20]},
            index=[30, 31, 32],
        ),
        3: pd.DataFrame(
            {"mz": [100, 80, 150], "parent_id": [30, 31, 32]},
            index=[40, 41, 42],
        ),
    }


def test_build_tree_joins_parent_indices_and_combines_duplicate_masses(indexed_ms_data):
    assert rma.rma_build_tree(indexed_ms_data, max_level=9) == {
        400.0: {200.0: {100.0: None, 80.0: None}},
        600.0: {300.0: {150.0: None}},
    }
    assert rma.rma_build_tree(indexed_ms_data, max_level=2) == {
        400.0: {200.0: None},
        600.0: {300.0: None},
    }
    assert rma.rma_build_tree(indexed_ms_data, max_level=1) == {
        400.0: {},
        600.0: {},
    }


def test_build_tree_nonroot_call_mutates_accumulator(indexed_ms_data):
    accumulator = {777.0: None}
    result = rma._build_tree(
        indexed_ms_data, level=2, acc=accumulator, parent=400.0, max_level=2
    )
    assert result is None
    assert accumulator == {777.0: None, 200.0: None}


def test_process_df_aggregates_bins_and_applies_strict_floors_before_parent_caps():
    peaks = pd.DataFrame(
        {
            "mz": [50.01, 50.02, 70, 80, 90, 199, 210, 40, 60, 70],
            "parent": [200.01, 200.02, 200, 200, 200, 200, 200, 300, 300, 300],
            "intensity": [20, 30, 25, 40, 100, 1000, 1000, 15, 20, 56],
        }
    )

    result = rma._process_df(2, peaks, 2, {2: 15}, 0.25, 1)

    assert result[["mz_bin", "parent_bin", "intensity"]].values.tolist() == [
        [600, 3000, 20],
        [500, 2000, 50],
        [700, 3000, 56],
        [900, 2000, 100],
    ]
    merged_peak = result.loc[result["mz_bin"] == 500].iloc[0]
    assert merged_peak["mz"] == pytest.approx(50.015)
    assert merged_peak["parent"] == pytest.approx(200.015)


@pytest.mark.parametrize("include_ms1", [False, True])
def test_process_preserves_input_and_synthesizes_ms1_from_total_intensity(include_ms1):
    sample = {
        2: pd.DataFrame(
            {
                "mz": [100.0, 150.0, 200.0],
                "parent": [500.129, 500.129, 600.0],
                "intensity": [40.0, 30.0, 60.0],
            },
            index=[7, 9, 11],
        )
    }
    if include_ms1:
        sample[1] = pd.DataFrame({"mz": [500.129], "intensity": [100.0]})
    original = {level: frame.copy(deep=True) for level, frame in sample.items()}

    processed = rma.rma_process(sample, n_digits=2)

    assert sample.keys() == original.keys()
    for level, frame in sample.items():
        pd.testing.assert_frame_equal(frame, original[level])
        assert processed[level] is not frame
    if include_ms1:
        assert processed[1]["parent"].tolist() == [1_000_000.0]
    else:
        pd.testing.assert_frame_equal(
            processed[1],
            pd.DataFrame(
                {
                    "mz": [500.129],
                    "intensity": [100_000.0],
                    "mz_bin": [50012],
                }
            ),
        )


def test_identify_parents_matches_nearest_indices_and_removes_orphan_descendants():
    dataset = {
        1: pd.DataFrame(
            {"mz": [200.0, 100.4, 100.0], "mz_bin": [2000, 1004, 1000]},
            index=[40, 20, 10],
        ),
        2: pd.DataFrame(
            {
                "mz": [50.0, 70.0, 60.0, 90.0, 40.0],
                "mz_bin": [500, 700, 600, 900, 400],
                "parent_bin": [1007, 1006, 1002, 1003, 2000],
            },
            index=[77, 88, 99, 66, 55],
        ),
        3: pd.DataFrame(
            {
                "mz": [20.0, 30.0, 25.0],
                "mz_bin": [200, 300, 250],
                "parent_bin": [500, 400, 600],
            },
            index=[9, 8, 7],
        ),
    }
    original = {level: frame.copy(deep=True) for level, frame in dataset.items()}

    result = rma.rma_identify_parents(dataset, mass_tol=0.2, ms_n_digits=1)

    assert result[1] is dataset[1]
    assert result[2][["mz_bin", "parent_id"]].values.tolist() == [
        [600, 10],
        [900, 20],
        [700, 20],
        [400, 40],
    ]
    assert result[2].index.tolist() == [0, 1, 2, 4]
    assert result[3][["mz_bin", "parent_id"]].values.tolist() == [[300, 4], [250, 0]]
    assert result[3].index.tolist() == [0, 2]
    for level, frame in dataset.items():
        pd.testing.assert_frame_equal(frame, original[level])


@pytest.mark.parametrize(
    ("ion", "adducts", "expected"),
    [
        (100.5, [0.0], {}),
        (100.25, [0.0], {40.0: None, 60.25: {}}),
        (101.0, [1.0], {40.0: None, 61.0: {}}),
    ],
)
def test_precursors_use_strict_tolerance_and_matched_ion_for_complements(
    ion, adducts, expected
):
    estimator = rma.MAEstimator(same_level=False, tol=0.5, adduct_masses=adducts)
    assert estimator.precursors({ion: {40.0: None}}, 100.0) == expected


def test_precursors_filter_nonpositive_and_heavier_fragments():
    estimator = rma.MAEstimator(same_level=False, adduct_masses=[0.0])
    data = {100.0: {20.0: None, 60.0: {}, 120.0: None, -10.0: None, 0.0: None}}
    assert estimator.precursors(data, 100.0) == {
        20.0: None,
        40.0: {},
        60.0: {},
        80.0: {},
    }
    assert estimator.precursors({19.0: {5.0: None}}, 19.0) == {}


def test_same_level_fallback_and_adduct_matching():
    data = {40.0: None, 60.0: {}}
    estimator = rma.MAEstimator(adduct_masses=[0.0], tol=0.5)
    assert estimator.precursors(data, 100.0) == data
    estimator.same_level = False
    assert estimator.precursors(data, 100.0) == {}
    assert estimator.same_level_precursors({40.0: None, 60.5: {}}, 100.0) == {}
    estimator.adduct_masses = [1.0]
    adduct_data = {40.0: None, 61.0: {}}
    assert estimator.same_level_precursors(adduct_data, 100.0) == adduct_data


def test_mw_estimates_cache_samples_and_only_zero_childless_isotopes(
    monkeypatch, capsys
):
    draws = []

    def samples(mw, n_samples):
        draws.append((mw, n_samples))
        return np.full(n_samples, mw)

    monkeypatch.setattr(rma, "ma_samples", samples)
    monkeypatch.setattr(rma, "ISOTOPES", {"Testium": 50.0})
    estimator = rma.MAEstimator(tol=0.5, n_samples=3)
    assert estimator.estimate_by_MW(50.0, False) is estimator.zero
    assert estimator.estimate_by_MW(50.0, False) is estimator.zero
    assert capsys.readouterr().out == "HIT: 50.0 ~ Testium (50.0)\n"
    sampled = estimator.estimate_by_MW(50.0, True)
    assert estimator.estimate_by_MW(50.0, True) is sampled
    estimator.estimate_by_MW(49.5, False)
    other = rma.MAEstimator(tol=0.5, n_samples=3)
    assert other.estimate_by_MW(50.0, True) is not sampled
    assert draws == [(50.0, 3), (49.5, 3), (50.0, 3)]


def test_joint_leaf_shares_default_prior_cache(monkeypatch):
    monkeypatch.setattr(rma, "ma_samples", lambda mw, n: np.arange(n, dtype=float))
    estimator = rma.MAEstimator(n_samples=3, same_level=False)
    tree = {300.0: None}
    assert estimator.estimate_MA(tree, 300.0, joint=True) is estimator.estimate_MA(
        tree, 300.0
    )


@pytest.mark.parametrize(
    ("tree", "mw", "costs", "joint", "expected"),
    [
        (
            {100.0: {40.0: {}, 60.0: {}}},
            100.0,
            {100.0: [3.0, 9.0], 40.0: [5.0, 0.0], 60.0: [0.0, 1.0]},
            False,
            [6.0, 2.0],
        ),
        (
            {300.0: {100.0: {40.0: {}, 60.0: {}}, 200.0: {}}},
            300.0,
            {100.0: [10.0], 200.0: [20.0], 40.0: [1.0], 60.0: [1.0]},
            True,
            [23.0],
        ),
        (
            {200.0: {120.0: {50.0: {}}, 80.0: {50.0: {}}}},
            200.0,
            {
                200.0: [100.0],
                120.0: [80.0],
                80.0: [80.0],
                70.0: [1.0],
                30.0: [1.0],
                50.0: [1.0],
            },
            False,
            [6.0],
        ),
    ],
    ids=[
        "choose-whole-sample-by-mean",
        "joint-only-at-root",
        "common-precursor-join-cost",
    ],
)
@pytest.mark.parametrize("progress_levels", [0, 3], ids=["quiet", "progress"])
def test_estimation_strategies_preserve_connection_costs(
    monkeypatch, capsys, tree, mw, costs, joint, expected, progress_levels
):
    monkeypatch.setattr(rma, "ma_samples", lambda mass, n: np.array(costs[mass]))
    monkeypatch.setattr(rma, "ISOTOPES", {})
    estimator = rma.MAEstimator(same_level=False, adduct_masses=[0.0])
    np.testing.assert_array_equal(
        estimator.estimate_MA(tree, mw, joint=joint, progress_levels=progress_levels),
        expected,
    )
    assert bool(capsys.readouterr().out) is (progress_levels > 0)


def test_public_wrappers_forward_options_and_preserve_return_types(monkeypatch):
    calls = []
    samples = np.array([2.0, 4.0])

    class Estimator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def estimate_MA(self, **kwargs):
            calls.append(kwargs)
            return samples

        def estimate_by_MW(self, **kwargs):
            calls.append(kwargs)
            return samples

        def common_precursors(self, **kwargs):
            calls.append(kwargs)
            return {40.0}

    monkeypatch.setattr(rma, "MAEstimator", Estimator)
    tree = {100.0: None}
    mean = rma.rma_estimate_ma(tree, 100.0, progress_levels=2, joint=True, n_samples=2)
    assert type(mean) is float
    assert mean == 3.0
    assert rma.rma_estimate_by_mw(100.0, has_children=True, tol=0.5) is samples
    assert rma.find_common_precursors(tree, 100.0, 80.0, False, 3, n_samples=2) == {
        40.0
    }
    assert calls == [
        {"n_samples": 2},
        {"tree": tree, "mw": 100.0, "progress_levels": 2, "joint": True},
        {"tol": 0.5},
        {"mw": 100.0, "has_children": True},
        {"same_level": False, "tol": 0.001, "n_samples": 2},
        {"data": tree, "parent1": 100.0, "parent2": 80.0},
    ]
