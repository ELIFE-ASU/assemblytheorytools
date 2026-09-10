"""Sampling, dataframe filtering, numerical models, and dataset downloads."""

import io

import numpy as np
import pandas as pd
import pytest

from assemblytheorytools import tools_data


@pytest.fixture
def seeded_numpy():
    """Keep deterministic sampling checks from changing other tests' RNG state."""
    state = np.random.get_state()
    np.random.seed(13)
    yield
    np.random.set_state(state)


@pytest.fixture(params=[1, 2], ids=["1d", "2d"])
def sample_data(request):
    if request.param == 1:
        return np.array([0.0, 2.0, 2.0, 7.0, 11.0])
    return np.array([[0.0, 10.0], [2.0, 4.0], [7.0, 3.0], [11.0, 9.0], [13.0, -2.0]])


def test_random_argmin_preserves_seeded_tie_choices(seeded_numpy):
    values = np.array([3, 1, 4, 1, 1])

    indices = [tools_data.random_argmin(values) for _ in range(8)]

    assert indices == [4, 1, 4, 1, 4, 4, 1, 3]
    assert all(isinstance(index, int) for index in indices)


def test_closest_index_preserves_coordinate_wise_2d_comparison(seeded_numpy):
    # The original API chooses a row by its closest coordinate, so a distant
    # row with one exact coordinate match wins over the nearest Euclidean row.
    data = np.array([[0.0, 100.0], [1.0, 1.0], [4.0, 4.0]])

    assert tools_data.get_close_random_index(data, np.array([0.0, 0.0])) == 0


@pytest.mark.parametrize(
    ("sampler", "kwargs", "expected_1d", "expected_2d"),
    [
        (tools_data.sample_boostrapping, {}, [2, 0, 2, 0, 2, 4], [2, 0, 2, 0, 2, 4]),
        (
            tools_data.sample_importance_sampling,
            {"n_bins": 3},
            [3, 1, 4, 4, 4, 1],
            [0, 4, 3, 3, 3, 2],
        ),
    ],
)
def test_discrete_sampling_preserves_seeded_values_and_indices(
    sampler, kwargs, expected_1d, expected_2d, sample_data, seeded_numpy
):
    original = sample_data.copy()

    samples, indices = sampler(sample_data, 6, **kwargs)

    expected = expected_1d if sample_data.ndim == 1 else expected_2d
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(samples, sample_data[expected])
    np.testing.assert_array_equal(sample_data, original)
    assert np.issubdtype(indices.dtype, np.integer)


def test_kde_sampling_preserves_seeded_samples_and_indices(sample_data, seeded_numpy):
    expected_samples = {
        1: [4.67366510, 9.46144865, -0.14532625, 3.47540790, 4.39247342, 12.73836671],
        2: [
            [4.06499051, 9.49331635],
            [2.92993605, 4.85064588],
            [8.59098341, 3.21981857],
            [-1.87974554, 9.95211133],
            [-5.57038179, 6.18354117],
            [14.00050095, -7.13135308],
        ],
    }
    expected_indices = {1: [3, 4, 0, 2, 1, 4], 2: [3, 1, 2, 0, 1, 4]}
    original = sample_data.copy()

    samples, indices = tools_data.sample_kde_resampling(sample_data, 6)

    np.testing.assert_allclose(samples, expected_samples[sample_data.ndim], atol=1e-8)
    np.testing.assert_array_equal(indices, expected_indices[sample_data.ndim])
    np.testing.assert_array_equal(sample_data, original)
    assert np.issubdtype(indices.dtype, np.integer)


@pytest.mark.parametrize(
    "sampler",
    [
        tools_data.sample_boostrapping,
        tools_data.sample_kde_resampling,
        tools_data.sample_importance_sampling,
    ],
)
def test_zero_samples_keep_value_shape_and_integer_indices(sampler, sample_data):
    samples, indices = sampler(sample_data, 0)

    assert samples.shape == (0, *sample_data.shape[1:])
    assert indices.shape == (0,)
    assert np.issubdtype(indices.dtype, np.integer)


@pytest.mark.parametrize(
    "sampler", [tools_data.sample_kde_resampling, tools_data.sample_importance_sampling]
)
@pytest.mark.parametrize("data", [np.array(1.0), np.ones((2, 2, 2))])
def test_distribution_sampling_rejects_unsupported_dimensions(sampler, data):
    with pytest.raises(ValueError, match="Data must be either 1D or 2D"):
        sampler(data, 1)


@pytest.mark.parametrize(
    ("function", "kwargs", "expected_values"),
    [
        (
            tools_data.filter_by_bonds,
            {"min_bonds": 4, "max_bonds": 7, "c_bonds": "metric"},
            [4, 7, 10],
        ),
        (
            tools_data.filter_by_nh_bonds,
            {"min_bonds": 0, "max_bonds": 1, "c_bonds": "metric"},
            [0, 1, 2],
        ),
        (
            tools_data.filter_by_mw,
            {"min_mw": 16.0, "max_mw": 31.0, "c_mw": "metric"},
            [16.043, 30.070, 44.097],
        ),
    ],
)
def test_filters_preserve_custom_columns_source_mutation_and_inclusive_bounds(
    function, kwargs, expected_values, serial_data_mp
):
    source = pd.DataFrame(
        {"structure": ["C", "CC", "CCC"], "metric": [-1, -1, -1]},
        index=[9, 3, 7],
    )

    result = function(source, c_smiles="structure", **kwargs)

    assert source.index.tolist() == [9, 3, 7]
    assert source["structure"].tolist() == ["C", "CC", "CCC"]
    np.testing.assert_allclose(source["metric"], expected_values)
    assert result["structure"].tolist() == ["C", "CC"]
    assert result.index.tolist() == [0, 1]
    assert result.columns.tolist() == ["structure", "metric"]


@pytest.mark.parametrize(
    ("function", "minimum", "maximum", "expected_indices", "result_column"),
    [
        (tools_data.filter_by_bonds, 1, 50, [2, 3], "n_bonds"),
        (tools_data.filter_by_nh_bonds, 1, 30, [1, 2, 3], "n_bonds"),
        (tools_data.filter_by_mw, 100, 300, [2, 3], "mw"),
    ],
)
def test_dataframe_filters(
    function,
    minimum,
    maximum,
    expected_indices,
    result_column,
    serial_data_mp,
):
    smiles = [
        "[Fe]",
        "CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C",
        "O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F",
        "Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1",
        "Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1",
    ]
    frame = pd.DataFrame({"smiles": smiles})
    bounds = (
        {"min_mw": minimum, "max_mw": maximum}
        if function is tools_data.filter_by_mw
        else {"min_bonds": minimum, "max_bonds": maximum}
    )

    result = function(frame.copy(), **bounds)

    assert result["smiles"].tolist() == [smiles[index] for index in expected_indices]
    assert result_column in result.columns
    assert result[result_column].between(minimum, maximum).all()


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        (tools_data.linear_func, (2, 3), [-1, 3, 7]),
        (tools_data.quadratic_func, (2, 3, 5), [7, 5, 19]),
        (tools_data.cubic_func, (2, 3, 5, 7), [-7, 7, 45]),
        (tools_data.quartic_func, (2, 3, 5, 7, 11), [25, 11, 101]),
        (tools_data.quintic_func, (2, 3, 5, 7, 11), [-50, 0, 202]),
    ],
)
def test_polynomial_models_accept_scalars_and_arrays(model, params, expected):
    x_values = np.array([-2.0, 0.0, 2.0])

    np.testing.assert_array_equal(model(x_values, *params), expected)
    for x, value in zip(x_values, expected):
        assert model(float(x), *params) == value


def test_fit_metrics_flatten_inputs_and_accept_lists():
    observed = [[1, 2], [3, 4]]
    predicted = [4, 3, 2, 1]

    assert tools_data.get_r(observed, predicted) == pytest.approx(-1.0)
    assert tools_data.get_r2(observed, predicted) == pytest.approx(-3.0)
    assert tools_data.get_rmsd(observed, predicted) == pytest.approx(np.sqrt(5))


@pytest.mark.parametrize("values", [[3.0, 3.0], [0.0, 1e-10]])
def test_fit_metrics_preserve_degenerate_variance_handling(values):
    assert np.isnan(tools_data.get_r(values, values))
    assert tools_data.get_r2(values, values) == 1.0
    assert tools_data.get_r2(values, [value + 1 for value in values]) == 0.0
    assert tools_data.get_rmsd(values, values) == 0.0


def test_peak_fit_uses_integer_truncation_without_clamping():
    params = [2.0, -1.9]

    assert tools_data._peaks_to_ai(0, tools_data.linear_func, params) == -1
    assert tools_data._peaks_to_ai(2, tools_data.linear_func, params) == 2
    assert (
        tools_data._func_min_helper(
            params, [0, 1, 2], [-1, 0, 2], tools_data.linear_func
        )
        == 0.0
    )


def test_peak_fit_preserves_calibrated_predictions():
    peaks = np.arange(1, 10)
    observed = 2 * peaks + 3
    initial = np.array([1.8, 2.8])

    params, predicted = tools_data.estimate_ai_from_ir_peaks(
        peaks, observed, tools_data.linear_func, initial
    )

    np.testing.assert_allclose(params, [2.1375, 2.695])
    np.testing.assert_array_equal(predicted, [4, 6, 9, 11, 13, 15, 17, 19, 21])
    np.testing.assert_array_equal(initial, [1.8, 2.8])
    assert np.issubdtype(predicted.dtype, np.integer)


def test_github_download_can_replace_an_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "data.csv"
    output.write_bytes(b"old data")
    monkeypatch.setattr(
        tools_data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"new data")
    )

    result = tools_data.get_github_file(
        "data.csv", "https://example.test/data", tmp_path, overwrite=True
    )

    assert result == output
    assert output.read_bytes() == b"new data"
    assert not output.with_suffix(".csv.part").exists()


def test_get_github_file_downloads_atomically_and_reuses_existing(
    tmp_path, monkeypatch
):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(b"downloaded data")

    monkeypatch.setattr(tools_data, "urlopen", fake_urlopen)

    path = tools_data.get_github_file(
        "dataset.csv", "https://example.test/repository/", tmp_path, timeout=7
    )
    reused = tools_data.get_github_file(
        "dataset.csv", "https://example.test/repository/", tmp_path
    )

    assert path == tmp_path / "dataset.csv"
    assert path.read_bytes() == b"downloaded data"
    assert reused == path
    assert calls == [
        ("https://example.test/repository/dataset.csv", "python-download/1.0", 7)
    ]
    assert not (tmp_path / "dataset.csv.part").exists()


def test_sample_cbrdb_filters_local_fixture(tmp_path, monkeypatch, serial_data_mp):
    dataset = tmp_path / "CBRdb_C.csv.zip"
    pd.DataFrame(
        {
            "compound_id": [1, 2, 3, 4, 5],
            "nickname": ["ethanol", "benzene", "heavy", "invalid", "methane"],
            "smiles": ["CCO", "c1ccccc1", "CCCCCCCCCCCC", "not-smiles", "C"],
            "molecular_weight": [46.1, 78.1, 400.0, 20.0, 16.0],
            "n_heavy_atoms": [3, 6, 12, 1, 1],
        }
    ).to_csv(dataset, index=False, compression="zip")
    monkeypatch.setattr(tools_data, "get_github_file", lambda *args, **kwargs: dataset)

    result = tools_data.sample_cbrdb(n_samples=2, max_mw=100, max_bonds=6)

    assert result["compound_id"].tolist() == [1, 2]
    assert result["molecular_weight"].le(100).all()
    assert result["n_bonds"].le(6).all()
    assert not dataset.exists()
