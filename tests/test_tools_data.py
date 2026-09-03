import io
import json
import shutil
import tarfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from rdkit import Chem
from types import SimpleNamespace

import assemblytheorytools as att
from assemblytheorytools import tools_data

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CHEMOTION_IR_TAR = Path("~/Downloads/10.22000-OGoEQGlsZGElrgst.tar").expanduser()


@pytest.fixture
def serial_data_mp(monkeypatch):
    """Exercise data transformations without retesting multiprocessing."""

    def serial_map(function, values):
        return [function(value) for value in values]

    monkeypatch.setattr(tools_data, "mp_calc", serial_map)


def test_pubchem_name_and_cid_wrappers(monkeypatch):
    compound = SimpleNamespace(
        cid=2244,
        smiles=ASPIRIN,
        synonyms=["aspirin"],
        iupac_name="2-acetyloxybenzoic acid",
    )
    queries = []

    def fake_get_compounds(identifier, namespace, **kwargs):
        queries.append((identifier, namespace, kwargs))
        return [compound]

    monkeypatch.setattr(tools_data.pcp, "get_compounds", fake_get_compounds)
    monkeypatch.setattr(
        tools_data.pcp.Compound,
        "from_cid",
        staticmethod(lambda cid: compound),
    )

    assert att.pubchem_name_to_smi("Aspirin") == ASPIRIN
    name_mol = att.pubchem_name_to_mol("Aspirin", add_hydrogens=True)
    name_graph = att.pubchem_name_to_nx("Aspirin", add_hydrogens=True)
    assert Chem.MolToSmiles(Chem.RemoveHs(name_mol)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(ASPIRIN)
    )
    assert name_graph.number_of_nodes() == name_mol.GetNumAtoms()

    assert att.pubchem_id_to_smi(2244) == ASPIRIN
    id_mol = att.pubchem_id_to_mol(2244, add_hydrogens=False)
    id_graph = att.pubchem_id_to_nx(2244, add_hydrogens=False)
    assert id_mol.GetNumAtoms() == id_graph.number_of_nodes()
    assert queries == [
        ("Aspirin", "name", {}),
        ("Aspirin", "name", {}),
        ("Aspirin", "name", {}),
    ]


def test_pubchem_sampling_is_deterministic_with_mocked_batches(monkeypatch):
    def fake_get_compounds(cids, namespace):
        assert namespace == "cid"
        return [SimpleNamespace(cid=cid, smiles="CCO") for cid in cids]

    monkeypatch.setattr(tools_data.pcp, "get_compounds", fake_get_compounds)

    random_ids, random_smis = att.sample_random_pubchem(
        3, seed=7, max_cid=100, delay_s=0, batch_size=3
    )
    first_ids, first_smis = att.sample_first_pubchem(
        3, start_cid=10, max_cid=20, delay_s=0, batch_size=3
    )

    assert random_ids == [42, 20, 51]
    assert random_smis == ["CCO"] * 3
    assert first_ids == [10, 11, 12]
    assert first_smis == ["CCO"] * 3


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (att.sample_random_pubchem, {"batch_size": 0}, "batch_size"),
        (
                att.sample_first_pubchem,
                {"start_cid": 0, "max_cid": 10},
                "start_cid",
        ),
    ],
)
def test_pubchem_sampling_validates_arguments(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(1, delay_s=0, **kwargs)


def test_pubchem_smi_to_name_uses_requested_field(monkeypatch):
    compound = SimpleNamespace(
        synonyms=["lidocaine"],
        iupac_name="2-(diethylamino)-n-(2,6-dimethylphenyl)acetamide",
    )
    monkeypatch.setattr(
        tools_data.pcp,
        "get_compounds",
        lambda *args, **kwargs: [compound],
    )

    assert att.pubchem_smi_to_name("CCN", prefer="synonym") == "Lidocaine"
    assert att.pubchem_smi_to_name("CCN", prefer="iupac_name") == (
        "2-(diethylamino)-N-(2, 6-Dimethylphenyl)acetamide"
    )
    with pytest.raises(ValueError, match="Unknown prefer option"):
        att.pubchem_smi_to_name("CCN", prefer="registry_number")


def test_pubchem_smi_to_name_returns_none_on_lookup_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(tools_data.pcp, "get_compounds", fail)

    assert att.pubchem_smi_to_name("CCN") is None
    assert att.pubchem_smi_to_name("") is None


@pytest.mark.integration
def test_pubchem_live_lookup():
    """Minimal contract check against the real service."""
    assert Chem.MolToSmiles(Chem.MolFromSmiles(att.pubchem_name_to_smi("Aspirin"))) == (
        Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))
    )


@pytest.mark.parametrize(
    ("function", "minimum", "maximum", "expected_count", "result_column"),
    [
        (att.filter_by_bonds, 1, 50, 2, "n_bonds"),
        (att.filter_by_nh_bonds, 1, 30, 3, "n_bonds"),
        (att.filter_by_mw, 100, 300, 2, "mw"),
    ],
)
def test_dataframe_filters(
        function,
        minimum,
        maximum,
        expected_count,
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
        if function is att.filter_by_mw
        else {"min_bonds": minimum, "max_bonds": maximum}
    )

    result = function(frame.copy(), **bounds)

    assert len(result) == expected_count
    assert result_column in result.columns
    assert result[result_column].between(minimum, maximum).all()


def test_load_ir_jcamp_data(data_dir):
    spectrum = att.load_ir_jcamp_data(data_dir / "ir_jcamp")

    assert spectrum.ndim == 2
    assert spectrum.shape[0] > 0
    assert spectrum.shape[1] == 2


def test_find_peak_indices_in_range(data_dir):
    spectrum = att.load_ir_jcamp_data(data_dir / "ir_jcamp")
    spectrum = att.apply_sg_filter(spectrum, window_length=35, polyorder=3)

    peaks = att.find_peak_indices_in_range(
        spectrum, min_x=400, max_x=1500, prominence=0.01, distance=5
    )
    fig, _ = att.plot_ir_spectrum(spectrum, peaks=peaks)

    assert len(peaks) == 14
    assert fig.axes


def test_calc_n_peaks_in_range(data_dir):
    spectrum = att.load_ir_jcamp_data(data_dir / "ir_jcamp")

    assert att.find_n_peak_indices_in_range(spectrum, min_x=500, max_x=1500) == 19


def test_get_github_file_downloads_atomically_and_reuses_existing(
        tmp_path, monkeypatch
):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_header("User-agent"), timeout))
        return io.BytesIO(b"downloaded data")

    monkeypatch.setattr(tools_data, "urlopen", fake_urlopen)

    path = att.get_github_file(
        "dataset.csv", "https://example.test/repository/", tmp_path, timeout=7
    )
    reused = att.get_github_file(
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

    result = att.sample_cbrdb(n_samples=2, max_mw=100, max_bonds=6)

    assert len(result) == 2
    assert set(result["compound_id"]).issubset({1, 2, 5})
    assert result["molecular_weight"].le(100).all()
    assert result["n_bonds"].le(6).all()
    assert not dataset.exists()


def test_enumerate_stereoisomers_selects_shortest_available_name(monkeypatch):
    seen = []

    def fake_name(smiles, *, prefer):
        seen.append((smiles, prefer))
        return "a descriptive compound name" if len(seen) == 1 else "X"

    monkeypatch.setattr(tools_data, "pubchem_smi_to_name", fake_name)
    mol = Chem.MolFromSmiles("FC(Cl)Br")

    result = att.enumerate_stereoisomers_shortest(mol)

    assert len(seen) == 2
    assert result == seen[1][0]
    assert all(prefer == "synonym" for _, prefer in seen)


def test_enumerate_stereoisomers_falls_back_when_names_are_missing(monkeypatch):
    monkeypatch.setattr(tools_data, "pubchem_smi_to_name", lambda *args, **kwargs: None)
    mol = Chem.MolFromSmiles("FC(Cl)Br")

    assert att.enumerate_stereoisomers_shortest(mol) == Chem.MolToSmiles(
        mol, isomericSmiles=True, canonical=True
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not CHEMOTION_IR_TAR.is_file(),
    reason="Chemotion IR dataset archive is not installed",
)
def test_process_chemotion_ir_archive(tmp_path):
    frame = att.process_chemotion_ir_data(CHEMOTION_IR_TAR)

    assert {"smiles", "spectrum"}.issubset(frame.columns)
    assert not frame.empty

    spectrum = att.apply_sg_filter(frame.iloc[0]["spectrum"])
    peaks = att.find_peak_indices_in_range(spectrum)
    fig, _ = att.plot_ir_spectrum(spectrum, peaks=peaks)
    output = tmp_path / "ir-spectrum.png"
    fig.savefig(output)

    assert output.stat().st_size > 0


@pytest.fixture
def chemotion_archive(tmp_path, data_dir):
    """Build a miniature Chemotion IR archive around one real JCAMP-DX spectrum.

    Mirrors the published layout: a tar holding ``meta_data.json`` and a nested
    ``IR_data.tar.xz``. The metadata's non-``.peak.jdx`` identifier has to match the
    spectrum's filename, which is what the two halves are merged on.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy(data_dir / "ir_jcamp", staging / "SPEC1.jdx")

    inner = staging / "IR_data.tar.xz"
    with tarfile.open(inner, "w:xz") as tar:
        tar.add(staging / "SPEC1.jdx", arcname="SPEC1.jdx")

    meta = staging / "meta_data.json"
    meta.write_text(json.dumps([{
        "cano_smiles": "CCO",
        "datasets": [{"attacments": [
            {"filename": "SPEC1.peak.jdx", "identifier": "a/b/SPEC1.peak.jdx"},
            {"filename": "SPEC1.jdx", "identifier": "a/b/SPEC1.jdx"},
        ]}],
    }]))

    archive = tmp_path / "10.22000-OGoEQGlsZGElrgst.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(meta, arcname="meta_data.json")
        tar.add(inner, arcname="IR_data.tar.xz")

    shutil.rmtree(staging)
    return archive


def _cache_path(archive):
    return archive.parent / "chemotion_ir_data" / "chemotion_ir_data.pkl.gz"


def test_process_chemotion_ir_data_builds_frame(chemotion_archive, serial_data_mp):
    frame = att.process_chemotion_ir_data(chemotion_archive)

    assert list(frame.columns) == ["smiles", "name", "spectrum"]
    assert frame["smiles"].tolist() == ["CCO"]
    spectrum = frame["spectrum"].iloc[0]
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.ndim == 2 and spectrum.shape[1] == 2


def test_process_chemotion_ir_data_does_not_cache_by_default(
        chemotion_archive, serial_data_mp, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    att.process_chemotion_ir_data(chemotion_archive)

    assert not _cache_path(chemotion_archive).exists()
    assert list(cwd.iterdir()) == []


def test_process_chemotion_ir_data_cache_round_trips_spectra(chemotion_archive, serial_data_mp):
    """A saved cache must reload spectra as arrays, not as their string repr."""
    first = att.process_chemotion_ir_data(chemotion_archive, save=True)
    assert _cache_path(chemotion_archive).is_file()

    second = att.process_chemotion_ir_data(chemotion_archive)

    spectrum = second["spectrum"].iloc[0]
    assert isinstance(spectrum, np.ndarray)
    np.testing.assert_allclose(spectrum, first["spectrum"].iloc[0])
    # The protocol's next step indexes the intensity column; strings would break it.
    assert np.all(np.isfinite(spectrum.T[1]))


def test_process_chemotion_ir_data_cache_follows_the_archive(
        chemotion_archive, serial_data_mp, tmp_path, monkeypatch):
    """The cache is keyed to the archive, not to the caller's working directory."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    att.process_chemotion_ir_data(chemotion_archive, save=True)

    # Saving writes beside the archive, leaving the working directory untouched ...
    assert list(cwd.iterdir()) == []
    assert _cache_path(chemotion_archive).is_file()

    # ... so the cache is still found from a different working directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    frame = att.process_chemotion_ir_data(chemotion_archive)

    assert isinstance(frame["spectrum"].iloc[0], np.ndarray)
    assert list(elsewhere.iterdir()) == []


def test_process_chemotion_ir_data_ignores_unreadable_cache(chemotion_archive, serial_data_mp):
    """A corrupt cache is reprocessed instead of raising."""
    att.process_chemotion_ir_data(chemotion_archive, save=True)
    _cache_path(chemotion_archive).write_bytes(b"not a pickle")

    frame = att.process_chemotion_ir_data(chemotion_archive)

    assert isinstance(frame["spectrum"].iloc[0], np.ndarray)
