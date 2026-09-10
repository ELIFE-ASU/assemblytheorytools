"""PubChem lookups and sampling, with offline service and archive fixtures."""

import gzip
import io
from types import SimpleNamespace

import pandas as pd
import pytest
from rdkit import Chem

from assemblytheorytools import tools_data

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


@pytest.fixture
def cid_smiles_archive(tmp_path):
    path = tmp_path / "CID-SMILES.gz"
    with gzip.open(path, "wt") as archive:
        archive.write("1\tCCO\n2\tC.C\n3\tinvalid\n4\tCCCCCC\n5\tC\n6\tCC\n")
    return path


@pytest.fixture
def serial_weights(monkeypatch, serial_data_mp):
    weights = {"C": 16.043, "CC": 30.07, "CCO": 46.069, "CCCCCC": 86.178}
    monkeypatch.setattr(
        tools_data, "_valid_mol_mw", lambda smiles: weights.get(smiles, 0.0)
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, None),
        (" \n\t ", None),
        ("N-acetyl-L-cysteine", "N-Acetyl-l-Cysteine"),
        ("ATP HCl NaCl", "Atp Hcl Nacl"),
        ("  IRON ( iii )  ", "Iron (III)"),
        ("sec - butyl, tert-amyl; CIS-2-BUTENE", "sec-Butyl, tert-Amyl; cis-2-Butene"),
        ("İı Kſ Σσς 𝐀𝐁", "İı Kſ Σσς 𝐀𝐁"),
    ],
)
def test_common_name_normalization_preserves_existing_case_rules(name, expected):
    assert tools_data._standardize_common_name(name) == expected


def test_complex_name_lookup_preserves_preference_and_synonym_ties(monkeypatch):
    compound = SimpleNamespace(
        synonyms=["a-very-long-systematic-name-with-digits-12345", "water", "aqua"],
        iupac_name=" oxidane ",
        title=" dihydrogen monoxide ",
    )
    queries = []

    def lookup(smiles, namespace, timeout):
        queries.append((smiles, namespace, timeout))
        return [compound]

    monkeypatch.setattr(tools_data.pcp, "get_compounds", lookup)

    assert tools_data.pubchem_smi_to_name_complex(" O ", timeout=7) == "Water"
    assert tools_data.pubchem_smi_to_name_complex("O", prefer=("title", "synonym")) == (
        "Dihydrogen Monoxide"
    )
    compound.synonyms = [" ", "123456789012345678901234567890123456789"]
    assert tools_data.pubchem_smi_to_name_complex("O") == "Oxidane"
    assert tools_data.pubchem_smi_to_name_complex("O", prefer=("unknown",)) is None
    assert queries[0] == ("O", "smiles", 7)


@pytest.mark.parametrize("response", [[], RuntimeError("unavailable")])
def test_complex_name_lookup_handles_missing_compounds(monkeypatch, response):
    queries = []

    def lookup(*args, **kwargs):
        queries.append(args)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(tools_data.pcp, "get_compounds", lookup)

    assert tools_data.pubchem_smi_to_name_complex(" \t ") is None
    assert queries == []
    assert tools_data.pubchem_smi_to_name_complex("O") is None
    assert len(queries) == 1


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

    assert tools_data.pubchem_name_to_smi("Aspirin") == ASPIRIN
    name_mol = tools_data.pubchem_name_to_mol("Aspirin", add_hydrogens=True)
    name_graph = tools_data.pubchem_name_to_nx("Aspirin", add_hydrogens=True)
    assert Chem.MolToSmiles(Chem.RemoveHs(name_mol)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(ASPIRIN)
    )
    assert name_graph.number_of_nodes() == name_mol.GetNumAtoms()

    assert tools_data.pubchem_id_to_smi(2244) == ASPIRIN
    id_mol = tools_data.pubchem_id_to_mol(2244, add_hydrogens=False)
    id_graph = tools_data.pubchem_id_to_nx(2244, add_hydrogens=False)
    assert id_mol.GetNumAtoms() == id_graph.number_of_nodes()
    assert queries == [
        ("Aspirin", "name", {}),
        ("Aspirin", "name", {}),
        ("Aspirin", "name", {}),
    ]


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

    assert tools_data.pubchem_smi_to_name("CCN", prefer="synonym") == "Lidocaine"
    assert tools_data.pubchem_smi_to_name("CCN", prefer="iupac_name") == (
        "2-(diethylamino)-N-(2, 6-Dimethylphenyl)acetamide"
    )
    with pytest.raises(ValueError, match="Unknown prefer option"):
        tools_data.pubchem_smi_to_name("CCN", prefer="registry_number")


def test_pubchem_smi_to_name_returns_none_on_lookup_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(tools_data.pcp, "get_compounds", fail)

    assert tools_data.pubchem_smi_to_name("CCN") is None
    assert tools_data.pubchem_smi_to_name("") is None


@pytest.mark.integration
def test_pubchem_live_lookup():
    """Minimal contract check against the real service."""
    assert Chem.MolToSmiles(
        Chem.MolFromSmiles(tools_data.pubchem_name_to_smi("Aspirin"))
    ) == (Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN)))


@pytest.mark.parametrize(
    ("sampler", "kwargs", "expected_queries"),
    [
        (tools_data.sample_random_pubchem, {"seed": 7}, [[42, 20, 51], [84, 7, 10]]),
        (tools_data.sample_first_pubchem, {}, [[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_pubchem_samplers_continue_after_bad_batches_and_compounds(
    monkeypatch, sampler, kwargs, expected_queries
):
    queries, validations, sleeps = [], [], []

    class UnexpectedCompound:
        @property
        def cid(self):
            raise AssertionError("Do not inspect compounds after completing the sample")

    def lookup(cids, namespace):
        assert namespace == "cid"
        queries.append(cids)
        if len(queries) == 1:
            raise RuntimeError("temporary request failure")
        return [
            SimpleNamespace(smiles="valid"),
            SimpleNamespace(cid=2),
            SimpleNamespace(cid="invalid id", smiles="valid"),
            SimpleNamespace(cid=3, smiles="invalid"),
            SimpleNamespace(cid=4, smiles="error"),
            SimpleNamespace(cid="8", smiles="valid"),
            SimpleNamespace(cid=9, smiles="valid"),
            UnexpectedCompound(),
        ]

    def validate(smiles, max_bonds):
        validations.append((smiles, max_bonds))
        if smiles == "error":
            raise ValueError("invalid molecule")
        return smiles == "valid"

    monkeypatch.setattr(tools_data.pcp, "get_compounds", lookup)
    monkeypatch.setattr(tools_data, "_is_valid_sampled_smiles", validate)
    monkeypatch.setattr(tools_data.time, "sleep", sleeps.append)

    assert sampler(
        2, max_cid=100, batch_size=3, delay_s=0.1, max_bonds=7, **kwargs
    ) == (
        [8, 9],
        ["valid", "valid"],
    )
    assert queries == expected_queries
    assert validations == [
        (smiles, 7) for smiles in ("valid", "invalid", "error", "valid", "valid")
    ]
    assert sleeps == [0.1, 0.1]


def test_random_pubchem_counts_duplicate_draws_against_attempt_limit(monkeypatch):
    queries = []
    monkeypatch.setattr(
        tools_data.pcp, "get_compounds", lambda *args: queries.append(args)
    )

    with pytest.raises(
        RuntimeError, match="Only collected 0 valid molecules after 3 attempts"
    ):
        tools_data.sample_random_pubchem(
            1, seed=0, max_cid=1, max_attempts=3, batch_size=2, delay_s=0
        )
    assert queries == []


def test_sequential_pubchem_finishes_batch_before_enforcing_attempt_limit(monkeypatch):
    queries = []

    def lookup(cids, namespace):
        queries.append(cids)
        return [SimpleNamespace(cid=cid, smiles="C") for cid in cids]

    monkeypatch.setattr(tools_data.pcp, "get_compounds", lookup)
    assert tools_data.sample_first_pubchem(
        2, batch_size=3, max_attempts=1, delay_s=0
    ) == (
        [1, 2],
        ["C", "C"],
    )
    assert queries == [[1, 2, 3]]


def test_sequential_pubchem_reports_exhausted_cids(monkeypatch):
    queries = []

    def lookup(cids, namespace):
        queries.append(cids)
        return []

    monkeypatch.setattr(tools_data.pcp, "get_compounds", lookup)
    with pytest.raises(
        RuntimeError, match="Reached max_cid=4 after 2 attempts; collected 0"
    ):
        tools_data.sample_first_pubchem(
            2, start_cid=3, max_cid=4, batch_size=5, delay_s=0
        )
    assert queries == [[3, 4]]


@pytest.mark.parametrize(
    "sampler", [tools_data.sample_random_pubchem, tools_data.sample_first_pubchem]
)
def test_pubchem_empty_sample_bypasses_argument_validation(sampler):
    assert sampler(0, batch_size=0, max_cid=0) == ([], [])


def test_pubchem_sampling_is_deterministic_with_mocked_batches(monkeypatch):
    def fake_get_compounds(cids, namespace):
        assert namespace == "cid"
        return [SimpleNamespace(cid=cid, smiles="CCO") for cid in cids]

    monkeypatch.setattr(tools_data.pcp, "get_compounds", fake_get_compounds)

    random_ids, random_smis = tools_data.sample_random_pubchem(
        3, seed=7, max_cid=100, delay_s=0, batch_size=3
    )
    first_ids, first_smis = tools_data.sample_first_pubchem(
        3, start_cid=10, max_cid=20, delay_s=0, batch_size=3
    )

    assert random_ids == [42, 20, 51]
    assert random_smis == ["CCO"] * 3
    assert first_ids == [10, 11, 12]
    assert first_smis == ["CCO"] * 3


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (tools_data.sample_random_pubchem, {"batch_size": 0}, "batch_size"),
        (
            tools_data.sample_first_pubchem,
            {"start_cid": 0, "max_cid": 10},
            "start_cid",
        ),
    ],
)
def test_pubchem_sampling_validates_arguments(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(1, delay_s=0, **kwargs)


def test_download_pubchem_streams_chunks_and_preserves_existing_file(
    monkeypatch, tmp_path
):
    requests, reads = [], []

    class Response(io.BytesIO):
        def read(self, size):
            reads.append(size)
            return super().read(size)

    def urlopen(request):
        requests.append((request.full_url, request.get_header("User-agent")))
        return Response(b"cid-smiles")

    monkeypatch.setattr(tools_data, "urlopen", urlopen)
    directory = tmp_path / "downloads"
    kwargs = {
        "target_dir": directory,
        "url": "https://example.test/cids.gz",
        "chunk_size": 3,
    }
    path = tools_data.download_pubchem_cid_smiles_gz(**kwargs)
    assert path.read_bytes() == b"cid-smiles"
    assert reads == [3] * 5
    path.write_bytes(b"existing")
    assert tools_data.download_pubchem_cid_smiles_gz(**kwargs) == path
    assert path.read_bytes() == b"existing"
    tools_data.download_pubchem_cid_smiles_gz(**kwargs, overwrite=True)
    assert path.read_bytes() == b"cid-smiles"
    assert requests == [("https://example.test/cids.gz", "python-download/1.0")] * 2


def test_gzip_sampling_preserves_seed_and_filters_molecules(cid_smiles_archive):
    result = tools_data.sample_pubchem_cid_smiles_gz(
        5, gz_path=cid_smiles_archive, seed=7, max_bonds=8
    )
    assert result == ([1, 5, 6], ["CCO", "C", "CC"])


def test_gzip_weight_sampling_filters_and_reuses_cache(
    cid_smiles_archive, tmp_path, serial_weights
):
    output = tmp_path / "sample.csv.gz"
    result = tools_data.sample_pubchem_cid_smiles_gz_mw(
        3, gz_path=cid_smiles_archive, out_file=output, seed=7, max_mw=50, max_bonds=2
    )
    assert result["cid"].tolist() == [5, 1, 6]
    assert result["n_bonds"].tolist() == [0, 2, 1]
    assert result["molecular_weight"].tolist() == pytest.approx([16.043, 46.069, 30.07])
    cached = tools_data.sample_pubchem_cid_smiles_gz_mw(
        100, gz_path="missing.gz", out_file=output, max_mw=0
    )
    pd.testing.assert_frame_equal(cached, result, check_dtype=False)


def test_gzip_weight_sampling_raises_when_too_few_candidates_survive(
    cid_smiles_archive, tmp_path, serial_weights
):
    output = tmp_path / "sample.csv.gz"
    with pytest.raises(ValueError, match="larger sample than population"):
        tools_data.sample_pubchem_cid_smiles_gz_mw(
            3, gz_path=cid_smiles_archive, out_file=output, max_mw=20
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("names", "selected"),
    [
        pytest.param(["a descriptive compound name", "X"], 1, id="shortest"),
        pytest.param(["same", "size"], 0, id="first-tie"),
        pytest.param(["", "valid"], 0, id="empty-name"),
        pytest.param([None, "valid"], 1, id="missing-name"),
        pytest.param([None, None], None, id="canonical-fallback"),
    ],
)
def test_stereoisomer_selection_uses_shortest_available_name(
    names, selected, monkeypatch
):
    queries = []

    def name_for_isomer(smiles, *, prefer):
        queries.append((smiles, prefer))
        return names[len(queries) - 1]

    monkeypatch.setattr(tools_data, "pubchem_smi_to_name", name_for_isomer)
    mol = Chem.MolFromSmiles("FC(Cl)Br")

    result = tools_data.enumerate_stereoisomers_shortest(mol)

    assert len(queries) == 2
    assert all(prefer == "synonym" for _, prefer in queries)
    expected = (
        Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if selected is None
        else queries[selected][0]
    )
    assert result == expected
