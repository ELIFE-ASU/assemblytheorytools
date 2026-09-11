"""Rust calculations, search options, pathway parsing and error translation."""

import networkx as nx
import pytest

import assemblytheorytools as att
from assemblytheorytools import assembly


def test_calculate_rust_ai():
    smi = "C1=CC=CC=C1"  # Benzene
    mol = att.smi_to_mol(smi)
    ai_r = att.calculate_assembly_index_rust(mol)
    ai_v5, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    assert ai_v5 == ai_r, f"Expected AI to be {ai_v5}, but got {ai_r}"


@pytest.mark.integration
@pytest.mark.slow
def test_rust_matches_default_calculator_on_random_molecules():
    """Survey the two backends over a random sample, not just known molecules.

    Both calculators implement the same definition, so a disagreement is a bug
    in one of them. Random PubChem compounds cover shapes no hand-picked
    fixture would, the bond limit keeps a hundred exact searches affordable,
    and the seed makes any failure reproducible.

    The sampler counts bonds on the molecule with hydrogens added, so a limit
    of 50 bonds leaves a smaller heavy-atom graph to search: this draw spans
    11 to 33 bonds once hydrogens are stripped, and indices 5 to 19.
    """
    _, smiles = att.sample_random_pubchem(100, seed=0, max_bonds=50)
    mols = [att.smi_to_mol(smi) for smi in smiles]

    assert len(mols) == 100
    assert max(mol.GetNumBonds() for mol in mols) <= 50

    rust = [att.calculate_assembly_index_rust(mol) for mol in mols]
    # The Rust backend always strips hydrogens, so the default calculator has
    # to search the same hydrogen-free graph, and it has to prove its minimum:
    # the bound it returns on a timeout is not a result worth comparing.
    default, _, _ = att.calculate_assembly_index_parallel(
        mols, dict(strip_hydrogen=True, exact=True)
    )

    unfinished = [smi for smi, ai in zip(smiles, default) if ai < 0]
    assert unfinished == [], f"The default calculator did not finish: {unfinished}"

    disagreements = {
        smi: (expected, found)
        for smi, expected, found in zip(smiles, default, rust)
        if expected != found
    }
    assert disagreements == {}, (
        f"Expected the Rust indices to match the default calculator, but "
        f"{len(disagreements)} differ (smiles: default, rust): {disagreements}"
    )


@pytest.mark.parametrize(
    "graph_type",
    [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
    ids=lambda cls: cls.__name__,
)
def test_rust_index_accepts_networkx_graph_types(graph_type):
    assert att.calculate_assembly_index_rust(graph_type(att.smi_to_nx("CCO"))) == 1


def test_calculate_rust_ai_errors():
    with pytest.raises(ValueError):
        att.calculate_assembly_index_rust("CCO")

    too_big = att.smi_to_mol("C" * 1200, add_hydrogens=False)
    with pytest.raises(ValueError):
        att.calculate_assembly_index_rust(too_big)


@pytest.mark.parametrize("smiles, expected", [("c1ccccc1", 3), ("CCO", 1), ("CC", 0)])
def test_rust_assembly_depth(smiles, expected):
    assert att.calculate_assembly_depth_rust(att.smi_to_nx(smiles)) == expected


def test_get_molecule_info_rust(data_dir):
    mol = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )
    info = att.get_molecule_info_rust(mol)
    counts = (
        info.count('label = "Atom'),
        info.count('label = "Single"'),
        info.count('label = "Double"'),
    )
    assert counts == (14, 9, 7), f"Expected (14, 9, 7), but got {counts}"

    with_hydrogens = att.get_molecule_info_rust(
        att.smi_to_mol("CCO", add_hydrogens=True)
    )
    assert with_hydrogens.count('label = "Atom') == 3


def test_calculate_assembly_index_rust_search(data_dir):
    mol = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )

    result = att.calculate_assembly_index_rust_search(
        mol, parallel="none", memoize="none", kernel="none"
    )

    assert (result.index, result.num_matches, result.states_searched) == (6, 466, 491)
    assert result.pathways == []
    index, num_matches, states, pathways = result
    assert (index, num_matches, states, pathways) == (6, 466, 491, [])


@pytest.mark.parametrize("smiles", ["C", "O", "[Fe+2]", "[13CH4]"])
def test_rust_index_normalizes_unsigned_underflow_for_bare_atoms(smiles):
    assert att.calculate_assembly_index_rust(att.smi_to_mol(smiles)) == 0


def test_rust_empty_graph_and_bare_atom_search_have_zero_index():
    assert att.calculate_assembly_index_rust(nx.Graph()) == 0
    assert att.calculate_assembly_index_rust_search(att.smi_to_mol("C")).index == 0


@pytest.mark.parametrize(
    "seconds, milliseconds",
    [(None, None), (0, 0), (0.0001, 1), (0.001, 1), (0.0011, 2)],
)
def test_rust_search_rounds_timeout_up_to_milliseconds(
    monkeypatch, seconds, milliseconds
):
    calls = []

    def search(mol_block, **options):
        calls.append(options)
        return 1, 0, None, []

    monkeypatch.setattr(assembly.at_rust, "index_search", search)

    result = att.calculate_assembly_index_rust_search(
        att.smi_to_nx("CCO"), timeout=seconds
    )

    assert len(calls) == 1
    assert calls[0]["timeout"] == milliseconds
    assert result == (1, 0, None, [])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"timeout": -1}, "must not be negative"),
        ({"bounds": "int"}, "not a single string"),
        ({"vo_type": "nope"}, "vo_type"),
        ({"memoize": "frags-index"}, "Invalid memoization mode"),
        ({"canonize": "nope"}, "canonization"),
        ({"parallel": "nope"}, "parallel"),
        ({"memoize": "nope"}, "memoization"),
        ({"kernel": "nope"}, "kernel"),
        ({"bounds": ["nope"]}, "bound"),
    ],
    ids=[
        "negative-timeout",
        "bare-bounds",
        "vo-type",
        "unsupported-memoization",
        "canonize",
        "parallel",
        "memoize",
        "kernel",
        "bounds",
    ],
)
def test_rust_search_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        att.calculate_assembly_index_rust_search(att.smi_to_nx("CCO"), **options)


def test_calculate_assembly_index_rust_search_unreadable_pathways(monkeypatch):
    """When the searched mol block cannot be parsed back by RDKit, pathway bond
    indices cannot be resolved to fragments, and the virtual objects would
    otherwise fall back to bare bond-set labels without any warning."""

    class FakeRust:
        @staticmethod
        def index_search(mol_block, **kwargs):
            return 1, 0, 1, ['digraph { 0 [ label = "{0}" ] }']

    monkeypatch.setattr(assembly, "at_rust", FakeRust)
    monkeypatch.setattr(assembly, "_rust_supports_pathways", lambda: True)
    monkeypatch.setattr(assembly.Chem, "MolFromMolBlock", lambda *a, **k: None)

    with pytest.raises(ValueError, match="cannot be read back"):
        att.calculate_assembly_index_rust_search(att.smi_to_nx("CCO"), max_pathways=1)


def test_calculate_assembly_index_rust_search_pathways():
    graph = att.smi_to_nx("c1ccccc1")

    result = att.calculate_assembly_index_rust_search(
        graph, parallel="none", max_pathways=1
    )

    assert len(result.pathways) == 1
    pathway = result.pathways[0]
    assert isinstance(pathway, nx.MultiDiGraph)

    target = [n for n in pathway.nodes if pathway.out_degree(n) == 0]
    assert len(target) == 1
    assert att.standardise_smiles(
        pathway.nodes[target[0]]["vo"], add_hydrogens=False
    ) == att.standardise_smiles("c1ccccc1", add_hydrogens=False)


def test_calculate_assembly_index_rust_search_pathway_parsing(data_dir, monkeypatch):
    """The known DOT fixture pins fragment-to-bond alignment during parsing."""
    mol = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()
    forwarded = {}

    class FakeRust:
        @staticmethod
        def index_search(mol_block, **kwargs):
            forwarded.update(kwargs)
            return 6, 466, 491, [dot]

    monkeypatch.setattr(assembly, "at_rust", FakeRust)
    monkeypatch.setattr(assembly, "_rust_supports_pathways", lambda: True)

    result = att.calculate_assembly_index_rust_search(mol, max_pathways=1)
    assert forwarded["max_pathways"] == 1

    assert len(result.pathways) == 1
    pathway = result.pathways[0]
    assert isinstance(pathway, nx.MultiDiGraph)
    assert (pathway.number_of_nodes(), pathway.number_of_edges()) == (8, 12)

    target = [n for n in pathway.nodes if pathway.out_degree(n) == 0]
    assert len(target) == 1
    assert att.standardise_smiles(
        pathway.nodes[target[0]]["vo"], add_hydrogens=False
    ) == att.standardise_smiles("c1ccc2cc3ccccc3cc2c1", add_hydrogens=False)


@pytest.mark.parametrize(
    "wrapper, backend_method",
    [
        (assembly.calculate_assembly_index_rust, "index"),
        (assembly.calculate_assembly_depth_rust, "depth"),
        (assembly.get_molecule_info_rust, "mol_info"),
    ],
)
@pytest.mark.parametrize("failure_stage", ["conversion", "backend"])
def test_rust_wrappers_translate_os_errors(
    monkeypatch, wrapper, backend_method, failure_stage
):
    error = OSError("unreadable mol block")

    def fail(*args, **kwargs):
        raise error

    if failure_stage == "conversion":
        monkeypatch.setattr(assembly, "_mol_to_molblock", fail)
    else:
        monkeypatch.setattr(assembly.at_rust, backend_method, fail)

    with pytest.raises(ValueError) as raised:
        wrapper(assembly.Chem.MolFromSmiles("CCO"))

    assert str(raised.value) == (
        "The Rust backend could not read this molecule: unreadable mol block"
    )
    assert raised.value.__cause__ is error


def test_rust_search_reports_unavailable_pathway_support(monkeypatch):
    monkeypatch.setattr(assembly, "_rust_supports_pathways", lambda: False)
    with pytest.raises(NotImplementedError, match="cannot reconstruct"):
        att.calculate_assembly_index_rust_search(att.smi_to_nx("CCO"), max_pathways=1)
