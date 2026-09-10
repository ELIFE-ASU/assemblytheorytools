"""Atomic conversions, calculator configuration, and simulation wrappers."""

import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import write
from ase.units import Hartree, Rydberg
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import tools_atoms as atoms_tools


def test_smiles_to_atoms_adds_hydrogens():
    atoms = att.smiles_to_atoms("c1ccccc1")

    assert atoms.get_chemical_formula() == "C6H6"
    assert len(atoms) == 12
    assert np.isfinite(atoms.positions).all()


def test_atoms_to_smiles_retains_explicit_hydrogens():
    assert att.atoms_to_smiles(molecule("H2O")) == "[H]O[H]"


@pytest.mark.parametrize(
    "smiles",
    [
        pytest.param("[H]O[H]", id="water"),
        pytest.param(
            "[H]C1([H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])C1([H])[H]",
            id="cyclohexane",
        ),
    ],
)
@pytest.mark.parametrize("via_graph", [False, True], ids=["atoms", "atoms-graph"])
def test_smiles_atoms_roundtrip(smiles, via_graph):
    atoms = att.smiles_to_atoms(smiles)
    if via_graph:
        atoms = att.nx_to_atoms(att.atoms_to_nx(atoms))

    assert att.atoms_to_smiles(atoms) == smiles


def test_mol_to_atoms_preserves_sdf_precision_and_metadata_behavior():
    mol = Chem.AddHs(Chem.MolFromSmiles("[13CH3][NH3+]"))
    coordinates = np.arange(mol.GetNumAtoms() * 3).reshape(-1, 3) / 7
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, position in enumerate(coordinates):
        conformer.SetAtomPosition(index, position)
    mol.AddConformer(conformer)

    atoms = atoms_tools.mol_to_atoms(mol, sanitize=False, optimise=False)

    assert atoms.get_chemical_symbols() == [atom.GetSymbol() for atom in mol.GetAtoms()]
    np.testing.assert_allclose(atoms.positions, coordinates.round(4))
    np.testing.assert_array_equal(atoms.get_initial_charges(), np.zeros(len(atoms)))
    assert atoms.get_masses()[0] == Atoms("C").get_masses()[0]
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coordinates)
    assert mol.GetAtomWithIdx(0).GetIsotope() == 13
    assert Chem.GetFormalCharge(mol) == 1


def test_mol_to_atoms_without_conformer_generates_planar_coordinates():
    mol = Chem.MolFromSmiles("CCC")
    atoms = atoms_tools.mol_to_atoms(mol, sanitize=False, optimise=False)
    assert atoms.get_chemical_symbols() == ["C", "C", "C"]
    assert atoms.get_distance(0, 1) == pytest.approx(1.5, abs=1e-4)
    assert atoms.get_distance(1, 2) == pytest.approx(1.5, abs=1e-4)
    np.testing.assert_array_equal(atoms.positions[:, 2], np.zeros(3))
    assert mol.GetNumConformers() == 0


def test_atoms_to_mol_warns_for_periodic_input_and_continues():
    water = molecule("H2O")
    water.cell = [10, 10, 10]
    water.pbc = True

    with pytest.warns(UserWarning, match="periodic boundary conditions") as record:
        mol = att.atoms_to_mol(water)
    assert mol.GetNumAtoms() == 3
    assert all("cell_to_nx" in str(warning.message) for warning in record)
    with pytest.warns(UserWarning, match="periodic boundary conditions"):
        assert att.atoms_to_nx(water).number_of_nodes() == 3

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert att.atoms_to_mol(molecule("H2O")).GetNumAtoms() == 3


def test_atoms_to_mol_keeps_coordinates_and_uses_supplied_charge():
    atoms = Atoms("OH", positions=[[0.123456, 0, 0], [0.123456, 0, 0.97]])
    mol = atoms_tools.atoms_to_mol(atoms, sanitize=False, charge=-1)
    assert Chem.GetFormalCharge(mol) == -1
    assert mol.GetNumAtoms() == 2
    assert mol.GetNumBonds() == 1
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), atoms.positions)


@pytest.mark.parametrize("smiles,charge", [("O", 0), ("[OH-]", -1)])
def test_molecular_charge(smiles, charge):
    assert atoms_tools.get_charge(Chem.MolFromSmiles(smiles)) == charge


@pytest.mark.parametrize(
    ("smiles", "multiplicity"),
    [
        ("C", 1),
        ("O", 1),
        ("[O]", 3),
        ("[CH3]", 2),
        ("[CH2][CH2]", 3),
        ("[Cr]", 7),
        ("[Cu]", 2),
        ("[Mo]", 7),
        ("[Ag]", 2),
    ],
)
def test_spin_multiplicity_with_implicit_hydrogens_and_atomic_exceptions(
    smiles, multiplicity
):
    mol = Chem.MolFromSmiles(smiles)
    atom_count = mol.GetNumAtoms()
    assert atoms_tools.get_spin_multiplicity(mol) == multiplicity
    assert mol.GetNumAtoms() == atom_count


def test_spin_multiplicity_override_precedence():
    mol = Chem.MolFromSmiles("O")
    mol.SetProp("SpinMultiplicity", "3")
    assert atoms_tools.get_spin_multiplicity(mol) == 3
    mol.SetProp("spinMultiplicity", "5")
    assert atoms_tools.get_spin_multiplicity(mol) == 5


@pytest.mark.parametrize(
    ("charge", "expected"), [(None, 10), (1.5, 8), (2.5, 8), (-1.5, 12)]
)
def test_total_electrons_rounds_info_charge(charge, expected):
    atoms = molecule("H2O")
    if charge is not None:
        atoms.info["charge"] = charge
    assert atoms_tools.get_total_electrons(atoms) == expected
    assert isinstance(atoms_tools.get_total_electrons(atoms), int)


@pytest.mark.parametrize(
    ("number", "expected"),
    [(-3, -4), (-1, 1), (0, 1), (1, 1), (3, 4), (5, 4), (7.1, 8)],
)
def test_round_to_nearest_two_preserves_ties_and_zero(number, expected):
    assert atoms_tools.round_to_nearest_two(number) == expected


@pytest.mark.parametrize(
    ("calc_type", "multiplicity", "expected"),
    [
        ("DFT", 1, "r2SCAN-3c  def2-SVP"),
        ("DFT", 2, "UKS  r2SCAN-3c  def2-SVP"),
        ("MP2", 1, "DLPNO-MP2 def2-SVP def2-SVP/C"),
        ("MP2", 3, "UKS DLPNO-MP2 def2-SVP def2-SVP/C"),
        ("CCSD", 1, "DLPNO-CCSD(T) def2-SVP def2-SVP/C"),
        ("CCSD", 2, "UKS DLPNO-CCSD(T) def2-SVP def2-SVP/C"),
        ("QM/XTB2", 1, "QM/XTB2 r2SCAN-3c  def2-SVP"),
        ("QM/XTB2", 2, "UKS  QM/XTB2 r2SCAN-3c  def2-SVP"),
        ("HF", 2, "HF def2-SVP"),
    ],
)
def test_orca_method_presets(tmp_path, calc_type, multiplicity, expected):
    calc = atoms_tools.orca_calc_preset(
        orca_path="/opt/orca",
        directory=tmp_path,
        calc_type=calc_type,
        multiplicity=multiplicity,
        basis_set="def2-SVP",
    )
    assert calc.parameters["orcasimpleinput"] == expected
    assert calc.parameters["orcablocks"] == ""


@pytest.mark.parametrize("option", [True, "custom"])
def test_orca_truthy_solvent_and_dispersion_options_use_water_and_d4(tmp_path, option):
    calc = atoms_tools.orca_calc_preset(
        orca_path="/opt/orca", directory=tmp_path, f_solv=option, f_disp=option
    )
    assert calc.parameters["orcasimpleinput"] == "r2SCAN-3c D4 "
    assert 'SMDSOLVENT "WATER"' in calc.parameters["orcablocks"]


def test_orca_qmmm_replaces_extra_blocks(tmp_path):
    calc = atoms_tools.orca_calc_preset(
        orca_path="/opt/orca",
        directory=tmp_path,
        calc_type="QM/XTB2",
        atom_list=[0, 2, 5],
        n_procs=4,
        blocks_extra="%scf maxiter 200 end",
    )
    blocks = calc.parameters["orcablocks"]
    assert "%pal nprocs 4 end" in blocks
    assert "%QMMM QMATOMS {} END END" in blocks
    assert "maxiter" not in blocks


def test_orca_configuration_forwards_calculation_and_scf_options(tmp_path):
    calc = atoms_tools.orca_calc_preset(
        orca_path="/opt/orca",
        directory=tmp_path,
        charge=-1,
        multiplicity=2,
        basis_set="def2-SVP",
        n_procs=4,
        f_solv=True,
        f_disp=True,
        calc_extra="OPT",
        blocks_extra="%scf maxiter 200 end",
        scf_option="TIGHTSCF",
    )

    assert calc.profile.command == "/opt/orca"
    assert Path(calc.directory) == tmp_path
    assert calc.parameters["charge"] == -1
    assert calc.parameters["mult"] == 2
    assert calc.parameters["orcasimpleinput"] == (
        "UKS  r2SCAN-3c D4 def2-SVP TIGHTSCF OPT"
    )
    assert "%pal nprocs 4 end" in calc.parameters["orcablocks"]
    assert 'SMDSOLVENT "WATER"' in calc.parameters["orcablocks"]
    assert "%scf maxiter 200 end" in calc.parameters["orcablocks"]


@pytest.mark.parametrize(
    "temperature,pressure,expected_options",
    [
        (None, None, []),
        (298.0, None, ["Temp 298.0"]),
        (None, 1.5, ["Pressure 1.5"]),
        (298.0, 1.5, ["Temp 298.0", "Pressure 1.5"]),
    ],
)
def test_orca_frequency_block_includes_all_requested_conditions(
    temperature, pressure, expected_options
):
    block = atoms_tools._orca_freq_block(temperature, pressure)

    if expected_options:
        assert [line.strip() for line in block.strip().splitlines()] == [
            "%freq",
            *expected_options,
            "end",
        ]
    else:
        assert block is None


def test_calculator_environment_and_generated_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCA_PATH", "environment-orca")
    monkeypatch.setenv("CP2K_COMMAND", "environment-cp2k")
    monkeypatch.setattr(atoms_tools.tempfile, "mkdtemp", lambda: str(tmp_path))
    cp2k = Mock()
    monkeypatch.setattr(atoms_tools, "CP2K", cp2k)

    orca = atoms_tools.orca_calc_preset()
    atoms_tools.cp2k_calc_preset()

    assert orca.profile.command == "environment-orca"
    assert Path(orca.directory) == tmp_path / "orca"
    assert cp2k.call_args.kwargs["command"] == "environment-cp2k"
    assert cp2k.call_args.kwargs["directory"] == str(tmp_path)
    assert cp2k.call_args.kwargs["cutoff"] == 400 * Rydberg


def test_cp2k_extra_blocks_override_shallowly(monkeypatch, tmp_path):
    cp2k = Mock()
    monkeypatch.setattr(atoms_tools, "CP2K", cp2k)
    force_eval = {"CHARGE": 5, "DFT": {"CUSTOM": True}}
    blocks = {"GLOBAL": {"RUN_TYPE": "GEO_OPT"}}

    result = atoms_tools.cp2k_calc_preset(
        "cp2k", tmp_path, 600, -1, 2, "custom-basis", "custom-xc", force_eval, blocks
    )

    assert result is cp2k.return_value
    assert cp2k.call_args.kwargs == {
        "command": "cp2k",
        "directory": tmp_path,
        "cutoff": 600 * Rydberg,
        "inp": {
            "GLOBAL": {"RUN_TYPE": "GEO_OPT"},
            "FORCE_EVAL": {
                "METHOD": "Quickstep",
                "DFT": {"CUSTOM": True},
                "CHARGE": 5,
                "MULTIPLICITY": 2,
            },
        },
    }
    assert force_eval == {"CHARGE": 5, "DFT": {"CUSTOM": True}}
    assert blocks == {"GLOBAL": {"RUN_TYPE": "GEO_OPT"}}


def test_grab_value_uses_last_matching_line_and_converts_hartree(tmp_path):
    output = tmp_path / "orca.out"
    output.write_text("Energy ... -1.0 Eh\nOther ... 200 Eh\nEnergy ... -2.5 Eh\n")
    assert atoms_tools.grab_value(output, "Energy", "...") == -2.5 * Hartree
    assert atoms_tools.grab_value(output, "Missing", "...") is None
    output.write_text("Energy ... not-a-number Eh\n")
    with pytest.raises(ValueError):
        atoms_tools.grab_value(output, "Energy", "...")


@pytest.mark.parametrize("terminator", ["\n", "Conformers remaining: 2\n"])
def test_conformer_table_ignores_unmatched_rows_and_stops_at_end(tmp_path, terminator):
    output = tmp_path / "orca.out"
    output.write_text(
        "Unrelated 9 -0.1 1 2.0 100.0\n"
        "CONFORMER Energy degeneracy % total % cumulative\n"
        "-----------------------------------\n"
        "  1 -0.25 2 75.0 75.0\n"
        "not a conformer\n"
        "  2 1.50 1 25.0 100.0\n" + terminator + "  3 2.50 1 99.0 100.0\n"
    )
    expected = pd.DataFrame(
        [(1, -0.25, 75.0), (2, 1.50, 25.0)],
        columns=["Conformer", "Energy_kcal_mol", "Percent_total"],
    )
    pd.testing.assert_frame_equal(atoms_tools.extract_conformer_info(output), expected)


@pytest.mark.parametrize(
    "content", ["no table", "Conformer Energy % total\n\n1 0.0 1 100.0 100.0\n"]
)
def test_conformer_table_requires_data(tmp_path, content):
    output = tmp_path / "orca.out"
    output.write_text(content)
    with pytest.raises(ValueError, match="Could not locate ensemble table"):
        atoms_tools.extract_conformer_info(output)


@pytest.fixture
def fake_orca(monkeypatch):
    """Write small real output files when ASE requests a mocked ORCA energy."""
    calls = []

    def make_calculator(**kwargs):
        calc = SimpleNamespace(**kwargs)

        def get_potential_energy(atoms):
            directory = Path(kwargs["directory"])
            assert directory.is_dir()
            calls.append(kwargs)
            optimized = atoms.copy()
            optimized.positions += 1
            write(directory / "orca.xyz", optimized, format="xyz")
            write(
                directory / "orca.finalensemble.xyz", [atoms, optimized], format="xyz"
            )
            (directory / "orca.hess").write_text("mock Hessian")
            (directory / "orca.out").write_text(
                "Total entropy correction ... -0.2 Eh\n"
                "Final Gibbs free energy ... -3.0 Eh\n"
                "G-E(el) ... 0.3 Eh\n"
                "Free-energy (cav+disp) : -0.1 Eh\n"
                "Conformer Energy degeneracy % total % cumulative\n"
                "1 0.0 1 75.0 75.0\n"
                "2 1.5 1 25.0 100.0\n\n"
            )
            return 99.0

        calc.get_potential_energy = get_potential_energy
        return calc

    monkeypatch.setenv("ORCA_PATH", "environment-orca")
    monkeypatch.setattr(atoms_tools, "ORCA", make_calculator)
    return calls


@pytest.mark.parametrize(
    ("tight_opt", "tight_scf", "flags"),
    [(False, False, "OPT"), (True, True, "TIGHTOPT TIGHTSCF")],
)
@pytest.mark.parametrize(
    "function", [atoms_tools.optimise_atoms, atoms_tools.calculate_hessian]
)
def test_geometry_wrappers_read_outputs_and_cleanup(
    fake_orca, function, tight_opt, tight_scf, flags
):
    atoms = molecule("H2O")
    initial_positions = atoms.positions.copy()
    result = function(
        atoms, charge=-1, multiplicity=2, tight_opt=tight_opt, tight_scf=tight_scf
    )
    if function is atoms_tools.calculate_hessian:
        optimized, hessian_path = result
        assert Path(hessian_path).name == "orca.hess"
        flags += " FREQ"
    else:
        optimized = result
    np.testing.assert_allclose(optimized.positions, initial_positions + 1)
    np.testing.assert_array_equal(atoms.positions, initial_positions)
    assert len(fake_orca) == 1
    call = fake_orca[0]
    assert call["orcasimpleinput"].endswith(flags)
    assert call["profile"].command == "environment-orca"
    assert call["charge"] == -1
    assert call["mult"] == 2
    assert atoms.calc.directory == call["directory"]
    assert not Path(call["directory"]).exists()


@pytest.mark.parametrize(
    ("n_procs", "expected"),
    [(1, ""), (8, "%pal nprocs 8 end"), (10, "%pal nprocs 6 end")],
)
def test_ccsd_limits_processes_using_atoms_info_charge(fake_orca, n_procs, expected):
    atoms = molecule("H2O")
    atoms.info["charge"] = 2
    assert atoms_tools.calculate_ccsd_energy(atoms, charge=-1, n_procs=n_procs) == 99.0
    call = fake_orca[0]
    assert call["orcablocks"] == expected
    assert call["charge"] == -1
    assert call["orcasimpleinput"] == "DLPNO-CCSD(T) def2-TZVPP def2-TZVPP/C"
    assert call["profile"].command == str(Path("environment-orca").resolve())
    assert not Path(call["directory"]).exists()


@pytest.mark.parametrize(
    ("use_ccsd", "ccsd_energy", "f_solv", "expected_energy", "runs"),
    [
        (False, None, False, -3 * Hartree, 1),
        (False, 12.0, True, -3 * Hartree, 1),
        (True, 12.0, False, 12.0 + 0.3 * Hartree, 1),
        (True, 12.0, True, 12.0 + 0.2 * Hartree, 1),
        (True, None, False, 99.0 + 0.3 * Hartree, 2),
    ],
)
def test_free_energy_corrections_and_precomputed_ccsd(
    fake_orca, use_ccsd, ccsd_energy, f_solv, expected_energy, runs
):
    result = atoms_tools.calculate_free_energy(
        molecule("H2O"),
        temperature=300.0,
        pressure=2.0,
        use_ccsd=use_ccsd,
        ccsd_energy=ccsd_energy,
        f_solv=f_solv,
    )
    assert result == pytest.approx(
        (expected_energy, expected_energy + 0.2 * Hartree, -0.2 * Hartree)
    )
    assert len(fake_orca) == runs
    call = fake_orca[-1]
    assert call["orcasimpleinput"].endswith("OPT  FREQ")
    assert "Temp 300.0" in call["orcablocks"]
    assert "Pressure 2.0" in call["orcablocks"]
    assert all(not Path(item["directory"]).exists() for item in fake_orca)


def test_free_energy_skips_optimization_for_isolated_atom(fake_orca):
    atoms_tools.calculate_free_energy(Atoms("He"), tight_opt=True, tight_scf=True)
    assert fake_orca[0]["orcasimpleinput"].endswith(" TIGHTSCF FREQ")
    assert "OPT" not in fake_orca[0]["orcasimpleinput"]


def test_free_energy_failed_ccsd_stops_before_frequency_calculation(
    monkeypatch, fake_orca
):
    monkeypatch.setattr(
        atoms_tools, "calculate_ccsd_energy", lambda *args, **kwargs: None
    )
    with pytest.raises(ValueError, match="CCSD energy calculation failed"):
        atoms_tools.calculate_free_energy(molecule("H2O"), use_ccsd=True)
    assert fake_orca == []


def test_goat_reads_all_conformers_and_table(fake_orca):
    atoms = molecule("H2O")
    conformers, table = atoms_tools.calculate_goat(
        atoms, charge=-1, multiplicity=2, n_procs=4
    )
    assert len(conformers) == 2
    np.testing.assert_allclose(conformers[0].positions, atoms.positions)
    np.testing.assert_allclose(conformers[1].positions, atoms.positions + 1)
    assert table["Percent_total"].tolist() == [75.0, 25.0]
    call = fake_orca[0]
    assert call["orcasimpleinput"] == "GOAT XTB"
    assert call["orcablocks"] == "%pal nprocs 4 end"
    assert call["charge"] == -1
    assert call["mult"] == 2
    assert not Path(call["directory"]).exists()


def test_virtual_objects_preserve_order_charge_spin_and_ccsd_forwarding(monkeypatch):
    molecules = [Chem.MolFromSmiles("[OH-]"), Chem.MolFromSmiles("[CH3]")]
    calculation = Mock(side_effect=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    monkeypatch.setattr(atoms_tools, "calculate_free_energy", calculation)
    monkeypatch.setattr(
        atoms_tools,
        "mol_to_atoms",
        lambda mol: Atoms(numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()]),
    )
    assert atoms_tools.get_virtual_objects_energy(molecules, ccsd_energy=True) == [
        1.0,
        4.0,
    ]
    first, second = calculation.call_args_list
    assert (first.kwargs["charge"], first.kwargs["multiplicity"]) == (-1, 1)
    assert (second.kwargs["charge"], second.kwargs["multiplicity"]) == (0, 2)
    assert len(first.args[0]) == 2
    assert len(second.args[0]) == 4
    assert first.kwargs["ccsd_energy"] is True
    assert "use_ccsd" not in first.kwargs
    assert all(mol.GetNumAtoms() == 1 for mol in molecules)


@pytest.mark.integration
@pytest.mark.parametrize(
    "n_procs",
    [
        1,
        pytest.param(2, marks=pytest.mark.slow),
        pytest.param(4, marks=pytest.mark.slow),
    ],
)
def test_orca_water_energy_is_independent_of_processor_count(
    orca_path, tmp_path, n_procs
):
    atoms = molecule("H2O")
    atoms.calc = atoms_tools.orca_calc_preset(
        orca_path=orca_path, directory=tmp_path, calc_extra="OPT", n_procs=n_procs
    )

    assert atoms.get_potential_energy() == pytest.approx(-2077.2584652288906, abs=0.1)


@pytest.mark.integration
def test_orca_optimizes_water_geometry(orca_path):
    atoms = molecule("H2O")
    positions = atoms.positions.copy()

    optimized = atoms_tools.optimise_atoms(atoms, orca_path=orca_path)

    assert optimized.get_chemical_formula() == "H2O"
    assert np.isfinite(optimized.positions).all()
    np.testing.assert_array_equal(atoms.positions, positions)
    assert optimized.get_distance(0, 1) == pytest.approx(0.96, abs=0.1)
    assert optimized.get_distance(0, 2) == pytest.approx(0.96, abs=0.1)


@pytest.mark.integration
def test_orca_ccsd_water_energy(orca_path):
    energy = atoms_tools.calculate_ccsd_energy(molecule("H2O"), orca_path=orca_path)

    assert energy == pytest.approx(-2077.230308940521, abs=0.1)


@pytest.mark.integration
@pytest.mark.parametrize(
    "use_ccsd,solvent,expected",
    [
        (False, False, -2079.5999124087302),
        (True, False, -2077.127788955219),
        (True, True, -2077.0724431372514),
    ],
)
def test_orca_water_free_energy(orca_path, use_ccsd, solvent, expected):
    energy, enthalpy, entropy = atoms_tools.calculate_free_energy(
        molecule("H2O"), orca_path=orca_path, use_ccsd=use_ccsd, f_solv=solvent
    )

    assert energy == pytest.approx(expected, abs=0.1)
    assert np.isfinite([energy, enthalpy, entropy]).all()
