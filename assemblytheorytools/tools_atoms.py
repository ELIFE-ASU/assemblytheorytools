"""
Interoperability with ASE ``Atoms`` objects.

This module converts between SMILES strings, RDKit molecules, NetworkX graphs and
ASE ``Atoms`` objects, and provides electronic-structure helpers: charge and spin
multiplicity determination, CP2K and ORCA calculator presets, geometry
optimisation, and CCSD and free-energy evaluation for virtual objects.
"""

import os
import re
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.cp2k import CP2K
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read
from ase.units import Hartree, Rydberg
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

from .tools_graph import mol_to_nx, nx_to_mol
from .tools_mol import standardize_mol


def smiles_to_atoms(
    smiles: str, sanitize: bool = True, add_hydrogens: bool = True
) -> Atoms:
    """
    Convert a SMILES string to an ASE Atoms object.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object. Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.

    Returns
    -------
    ase.Atoms
        The ASE Atoms object representing the molecule.

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed into a valid RDKit Mol object.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if sanitize:
        mol = standardize_mol(mol)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES string: {smiles}")
    return mol_to_atoms(mol)


def mol_to_atoms(
    mol: Mol, sanitize: bool = True, add_hydrogens: bool = False, optimise: bool = True
) -> Atoms:
    """
    Convert an RDKit molecule to ASE Atoms through in-memory SDF data.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit Mol object to be converted.
    sanitize : bool, optional
        Whether to sanitize the molecule (e.g., standardize its structure).
        Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    optimise : bool, optional
        Whether to optimize the molecule's geometry using RDKit's MMFF force
        field. Default is True.

    Returns
    -------
    ase.Atoms
        The ASE Atoms object representing the molecule.

    Raises
    ------
    ValueError
        If the molecule cannot be embedded or optimized.
    """
    if sanitize:
        mol = standardize_mol(mol)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    if optimise:
        AllChem.EmbedMolecule(
            mol, maxAttempts=5000, useRandomCoords=True, randomSeed=0xF00D
        )
        AllChem.MMFFOptimizeMolecule(mol)

    # Keep SDF's coordinate precision and support for missing conformers.
    with StringIO() as sdf:
        with Chem.SDWriter(sdf) as writer:
            writer.write(mol)
        sdf.seek(0)
        return read(sdf, format="sdf")


def atoms_to_mol(
    atoms: Atoms, sanitize: bool = True, add_hydrogens: bool = False, charge: int = 0
) -> Mol:
    """
    Build an RDKit molecule from ASE atomic numbers and coordinates.

    Bonds and bond orders are inferred from the geometry and formal charge.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its
        structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The RDKit Mol object representing the molecule.

    Raises
    ------
    ValueError
        If bonds cannot be inferred from the atomic coordinates.
    """
    mol = Chem.RWMol()
    for atomic_number in atoms.get_atomic_numbers():
        mol.AddAtom(Chem.Atom(int(atomic_number)))

    conformer = Chem.Conformer(len(atoms))
    for index, (x, y, z) in enumerate(atoms.get_positions()):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    mol = mol.GetMol()

    if sanitize:
        mol = standardize_mol(mol)
        Chem.Kekulize(mol)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    return mol


def atoms_to_smiles(
    atoms: Atoms, sanitize: bool = True, add_hydrogens: bool = True, charge: int = 0
) -> str:
    """
    Convert an ASE Atoms object to a SMILES string.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its
        structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns
    -------
    str
        The SMILES string representation of the molecule.
    """
    mol = atoms_to_mol(
        atoms, sanitize=sanitize, add_hydrogens=add_hydrogens, charge=charge
    )
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=True)


def atoms_to_nx(
    atoms: Atoms, sanitize: bool = True, add_hydrogen: bool = False, charge: int = 0
) -> nx.Graph:
    """
    Convert an ASE Atoms object to a NetworkX graph.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its
        structure). Default is True.
    add_hydrogen : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns
    -------
    nx.Graph
        A NetworkX graph representation of the molecule.
    """
    mol = atoms_to_mol(
        atoms, sanitize=sanitize, add_hydrogens=add_hydrogen, charge=charge
    )
    return mol_to_nx(mol, sanitize=sanitize, add_hydrogens=add_hydrogen)


def nx_to_atoms(
    graph: nx.Graph, sanitize: bool = True, add_hydrogens: bool = False
) -> Atoms:
    """
    Convert a NetworkX graph to an ASE Atoms object.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its
        structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.

    Returns
    -------
    ase.Atoms
        The ASE Atoms object representing the molecule.
    """
    mol = nx_to_mol(graph, sanitize=sanitize, add_hydrogens=add_hydrogens)
    return mol_to_atoms(mol, sanitize=sanitize, add_hydrogens=add_hydrogens)


def get_charge(mol: Mol) -> int:
    """
    Calculate the formal charge of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    int
        The formal charge of the molecule.
    """
    return Chem.GetFormalCharge(mol)


def _calc_unpaired(capacity: int, electrons: int) -> int:
    """Return the unpaired electron count for a subshell capacity."""
    orbitals = capacity // 2
    return min(electrons, 2 * orbitals - electrons)


def _aufbau_multiplicity(z: int) -> int:
    """Return atomic spin multiplicity from Aufbau subshell filling."""
    # Subshell capacities in Aufbau filling order, from 1s through 7p.
    capacities = (2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6)
    remaining, unpaired = z, 0
    for capacity in capacities:
        if remaining == 0:
            break
        electrons = min(capacity, remaining)
        remaining -= electrons
        unpaired += _calc_unpaired(capacity, electrons)
    return unpaired + 1


# Ground-state spin multiplicities for isolated atoms, by atomic number.
# Cr, Cu, Mo, Ag are known exceptions to the Aufbau-predicted value.
_GROUND_STATE_MULTIPLICITY_EXCEPTIONS = {24: 7, 29: 2, 42: 7, 47: 2}
_GROUND_STATE_MULTIPLICITY = {
    z: _GROUND_STATE_MULTIPLICITY_EXCEPTIONS.get(z, _aufbau_multiplicity(z))
    for z in range(1, 118 + 1)
}


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Determine spin multiplicity from an override, atom, or radical count.

    The ``spinMultiplicity`` and ``SpinMultiplicity`` properties take
    precedence. Isolated atoms use ground-state multiplicities; molecules
    use the number of radical electrons plus one.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.

    Returns
    -------
    int
        The spin multiplicity of the molecule.
    """
    mol = Chem.AddHs(mol)
    for key in ("spinMultiplicity", "SpinMultiplicity"):
        if mol.HasProp(key):
            return int(mol.GetProp(key))

    if mol.GetNumAtoms() == 1:
        atomic_number = mol.GetAtomWithIdx(0).GetAtomicNum()
        return _GROUND_STATE_MULTIPLICITY.get(atomic_number, 1)

    return 1 + sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())


def cp2k_calc_preset(
    cp2k_command: Optional[str] = None,
    directory: Optional[str] = None,
    cutoff: int = 400,
    charge: int = 0,
    multiplicity: int = 1,
    basis_set: str = "DZVP-MOLOPT-SR-GTH",
    xc: str = "PBE",
    calc_extra: Optional[Dict[str, Any]] = None,
    blocks_extra: Optional[Dict[str, Any]] = None,
) -> CP2K:
    """
    Create a CP2K calculator with optional input overrides.

    Parameters
    ----------
    cp2k_command : str, optional
        Path to the CP2K executable. Defaults to the 'CP2K_COMMAND'
        environment variable or 'cp2k.popt'.
    directory : str, optional
        Directory to store calculation files. Defaults to a temporary
        directory.
    cutoff : int, optional
        Plane-wave cutoff energy in Rydberg. Default is 400.
    charge : int, optional
        Molecular charge. Default is 0.
    multiplicity : int, optional
        Spin multiplicity. Default is 1.
    basis_set : str, optional
        Basis set to use. Default is 'DZVP-MOLOPT-SR-GTH'.
    xc : str, optional
        Exchange-correlation functional. Default is 'PBE'.
    calc_extra : dict, optional
        Additional calculation options to update the FORCE_EVAL section.
        Default is None.
    blocks_extra : dict, optional
        Additional CP2K input blocks. Default is None.

    Returns
    -------
    ase.calculators.cp2k.CP2K
        Configured CP2K calculator object.
    """
    if cp2k_command is None:
        cp2k_command = os.environ.get("CP2K_COMMAND", "cp2k.popt")
    if directory is None:
        directory = tempfile.mkdtemp()

    input_data = {
        "GLOBAL": {"PROJECT": "cp2k_calc", "RUN_TYPE": "ENERGY"},
        "FORCE_EVAL": {
            "METHOD": "Quickstep",
            "DFT": {"BASIS_SET": basis_set, "XC": {"XC_FUNCTIONAL": xc}},
            "CHARGE": charge,
            "MULTIPLICITY": multiplicity,
        },
    }
    if calc_extra:
        input_data["FORCE_EVAL"].update(calc_extra)
    if blocks_extra:
        input_data.update(blocks_extra)

    return CP2K(
        cutoff=cutoff * Rydberg,
        command=cp2k_command,
        directory=directory,
        inp=input_data,
    )


def orca_calc_preset(
    orca_path: Optional[str] = None,
    directory: Optional[str] = None,
    calc_type: str = "DFT",
    xc: str = "r2SCAN-3c",
    charge: int = 0,
    multiplicity: int = 1,
    basis_set: str = "",
    n_procs: int = 1,
    f_solv: Union[bool, str] = False,
    f_disp: Union[bool, str] = False,
    atom_list: Optional[List[int]] = None,
    calc_extra: Optional[str] = None,
    blocks_extra: Optional[str] = None,
    scf_option: Optional[str] = None,
) -> ORCA:
    """
    Create an ORCA calculator with method, solvent, and input options.

    Parameters
    ----------
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from
        the environment variable 'ORCA_PATH'.
    directory : str, optional
        Directory where the calculation will be performed. Defaults to a
        temporary directory.
    calc_type : str, optional
        Type of calculation to perform (e.g., 'DFT', 'MP2', 'CCSD',
        'QM/XTB2'). Default is 'DFT'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    charge : int, optional
        Total charge of the system. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        Basis set to use for the calculation. Default is an empty string.
    n_procs : int, optional
        Number of processors to use. Default is 1.
    f_solv : bool or str, optional
        Solvent option. Truthy values, including strings, select 'WATER'.
        False or None disables solvent effects. Default is False.
    f_disp : bool or str, optional
        Dispersion option. Truthy values, including strings, select 'D4'.
        False or None disables dispersion correction. Default is False.
    atom_list : list, optional
        For 'QM/XTB2', a non-None value enables the legacy QMATOMS block.
        The block retains its empty placeholder. Default is None.
    calc_extra : str, optional
        Additional calculation options to include in the ORCA input. Default
        is None.
    blocks_extra : str, optional
        Additional ORCA input blocks, ignored for 'QM/XTB2'. Default is
        None.
    scf_option : str, optional
        Additional SCF options to include in the ORCA input. Default is
        None.

    Returns
    -------
    ase.calculators.orca.ORCA
        Configured ORCA calculator object.
    """
    if orca_path is None:
        orca_path = os.environ.get("ORCA_PATH")
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), "orca")
    profile = OrcaProfile(command=orca_path)

    processors = f"%pal nprocs {n_procs} end" if n_procs > 1 else ""
    solvent = ""
    if f_solv is not None and f_solv is not False:
        # Preserve the preset's historical treatment of truthy options.
        solvent_name = "WATER" if f_solv else f_solv
        solvent = f"""
                                              %CPCM SMD TRUE
                                                  SMDSOLVENT "{solvent_name}"
                                              END"""
    dispersion = ""
    if f_disp is not None and f_disp is not False:
        dispersion = "D4" if f_disp else f_disp
    blocks = processors + solvent + ("" if blocks_extra is None else blocks_extra)

    if calc_type == "DFT":
        simple_input = f"{xc} {dispersion} {basis_set}"
    elif calc_type == "MP2":
        simple_input = f"DLPNO-{calc_type} {basis_set} {basis_set}/C"
    elif calc_type == "CCSD":
        simple_input = f"DLPNO-{calc_type}(T) {basis_set} {basis_set}/C"
    elif calc_type == "QM/XTB2":
        simple_input = f"{calc_type} {xc} {dispersion} {basis_set}"
        # Preserve the legacy QM/MM block and its replacement of extra blocks.
        qmmm = (
            ""
            if atom_list is None
            else """
                                              %QMMM QMATOMS {} END END
                                              """
        )
        blocks = processors + solvent + qmmm
    else:
        simple_input = f"{calc_type} {basis_set}"

    if multiplicity > 1:
        if calc_type in ("DFT", "QM/XTB2"):
            simple_input = "UKS  " + simple_input
        elif calc_type in ("MP2", "CCSD"):
            simple_input = "UKS " + simple_input
    for option in (scf_option, calc_extra):
        if option is not None:
            simple_input += " " + option

    return ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=simple_input,
        orcablocks=blocks,
    )


def _resolve_orca_path(orca_path: Optional[str]) -> Optional[str]:
    """Resolve explicit executable paths, preserving environment commands."""
    if orca_path is None:
        return os.environ.get("ORCA_PATH")
    return os.path.abspath(orca_path)


def _run_orca(atoms: Atoms, **calculator_options: Any) -> float:
    """Attach a preset calculator and trigger its energy calculation."""
    atoms.calc = orca_calc_preset(**calculator_options)
    return atoms.get_potential_energy()


def optimise_atoms(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: Optional[str] = None,
    xc: str = "r2SCAN-3c",
    basis_set: str = "def2-mTZVPP",
    tight_opt: bool = False,
    tight_scf: bool = False,
    f_solv: Union[bool, str] = False,
    f_disp: Union[bool, str] = False,
    n_procs: int = 10,
) -> Atoms:
    """
    Optimize the geometry of a molecule using ORCA.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule to be optimized.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read
        it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-mTZVPP'.
    tight_opt : bool, optional
        Whether to use tight optimization parameters. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence parameters. Default is False.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False
        (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is
        False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns
    -------
    ase.Atoms
        The ASE `Atoms` object representing the optimized geometry of the
        molecule.
    """
    orca_path = _resolve_orca_path(orca_path)
    calc_extra = "TIGHTOPT" if tight_opt else "OPT"
    if tight_scf:
        calc_extra += " TIGHTSCF"

    with tempfile.TemporaryDirectory() as directory:
        _run_orca(
            atoms,
            orca_path=orca_path,
            directory=directory,
            charge=charge,
            multiplicity=multiplicity,
            xc=xc,
            basis_set=basis_set,
            n_procs=n_procs,
            f_solv=f_solv,
            f_disp=f_disp,
            calc_extra=calc_extra,
        )
        return read(os.path.join(directory, "orca.xyz"), format="xyz")


def get_total_electrons(atoms: Atoms) -> int:
    """
    Count electrons, subtracting the rounded ``atoms.info['charge']``.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.

    Returns
    -------
    int
        The total number of electrons in the molecule, corrected for its
        charge.
    """
    charge = atoms.info.get("charge", 0.0)
    return int(np.sum(atoms.get_atomic_numbers())) - int(round(charge))


def round_to_nearest_two(number: Union[int, float]) -> int:
    """
    Round to the nearest multiple of two, returning one instead of zero.

    Ties follow Python's round-to-even rule.

    Parameters
    ----------
    number : float or int
        The number to be rounded.

    Returns
    -------
    int
        The nearest multiple of 2, or 1 if result would be 0.
    """
    return round(number / 2) * 2 or 1


def calculate_ccsd_energy(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: Optional[str] = None,
    basis_set: str = "def2-TZVPP",
    n_procs: int = 1,
) -> float:
    """
    Calculate the DLPNO-CCSD(T) energy with ORCA.

    If the requested processor count exceeds the electron count, reduce it
    using ``round_to_nearest_two(total_electrons - 2)``.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read
        it from the environment variable 'ORCA_PATH'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-TZVPP'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 1.

    Returns
    -------
    float
        The CCSD energy of the molecule in eV.
    """
    orca_path = os.path.abspath(orca_path or os.getenv("ORCA_PATH", "orca"))
    total_electrons = get_total_electrons(atoms)
    if n_procs > total_electrons:
        n_procs = round_to_nearest_two(total_electrons - 2)

    with tempfile.TemporaryDirectory() as directory:
        return _run_orca(
            atoms,
            orca_path=orca_path,
            directory=directory,
            calc_type="CCSD",
            charge=charge,
            multiplicity=multiplicity,
            basis_set=basis_set,
            n_procs=n_procs,
        )


def grab_value(orca_file: str, term: str, splitter: str) -> Optional[float]:
    """
    Read the last matching ORCA output value and convert Hartree to eV.

    Parameters
    ----------
    orca_file : str
        Path to the ORCA output file.
    term : str
        The search term to look for in the file.
    splitter : str
        The string used to split the line containing the term.

    Returns
    -------
    float or None
        The extracted value converted to eV, or None if the term is not
        found.
    """
    with open(orca_file) as output:
        for line in reversed(output.readlines()):
            if term in line:
                return float(line.split(splitter)[-1].split("Eh")[0]) * Hartree
    return None


def _orca_freq_block(
    temperature: Optional[float], pressure: Optional[float]
) -> Optional[str]:
    """Build a frequency block, omitting unspecified temperature and pressure."""
    options = [
        f"    {name} {value}"
        for name, value in (("Temp", temperature), ("Pressure", pressure))
        if value is not None
    ]
    if not options:
        return None
    return "\n".join(["", "%freq", *options, "end", ""])


def calculate_free_energy(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    temperature: Optional[float] = None,
    pressure: Optional[float] = None,
    orca_path: Optional[str] = None,
    xc: str = "r2SCAN-3c",
    basis_set: str = "def2-mTZVPP",
    tight_opt: bool = False,
    tight_scf: bool = False,
    f_solv: bool = False,
    f_disp: bool = False,
    n_procs: int = 1,
    use_ccsd: bool = False,
    ccsd_energy: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Calculate Gibbs free energy, enthalpy, and entropy correction with ORCA.

    A single atom skips geometry optimization. When ``use_ccsd`` is true,
    combine the CCSD energy with the DFT thermal and solvation corrections.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    temperature : float, optional
        Temperature in Kelvin for the calculation. Default is None.
    pressure : float, optional
        Pressure in atm for the calculation. Default is None.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from
        the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-mTZVPP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimization. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is
        False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation.
        Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 1.
    use_ccsd : bool, optional
        Whether to use CCSD energy calculations. Default is False.
    ccsd_energy : float, optional
        Precomputed CCSD energy in eV. If None, CCSD energy will be
        calculated if `use_ccsd` is True.

    Returns
    -------
    energy : float
        The Gibbs free energy in eV.
    enthalpy : float
        The enthalpy in eV.
    entropy : float
        The entropy correction in eV.

    Raises
    ------
    ValueError
        If the CCSD energy calculation fails or the ORCA setup is invalid.
    """
    orca_path = os.path.abspath(orca_path or os.getenv("ORCA_PATH", "orca"))
    opt_flag = "" if len(atoms) == 1 else ("TIGHTOPT" if tight_opt else "OPT")
    scf_flag = "TIGHTSCF" if tight_scf else ""
    calc_extra = f"{opt_flag} {scf_flag} FREQ".strip()
    blocks_extra = _orca_freq_block(temperature, pressure)

    if use_ccsd and ccsd_energy is None:
        ccsd_energy = calculate_ccsd_energy(
            atoms,
            orca_path=orca_path,
            charge=charge,
            multiplicity=multiplicity,
            n_procs=n_procs,
        )
        if ccsd_energy is None:
            raise ValueError(
                "CCSD energy calculation failed. Please check the ORCA setup."
            )

    with tempfile.TemporaryDirectory() as directory:
        _run_orca(
            atoms,
            orca_path=orca_path,
            directory=directory,
            charge=charge,
            multiplicity=multiplicity,
            xc=xc,
            basis_set=basis_set,
            n_procs=n_procs,
            f_solv=f_solv,
            f_disp=f_disp,
            calc_extra=calc_extra,
            blocks_extra=blocks_extra,
        )
        orca_file = os.path.join(directory, "orca.out")
        entropy = grab_value(orca_file, "Total entropy correction", "...")
        if use_ccsd:
            correction = grab_value(orca_file, "G-E(el)", "...")
            solvation = (
                grab_value(orca_file, "Free-energy (cav+disp)", ":") if f_solv else 0.0
            )
            energy = ccsd_energy + correction + solvation
        else:
            energy = grab_value(orca_file, "Final Gibbs free energy", "...")
        return energy, energy - entropy, entropy


def list_to_str(lst: List[Any]) -> str:
    """
    Convert a list to a comma-separated string.

    Parameters
    ----------
    lst : list
        The input list to be converted.

    Returns
    -------
    str
        A comma-separated string representation of the list items.
    """
    return ", ".join(str(item) for item in lst)


def calculate_hessian(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: Optional[str] = None,
    xc: str = "r2SCAN-3c",
    basis_set: str = "def2-mTZVPP",
    tight_opt: bool = False,
    tight_scf: bool = False,
    f_solv: Union[bool, str] = False,
    f_disp: Union[bool, str] = False,
    n_procs: int = 1,
) -> Tuple[Atoms, str]:
    """
    Perform a Hessian matrix calculation for a molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read
        it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-mTZVPP'.
    tight_opt : bool, optional
        Whether to use tight optimization parameters. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence parameters. Default is False.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False
        (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is
        False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 1.

    Returns
    -------
    optimized_atoms : ase.Atoms
        The ASE `Atoms` object representing the optimized geometry.
    hessian_file : str
        Temporary Hessian filename. Its directory is removed on return.

    Raises
    ------
    ValueError
        If the ORCA executable path is not provided and cannot be determined
        from the environment.
    """
    orca_path = _resolve_orca_path(orca_path)
    opt_flag = "TIGHTOPT" if tight_opt else "OPT"
    calc_extra = f"{opt_flag} TIGHTSCF FREQ" if tight_scf else f"{opt_flag} FREQ"

    with tempfile.TemporaryDirectory() as directory:
        _run_orca(
            atoms,
            orca_path=orca_path,
            directory=directory,
            charge=charge,
            multiplicity=multiplicity,
            xc=xc,
            basis_set=basis_set,
            n_procs=n_procs,
            f_solv=f_solv,
            f_disp=f_disp,
            calc_extra=calc_extra,
        )
        return (
            read(os.path.join(directory, "orca.xyz"), format="xyz"),
            os.path.join(directory, "orca.hess"),
        )


def extract_conformer_info(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Extract conformer information from an ORCA output file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the ORCA output file containing the ensemble table.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the following columns:
        - 'Conformer': Conformer index (int).
        - 'Energy_kcal_mol': Energy in kcal/mol (float).
        - 'Percent_total': Percentage of the total (float).

    Raises
    ------
    ValueError
        If the ensemble table cannot be located in the file.
    """
    header_pattern = re.compile(r"Conformer\s+Energy.*% total", re.I)
    row_pattern = re.compile(
        r"^\s*(?P<conformer>\d+)\s+"
        r"(?P<energy>-?\d+\.\d+)\s+"
        r"\d+\s+(?P<ptotal>\d+\.\d+)\s+\d+\.\d+\s*?$"
    )
    rows = []
    with open(filepath, encoding="utf-8", errors="ignore") as output:
        for line in output:
            if header_pattern.search(line):
                break
        for line in output:
            if not line.strip() or line.strip().startswith("Conformers"):
                break
            match = row_pattern.match(line)
            if match:
                rows.append(
                    (
                        int(match["conformer"]),
                        float(match["energy"]),
                        float(match["ptotal"]),
                    )
                )

    if not rows:
        raise ValueError(
            "Could not locate ensemble table. Check that the file is complete."
        )
    return pd.DataFrame(rows, columns=["Conformer", "Energy_kcal_mol", "Percent_total"])


def calculate_goat(
    atoms: Atoms,
    charge: int = 0,
    multiplicity: int = 1,
    orca_path: Optional[str] = None,
    n_procs: int = 1,
) -> Tuple[List[Atoms], pd.DataFrame]:
    """
    Run ORCA GOAT and return the conformer geometries and ensemble table.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object representing the molecule to be optimized.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from
        the environment variable 'ORCA_PATH'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 1.

    Returns
    -------
    atoms : list of ase.Atoms
        List of ASE Atoms objects representing the optimized conformers.
    df : pandas.DataFrame
        DataFrame containing conformer information, including:
        - 'Conformer': Conformer index (int).
        - 'Energy_kcal_mol': Energy in kcal/mol (float).
        - 'Percent_total': Percentage of the total (float).
    """
    profile = OrcaProfile(command=_resolve_orca_path(orca_path))
    processors = f"%pal nprocs {n_procs} end" if n_procs > 1 else ""
    with tempfile.TemporaryDirectory() as directory:
        atoms.calc = ORCA(
            profile=profile,
            charge=charge,
            mult=multiplicity,
            directory=directory,
            orcasimpleinput="GOAT XTB",
            orcablocks=processors,
        )
        atoms.get_potential_energy()

        conformers = extract_conformer_info(os.path.join(directory, "orca.out"))
        geometries = read(
            os.path.join(directory, "orca.finalensemble.xyz"), format="xyz", index=":"
        )
        return geometries, conformers


def get_virtual_objects_energy(
    mol_list: List[Mol],
    orca_path: Optional[str] = None,
    xc: str = "r2SCAN-3c",
    basis_set: str = "def2-mTZVPP",
    f_solv: bool = False,
    f_disp: bool = False,
    n_procs: int = 1,
    ccsd_energy: bool = False,
) -> List[float]:
    """
    Calculate free energies for a list of RDKit molecules.

    Parameters
    ----------
    mol_list : list of rdkit.Chem.rdchem.Mol
        List of RDKit molecule objects to process.
    orca_path : str, optional
        Path to the ORCA executable. If None, reads from environment
        variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for calculations. Default is 'def2-mTZVPP'.
    f_solv : bool, optional
        Whether to include solvent effects. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections. Default is False.
    n_procs : int, optional
        Number of processors to use. Default is 1.
    ccsd_energy : bool, optional
        Forwarded as ``calculate_free_energy``'s ``ccsd_energy`` argument.
        This does not enable ``use_ccsd``. Default is False.

    Returns
    -------
    list of float
        List of calculated energies in eV for each molecule.
    """
    energies = []
    for mol in mol_list:
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        charge = get_charge(mol)
        multiplicity = get_spin_multiplicity(mol)
        atoms = mol_to_atoms(mol)
        energy, _, _ = calculate_free_energy(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            orca_path=orca_path,
            xc=xc,
            basis_set=basis_set,
            f_solv=f_solv,
            f_disp=f_disp,
            n_procs=n_procs,
            ccsd_energy=ccsd_energy,
        )
        energies.append(energy)
    return energies
