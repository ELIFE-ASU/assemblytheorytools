import os
import re
import tempfile
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.cp2k import CP2K
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read
from ase.units import Hartree
from ase.units import Rydberg
from rdkit import Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

from .tools_graph import mol_to_nx, nx_to_mol
from .tools_mol import standardize_mol


def smiles_to_atoms(smiles: str,
                    sanitize: bool = True,
                    add_hydrogens: bool = True) -> Atoms:
    """
    Convert a SMILES string to an ASE Atoms object.

    This function parses a SMILES string into an RDKit Mol object, optionally sanitizes it,
    adds hydrogens, and converts it to an ASE Atoms object.

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
    Atoms
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


def mol_to_atoms(mol: Mol,
                 sanitize: bool = True,
                 add_hydrogens: bool = False,
                 optimise: bool = True) -> Atoms:
    """
    Convert an RDKit Mol object to an ASE Atoms object.

    This function processes an RDKit Mol object by optionally sanitizing it, adding hydrogens,
    and optimizing its geometry. The molecule is then written to a temporary SDF file, which
    is read back and converted into an ASE Atoms object.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit Mol object to be converted.
    sanitize : bool, optional
        Whether to sanitize the molecule (e.g., standardize its structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    optimise : bool, optional
        Whether to optimize the molecule's geometry using RDKit's MMFF force field. Default is True.

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
        # Standardize the molecule structure
        mol = standardize_mol(mol)
    if add_hydrogens:
        # Add explicit hydrogens to the molecule
        mol = Chem.AddHs(mol)
    # If optimisation is enabled, embed and optimise the molecule using RDKit
    if optimise:
        # Embed the molecule in 3D space
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True, randomSeed=0xf00d)
        # Optimize the molecule's geometry using MMFF force field
        AllChem.MMFFOptimizeMolecule(mol)

    # Create a temporary file to store the molecule in SDF format
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as temp_file:
        sdf_path = temp_file.name  # Get the path to the temporary file

    # Write the RDKit molecule to the SDF file
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    # Read the SDF file and convert it to an ASE Atoms object
    atoms = read(sdf_path)

    # Remove the temporary SDF file to clean up
    os.remove(sdf_path)

    # Return the ASE Atoms object
    return atoms


def atoms_to_mol(atoms,
                 sanitize: bool = True,
                 add_hydrogens: bool = False,
                 charge: int = 0) -> Chem.Mol:
    """
    Convert an ASE Atoms object to an RDKit Mol object.

    This function writes the ASE Atoms object to a temporary XYZ file, reads it back as an RDKit Mol object,
    determines the bonds, and optionally sanitizes and adds hydrogens to the molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its structure). Default is True.
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
        If the RDKit Mol object cannot be created from the XYZ file.
    """
    rw = Chem.RWMol()
    for z in atoms.get_atomic_numbers():
        rw.AddAtom(Chem.Atom(int(z)))

    conf = Chem.Conformer(len(atoms))
    for i, (x, y, z) in enumerate(np.asarray(atoms.get_positions())):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rw.AddConformer(conf)

    # This is the key step: RDKit infers which atoms are bonded and the bond order.
    rdDetermineBonds.DetermineBonds(rw, charge=charge)  # modifies in place

    mol = rw.GetMol()

    if sanitize:
        mol = standardize_mol(mol)  # Standardize the molecule structure
        Chem.Kekulize(mol)  # Ensure correct aromaticity
    if add_hydrogens:
        mol = Chem.AddHs(mol)  # Add explicit hydrogens to the molecule
    return mol


def atoms_to_smiles(atoms: Atoms,
                    sanitize: bool = True,
                    add_hydrogens: bool = True,
                    charge: int = 0) -> str:
    """
    Convert an ASE Atoms object to a SMILES string.

    This function converts an ASE Atoms object into an RDKit Mol object,
    and then generates a SMILES string representation of the molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns
    -------
    str
        The SMILES string representation of the molecule.
    """
    # Convert the ASE Atoms object to an RDKit Mol object
    mol = atoms_to_mol(atoms,
                       sanitize=sanitize,
                       add_hydrogens=add_hydrogens,
                       charge=charge)
    # Generate the SMILES string
    return Chem.MolToSmiles(mol,
                            isomericSmiles=True,
                            kekuleSmiles=True,
                            canonical=True)


def atoms_to_nx(atoms,
                sanitize: bool = True,
                add_hydrogen: bool = False,
                charge: int = 0) -> nx.Graph:
    """
    Convert an ASE Atoms object to a NetworkX graph.

    This function converts an ASE Atoms object into an RDKit Mol object,
    and then transforms the RDKit Mol object into a NetworkX graph.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its structure). Default is True.
    add_hydrogen : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns
    -------
    nx.Graph
        A NetworkX graph representation of the molecule.
    """
    mol = atoms_to_mol(atoms,
                       sanitize=sanitize,
                       add_hydrogens=add_hydrogen,
                       charge=charge)
    return mol_to_nx(mol,
                     sanitize=sanitize,
                     add_hydrogens=add_hydrogen)


def nx_to_atoms(graph: nx.Graph,
                sanitize: bool = True,
                add_hydrogens: bool = False) -> Atoms:
    """
    Convert a NetworkX graph to an ASE Atoms object.

    This function converts a NetworkX graph into an RDKit Mol object,
    and then transforms the RDKit Mol object into an ASE Atoms object.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its structure). Default is True.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.

    Returns
    -------
    ase.Atoms
        The ASE Atoms object representing the molecule.
    """
    mol = nx_to_mol(graph,
                    sanitize=sanitize,
                    add_hydrogens=add_hydrogens)
    return mol_to_atoms(mol,
                        sanitize=sanitize,
                        add_hydrogens=add_hydrogens)


def get_charge(mol: Mol) -> int:
    """
    Calculate the formal charge of a molecule.

    This function retrieves the formal charge of a molecule represented
    by an RDKit `Mol` object using RDKit's `GetFormalCharge` method.

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
    """
    Calculate the number of unpaired electrons in an orbital.

    This function determines the number of unpaired electrons based on the
    orbital capacity and the number of electrons present.

    Parameters
    ----------
    capacity : int
        The maximum number of electrons the orbital can hold.
    electrons : int
        The number of electrons present in the orbital.

    Returns
    -------
    int
        The number of unpaired electrons. If the number of electrons is less
        than or equal to the orbital capacity, it returns the number of electrons.
        Otherwise, it calculates the unpaired electrons based on the orbital capacity.
    """
    orbitals = capacity // 2
    return electrons if electrons <= orbitals else 2 * orbitals - electrons


def _aufbau_multiplicity(z: int) -> int:
    """
    Calculate the spin multiplicity of an atom based on the Aufbau principle.

    This function determines the spin multiplicity (2S+1) of an atom by filling
    its electron subshells according to the Aufbau principle. It calculates the
    number of unpaired electrons in each subshell and sums them to compute the
    total spin multiplicity.

    Parameters
    ----------
    z : int
        The atomic number of the atom.

    Returns
    -------
    int
        The spin multiplicity of the atom (2S+1).
    """
    subshells = [
        ('1s', 2), ('2s', 2), ('2p', 6), ('3s', 2), ('3p', 6),
        ('4s', 2), ('3d', 10), ('4p', 6), ('5s', 2), ('4d', 10),
        ('5p', 6), ('6s', 2), ('4f', 14), ('5d', 10), ('6p', 6),
        ('7s', 2), ('5f', 14), ('6d', 10), ('7p', 6),
    ]
    remaining, unpaired = z, 0
    for _, cap in subshells:
        if remaining == 0:
            break
        n = min(cap, remaining)
        remaining -= n
        unpaired += _calc_unpaired(cap, n)
    return unpaired + 1  # 2S+1


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Determine the spin multiplicity of a molecule.

    This function calculates the spin multiplicity (2S+1) of a molecule based on its structure.
    It first checks for explicit spin multiplicity properties, handles isolated atoms using
    predefined exceptions and the Aufbau principle, and finally calculates the multiplicity
    based on the radical count for molecules.

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
    # 1 – explicit override
    for key in ("spinMultiplicity", "SpinMultiplicity"):
        if mol.HasProp(key):
            return int(mol.GetProp(key))

    # 2 – isolated atom
    if mol.GetNumAtoms() == 1:
        _EXCEPTIONS = {24: 7, 29: 2, 42: 7, 47: 2}  # Cr, Cu, Mo, Ag
        _GROUND_STATE_MULTIPLICITY = {
            z: _EXCEPTIONS.get(z, _aufbau_multiplicity(z))
            for z in range(1, 118 + 1)
        }
        z = mol.GetAtomWithIdx(0).GetAtomicNum()
        return _GROUND_STATE_MULTIPLICITY.get(z, 1)

    # 3 – molecule: use radical count if present
    n_rad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return (n_rad + 1) if n_rad else 1


def cp2k_calc_preset(cp2k_command=None,
                     directory=None,
                     cutoff=400,
                     charge=0,
                     multiplicity=1,
                     basis_set='DZVP-MOLOPT-SR-GTH',
                     xc='PBE',
                     calc_extra=None,
                     blocks_extra=None):
    """
    Create and configure a CP2K calculator preset for quantum chemistry calculations.

    Parameters
    ----------
    cp2k_command : str, optional
        Path to the CP2K executable. Defaults to the 'CP2K_COMMAND' environment variable or 'cp2k.popt'.
    directory : str, optional
        Directory to store calculation files. Defaults to a temporary directory.
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
        Additional calculation options to update the FORCE_EVAL section. Default is None.
    blocks_extra : dict, optional
        Additional CP2K input blocks. Default is None.

    Returns
    -------
    CP2K
        Configured CP2K calculator object.
    """
    if cp2k_command is None:
        cp2k_command = os.environ.get('CP2K_COMMAND', 'cp2k.popt')
    if directory is None:
        directory = tempfile.mkdtemp()

    # Basic input setup
    input_data = {
        'GLOBAL': {
            'PROJECT': 'cp2k_calc',
            'RUN_TYPE': 'ENERGY'
        },
        'FORCE_EVAL': {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET': basis_set,
                'XC': {'XC_FUNCTIONAL': xc}
            },
            'CHARGE': charge,
            'MULTIPLICITY': multiplicity
        }
    }

    # Add extra input if provided
    if calc_extra:
        input_data['FORCE_EVAL'].update(calc_extra)
    if blocks_extra:
        input_data.update(blocks_extra)

    calc = CP2K(
        cutoff=cutoff * Rydberg,
        command=cp2k_command,
        directory=directory,
        inp=input_data
    )
    return calc


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='wB97X',
                     charge=0,
                     multiplicity=1,
                     basis_set='def2-SVP',
                     n_procs=10,
                     f_solv=False,
                     f_disp=False,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    """
    Create and configure an ORCA calculator preset for quantum chemistry calculations.

    Parameters
    ----------
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    directory : str, optional
        Directory where the calculation will be performed. Defaults to a temporary directory.
    calc_type : str, optional
        Type of calculation to perform (e.g., 'DFT', 'MP2', 'CCSD', 'QM/XTB2'). Default is 'DFT'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'wB97X'.
    charge : int, optional
        Total charge of the system. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the system. Default is 1.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-SVP'.
    n_procs : int, optional
        Number of processors to use. Default is 10.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    atom_list : list, optional
        List of atoms for QM/MM calculations. Only used if `calc_type` is 'QM/XTB2'. Default is None.
    calc_extra : str, optional
        Additional calculation options to include in the ORCA input. Default is None.
    blocks_extra : str, optional
        Additional ORCA input blocks to include. Default is None.
    scf_option : str, optional
        Additional SCF options to include in the ORCA input. Default is None.

    Returns
    -------
    ORCA
        Configured ORCA calculator object.
    """
    if orca_path is None:
        # Try and read the path from the environment
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        # Create a temporary directory for the calculation
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    # Create an ORCA profile with the specified command
    profile = OrcaProfile(command=orca_path)

    # Configure the number of processors
    if n_procs > 1:
        inpt_procs = '%pal nprocs {} end'.format(n_procs)
    else:
        inpt_procs = ''

    # Configure the solvent model
    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = 'WATER'
        inpt_solv = '''
                                              %CPCM SMD TRUE
                                                  SMDSOLVENT "{}"
                                              END'''.format(f_solv)
    else:
        inpt_solv = ''

    # Configure the dispersion correction
    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        if f_disp:
            f_disp = 'D4'
        inpt_disp = f_disp

    # Configure QM/MM atom list for QM/XTB2 calculations
    if atom_list is not None and calc_type == 'QM/XTB2':
        inpt_xtb = '''
                                              %QMMM QMATOMS {{}} END END
                                              '''.format(str(atom_list).strip('[').strip(']'))
    else:
        inpt_xtb = ''

    # Add any additional input blocks
    if blocks_extra is None:
        blocks_extra = ''

    # Combine all input blocks
    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

    # Configure the main calculation input based on the calculation type
    if calc_type == 'DFT':
        inpt_simple = '{} {} {}'.format(xc, inpt_disp, basis_set)
    elif calc_type == 'MP2':
        inpt_simple = 'DLPNO-{} {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'CCSD':
        inpt_simple = 'DLPNO-{}(T) {} {}/C'.format(calc_type, basis_set, basis_set)
    elif calc_type == 'QM/XTB2':
        inpt_simple = '{} {} {} {}'.format(calc_type, xc, inpt_disp, basis_set)
        inpt_blocks = inpt_procs + inpt_solv + inpt_xtb
    else:
        inpt_simple = '{} {}'.format(calc_type, basis_set)

    if multiplicity > 1:
        if calc_type == 'DFT' or calc_type == 'QM/XTB2':
            inpt_simple = 'UKS  ' + inpt_simple
        elif calc_type == 'MP2' or calc_type == 'CCSD':
            inpt_simple = 'UKS ' + inpt_simple

    # Add the SCF option if provided
    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    # Add any extra calculation options
    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    # Create and return the ORCA calculator object
    calc = ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple,
        orcablocks=inpt_blocks
    )
    return calc


def optimise_atoms(atoms,
                   charge=0,
                   multiplicity=1,
                   orca_path=None,
                   xc='r2SCAN-3c',
                   basis_set='def2-QZVP',
                   tight_opt=False,
                   tight_scf=False,
                   f_solv=False,
                   f_disp=False,
                   n_procs=10):
    """
    Optimize the geometry of a molecule using ORCA.

    This function sets up an ORCA quantum chemistry calculation to optimize the geometry
    of a molecule represented by an ASE `Atoms` object. It allows customization of various
    calculation parameters, including charge, spin multiplicity, basis set, and exchange-correlation
    functional.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule to be optimized.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight optimization parameters. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence parameters. Default is False.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns
    -------
    ase.Atoms
        The ASE `Atoms` object representing the optimized geometry of the molecule.
    """
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option}'

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, "orca.xyz")

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)
        # Assign the calculator to the molecule
        atoms.calc = calc

        # Trigger the calculation to optimise the geometry
        _ = atoms.get_potential_energy()

        # Load the optimised geometry from the ORCA output file
        return read(orca_file, format="xyz")


def get_total_electrons(atoms: Atoms) -> int:
    """
    Calculate the total number of electrons in a molecule.

    This function computes the total number of electrons in a molecule
    represented by an ASE `Atoms` object. It sums the atomic numbers (Z)
    of all atoms in the molecule and adjusts for the explicit charge
    provided in the `Atoms.info` dictionary.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.

    Returns
    -------
    int
        The total number of electrons in the molecule, corrected for its charge.
    """
    # Sum atomic numbers (Z) for every atom in the molecule
    n_electrons = int(np.sum(atoms.get_atomic_numbers()))

    # Correct for explicit total charge, if provided in the `Atoms.info` dictionary
    charge = atoms.info.get('charge', 0.0)
    n_electrons -= int(round(charge))

    return n_electrons


def round_to_nearest_two(number):
    """
    Round a number to the nearest multiple of 2.
    
    If the result would be 0, return 1 instead.

    Parameters
    ----------
    number : float or int
        The number to be rounded.

    Returns
    -------
    int
        The nearest multiple of 2, or 1 if result would be 0.
    """
    # Round to nearest multiple of 2
    result = round(number / 2) * 2

    # If result is 0, set it to 1
    if result == 0:
        result = 1

    return result


def calculate_ccsd_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          orca_path=None,
                          basis_set='def2-TZVPP',
                          n_procs=10):
    """
    Perform a CCSD energy calculation for a molecule.

    This function sets up and executes a CCSD (Coupled Cluster Singles and Doubles) energy calculation
    using the ORCA quantum chemistry package. It ensures that the number of processors used does not
    exceed the total number of electrons in the system.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-TZVPP'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns
    -------
    float
        The CCSD energy of the molecule in eV.
    """
    # If no ORCA path is provided, try to read it from the environment variable
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    # Get the total number of electrons in the system
    total_electrons = get_total_electrons(atoms)
    # Prevent too many processors being used
    if n_procs > total_electrons:
        n_procs = round_to_nearest_two(total_electrons - 2)

    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir = os.path.join(tempfile.mkdtemp())

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                calc_type='CCSD',
                                charge=charge,
                                multiplicity=multiplicity,
                                basis_set=basis_set,
                                n_procs=n_procs)
        # Attach the ORCA calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the energy calculation
        return atoms.get_potential_energy()


def grab_value(orca_file, term, splitter):
    """
    Extract a numerical value from an ORCA output file.
    
    This function searches for a specific term in an ORCA output file and extracts
    the associated numerical value, converting it from Hartree to eV.

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
        The extracted value converted to eV, or None if the term is not found.
    """
    with open(orca_file, 'r') as f:
        for line in reversed(f.readlines()):
            if term in line:
                return float(line.split(splitter)[-1].split('Eh')[0]) * Hartree
        return None


def calculate_free_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          temperature=None,
                          pressure=None,
                          orca_path=None,
                          xc='r2SCAN-3c',
                          basis_set='def2-QZVP',
                          tight_opt=False,
                          tight_scf=False,
                          f_solv=False,
                          f_disp=False,
                          n_procs=10,
                          use_ccsd=False,
                          ccsd_energy=None):
    """
    Calculate the Gibbs free energy, enthalpy, and entropy of a molecule.

    This function performs a quantum chemistry calculation using the ORCA package to compute
    the Gibbs free energy, enthalpy, and entropy of a molecule represented by an ASE `Atoms` object.
    It supports temperature and pressure adjustments, CCSD energy calculations, and various ORCA options.

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
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight geometry optimization. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence criteria. Default is False.
    f_solv : bool, optional
        Whether to include solvent effects in the calculation. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections in the calculation. Default is False.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.
    use_ccsd : bool, optional
        Whether to use CCSD energy calculations. Default is False.
    ccsd_energy : float, optional
        Precomputed CCSD energy in eV. If None, CCSD energy will be calculated if `use_ccsd` is True.

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
    # Determine the ORCA path
    orca_path = os.path.abspath(orca_path or os.getenv('ORCA_PATH', 'orca'))

    # Set optimization flags
    opt_flag = 'TIGHTOPT' if tight_opt else 'OPT'
    if len(atoms) == 1:  # Skip optimization for single atoms
        opt_flag = ''

    # Set SCF flags
    scf_flag = 'TIGHTSCF' if tight_scf else ''
    calc_extra = f'{opt_flag} {scf_flag} FREQ'.strip()

    # Set up the %thermo block for this temperature and pressure
    if temperature is not None and pressure is None:
        blocks_extra = f'''
                                  %freq
                                      Temp {temperature}
                                  end
                                  '''
    elif pressure is not None and temperature is None:
        blocks_extra = f'''
                                          %freq
                                              Pressure {pressure}
                                          end
                                          '''
    elif pressure is None and temperature is not None:
        blocks_extra = f'''
                                          %freq
                                              Temp {temperature}
                                              Pressure {pressure}
                                          end
                                          '''
    else:
        blocks_extra = None

    # Perform CCSD energy calculation if required and not provided
    if use_ccsd and ccsd_energy is None:
        ccsd_energy = calculate_ccsd_energy(atoms,
                                            orca_path=orca_path,
                                            charge=charge,
                                            multiplicity=multiplicity,
                                            n_procs=n_procs)
        if ccsd_energy is None:
            raise ValueError("CCSD energy calculation failed. Please check the ORCA setup.")

    # Create a temporary directory for the calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

        # Set up the ORCA calculator
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra,
                                blocks_extra=blocks_extra)
        atoms.calc = calc

        # Trigger the calculation
        _ = atoms.get_potential_energy()

        # Extract entropy correction
        entropy = grab_value(orca_file, 'Total entropy correction', '...')

        # Calculate Gibbs free energy based on CCSD or DFT results
        if use_ccsd:
            g_e_ele = grab_value(orca_file, 'G-E(el)', '...')
            g_e_solv = grab_value(orca_file, 'Free-energy (cav+disp)', ':') if f_solv else 0.0
            energy = ccsd_energy + g_e_ele + g_e_solv
        else:
            energy = grab_value(orca_file, 'Final Gibbs free energy', '...')

        # Return energy, enthalpy, and entropy
        return energy, energy - entropy, entropy


def list_to_str(lst):
    """
    Convert a list to a comma-separated string.
    
    This function converts all items in a list to strings and joins them
    with commas and spaces.

    Parameters
    ----------
    lst : list
        The input list to be converted.

    Returns
    -------
    str
        A comma-separated string representation of the list items.
    """
    lst = [str(item) for item in lst]
    return ', '.join(lst)


def calculate_hessian(atoms,
                      charge=0,
                      multiplicity=1,
                      orca_path=None,
                      xc='r2SCAN-3c',
                      basis_set='def2-QZVP',
                      tight_opt=False,
                      tight_scf=False,
                      f_solv=False,
                      f_disp=False,
                      n_procs=10):
    """
    Perform a Hessian matrix calculation for a molecule.

    This function sets up and executes a quantum chemistry calculation using the ORCA package
    to compute the Hessian matrix and optimized geometry of a molecule represented by an ASE `Atoms` object.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecule.
    charge : int, optional
        The total charge of the molecule. Default is 0.
    multiplicity : int, optional
        The spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'r2SCAN-3c'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-QZVP'.
    tight_opt : bool, optional
        Whether to use tight optimization parameters. Default is False.
    tight_scf : bool, optional
        Whether to use tight SCF convergence parameters. Default is False.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns
    -------
    optimized_atoms : ase.Atoms
        The ASE `Atoms` object representing the optimized geometry.
    hessian_file : str
        Path to the Hessian matrix file.

    Raises
    ------
    ValueError
        If the ORCA executable path is not provided and cannot be determined from the environment.
    """
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    if tight_opt:
        # Set up geometry optimization and frequency calculation parameters
        opt_option = 'TIGHTOPT'
    else:
        # Set up frequency calculation parameters only
        opt_option = 'OPT'

    if tight_scf:
        # Set up tight SCF convergence parameters
        calc_extra = f'{opt_option} TIGHTSCF FREQ'
    else:
        # Use default SCF convergence parameters
        calc_extra = f'{opt_option} FREQ'

    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                charge=charge,
                                multiplicity=multiplicity,
                                xc=xc,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=calc_extra)

        # Attach the ORCA calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the energy calculation
        _ = atoms.get_potential_energy()

        # Load the optimized geometry from the ORCA output file
        atoms_file = os.path.join(temp_dir, "orca.xyz")
        hessian_file = os.path.join(temp_dir, "orca.hess")
        return read(atoms_file, format="xyz"), hessian_file


def extract_conformer_info(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Extract conformer information from an ORCA output file.

    This function reads an ORCA output file and parses the ensemble table to extract
    conformer data, including conformer index, energy, and percentage of the total.

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
    # Compile a regex pattern to match a data line in the ensemble table
    line_pat = re.compile(
        r"""^\s*
            (?P<conformer>\d+)\s+          # integer index
            (?P<energy>-?\d+\.\d+)\s+      # energy in kcal/mol
            \d+\s+                         # degeneracy (ignored)
            (?P<ptotal>\d+\.\d+)\s+        # % total
            \d+\.\d+\s*?$                  # % cumulative (ignored)
        """,
        re.VERBOSE,
    )

    # Compile a regex pattern to locate the table header
    header_pat = re.compile(r"Conformer\s+Energy.*% total", re.I)

    # Initialize variables for parsing
    rows = []
    in_table = False

    # Open the file and read its contents
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            # Check for the table header to start reading data
            if not in_table and header_pat.search(line):
                in_table = True  # Start reading on the next lines
                continue

            if in_table:
                # Stop reading when the table ends
                if line.strip() == "" or line.strip().startswith("Conformers"):
                    break
                # Match a data line and extract values
                m = line_pat.match(line)
                if m:
                    rows.append(
                        (
                            int(m["conformer"]),
                            float(m["energy"]),
                            float(m["ptotal"]),
                        )
                    )

    # Raise an error if no data was found
    if not rows:
        raise ValueError(
            "Could not locate ensemble table. Check that the file is complete."
        )

    # Return the extracted data as a pandas DataFrame
    return pd.DataFrame(
        rows, columns=["Conformer", "Energy_kcal_mol", "Percent_total"]
    )


def calculate_goat(atoms,
                   charge=0,
                   multiplicity=1,
                   orca_path=None,
                   n_procs=10):
    """
    Perform a GOAT (Global Optimization of Atomic Topologies) calculation using ORCA.

    This function sets up and executes a GOAT calculation to optimize molecular conformers
    and extract conformer information from the ORCA output file.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object representing the molecule to be optimized.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, it will attempt to read from the environment variable 'ORCA_PATH'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

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
    # Determine the ORCA path
    if orca_path is None:
        # Try to read the path from the environment variable
        orca_path = os.environ.get('ORCA_PATH')
    else:
        # Convert the provided path to an absolute path
        orca_path = os.path.abspath(orca_path)

    # Create an ORCA profile with the specified command
    profile = OrcaProfile(command=orca_path)

    # Configure the number of processors
    if n_procs > 1:
        inpt_procs = '%pal nprocs {} end'.format(n_procs)
    else:
        inpt_procs = ''

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and configure the ORCA calculator object
        calc = ORCA(
            profile=profile,
            charge=charge,
            mult=multiplicity,
            directory=temp_dir,
            orcasimpleinput='GOAT XTB',
            orcablocks=inpt_procs
        )
        # Assign the calculator to the ASE Atoms object
        atoms.calc = calc

        # Trigger the calculation to optimize the geometry
        _ = atoms.get_potential_energy()

        # Define paths for the output files
        xyz_file = os.path.join(temp_dir, "orca.finalensemble.xyz")  # Path to the final ensemble file
        orca_file = os.path.join(temp_dir, "orca.out")  # Path to the ORCA output file

        # Extract conformer information from the ORCA output file
        df = extract_conformer_info(orca_file)

        # Read the optimized conformers from the ensemble file
        atoms = read(xyz_file, format="xyz", index=':')

        # Return the optimized conformers and conformer information
        return atoms, df


def get_virtual_objects_energy(mol_list,
                               orca_path=None,
                               xc='wB97X',
                               basis_set='def2-SVP',
                               f_solv=False,
                               f_disp=False,
                               n_procs=10,
                               ccsd_energy=False):
    """
    Calculate free energies for a list of RDKit molecules.
    
    This function processes a list of RDKit molecules, converting each to ASE Atoms
    objects and calculating their free energies using quantum chemistry methods.

    Parameters
    ----------
    mol_list : list of rdkit.Chem.rdchem.Mol
        List of RDKit molecule objects to process.
    orca_path : str, optional
        Path to the ORCA executable. If None, reads from environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'wB97X'.
    basis_set : str, optional
        Basis set to use for calculations. Default is 'def2-SVP'.
    f_solv : bool, optional
        Whether to include solvent effects. Default is False.
    f_disp : bool, optional
        Whether to include dispersion corrections. Default is False.
    n_procs : int, optional
        Number of processors to use. Default is 10.
    ccsd_energy : bool, optional
        Whether to use CCSD energy calculations. Default is False.

    Returns
    -------
    list of float
        List of calculated energies in eV for each molecule.
    """
    energy_list = []
    for mol in mol_list:
        # Sanitise the molecule
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)

        # Get the formal charge of the molecule
        charge = Chem.GetFormalCharge(mol)

        # Get the spin multiplicity
        multiplicity = get_spin_multiplicity(mol)

        # Convert the RDKit molecule to an ASE Atoms object
        atoms = mol_to_atoms(mol)

        # Perform the calculation
        energy, _, _ = calculate_free_energy(atoms,
                                             charge=charge,
                                             multiplicity=multiplicity,
                                             orca_path=orca_path,
                                             xc=xc,
                                             basis_set=basis_set,
                                             f_solv=f_solv,
                                             f_disp=f_disp,
                                             n_procs=n_procs,
                                             ccsd_energy=ccsd_energy)
        # Append the energy to the list
        energy_list.append(energy)
    return energy_list
