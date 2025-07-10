import os
import tempfile

from ase.atoms import Atoms
from ase.calculators.cp2k import CP2K
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read, write
from ase.units import Hartree
from ase.units import Rydberg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol

from .tools_mol import standardize_mol


def smiles_to_atoms(smiles: str,
                    sanitize: bool = True,
                    add_hydrogen: bool = True) -> Atoms:
    """
    Convert a SMILES string to an ASE Atoms object.

    This function parses a SMILES string into an RDKit Mol object, optionally sanitizes it,
    adds hydrogens, and converts it to an ASE Atoms object.

    Parameters:
    -----------
    smiles : str
        The SMILES string representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object. Default is True.
    add_hydrogen : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.

    Returns:
    --------
    Atoms
        The ASE Atoms object representing the molecule.

    Raises:
    -------
    ValueError
        If the SMILES string cannot be parsed into a valid RDKit Mol object.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if sanitize:
        mol = standardize_mol(mol)
    if add_hydrogen:
        mol = Chem.AddHs(mol)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES string: {smiles}")

    return mol_to_atoms(mol)


def mol_to_atoms(mol: Mol,
                 sanitize: bool = True,
                 add_hydrogen: bool = True,
                 optimise: bool = True) -> Atoms:
    """
    Convert an RDKit Mol object to an ASE Atoms object.

    This function processes an RDKit Mol object by optionally sanitizing it, adding hydrogens,
    and optimizing its geometry. The molecule is then written to a temporary SDF file, which
    is read back and converted into an ASE Atoms object.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit Mol object to be converted.
    sanitize : bool, optional
        Whether to sanitize the molecule (e.g., standardize its structure). Default is True.
    add_hydrogen : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    optimise : bool, optional
        Whether to optimize the molecule's geometry using RDKit's MMFF force field. Default is True.

    Returns:
    --------
    ase.Atoms
        The ASE Atoms object representing the molecule.

    Raises:
    -------
    ValueError
        If the molecule cannot be embedded or optimized.
    """
    if sanitize:
        mol = standardize_mol(mol)
    if add_hydrogen:
        mol = Chem.AddHs(mol)
    # If optimisation is enabled, embed and optimise the molecule using RDKit
    if optimise:
        AllChem.EmbedMolecule(mol, maxAttempts=5000, useRandomCoords=True, randomSeed=0xf00d)
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
                 add_hydrogen: bool = False,
                 charge: int = 0) -> Chem.Mol:
    """
    Convert an ASE Atoms object to an RDKit Mol object.

    This function writes the ASE Atoms object to a temporary XYZ file, reads it back as an RDKit Mol object,
    determines the bonds, and optionally sanitizes and adds hydrogens to the molecule.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the RDKit Mol object (e.g., standardize its structure). Default is True.
    add_hydrogen : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is False.
    charge : int, optional
        The formal charge of the molecule. Default is 0.

    Returns:
    --------
    rdkit.Chem.rdchem.Mol
        The RDKit Mol object representing the molecule.

    Raises:
    -------
    ValueError
        If the RDKit Mol object cannot be created from the XYZ file.
    """
    # Open a temporary file to write the Atoms object in XYZ format
    write('tmp.xyz', atoms, format='xyz')
    raw_mol = Chem.MolFromXYZFile('tmp.xyz')
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    # Remove the temporary file
    os.remove('tmp.xyz')

    if sanitize:
        mol = standardize_mol(mol)
        # Make sure the aromaticity is correct
        Chem.Kekulize(mol)
    if add_hydrogen:
        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)
    return mol


def atoms_to_smiles(atoms: Atoms) -> str:
    """
    Convert an ASE Atoms object to a SMILES string.

    Parameters:
    -----------
    atoms : ase.Atoms
        An ASE Atoms object representing the molecule.

    Returns:
    --------
    str
        The SMILES representation of the molecule.
    """
    mol = atoms_to_mol(atoms)
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=True)


def get_charge(mol: Mol) -> int:
    """
    Calculate the formal charge of a molecule.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object

    Returns:
    --------
    int
        The formal charge of the molecule.
    """
    return Chem.GetFormalCharge(mol)


def get_spin_multiplicity(mol: Mol) -> int:
    """
    Calculate the spin multiplicity of a molecule based on the number of radical electrons.

    Spin multiplicity = 2 S + 1, where S is the total spin quantum number.
    For radical electrons, S = n/2 where n is the number of unpaired electrons.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object

    Returns:
    --------
    int
        The spin multiplicity of the molecule (1 for singlet, 2 for doublet, etc.)
    """
    # Get the number of radical electrons
    num_radical_electrons = Descriptors.NumRadicalElectrons(mol)

    # Calculate spin multiplicity: 2 S + 1 = n + 1, where n is the number of unpaired electrons
    multiplicity = num_radical_electrons + 1

    return multiplicity


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
    Creates and configures a CP2K calculator preset for quantum chemistry calculations.

    Parameters:
        cp2k_command (str, optional): Path to the CP2K executable. Defaults to the 'CP2K_COMMAND' environment variable or 'cp2k.popt'.
        directory (str, optional): Directory to store calculation files. Defaults to a temporary directory.
        cutoff (int, optional): Plane-wave cutoff energy in Rydberg. Defaults to 400.
        charge (int, optional): Molecular charge. Defaults to 0.
        multiplicity (int, optional): Spin multiplicity. Defaults to 1.
        basis_set (str, optional): Basis set to use. Defaults to 'DZVP-MOLOPT-SR-GTH'.
        xc (str, optional): Exchange-correlation functional. Defaults to 'PBE'.
        calc_extra (dict, optional): Additional calculation options to update the FORCE_EVAL section. Defaults to None.
        blocks_extra (dict, optional): Additional CP2K input blocks. Defaults to None.

    Returns:
        CP2K: Configured CP2K calculator object.
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

    Parameters:
    -----------
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

    Returns:
    --------
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


def optimise_atoms(atoms, calc_settings=None):
    """
    Optimise the geometry of an ASE Atoms object using the ORCA quantum chemistry package.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule to be optimised.
    calc_settings : dict, optional
        A dictionary of settings for the ORCA calculator. If None, defaults to {'calc_extra': 'TIGHTOPT'}.

    Returns:
    --------
    ase.Atoms
        The optimised ASE Atoms object, loaded from the ORCA output file.
    """
    if calc_settings is None:
        # Default calculation settings if none are provided
        calc_settings = {'calc_extra': 'TIGHTOPT'}

    # Create a temporary directory for the calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(directory=temp_dir, **calc_settings)

        # Assign the calculator to the molecule
        atoms.calc = calc

        # Trigger the calculation to optimise the geometry
        _ = atoms.get_potential_energy()

        # Load the optimised geometry from the ORCA output file
        orca_file = os.path.join(temp_dir, "orca.xyz")
        return read(orca_file, format="xyz")


def calculate_ccsd_energy(atoms,
                          orca_path=None,
                          charge=0,
                          multiplicity=1,
                          basis_set='aug-cc-pVTZ',
                          n_procs=10):
    """
    Calculate the CCSD (Coupled Cluster with Single and Double excitations) energy of a molecule.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule for which the energy is calculated.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'aug-cc-pVTZ'.
    n_procs : int, optional
        Number of processors to use for the calculation. Default is 10.

    Returns:
    --------
    float
        The CCSD energy of the molecule in eV.
    """
    # If no ORCA path is provided, try to read it from the environment variable
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:
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
        energy = atoms.get_potential_energy()

        return energy


def calculate_free_energy(atoms,
                          charge=0,
                          multiplicity=1,
                          orca_path=None,
                          xc='wB97X',
                          basis_set='def2-SVP',
                          calc_extra='TIGHTOPT FREQ',
                          f_solv=False,
                          f_disp=False,
                          n_procs=10,
                          ccsd_energy=False,
                          atom_list=None,
                          blocks_extra=None,
                          scf_option=None):
    """
    Calculate the Gibbs free energy of a molecule using the ORCA quantum chemistry package.

    Parameters:
    -----------
    atoms : ase.Atoms
        The ASE Atoms object representing the molecule.
    charge : int, optional
        Total charge of the molecule. Default is 0.
    multiplicity : int, optional
        Spin multiplicity of the molecule. Default is 1.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'wB97X'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-SVP'.
    calc_extra : str, optional
        Additional calculation options for ORCA. Default is 'TIGHTOPT FREQ'.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use. Default is 10.
    ccsd_energy : bool, optional
        Whether to include CCSD energy correction. Default is False.
    atom_list : list, optional
        List of atoms for QM/MM calculations. Default is None.
    blocks_extra : str, optional
        Additional ORCA input blocks. Default is None.
    scf_option : str, optional
        Additional SCF options for ORCA. Default is None.

    Returns:
    --------
    float
        The Gibbs free energy of the molecule in eV.
    """
    # If no ORCA path is provided, try to read it from the environment variable
    if orca_path is None:
        orca_path = os.environ.get('ORCA_PATH')
    else:
        orca_path = os.path.abspath(orca_path)

    # Remove 'FREQ' from calc_extra for geometry optimization
    opti_calc_extra = calc_extra.replace('FREQ', '').strip()
    # Ensure the atoms object is optimized before calculating the free energy
    atoms = optimise_atoms(atoms, calc_settings={'xc': xc,
                                                 'basis_set': basis_set,
                                                 'calc_extra': opti_calc_extra,
                                                 'charge': charge,
                                                 'multiplicity': multiplicity,
                                                 'n_procs': n_procs,
                                                 'f_solv': f_solv,
                                                 'f_disp': f_disp,
                                                 'atom_list': atom_list,
                                                 'blocks_extra': blocks_extra,
                                                 'scf_option': scf_option})

    # Remove 'TIGHTOPT' and 'OPT' from calc_extra for frequency calculation
    freq_calc_extra = calc_extra.replace('TIGHTOPT', '').replace('OPT', '').strip()
    # Create a temporary directory for the ORCA calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        orca_file = os.path.join(temp_dir, 'orca.out')

        # Set up the ORCA calculator with the specified parameters
        calc = orca_calc_preset(orca_path=orca_path,
                                directory=temp_dir,
                                xc=xc,
                                charge=charge,
                                multiplicity=multiplicity,
                                basis_set=basis_set,
                                n_procs=n_procs,
                                f_solv=f_solv,
                                f_disp=f_disp,
                                calc_extra=freq_calc_extra,
                                atom_list=atom_list,
                                blocks_extra=blocks_extra,
                                scf_option=scf_option)

        # Attach the ORCA calculator to the ASE Atoms object
        atoms.calc = calc

        # Perform the energy calculation
        _ = atoms.get_potential_energy()

        # If CCSD energy is requested, calculate the correction
        if ccsd_energy:
            # Calculate the CCSD energy
            ccsd_energy = calculate_ccsd_energy(atoms,
                                                orca_path=orca_path,
                                                charge=charge,
                                                multiplicity=multiplicity,
                                                basis_set=basis_set,
                                                n_procs=n_procs)

            # Read the ORCA output file to extract the DFT Gibbs free energy
            with open(orca_file, 'r') as f:
                for line in reversed(f.readlines()):
                    if 'G-E(el)' in line:
                        g_e_ele = float(line.split('...')[-1].split('Eh')[0])
                        # Convert the energy from Hartree to eV
                        g_e_ele *= Hartree
                        break

            # Find the solvent free energy correction
            if f_solv:
                # Find the Free-energy (cav+disp) from the ORCA output file
                with open(orca_file, 'r') as f:
                    for line in reversed(f.readlines()):
                        if 'Free-energy (cav+disp)' in line:
                            g_e_solv = float(line.split(':')[-1].split('Eh')[0])
                            # Convert the energy from Hartree to eV
                            g_e_solv *= Hartree
                            break
                return ccsd_energy + g_e_ele + g_e_solv
            else:
                # If no solvent correction is applied, return the CCSD energy
                return ccsd_energy + g_e_ele

        else:
            # Read the ORCA output file to extract the final Gibbs free energy
            with open(orca_file, 'r') as f:
                for line in reversed(f.readlines()):
                    if 'Final Gibbs free energy' in line:
                        energy = float(line.split('...')[-1].split('Eh')[0])
                        # Convert the energy from Hartree to eV
                        energy *= Hartree
                        break

        return energy


def get_virtual_objects_energy(mol_list,
                               orca_path=None,
                               xc='wB97X',
                               basis_set='def2-SVP',
                               calc_extra='TIGHTOPT FREQ',
                               f_solv=False,
                               f_disp=False,
                               n_procs=10,
                               ccsd_energy=False,
                               atom_list=None,
                               blocks_extra=None,
                               scf_option=None):
    """
    Calculate the free energy of a list of molecules.

    This function processes a list of RDKit molecule objects, converts them to ASE Atoms objects,
    and calculates their free energy using the ORCA quantum chemistry package.

    Parameters:
    -----------
    mol_list : list
        A list of RDKit molecule objects to process.
    orca_path : str, optional
        Path to the ORCA executable. If None, the function attempts to read it from the environment variable 'ORCA_PATH'.
    xc : str, optional
        Exchange-correlation functional to use. Default is 'wB97X'.
    basis_set : str, optional
        Basis set to use for the calculation. Default is 'def2-SVP'.
    calc_extra : str, optional
        Additional calculation options for ORCA. Default is 'TIGHTOPT FREQ'.
    f_solv : bool or str, optional
        Solvent model to use. If True, defaults to 'WATER'. Default is False (no solvent).
    f_disp : bool or str, optional
        Dispersion correction to use. If True, defaults to 'D4'. Default is False (no dispersion correction).
    n_procs : int, optional
        Number of processors to use. Default is 10.
    ccsd_energy : bool, optional
        Whether to include CCSD energy correction. Default is False.
    atom_list : list, optional
        List of atoms for QM/MM calculations. Default is None.
    blocks_extra : dict, optional
        Additional ORCA input blocks. Default is None.
    scf_option : str, optional
        Additional SCF options for ORCA. Default is None.

    Returns:
    --------
    list
        A list of free energy values (in eV) for the input molecules.
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
        energy = calculate_free_energy(atoms,
                                       charge=charge,
                                       multiplicity=multiplicity,
                                       orca_path=orca_path,
                                       xc=xc,
                                       basis_set=basis_set,
                                       calc_extra=calc_extra,
                                       f_solv=f_solv,
                                       f_disp=f_disp,
                                       n_procs=n_procs,
                                       ccsd_energy=ccsd_energy,
                                       atom_list=atom_list,
                                       blocks_extra=blocks_extra,
                                       scf_option=scf_option)
        # Append the energy to the list
        energy_list.append(energy)
    return energy_list
