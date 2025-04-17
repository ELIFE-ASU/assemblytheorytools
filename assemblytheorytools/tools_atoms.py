import os
import tempfile

from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def smi_to_atoms(smiles):
    """
    Convert a SMILES string to an ASE Atoms object via an SDF file.

    Parameters:
    -----------
    smiles : str
        SMILES string representing the molecule.
    sanitize : bool, optional
        Whether to sanitize the molecule during conversion. Default is True.

    Returns:
    --------
    ase.Atoms
        Atoms object representing the molecule.

    Raises:
    -------
    ValueError
        If SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    mol = Chem.AddHs(mol)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES string: {smiles}")

    return mol_to_atoms(mol)


def mol_to_atoms(mol):
    """
    Convert an RDKit molecule to an ASE Atoms object via an SDF file.

    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        RDKit molecule object to be converted.

    Returns:
    --------
    ase.Atoms
        Atoms object representing the molecule.
    """
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as temp_file:
        sdf_path = temp_file.name

    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    atoms = read(sdf_path)
    os.remove(sdf_path)

    return atoms


def get_spin_multiplicity(mol):
    """
    Calculate the spin multiplicity of a molecule based on the number of radical electrons.

    Spin multiplicity = 2S + 1, where S is the total spin quantum number.
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

    # Calculate spin multiplicity: 2S + 1 = n + 1, where n is number of unpaired electrons
    multiplicity = num_radical_electrons + 1

    return multiplicity


def orca_calc_preset(orca_path=None,
                     directory=None,
                     calc_type='DFT',
                     xc='B3LYP',
                     charge=0,
                     multiplicity=1,
                     basis_set='6-311G',
                     nprocs=1,
                     f_solv=False,
                     f_disp=False,
                     atom_list=None,
                     calc_extra=None,
                     blocks_extra=None,
                     scf_option=None):
    if orca_path is None:
        # try and read the path from the environment
        orca_path = os.environ.get('ORCA_PATH')
    if directory is None:
        directory = os.path.join(tempfile.mkdtemp(), 'orca')

    profile = OrcaProfile(command=orca_path)

    if nprocs > 1:
        inpt_procs = '%pal nprocs {} end'.format(nprocs)
    else:
        inpt_procs = ''

    if f_solv is not None and f_solv is not False:
        if f_solv:
            f_solv = 'WATER'
        inpt_solv = '''
        %CPCM SMD TRUE
            SMDSOLVENT "{}"
        END'''.format(f_solv)
    else:
        inpt_solv = ''

    if f_disp is None or f_disp is False:
        inpt_disp = ''
    else:
        if f_disp:
            f_disp = 'D4'
        inpt_disp = f_disp

    if atom_list is not None and calc_type == 'QM/XTB2':
        inpt_xtb = '''
        %QMMM QMATOMS {{}} END END
        '''.format(str(atom_list).strip('[').strip(']'))
    else:
        inpt_xtb = ''

    if blocks_extra is None:
        blocks_extra = ''

    inpt_blocks = inpt_procs + inpt_solv + blocks_extra

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

    # Add the scf option
    if scf_option is not None:
        inpt_simple += ' ' + scf_option

    # Add the extra options
    if calc_extra is not None:
        inpt_simple += ' ' + calc_extra

    calc = ORCA(
        profile=profile,
        charge=charge,
        mult=multiplicity,
        directory=directory,
        orcasimpleinput=inpt_simple,
        orcablocks=inpt_blocks
    )
    return calc


def get_virtual_objects_energy(mol):
    # Sanitize the molecule
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    # Get the formal charge of the molecule
    charge = Chem.GetFormalCharge(mol)

    # Get the spin multiplicity (note that this assumes the molecule has been correctly assigned a spin state)
    multiplicity = get_spin_multiplicity(mol)

    # Convert the RDKit molecule to an ASE Atoms object
    atoms = mol_to_atoms(mol)

    # Set up the ORCA calculator
    calc = orca_calc_preset(
        xc='B3LYP',
        charge=charge,
        multiplicity=multiplicity,
        basis_set='6-311G',
        nprocs=1,
    )

    # Attach the calculator to the atoms object
    atoms.calc = calc

    # Perform the calculation
    energy = atoms.get_potential_energy()
    return energy
