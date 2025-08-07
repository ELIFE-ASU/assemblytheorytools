import shutil
import tempfile
import time as t

import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule
from rdkit import Chem

import assemblytheorytools as att


def test_smi_to_atoms():
    """
    Test the conversion of a SMILES string to an ASE Atoms object.

    This function performs the following steps:
    1. Defines a SMILES string for benzene ('c1ccccc1').
    2. Converts the SMILES string to an ASE Atoms object using the `smi_to_atoms` function.
    3. Asserts that the chemical formula of the resulting Atoms object is 'C6H6'.
    4. Asserts that the number of atoms in the Atoms object is 12 (6 carbons and 6 hydrogens).

    Asserts:
        - The chemical formula of the Atoms object matches 'C6H6'.
        - The total number of atoms in the Atoms object is 12.
    """
    print(flush=True)
    smi = 'c1ccccc1'  # SMILES string for benzene
    atoms = att.smiles_to_atoms(smi)  # Convert the SMILES string to an ASE Atoms object

    # Assert that the chemical formula is correct
    assert atoms.get_chemical_formula() == 'C6H6'
    # Assert that the total number of atoms is correct
    assert len(atoms) == 12


def test_atoms_interconvert():
    """
    Test the interconversion between SMILES strings and ASE Atoms objects.

    This function performs the following steps:
    1. Converts a SMILES string to an ASE Atoms object using the `smiles_to_atoms` function.
    2. Converts the resulting Atoms object back to a SMILES string using the `atoms_to_smiles` function.
    3. Asserts that the output SMILES string matches the input SMILES string.

    The test is repeated for two different SMILES strings:
    - Benzene ('[H]C1=C([H])C([H])=C([H])C([H])=C1[H]')
    - Water ('[H]O[H]')

    Asserts:
        - The output SMILES string matches the input SMILES string for each test case.
    """
    print(flush=True)
    smi = '[H]C1=C([H])C([H])=C([H])C([H])=C1[H]'  # SMILES string for benzene
    atoms = att.smiles_to_atoms(smi)  # Convert the SMILES string to an ASE Atoms object
    smi_out = att.atoms_to_smiles(atoms)  # Convert the Atoms object back to a SMILES string
    assert smi_out == smi  # Assert that the output matches the input

    smi = '[H]O[H]'  # SMILES string for water
    atoms = att.smiles_to_atoms(smi)  # Convert the SMILES string to an ASE Atoms object
    smi_out = att.atoms_to_smiles(atoms)  # Convert the Atoms object back to a SMILES string
    assert smi_out == smi  # Assert that the output matches the input


def test_get_charge():
    """
    Test the `get_charge` function for calculating the charge of a molecule.

    This function performs the following steps:
    1. Defines a SMILES string for a neutral water molecule ('[H]O[H]').
    2. Converts the SMILES string to an RDKit Mol object.
    3. Calculates the charge of the molecule using the `get_charge` function.
    4. Asserts that the charge is 0 for the neutral water molecule.
    5. Repeats the process for a charged molecule (hydroxide ion, '[OH-]') and
       asserts that the charge is -1.

    Asserts:
        - The charge of the water molecule is 0.
        - The charge of the hydroxide ion is -1.
    """
    print(flush=True)
    smi = '[H]O[H]'  # SMILES string for water
    mol = Chem.MolFromSmiles(smi)
    charge = att.get_charge(mol)  # Calculate the charge of the molecule
    assert charge == 0, "Charge should be zero for a neutral water molecule"  # Verify the charge

    # Test with a charged molecule (e.g., hydroxide ion)
    smi = '[OH-]'  # SMILES string for hydroxide ion
    mol = Chem.MolFromSmiles(smi)
    charge = att.get_charge(mol)  # Calculate the charge of the hydroxide ion
    assert charge == -1, "Charge should be -1 for hydroxide ion"  # Verify the charge


def test_get_spin_multiplicity():
    """
    Test the `get_spin_multiplicity` function for calculating the spin multiplicity of a molecule.

    This function performs the following steps:
    1. Defines a SMILES string for a water molecule ('[H]O[H]').
    2. Converts the SMILES string to an RDKit Mol object.
    3. Calculates the spin multiplicity of the molecule using the `get_spin_multiplicity` function.
    4. Asserts that the spin multiplicity is 1 for the singlet state of the water molecule.
    5. Repeats the process for a molecule with unpaired electrons (oxygen radical, '[O]') and
       asserts that the spin multiplicity is 3 for the triplet state.

    Asserts:
        - The spin multiplicity of the water molecule is 1.
        - The spin multiplicity of the oxygen radical is 3.
    """
    print(flush=True)
    smi = '[H]O[H]'  # SMILES string for water
    mol = Chem.MolFromSmiles(smi)
    spin_mult = att.get_spin_multiplicity(mol)  # Calculate the spin multiplicity
    assert spin_mult == 1, "Spin multiplicity should be 1 for a singlet state"  # Verify the result

    # Test with a molecule that has unpaired electrons (e.g., oxygen radical)
    smi = '[O]'  # SMILES string for oxygen radical
    mol = Chem.MolFromSmiles(smi)
    spin_mult = att.get_spin_multiplicity(mol)  # Calculate the spin multiplicity
    assert spin_mult == 3, "Spin multiplicity should be 2 for a doublet state (one unpaired electron)"  # Verify the result


def test_orca_calc_preset():
    """
    Test the `orca_calc_preset` function by setting up an ORCA calculator,
    computing the potential energy of a water molecule, and verifying the result.

    Steps:
    1. Create a water molecule using ASE's `molecule` function.
    2. Set up a temporary directory for the ORCA calculation.
    3. Configure the ORCA calculator with the specified parameters.
    4. Assign the calculator to the molecule and compute its potential energy.
    5. Verify that the computed energy matches the expected value within a tolerance.
    6. Clean up the temporary directory after the test.

    Parameters:
    None

    Returns:
    None
    """
    print(flush=True)
    atoms = molecule('H2O')  # Create a water molecule

    temp_dir = tempfile.mkdtemp()  # Create a temporary directory

    # Set up the ORCA calculator with the specified parameters
    calc = att.orca_calc_preset(directory=temp_dir, calc_extra='OPT')
    atoms.calc = calc  # Assign the calculator to the molecule

    energy = atoms.get_potential_energy()  # Compute the potential energy
    print(f"Energy: {energy}", flush=True)

    # Verify that the computed energy matches the expected value within a tolerance
    assert np.allclose(-2077.2584652288906, energy, atol=0.1), "Energy does not match expected value"

    # Clean up the temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)  # Remove the temporary directory


def test_orca_speed():
    """
    Tests the performance of the ORCA quantum chemistry calculator by measuring
    the time taken to compute the potential energy of a benzene molecule using
    different numbers of processors.

    The function performs the following steps:
    1. Creates a benzene molecule (`C6H6`) using ASE.
    2. Iterates over a list of processor counts (`n_procs`).
    3. For each processor count, runs multiple calculations (`n_reps`) to compute
       the potential energy and records the average and standard deviation of the times.
    4. Plots the average computation time with error bars for each processor count.

    Parameters:
    None

    Returns:
    None
    """
    print(flush=True)
    atoms = molecule('C6H6')  # Create a benzene molecule
    n_reps = 5  # Number of repetitions for each processor count
    n_procs = [1, 2, 4, 6, 8, 10]  # List of processor counts to test
    time_avg = []  # List to store average times
    time_std = []  # List to store standard deviations of times

    for n in n_procs:
        times_i = []  # List to store times for the current processor count
        for i in range(n_reps):
            # Set up the ORCA calculator with the specified parameters
            temp_dir = tempfile.mkdtemp()  # Create a temporary directory
            calc = att.orca_calc_preset(xc='r2SCAN-3c',
                                        basis_set='def2-QZVP',
                                        f_solv=False,
                                        f_disp=False,
                                        n_procs=n,
                                        )
            atoms.calc = calc  # Assign the calculator to the molecule
            t1 = t.perf_counter()  # Start timing
            _ = atoms.get_potential_energy()  # Compute the potential energy
            t2 = t.perf_counter()  # End timing
            times_i.append(t2 - t1)  # Record the time taken
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)  # Remove the temporary directory

        time_avg.append(np.mean(times_i))  # Compute the average time
        time_std.append(np.std(times_i))  # Compute the standard deviation
        print(f"Time taken for {n} processors: {time_avg[-1]}+/{time_std[-1]}  seconds", flush=True)

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(n_procs, time_avg, yerr=time_std, fmt='o', capsize=5)  # Plot with error bars
    att.ax_plot(fig, ax, xlab='Number of processors', ylab='Time (s)')  # Add axis labels
    plt.show()  # Display the plot
    pass


def test_optimise_atoms():
    """
    Test the `optimise_atoms` function by optimising the geometry of a water molecule.

    Steps:
    1. Create a water molecule using ASE's `molecule` function.
    2. Optimise the geometry of the molecule using the `optimise_atoms` function.
    3. Verify that the optimised atoms object is not `None`.

    Parameters:
    None

    Returns:
    None
    """
    print(flush=True)  # Ensure immediate output to the console
    atoms = molecule('H2O')  # Create a water molecule
    atoms = att.optimise_atoms(atoms)  # Optimise the geometry of the molecule
    assert atoms is not None, "Optimised atoms should not be None"  # Ensure atoms are optimised


def test_calculate_ccsd_energy():
    """
    Test the `calculate_ccsd_energy` function by calculating the CCSD energy of a water molecule.

    Steps:
    1. Create a water molecule using ASE's `molecule` function.
    2. Call the `calculate_ccsd_energy` function to compute the CCSD energy.
    3. Verify that the computed energy matches the expected value within a tolerance.

    Parameters:
    None

    Returns:
    None
    """
    print(flush=True)  # Ensure immediate output to the console
    atoms = molecule('H2O')  # Create a water molecule
    energy = att.calculate_ccsd_energy(atoms)  # Calculate the CCSD energy
    print(f"CCSD Energy: {energy}", flush=True)  # Print the calculated energy
    assert np.allclose(-2077.230308940521, energy, atol=0.1), "Energy does not match expected value"


def test_calculate_free_energy():
    """
    Test the `calculate_free_energy` function by calculating the Gibbs free energy of a water molecule
    under different conditions.

    Steps:
    1. Create a water molecule using ASE's `molecule` function.
    2. Calculate the Gibbs free energy using the `calculate_free_energy` function.
    3. Verify that the calculated energy matches the expected value within a tolerance.
    4. Repeat the calculation with CCSD energy correction and verify the result.
    5. Perform the calculation with CCSD energy correction and solvent correction, then verify the result.

    Parameters:
    None

    Returns:
    None
    """
    print(flush=True)
    atoms = molecule('H2O')  # Create a water molecule

    # Calculate the Gibbs free energy
    energy, _, _ = att.calculate_free_energy(atoms)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2079.5999124087302, energy, atol=0.1), "Energy does not match expected value"

    # Calculate the Gibbs free energy with CCSD energy correction
    energy, _, _ = att.calculate_free_energy(atoms, use_ccsd=True)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2077.127788955219, energy, atol=0.1), "Energy does not match expected value"

    # Calculate the Gibbs free energy with CCSD energy correction and solvent correction
    energy, _, _ = att.calculate_free_energy(atoms, use_ccsd=True, f_solv=True)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2077.0724431372514, energy, atol=0.1), "Energy does not match expected value"
