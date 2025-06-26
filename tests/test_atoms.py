import shutil
import tempfile

import numpy as np
from ase.build import molecule

import assemblytheorytools as att


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
    assert np.allclose(-2077.373806754082, energy, atol=0.1), "Energy does not match expected value"


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
    energy = att.calculate_free_energy(atoms)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2077.148656604997, energy, atol=0.1), "Energy does not match expected value"

    # Calculate the Gibbs free energy with CCSD energy correction
    energy = att.calculate_free_energy(atoms, ccsd_energy=True)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2072.711389429442, energy, atol=0.1), "Energy does not match expected value"

    # Calculate the Gibbs free energy with CCSD energy correction and solvent correction
    energy = att.calculate_free_energy(atoms, ccsd_energy=True, f_solv=True)
    print(f"Gibbs Free Energy: {energy}", flush=True)
    assert np.allclose(-2072.6569718445553, energy, atol=0.1), "Energy does not match expected value"
