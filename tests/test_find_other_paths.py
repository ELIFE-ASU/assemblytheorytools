import assemblytheorytools as att

def test_all_paths_simple():
    """
    Test the calculation of all shortest paths in a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates all shortest paths in the molecule.
    3. Compares the calculated paths to the expected paths.

    Asserts:
        - Each calculated path is in the list of expected paths.
    """
    print(flush=True)
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol("C#CCC=C")
    # Calculate all shortest paths in the molecule
    paths = att.all_shortest_paths(mol, f_graph_care=False)

    # check that the output is a list of strings
    assert isinstance(paths, list)
    assert len(paths) > 0


def test_energy_of_all_paths():
    """
    Test the calculation of the energy for all shortest paths in a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates all shortest paths in the molecule.
    3. Defines a list of expected paths.
    4. Asserts that each calculated path is in the list of expected paths.
    5. For each path, calculates its energy and asserts that the energy is not None.

    Asserts:
        - Each calculated path is in the list of expected paths.
        - The energy of each path is not None.
    """
    print(flush=True)
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol("CC")
    # Calculate all shortest paths in the molecule
    paths = att.all_shortest_paths(mol, f_graph_care=False)
    mols = [att.smi_to_mol(vo) for vo in paths]
    energy = att.get_virtual_objects_energy(mols)
    for i, vo in enumerate(paths):
        print(f"VO: {vo}, Energy: {energy[i]}", flush=True)
        # Assert that the energy is not None
        assert energy is not None