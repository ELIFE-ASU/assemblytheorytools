import assemblytheorytools as att


def test_all_paths_simple():
    """
    Test the calculation of all shortest paths in a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates all shortest paths in the molecule.
    3. Asserts that the output is a list of strings and is not empty.

    Asserts:
        - The output is a list of strings.
        - The list of paths is not empty.
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
    3. Converts each path to a molecule object.
    4. Calculates the energy for each virtual object.
    5. Asserts that the energy of each path is not None.

    Asserts:
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
