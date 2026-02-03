import assemblytheorytools as att

if __name__ == "__main__":
    # Convert a SMILES string to an RDKit Mol object
    # The SMILES string "c1ccccc1" represents benzene
    mol = att.smi_to_mol("c1ccccc1")

    # Calculate the assembly index, virtual objects, and pathway for the molecule
    # The assembly index is a measure of molecular complexity
    # Virtual objects represent intermediate structures in the assembly process
    # The pathway describes the sequence of steps in the assembly process
    ai, virt_obj, path = att.calculate_assembly_index(mol)

    # Print the calculated assembly index to the console
    print(f"Assembly index: {ai}", flush=True)

    # Print the virtual objects to the console
    print(f"virt_obj: {virt_obj}", flush=True)
