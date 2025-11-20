import assemblytheorytools as att

if __name__ == "__main__":
    # Convert all the smile to mol
    mol = att.smi_to_mol("c1ccccc1")
    # Calculate the assembly index
    ai, virt_obj, path = att.calculate_assembly_index(mol)

    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
