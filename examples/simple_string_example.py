import assemblytheorytools as att

if __name__ == "__main__":
    # Define a single input string for analysis
    s_inpt = "abracadabra"

    # Calculate the string assembly index
    ai, virt_obj, path = att.calculate_string_assembly_index(
        s_inpt,
        dir_code=None,     # No direction code applied
        timeout=100.0,     # Maximum time allowed for calculation
        directed=False,    # Treat as undirected assembly
        mode="mol"         # Use molecular assembly logic (general case)
    )

    # Output the results
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {path}", flush=True)
