import assemblytheorytools as att

if __name__ == "__main__":
    # Define a single input string for analysis
    # The input string "abracadabra" will be analyzed to calculate its assembly index
    s_inpt = "abracadabra"

    # Calculate the string assembly index
    # The function `calculate_string_assembly_index` computes the assembly index, virtual objects,
    # and assembly pathway for the given input string. The parameters are:
    # - dir_code: Direction code (set to None for no specific direction)
    # - timeout: Maximum time allowed for the calculation (100.0 seconds)
    # - directed: Boolean indicating whether the assembly is directed (False for undirected)
    # - mode: Specifies the assembly logic mode ("mol" for molecular assembly logic)
    ai, virt_obj, path = att.calculate_string_assembly_index(
        s_inpt,
        dir_code=None,  # No direction code applied
        timeout=100.0,  # Maximum time allowed for calculation
        directed=False,  # Treat as undirected assembly
        mode="mol"  # Use molecular assembly logic (general case)
    )

    # Output the results
    # Print the calculated assembly index, virtual objects, and assembly pathway to the console
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {path}", flush=True)
