import assemblytheorytools as att

if __name__ == "__main__":
    # Define a list of input strings to be analyzed for assembly index.
    # Each string in the list will be processed to calculate its assembly index,
    # virtual objects, and assembly pathway.
    s_inpt = ["abracadabra", "abra"]

    # Calculate the string assembly index for the input strings.
    # The function `calculate_string_assembly_index` computes the assembly index,
    # virtual objects, and assembly pathway for the given list of strings.
    # Parameters:
    # - dir_code: Direction code (set to None for no specific direction).
    # - timeout: Maximum time allowed for the calculation (100.0 seconds).
    # - directed: Boolean indicating whether the assembly is directed (False for undirected).
    # - mode: Specifies the assembly logic mode ("mol" for molecular assembly logic).
    ai, virt_obj, path = att.calculate_string_assembly_index(
        s_inpt,
        dir_code=None,
        timeout=100.0,
        directed=False,
        mode="mol"
    )

    # Output the results to the console.
    # Print the calculated assembly index, virtual objects, and assembly pathway.
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {path}", flush=True)
