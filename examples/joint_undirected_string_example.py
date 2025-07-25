import assemblytheorytools as att

if __name__ == "__main__":
    # Define a list of input strings to be analyzed for assembly index.
    s_inpt = ["abracadabra", "abra"]

    # Call the calculate_string_assembly_index function with:
    # - s_inpt: input list of strings
    # - dir_code: None (no pre-defined direction coding is provided)
    # - timeout: 100.0 seconds max computation time
    # - directed: False (treat the structure as undirected)
    # - mode: "mol" (assumes molecule-like assembly logic)
    ai, virt_obj, path = att.calculate_string_assembly_index(
        s_inpt,
        dir_code=None,
        timeout=100.0,
        directed=False,
        mode="mol"
    )

    # Output the results
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {path}", flush=True)
