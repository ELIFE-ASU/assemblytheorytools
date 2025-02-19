import assemblytheorytools as att

if __name__ == "__main__":
    s_inpt = ["abracadabra", "abra"]
    ai, virt_obj, path = att.calculate_string_assembly_index(s_inpt,
                                                             dir_code=None,
                                                             timeout=100.0,
                                                             directed=False,
                                                             mode="mol")
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {path}", flush=True)
