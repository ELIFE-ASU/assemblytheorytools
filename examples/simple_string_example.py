import assemblytheorytools as att

if __name__ == "__main__":
    s_inpt = "abracadabra"
    ai, path = att.calculate_assembly_index(s_inpt)
    print(f"Assembly index: {ai}",flush=True)
    print(f"Path: {path}", flush=True)