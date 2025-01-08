import assemblytheorytools as att

if __name__ == "__main__":
    s_inpt = "abracadabra"
    ai, virt_obj, path = att.calculate_assembly_index(s_inpt)
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
