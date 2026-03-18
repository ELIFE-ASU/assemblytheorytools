import io
import re
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    smiles = 'COC(=O)C(NC(=O)OC(C)(C)C)P(=O)(OC)OC'
    mw = 297.2
    ma_reference = 14
    target_parent_mz = 296.26

    print(f"Compound: {smiles}", flush=True)
    print(f"Molecular Weight: {mw}", flush=True)
    print(f"Reference MA: {ma_reference}", flush=True)
    print(f"Target Parent m/z: {target_parent_mz}", flush=True)

    atoms = att.smiles_to_atoms(smiles)
    # Plot the 3D atomic structure and save it as a PNG file
    att.plot_ase_atoms(atoms, 'example_atoms.png', rotation='30x,30y,0z')
    plt.show()

    with tarfile.open('Sample_#15_Stepped_MS3.tar.xz', "r:xz") as tar:
        tar.extractall(path='.')

    # Parse the mzML file
    mzml_file = Path('Sample_#15_Stepped_MS3.mzML')
    output_dir = Path('temp_mzml_output')
    output_dir.mkdir(exist_ok=True)

    att.process_mzml_file(
        filename=str(mzml_file),
        out_dir=str(output_dir),
    )

    json_file = output_dir / f"ripper_{mzml_file.stem}.json"

    # Process JSON to structured DataFrames
    print("Converting JSON to DataFrames...", flush=True)
    raw_data = att.process_mzml_json(json_file)

    print(f"✓ MS levels: {list(raw_data.keys())}", flush=True)
    for level, df in raw_data.items():
        print(f"  MS{level}: {len(df):,} peaks", flush=True)

    # Processing parameters
    min_rel_intensity = 0.01
    max_num_peaks = 20
    mass_tol = 0.05
    min_abs_intensity = {
        1: 10 ** 6,
        2: 10 ** 4,
        3: 10 ** 3
    }
    processed_data = att.rma_process(
        sample=raw_data,
        max_num_peaks=max_num_peaks,
        min_abs_intensity=min_abs_intensity,
        min_rel_intensity=min_rel_intensity,
    )

    for level, df in processed_data.items():
        print(f"  MS{level}: {len(df):,} peaks (filtered from {len(raw_data[level]):,})", flush=True)

    # Link parent-child relationships
    rooted_data = att.rma_identify_parents(
        processed_data,
        mass_tol=mass_tol,
    )

    # Build tree structure
    tree = att.rma_build_tree(rooted_data, max_level=3)

    # Filter out MS1 parents with no fragments (empty entries)
    tree = {parent: children for parent, children in tree.items() if children}

    # Find the parent closest to our target
    parent_mz = None
    for mz in tree.keys():
        if abs(mz - target_parent_mz) < mass_tol:
            parent_mz = mz
            break

    if parent_mz is None:
        parent_mz = list(tree.keys())[0]

    n_fragments = len(tree[parent_mz]) if parent_mz in tree else 0

    print(f"✓ Tree created: {len(tree)} parent(s) with MS2 fragments", flush=True)
    print(f"  Parent m/z: {parent_mz:.4f} ({n_fragments} MS2 fragments)", flush=True)

    # Plot MS2 spectrum - showing all fragments after processing (that go into the tree)
    att.plot_ms2_spectrum(processed_data[2], parent_mz, tree)
    plt.savefig("processed_MS2.svg")
    plt.savefig("processed_MS2.png", dpi=300)
    plt.show()

    # Initialize MA estimator
    est = att.MAEstimator(tol=mass_tol)

    # First approximation (mass-only)
    ma_first = float(est.estimate_by_MW(parent_mz, has_children=False).mean())

    # Recursive MA (fragment-informed)
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    ma_recursive = float(est.estimate_MA(tree, parent_mz, progress_levels=2).mean())
    sys.stdout = old_stdout
    # Extract diagnostic information
    interesting = [
        line
        for line in buffer.getvalue().split("\n")
        if "Common precursors" in line or "HIT:" in line
    ]

    if interesting:
        print("Numbers in [brackets] are shared sub-fragments from both MS2 fragments.", flush=True)
        print("Common precursors found:", flush=True)

        for line in interesting:
            formatted_line = re.sub(r"(\d+\.\d{3})\d+", r"\1", line)
            print(f"  {formatted_line}")

    print(f"Parent m/z: {parent_mz:.4f}", flush=True)
    print(f"First approximation (mass-only):       {ma_first:.2f} Da", flush=True)
    print(f"Recursive MA (fragment-informed):      {ma_recursive:.2f} Da", flush=True)
    print(f"Reference MA (known value):            {ma_reference} Da", flush=True)
