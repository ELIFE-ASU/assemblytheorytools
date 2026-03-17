import io
import re
import sys
import tarfile
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Fix for pandas compatibility with older pickle files
mod = types.ModuleType('pandas.core.indexes.numeric')
mod.Int64Index = pd.Index
mod.UInt64Index = pd.Index
mod.Float64Index = pd.Index
sys.modules['pandas.core.indexes.numeric'] = mod
from rdkit import Chem
from rdkit.Chem import Draw
import assemblytheorytools as att

if __name__ == "__main__":
    SMILES = 'COC(=O)C(NC(=O)OC(C)(C)C)P(=O)(OC)OC'
    MW = 297.2
    MA_REFERENCE = 14
    TARGET_PARENT_MZ = 296.26

    print(f"Compound: {SMILES}")
    print(f"Molecular Weight: {MW}")
    print(f"Reference MA: {MA_REFERENCE}")
    print(f"Target Parent m/z: {TARGET_PARENT_MZ}")

    mol = Chem.MolFromSmiles(SMILES)
    display(Draw.MolToImage(mol, size=(320, 220)))

    with tarfile.open('Sample_#15_Stepped_MS3.tar.xz', "r:xz") as tar:
        tar.extractall(path='.')

    # Parse the mzML file
    mzml_file = Path('Sample_#15_Stepped_MS3.mzML')
    output_dir = Path('temp_mzml_output')
    output_dir.mkdir(exist_ok=True)

    att.process_mzml_file(
        filename=str(mzml_file),
        out_dir=str(output_dir),
        rt_units='min',
        int_threshold=1000,
        relative=False
    )

    json_file = output_dir / f"ripper_{mzml_file.stem}.json"

    # Process JSON to structured DataFrames
    print("Converting JSON to DataFrames...")
    raw_data = att.process_mzml_json(json_file)

    print(f"✓ MS levels: {list(raw_data.keys())}")
    for level, df in raw_data.items():
        print(f"  MS{level}: {len(df):,} peaks")

    # Processing parameters (matching original pickle files)
    MIN_REL_INTENSITY = 0.01
    MAX_NUM_PEAKS = 20
    MASS_TOL = 0.05
    MS_N_DIGITS = 3
    MIN_ABS_INTENSITY = {
        1: 10 ** 6,
        2: 10 ** 4,
        3: 10 ** 3
    }
    processed_data = att.rma_process(
        sample=raw_data,
        max_num_peaks=MAX_NUM_PEAKS,
        min_abs_intensity=MIN_ABS_INTENSITY,
        min_rel_intensity=MIN_REL_INTENSITY,
        n_digits=MS_N_DIGITS
    )

    for level, df in processed_data.items():
        print(f"  MS{level}: {len(df):,} peaks (filtered from {len(raw_data[level]):,})")

    # Link parent-child relationships
    rooted_data = att.rma_identify_parents(
        processed_data,
        mass_tol=MASS_TOL,
        ms_n_digits=MS_N_DIGITS
    )

    # Build tree structure
    tree = att.rma_build_tree(rooted_data, max_level=3)

    # Filter out MS1 parents with no fragments (empty entries)
    tree = {parent: children for parent, children in tree.items() if children}

    # Find the parent closest to our target
    parent_mz = None
    for mz in tree.keys():
        if abs(mz - TARGET_PARENT_MZ) < 0.01:
            parent_mz = mz
            break

    if parent_mz is None:
        parent_mz = list(tree.keys())[0]

    n_fragments = len(tree[parent_mz]) if parent_mz in tree else 0

    print(f"✓ Tree created: {len(tree)} parent(s) with MS2 fragments")
    print(f"  Parent m/z: {parent_mz:.4f} ({n_fragments} MS2 fragments)")

    # Plot MS2 spectrum - showing all fragments after processing (that go into the tree)
    ms2_processed = processed_data[2]
    parent_data = ms2_processed[ms2_processed['parent'].between(parent_mz - 0.01, parent_mz + 0.01)]
    plt.figure(figsize=(10, 4))
    if len(parent_data) > 0:
        plot_df = parent_data.sort_values('mz')
        plt.vlines(plot_df['mz'], 0, plot_df['intensity'], color='black', linewidth=1.5)
        print(f"Plotting all {len(plot_df)} processed MS2 fragments")
    else:
        fragments = sorted(tree[parent_mz].keys())
        plt.vlines(fragments, 0, 1, color='black', linewidth=1.5)
        print(f"Plotting {len(fragments)} MS2 fragments from tree")
    plt.axhline(y=0, color='gray', linewidth=1)
    plt.xlabel('MS2 m/z', fontsize=11)
    plt.ylabel('Intensity', fontsize=11)
    plt.title(f'Sample #15 MS2 Spectrum - Processed Fragments (parent m/z {parent_mz:.2f})', fontsize=12,
              fontweight='bold')
    plt.xlim(0, max(plot_df['mz']) + 20 if len(plot_df) > 0 else 300)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Initialize MA estimator
    est = att.MAEstimator(same_level=True, n_samples=300, tol=0.05)
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
        print("Numbers in [brackets] are shared sub-fragments from both MS2 fragments.")
        print("Common precursors found:")

        for line in interesting:
            formatted_line = re.sub(r"(\d+\.\d{3})\d+", r"\1", line)
            print(f"  {formatted_line}")

    print(f"Parent m/z: {parent_mz:.4f}")
    print(f"First approximation (mass-only):       {ma_first:.2f} Da")
    print(f"Recursive MA (fragment-informed):      {ma_recursive:.2f} Da")
    print(f"Reference MA (known value):            {MA_REFERENCE} Da")
    print(tree)
