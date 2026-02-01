import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Sample 10,000 entries from the CBRDB database
    sample = att.sample_cbrdb(n_samples=10_000)

    # Convert SMILES strings to NetworkX graph representations in parallel
    graphs = att.mp_calc(att.smi_to_nx, sample['smiles'])

    # Calculate the Assembly Index (AI) for each graph in parallel
    # Settings:
    # - strip_hydrogen: Removes hydrogen atoms from the graph
    # - timeout: Maximum time allowed for the calculation per graph (in seconds)
    sample['assembly_index'] = att.calculate_assembly_index_parallel(
        graphs,
        settings={'strip_hydrogen': True, 'timeout': 120.0}
    )[0]

    # Filter the sample to include only molecules with AI >= 1 and reset the index
    sample = sample[sample['assembly_index'] >= 1].reset_index(drop=True)

    # Determine the number of bins for the heatmap
    n_x_bins = len(set(int(x) // 10 * 10 for x in sample['molecular_weight']))
    n_y_bins = len(set(sample['assembly_index']))

    # Plot a heatmap of Molecular Weight (MW) vs Assembly Index (AI)
    fig, ax = att.plot_heatmap(
        sample['molecular_weight'].to_numpy(),
        sample['assembly_index'].to_numpy(),
        "Molecular Weight, (MW), [Da]",
        "Assembly Index, (AI)",
        nbins=(n_x_bins, n_y_bins)
    )

    # Save the heatmap as a PNG file
    plt.savefig("assembly_index_heatmap.png", dpi=300)
    plt.savefig("assembly_index_heatmap.svg", dpi=300)

    # Display the heatmap
    plt.show()

    # Generate labels for the first 9 molecules in the sample
    labs = [f"Name: {sample['nickname'][i]}, AI: {sample['assembly_index'][i]}" for i in range(9)]

    # Extract the SMILES strings for the first 9 molecules
    smis = [sample['smiles'][i] for i in range(len(labs))]

    # Create a grid of molecule images with legends
    img = att.draw_mol_grid_box(smis, legends=labs, n_cols=3)

    # Save the molecule grid as a PNG file
    img.save("molecule_grid.png")

    # Display the molecule grid
    img.show()
