import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    sample = att.sample_cbrdb(n_samples=10_000)

    graphs = att.mp_calc(att.smi_to_nx, sample['smiles'])
    sample['assembly_index'] = att.calculate_assembly_index_parallel(graphs,
                                                                     settings={'strip_hydrogen': True,
                                                                               'timeout': 120.0})[0]
    sample = sample[sample['assembly_index'] >= 1].reset_index(drop=True)

    n_x_bins = len(set(int(x) // 10 * 10 for x in sample['molecular_weight']))
    n_y_bins = len(set(sample['assembly_index']))

    fig, ax = att.plot_heatmap(sample['molecular_weight'].to_numpy(),
                               sample['assembly_index'].to_numpy(),
                               "Molecular Weight, (MW), [Da]",
                               "Assembly Index, (AI)",
                               nbins=(n_x_bins, n_y_bins))
    plt.savefig("assembly_index_heatmap.png", dpi=300)
    plt.show()

    labs = [f"Name: {sample['nickname'][i]}, AI: {sample['assembly_index'][i]}" for i in range(9)]
    smis = [sample['smiles'][i] for i in range(len(labs))]
    img = att.draw_mol_grid_box(smis, legends=labs, n_cols=3)
    img.save("molecule_grid.png")
    img.show()
