import os

import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

if __name__ == "__main__":
    timeout = 2.0 * 60.0
    max_bonds = 30
    n_peaks_range = (1, 40)
    view_idx = 5
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    df = att.process_chemotion_ir_data('/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar')

    df = att.filter_by_nh_bonds(df, max_bonds=max_bonds)

    df['spectrum'] = att.mp_calc(att.apply_sg_filter,
                                 df['spectrum'])

    peaks = att.find_peak_indices_in_range(df['spectrum'].iloc[view_idx])
    att.plot_ir_spectrum(df['spectrum'].iloc[view_idx], peaks=peaks)
    plt.savefig("example_ir_spectrum.svg")
    plt.savefig("example_ir_spectrum.png", dpi=300)
    plt.show()

    atoms = att.smiles_to_atoms(df['smiles'].iloc[view_idx])
    att.plot_ase_atoms(atoms, 'example_atoms.png', rotation='30x,30y,0z')
    plt.show()

    # Calculate number of peaks
    df['n_peaks'] = np.array(att.mp_calc(att.find_n_peak_indices_in_range,
                                         df['spectrum']), dtype=int)
    df = df[df['n_peaks'].between(*n_peaks_range)].reset_index(drop=True)

    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())
    df['ai'] = att.calculate_assembly_index_parallel(graphs, settings={'strip_hydrogen': True,
                                                                       'timeout': timeout})[0]
    n_peaks = df['n_peaks'].to_numpy()
    ai_obs = df['ai'].to_numpy()

    params, ai_pred = att.estimate_ai_from_ir_peaks(n_peaks, ai_obs, att.linear_func, [0.5, 0.0])

    print(f'Number of data points: {len(ai_obs)}', flush=True)
    print('Linear Fit:', flush=True)
    print(f'params: {params}', flush=True)
    r = att.get_r(ai_obs, ai_pred)
    rmsd = att.get_rmsd(ai_obs, ai_pred)
    print(f'r: {r:.3f}, RMSD: {rmsd:.3f}', flush=True)

    mask = (ai_pred >= min(ai_obs)) & (ai_pred <= max(ai_obs))
    ai_obs = ai_obs[mask]
    ai_pred = ai_pred[mask]

    att.plot_heatmap(ai_obs,
                     ai_pred,
                     "Assembly Index",
                     "IR-Predicted Assembly Index",
                     nbins=(len(set(ai_obs)),
                            len(set(ai_pred))))

    plt.plot([min(ai_obs), max(ai_obs)],
             [min(ai_pred), max(ai_pred)],
             color='black',
             linestyle='--')
    plt.savefig("ir_ai_correlation_heatmap.svg")
    plt.savefig("ir_ai_correlation_heatmap.png", dpi=300)
    plt.show()
