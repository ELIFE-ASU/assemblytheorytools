import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

if __name__ == "__main__":
    timeout = 2.0 * 60.0
    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    os.remove('chemotion_ir_data.csv.gz') if os.path.exists('chemotion_ir_data.csv.gz') else None
    df = att.process_chemotion_ir_data(target_file)

    max_bonds = 30
    n_peaks_range = (1, 40)
    df = att.filter_by_nh_bonds(df, max_bonds=max_bonds)

    # Preprocess spectra by applying a Savitzky-Golay filter
    func_filter = partial(att.apply_sg_filter, window_length=9, polyorder=3)
    df['spectrum'] = att.mp_calc(func_filter, df['spectrum'])

    # Calculate number of peaks
    func_peaks = partial(att.find_n_peak_indices_in_range,
                         min_x=400.0,
                         max_x=1500.0,
                         prominence=0.02,
                         distance=10)
    df['n_peaks'] = np.array(att.mp_calc(func_peaks, df['spectrum']), dtype=int)

    # Only keep rows with n_peaks that make sense
    df = df[df['n_peaks'].between(*n_peaks_range)].reset_index(drop=True)

    # Calculate assembly index
    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())
    df['ai'] = att.calculate_assembly_parallel(graphs, settings={'strip_hydrogen': True,
                                                                 'timeout': timeout})[0]
    n_peaks = np.array(df['n_peaks'], dtype=int)
    ai_obs = np.array(df['ai'], dtype=int)

    params, ai_pred = att.estimate_ai_from_ir_peaks(n_peaks, ai_obs, att.linear_func, [0.5, 0.0])

    print('Linear Fit:')
    print(f'p={params}')
    r = att.get_r(ai_obs, ai_pred)
    rmsd = att.get_rmsd(ai_obs, ai_pred)
    print(f'r: {r:.3f}, RMSD: {rmsd:.3f}')

    att.plot_heatmap(ai_obs,
                     n_peaks,
                     "Assembly Index",
                     "Number of Peaks",
                     nbins=(len(set(ai_obs)),
                            len(set(n_peaks))),
                     c_map='Blues')
    plt.show()

    # Filter out cases where prediction is outside observed range
    mask = (ai_pred >= min(ai_obs)) & (ai_pred <= max(ai_obs))
    ai_obs = ai_obs[mask]
    ai_pred = ai_pred[mask]

    att.plot_heatmap(ai_obs,
                     ai_pred,
                     "Assembly Index",
                     "Predicted",
                     nbins=(len(set(ai_obs)),
                            len(set(ai_pred))),
                     c_map='Blues')

    plt.plot([min(ai_obs), max(ai_obs)],
             [min(ai_pred), max(ai_pred)],
             color='black',
             linestyle='--')
    plt.show()
