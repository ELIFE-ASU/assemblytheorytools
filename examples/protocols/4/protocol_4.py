import os

import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

if __name__ == "__main__":
    timeout = 5.0 * 60.0

    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    os.remove('chemotion_ir_data.csv.gz') if os.path.exists('chemotion_ir_data.csv.gz') else None
    df = att.process_chemotion_ir_data(target_file)

    max_bonds = 20
    df = att.filter_by_nh_bonds(df, max_bonds=max_bonds)

    # calculate number of peaks
    df['n_peaks'] = att.mp_calc(att.calc_n_peaks_in_range, df['spectrum'])

    # only keep rows with n_peaks > 0
    df = df[df['n_peaks'] > 0].reset_index(drop=True)
    # only keep rows with n_peaks <= 20
    df = df[df['n_peaks'] <= 60].reset_index(drop=True)

    # calculate assembly index
    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())
    df['ai'] = att.calculate_assembly_parallel(graphs, settings={'strip_hydrogen': True,
                                                                 'timeout': timeout})[0]

    n_x_bins = len(set(df['ai']))
    n_y_bins = len(set(df['n_peaks']))

    fig, ax = att.plot_heatmap(np.array(df['ai']),
                               np.array(df['n_peaks']),
                               "Assembly Index",
                               "Number of Peaks",
                               nbins=(n_x_bins, n_y_bins),
                               )
    plt.show()
