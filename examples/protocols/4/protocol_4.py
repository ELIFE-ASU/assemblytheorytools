import os

import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att
from scipy.signal import savgol_filter

def fit_line_2d(points: np.ndarray, return_stats: bool = True):
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape {pts.shape}")

    # Accept (N,2) or (2,N)
    if pts.shape[1] == 2:
        x, y = pts[:, 0], pts[:, 1]
    elif pts.shape[0] == 2:
        x, y = pts[0, :], pts[1, :]
    else:
        raise ValueError(
            f"`points` must have shape (N,2) or (2,N); got {pts.shape}"
        )

    if x.size < 2:
        raise ValueError("Need at least 2 points to fit a line.")

    # Least squares fit: minimize ||(m*x + b) - y||^2
    A = np.column_stack([x, np.ones_like(x)])
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)

    if not return_stats:
        return m, b

    y_hat = m * x + b
    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # R^2 (handle constant y gracefully)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    stats = {"r2": r2, "rmse": rmse, "residuals": residuals}
    return m, b, stats

def apply_filter(data, window_length=11, polyorder=3):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

if __name__ == "__main__":
    timeout = 5.0 * 60.0

    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    os.remove('chemotion_ir_data.csv.gz') if os.path.exists('chemotion_ir_data.csv.gz') else None
    df = att.process_chemotion_ir_data(target_file)

    max_bonds = 50
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

    m, b, stats = fit_line_2d(np.array(df[['ai', 'n_peaks']]))

    y = np.array(m * np.array(df['ai']) + b, dtype=int)
    plt.scatter(df['ai'], df['n_peaks'], alpha=0.5)
    plt.plot(df['ai'], y, color='red')
    plt.xlabel('Assembly Index')
    plt.ylabel('Number of Peaks')
    plt.title('Number of Peaks vs Assembly Index with Fitted Line')
    plt.show()

    n_x_bins = len(set(df['ai']))
    n_y_bins = len(set(y))

    fig, ax = att.plot_heatmap(np.array(df['ai']),
                               y,
                               "Assembly Index",
                               "Predicted Assembly Index",
                               # nbins=(n_x_bins, n_y_bins),
                               )
    plt.show()
