import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import assemblytheorytools as att


def linear_func(x, m, b):
    """
    Linear function to model a straight line.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    m : float
        The slope of the line.
    b : float
        The y-intercept of the line.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as m * x + b.
    """
    return m * x + b


def quadratic_func(x, a, b, c):
    """
    Quadratic function to model a parabola.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quadratic term (x^2).
    b : float
        The coefficient for the linear term (x).
    c : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^2 + b * x + c.
    """
    return a * x ** 2 + b * x + c


def cubic_func(x, a, b, c, d):
    """
    Cubic function to model a polynomial of degree 3.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the cubic term (x^3).
    b : float
        The coefficient for the quadratic term (x^2).
    c : float
        The coefficient for the linear term (x).
    d : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^3 + b * x^2 + c * x + d.
    """
    return a * x ** 3 + b * x ** 2 + c * x + d


def quartic_func(x, a, b, c, d, e):
    """
    Quartic function to model a polynomial of degree 4.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quartic term (x^4).
    b : float
        The coefficient for the cubic term (x^3).
    c : float
        The coefficient for the quadratic term (x^2).
    d : float
        The coefficient for the linear term (x).
    e : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^4 + b * x^3 + c * x^2 + d * x + e.
    """
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def quintic_func(x, a, b, c, d, e):
    """
    Quintic function to model a polynomial of degree 5.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quintic term (x^5).
    b : float
        The coefficient for the quartic term (x^4).
    c : float
        The coefficient for the cubic term (x^3).
    d : float
        The coefficient for the quadratic term (x^2).
    e : float
        The coefficient for the linear term (x).

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x.
    """
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x


def get_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.sum(yt * yp) / denom)


def get_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle the degenerate case
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
    return 1.0 - (ss_res / ss_tot)


def get_rmsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _peaks_to_ai(n_peaks, model, params):
    return int(model(n_peaks, *params))


def _func_min_helper(x, *args):
    n_peaks, obs, model_fit = args
    pred = np.array([_peaks_to_ai(n, model_fit, x) for n in n_peaks], dtype=int)
    return get_rmsd(obs, pred)


def estimate_ai_from_peaks(peaks_data,
                           ai_obs,
                           model,
                           params_0):
    res = minimize(_func_min_helper,
                   np.array(params_0),
                   args=(peaks_data,
                         ai_obs,
                         model),
                   method='Nelder-Mead',
                   tol=1e-6)
    data_pred = [_peaks_to_ai(x_i, model, res.x) for x_i in peaks_data]
    return res.x, np.array(data_pred, dtype=int)


if __name__ == "__main__":
    timeout = 2.0 * 60.0

    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    os.remove('chemotion_ir_data.csv.gz') if os.path.exists('chemotion_ir_data.csv.gz') else None
    df = att.process_chemotion_ir_data(target_file)

    max_bonds = 30
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
    df = df[df['n_peaks'].between(1, 40)].reset_index(drop=True)

    # Calculate assembly index
    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())
    df['ai'] = att.calculate_assembly_parallel(graphs, settings={'strip_hydrogen': True,
                                                                 'timeout': timeout})[0]
    n_peaks = np.array(df['n_peaks'], dtype=int)
    ai_obs = np.array(df['ai'], dtype=int)

    params, ai_pred = estimate_ai_from_peaks(n_peaks, ai_obs, linear_func, [0.5, 0.0])

    print('Linear Fit:')
    print(f'p={params}')
    r = get_r(ai_obs, ai_pred)
    rmsd = get_rmsd(ai_obs, ai_pred)
    print(f'r: {r:.3f}, RMSD: {rmsd:.3f}')

    fig, ax = att.plot_heatmap(ai_obs,
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

    fig, ax = att.plot_heatmap(ai_obs,
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
