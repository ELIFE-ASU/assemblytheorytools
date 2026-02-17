import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

if __name__ == "__main__":
    # Index of the spectrum to visualize
    view_idx = 5

    # Process Chemotion IR data from a given file path
    # The dataset is filtered to include molecules with a maximum of 30 NH bonds
    df = att.process_chemotion_ir_data('/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar')
    df = att.filter_by_nh_bonds(df, max_bonds=30)

    # Apply Savitzky-Golay filter to smooth the IR spectra in parallel
    df['spectrum'] = att.mp_calc(att.apply_sg_filter, df['spectrum'])

    # Find peak indices in the spectrum of the selected molecule
    peaks = att.find_peak_indices_in_range(df['spectrum'].iloc[view_idx])

    # Plot the IR spectrum with the identified peaks
    att.plot_ir_spectrum(df['spectrum'].iloc[view_idx], peaks=peaks)
    plt.savefig("example_ir_spectrum.svg")  # Save the plot as an SVG file
    plt.savefig("example_ir_spectrum.png", dpi=300)  # Save the plot as a PNG file
    plt.show()  # Display the plot

    # Convert the SMILES string of the selected molecule to atomic coordinates
    smi = df['smiles'].iloc[view_idx]
    print(f"SMILES: {smi}", flush=True)
    atoms = att.smiles_to_atoms(smi)

    # Plot the 3D atomic structure and save it as a PNG file
    att.plot_ase_atoms(atoms, 'example_atoms.png', rotation='30x,30y,0z')
    plt.show()  # Display the atomic structure

    # Calculate the number of peaks in each spectrum and filter the dataset
    # to include only molecules with 1 to 40 peaks
    df['n_peaks'] = np.array(att.mp_calc(att.find_n_peak_indices_in_range,
                                         df['spectrum']), dtype=int)
    df = df[df['n_peaks'].between(*(1, 40))].reset_index(drop=True)

    # Convert SMILES strings to NetworkX graph representations in parallel
    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())

    # Calculate the Assembly Index (AI) for each molecule in parallel
    df['ai'] = att.calculate_assembly_index_parallel(graphs, settings={'strip_hydrogen': True})[0]

    # Extract the number of peaks and observed AI values as NumPy arrays
    n_peaks = df['n_peaks'].to_numpy()
    ai_obs = df['ai'].to_numpy()

    # Perform a linear fit to estimate AI from the number of peaks
    params, ai_pred = att.estimate_ai_from_ir_peaks(n_peaks,
                                                    ai_obs,
                                                    att.linear_func,
                                                    params_0=[0.5, 0.0])

    # Print the number of data points, fit parameters, correlation coefficient, and RMSD
    print(f'Number of data points: {len(ai_obs)}', flush=True)
    print('Linear Fit:', flush=True)
    print(f'params: {params}', flush=True)
    r = att.get_r(ai_obs, ai_pred)  # Calculate the correlation coefficient
    rmsd = att.get_rmsd(ai_obs, ai_pred)  # Calculate the root mean square deviation
    print(f'r: {r:.3f}, RMSD: {rmsd:.3f}', flush=True)

    # Mask predicted AI values to ensure they fall within the observed range
    mask = (ai_pred >= min(ai_obs)) & (ai_pred <= max(ai_obs))
    ai_obs = ai_obs[mask]
    ai_pred = ai_pred[mask]

    # Plot a heatmap comparing observed and predicted AI values
    att.plot_heatmap(ai_obs,
                     ai_pred,
                     "Assembly Index",
                     "IR-Predicted Assembly Index",
                     nbins=(len(set(ai_obs)),
                            len(set(ai_pred))),
                     c_map='Greys',
                     )

    # Overlay a diagonal line representing perfect correlation
    plt.plot([min(ai_obs), max(ai_obs)],
             [min(ai_pred), max(ai_pred)],
             color='black',
             linestyle='--')
    plt.savefig("ir_ai_correlation_heatmap.svg")  # Save the heatmap as an SVG file
    plt.savefig("ir_ai_correlation_heatmap.png", dpi=300)  # Save the heatmap as a PNG file
    plt.show()  # Display the heatmap
