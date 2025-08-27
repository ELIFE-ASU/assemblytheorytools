import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import gaussian_kde

import assemblytheorytools as att

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


def scatter_plot(x,
                 y,
                 xlab='x',
                 ylab='y',
                 figsize=(8, 5),
                 fontsize=16,
                 alpha=0.5,
                 ):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, color='black', alpha=alpha, s=50)

    att.ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_with_colorbar(x,
                               y,
                               xlab='x',
                               ylab='y',
                               cmap='viridis',
                               figsize=(8, 5),
                               fontsize=16,
                               ):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Stack the data and calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density so that high-density points are plotted last
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]

    # Create the scatter plot with colour determined by point density
    scatter = ax.scatter(x_sorted,
                         y_sorted,
                         c=z_sorted,
                         cmap=cmap,
                         s=50,
                         alpha=0.8)

    # Add colour bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Point Density', fontsize=fontsize)

    # Configure the plot
    att.ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_contourf_full(x, y, xlab, ylab, c_map="Purples", name="name"):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    lims = [min(x), max(x)]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[lims[0]:lims[1]:x.size ** 0.6 * 1j, lims[0]:lims[1]:y.size ** 0.6 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.9, cmap=c_map)  # , levels=20)

    # set the axis limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # # draw the y=x line
    # ax.plot(lims, lims, color="black", linestyle="--")
    ax.set_title(name, fontsize=24)
    # add axis labels
    att.ax_plot(fig, ax, xlab, ylab, 22, 22)
    # save the plot
    plt.savefig(name + "_contourf.png", dpi=600)
    plt.savefig(name + "_contourf.pdf", dpi=600)
    plt.show()
    return None


def plot_heatmap(x, y, xlab, ylab, name, c_map='viridis', nbins=50):
    # Create a 2D histogram of the data
    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=(nbins, nbins))

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data.T,
               origin='lower',
               cmap=c_map,
               aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Counts')
    att.n_plot(xlab, ylab)
    plt.savefig(f"{name}_heatmap.png", dpi=600)
    plt.show()


def plot_heatmap_line(x, y, xlab, ylab, name, c_map='viridis', nbins=50):
    # For each n_heavy_atoms value calculate the average assembly index
    ave_y = []
    std_y = []
    x_range = np.arange(min(x), max(x))
    for i in x_range:
        ave_y.append(np.mean(y[x == i]))
        std_y.append(np.std(y[x == i]))

    # Create a 2D histogram of the data
    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=(nbins, nbins))

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data.T,
               origin='lower',
               cmap=c_map,
               aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Counts')

    # Plot the assembly index
    plt.plot(x_range, ave_y, c="black", alpha=0.5)
    # add the error bars
    plt.errorbar(x_range, ave_y, yerr=std_y, fmt='o', c="black", alpha=0.5)
    att.n_plot("Heavy atom count", "Assembly index")
    # set the x-axis limits
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])

    att.n_plot(xlab, ylab)
    plt.savefig(f"{name}_heatmap_line.png", dpi=600)
    plt.show()


def get_ai(smi):
    ai, _, _ = att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True)
    return ai


def get_bertz_complexity(smi):
    return att.bertz_complexity(Chem.MolFromSmiles(smi, sanitize=True))


def get_bottcher_complexity(smi):
    return att.bottcher(Chem.MolFromSmiles(smi, sanitize=True))


if __name__ == "__main__":
    max_heavy = 30
    data_file_in = "CBRdb_C.csv"
    kegg_data_in_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_in}"))
    target_url = f'https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main/{data_file_in}'

    if not os.path.exists(kegg_data_in_path):
        os.system(f"wget {target_url} -O ../../{data_file_in}")
    else:
        print("File already exists, skipping download.")

    df = pd.read_csv(kegg_data_in_path)
    # remove duplicates based on the smiles column
    df = df.drop_duplicates(subset=['smiles'])
    # Remove . in the smiles column
    df = df[~df['smiles'].str.contains(r"\.")]
    # Remove * in the smiles column
    df = df[~df['smiles'].str.contains(r"\*")]
    # Remove diative bonds in the smiles column
    df = df[~df['smiles'].str.contains(r"\->")]
    # Only select the cases where there are less than n heavy atoms
    df = df[df['n_heavy_atoms'] <= max_heavy]
    # Only select the cases where there are more than 2 heavy atoms
    df = df[df['n_heavy_atoms'] >= 2]
    # Remove rows which cannot be parsed by rdkit
    df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    # Remove all the rows that will not sanitize
    df = df[df['smiles'].apply(
        lambda x: Chem.SanitizeMol(Chem.MolFromSmiles(x), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE)]
    # remove nan values
    df = df.dropna(subset=['smiles'])
    print(f"Number of molecules: {len(df)}")

    # Only select the compound_id, smiles, n_heavy_atoms, n_chiral_centers columns
    df = df[['compound_id', 'smiles', 'n_heavy_atoms', 'molecular_weight', 'n_chiral_centers']]

    df['assembly_index'] = att.mp_calc(get_ai, df['smiles'])
    df['bertz_complexity'] = att.mp_calc(get_bertz_complexity, df['smiles'])
    df['bottcher_complexity'] = att.mp_calc(get_bottcher_complexity, df['smiles'])

    # Drop rows with nan or inf values in the assembly_index column
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['bertz_complexity', 'bottcher_complexity'])

    # Remove rows with assembly index greater than 20
    df = df[df['assembly_index'] <= 30]

    plot_heatmap(df['n_heavy_atoms'], df['assembly_index'],
                 "Heavy atom count",
                 "Assembly index",
                 "heavy_ai",
                 c_map='Blues',
                 nbins=100)

    scatter_plot(df['n_heavy_atoms'], df['assembly_index'], xlab="Heavy atom count", ylab="Assembly index")
    plt.show()

    scatter_plot(df['n_chiral_centers'], df['assembly_index'], xlab="Number of Chiral centers", ylab="Assembly index")
    plt.show()

    scatter_plot_with_colorbar(df['bottcher_complexity'], df['bertz_complexity'], xlab="Bottcher Complexity",
                               ylab="Assembly Index")
    plt.savefig("bottcher_ai_colorbar.svg")
    plt.show()

    # plot_contourf_full(df['n_heavy_atoms'], df['assembly_index'], "Heavy atom count", "Assembly index", name="heavy_atom")
