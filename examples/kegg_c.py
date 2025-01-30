import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from scipy.stats import gaussian_kde

import assemblytheorytools as att

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


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
    if "." in smi:
        return -1
    else:
        # Convert the smile to mol
        mol = Chem.MolFromSmiles(smi, sanitize=False)

        # convert to a networkx graph
        graph = att.mol_to_nx(mol)

        # Calculate the assembly index
        ai, _, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
        return ai


if __name__ == "__main__":
    f_run = True
    max_heavy = 15
    data_file_in = "kegg_data_C.csv.zip"
    kegg_data_in_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_in}"))
    data_file_out = "kegg_data_C_AI.csv.zip"
    kegg_data_out_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_out}"))

    if f_run:
        # Load the KEGG data
        kegg_data = pd.read_csv(kegg_data_in_path)

        # Sort by the number of heavy atoms
        kegg_data = kegg_data.sort_values(by="n_heavy_atoms")

        # Add a column for the assembly index
        kegg_data["assembly_index"] = -1  # Initialize to -1

        # only select the data that has less than 20 heavy atoms
        kegg_data = kegg_data[kegg_data["n_heavy_atoms"] < max_heavy]

        # Apply the get_ai function to each row in the kegg_data DataFrame
        kegg_data['assembly_index'] = kegg_data['smiles'].apply(get_ai)

        # Save the updated DataFrame file
        kegg_data.to_csv(kegg_data_out_path, index=False)

    # Load the KEGG data
    kegg_data = pd.read_csv(kegg_data_out_path)

    # filter out the data that has an assembly index of -1
    kegg_data = kegg_data[kegg_data["assembly_index"] > -1]

    n_heavy_atoms = np.array(kegg_data["n_heavy_atoms"].values)
    n_chiral_centers = np.array(kegg_data["n_chiral_centers"].values)
    array_ai = np.array(kegg_data["assembly_index"].values)

    plot_heatmap(n_heavy_atoms, array_ai,
                 "Heavy atom count",
                 "Assembly index",
                 "heavy_ai",
                 c_map='viridis',
                 nbins=100)
    plot_heatmap_line(n_heavy_atoms, array_ai,
                      "Heavy atom count",
                      "Assembly index",
                      "heavy_ai",
                      c_map='viridis',
                      nbins=100)
    plot_heatmap(n_chiral_centers, array_ai,
                 "Chiral center count",
                 "Assembly index",
                 "chiral_ai",
                 c_map='viridis',
                 nbins=100)
    plot_heatmap_line(n_chiral_centers, array_ai,
                      "Chiral center count",
                      "Assembly index",
                      "chiral_ai",
                      c_map='viridis',
                      nbins=100)

    # plot_contourf_full(n_heavy_atoms, array_ai, "Heavy atom count", "Assembly index", name="heavy_atom")
