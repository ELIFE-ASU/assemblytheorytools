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


def plot_heatmap(x, y, xlab, ylab, c_map='viridis', nbins=50):
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
    plt.show()


def plot_heatmap_line(x, y, xlab, ylab, c_map='viridis', nbins=50):
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
    plt.show()


if __name__ == "__main__":
    f_run = True
    max_heavy = 25

    if f_run:
        kegg_data_path = os.path.expanduser(os.path.abspath("..//..//kegg_data_C.csv.zip"))

        kegg_data = pd.read_csv(kegg_data_path)
        # Print the size
        print(kegg_data.shape, flush=True)
        print(kegg_data.columns, flush=True)

        # sort by the number of heavy atoms
        kegg_data = kegg_data.sort_values(by="n_heavy_atoms")

        # only select the data that has less than 20 heavy atoms
        kegg_data = kegg_data[kegg_data["n_heavy_atoms"] < max_heavy]

        # Get the smiles
        smi = kegg_data["smiles"].tolist()
        # Get the number of heavy atoms
        n_heavy_atoms = kegg_data["n_heavy_atoms"].tolist()
        # Get the number of chiral centers
        n_chiral_centers = kegg_data["n_chiral_centers"].tolist()
        array_ai = []
        array_heavy_atoms = []
        array_index = []
        print(kegg_data.shape, flush=True)
        for i, s in enumerate(smi):
            print(i, s, flush=True)
            # Convert all the smile to mol

            if "." in s:
                continue
            else:
                mol = Chem.MolFromSmiles(s, sanitize=False)

                # convert to a networkx graph
                graph = att.mol_to_nx(mol)

                # Calculate the assembly index
                ai, _, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
                print(f"Assembly index: {ai}", flush=True)
                if ai <= 0:
                    continue
                else:
                    array_ai.append(ai)
                    array_index.append(i)
                    # array_heavy_atoms.append(n_heavy_atoms[i])
        array_ai = np.array(array_ai)

        # Select only the array_index
        n_heavy_atoms = np.array([n_heavy_atoms[i] for i in array_index])
        n_chiral_centers = np.array([n_chiral_centers[i] for i in array_index])

        # Save the data
        np.savetxt("data.csv", np.vstack([n_heavy_atoms, n_chiral_centers, array_ai]).T, delimiter=",")

    # Load the data
    data = np.loadtxt("data.csv", delimiter=",")
    n_heavy_atoms = data[:, 0]
    n_chiral_centers = data[:, 1]
    array_ai = data[:, 2]

    plot_heatmap(n_heavy_atoms, array_ai, "Heavy atom count", "Assembly index", c_map='viridis', nbins=100)
    plot_heatmap_line(n_heavy_atoms, array_ai, "Heavy atom count", "Assembly index", c_map='viridis', nbins=100)
    plot_heatmap(n_chiral_centers, array_ai, "Chiral center count", "Assembly index", c_map='viridis', nbins=100)
    plot_heatmap_line(n_chiral_centers, array_ai, "Chiral center count", "Assembly index", c_map='viridis', nbins=100)

    # plot_contourf_full(n_heavy_atoms, array_ai, "Heavy atom count", "Assembly index", name="heavy_atom")
