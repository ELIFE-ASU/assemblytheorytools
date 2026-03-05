import random
import time

import CFG
import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


def generate_random_string(length, char_set):
    """
    Generate a random string of a specified length from a given set of characters.

    Args:
        length (int): The length of the random string to generate.
        char_set (str): The set of characters to use for generating the string.

    Returns:
        str: A random string of the specified length.
    """
    return ''.join(random.choice(char_set) for _ in range(length))


if __name__ == "__main__":

    string_lengths = np.arange(2, 40, 1)

    ai_list = []
    ai_cfg_list = []
    time_list = []
    time_cfg_list = []

    for i, length in enumerate(string_lengths):
        s_inpt = generate_random_string(int(length), "atgc")

        start_time = time.time()
        ai, _, _ = att.calculate_string_assembly_index(s_inpt, timeout=1000.0, directed=True, mode="mol")
        end_time = time.time()
        time_list.append(end_time - start_time)

        start_time = time.time()
        ai_cfg, _, _ = CFG.ai_with_pathways(s_inpt, f_print=False)
        end_time = time.time()
        time_cfg_list.append(end_time - start_time)

        print(f"{i}, String: {s_inpt}, AI: {ai}, AI_CFG: {ai_cfg}", flush=True)
        ai_list.append(ai)
        ai_cfg_list.append(ai_cfg)
    # force the x ticks to be integers
    plt.xticks(np.arange(min(string_lengths), max(string_lengths) + 1, 2.0))
    # force the y ticks to be integers
    plt.yticks(np.arange(min(ai_list), max(ai_list) + 1, 2.0))
    plt.plot(string_lengths, ai_list, 'o-', label="AssCPP", lw=2, color='black')
    plt.plot(string_lengths, ai_cfg_list, 'o-', label="CFG", lw=2, color='red')
    plt.legend()
    # att.n_plot("String length", "Assembly index")
    ys = 14
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.xlabel("String length", fontsize=ys)
    plt.ylabel("Assembly index", fontsize=ys)
    plt.tight_layout()
    plt.savefig("string_comparison.png", dpi=600)
    plt.savefig("string_comparison.pdf")
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("String length")
    ax1.set_ylabel("Assembly index")
    ax1.plot(string_lengths, ai_list, 'o-', label="AssCPP", lw=2, color='black')
    ax1.plot(string_lengths, ai_cfg_list, 'o--', label="CFG", lw=2, color='black')
    att.ax_plot(fig, ax1, "String length", "Assembly index")

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.plot(string_lengths, time_list, 'o-', label="AssCPP Time", lw=2, color='red')
    ax2.plot(string_lengths, time_cfg_list, 'o--', label="CFG Time", lw=2, color='red')
    att.ax_plot(fig, ax2, "String length", "Execution time (s)")
    # Make the axis text colour match the line colour
    ax2.set_ylabel('Execution time (s)', color='red')
    ax2.legend()

    fig.tight_layout()
    plt.savefig("string_comparison_with_time.png", dpi=600)
    plt.savefig("string_comparison_with_time.pdf")
    plt.show()
