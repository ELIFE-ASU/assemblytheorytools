import random

import matplotlib.pyplot as plt
import numpy as np

import CFG
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

    string_lengths = np.arange(2, 50, 1)

    ai_list = []
    ai_cfg_list = []

    for i, length in enumerate(string_lengths):
        s_inpt = generate_random_string(int(length), "atgc")
        ai, _, _ = att.calculate_string_assembly_index(s_inpt, timeout=1000.0, directed=True, mode="mol")
        ai_cfg, _, _ = CFG.ai_with_pathways(s_inpt, f_print=False)
        print(f"{i}, String: {s_inpt}, AI: {ai}, AI_CFG: {ai_cfg}")
        ai_list.append(ai)
        ai_cfg_list.append(ai_cfg)

    plt.plot(string_lengths, ai_list, 'o-', label="AssCPP", lw=2, color='black')
    plt.plot(string_lengths, ai_cfg_list, 'o-', label="CFG", lw=2, color='black')
    plt.legend()
    att.n_plot("String length", "Assembly index")

    plt.savefig("string_comparison.png", dpi=600)
    plt.savefig("string_comparison.pdf")
    plt.show()
