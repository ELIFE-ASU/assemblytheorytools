import random
from functools import partial

import matplotlib.pyplot as plt

import assemblytheorytools as att


def random_string(length, pool):
    """
    Generate a random string of a specified length using characters from the given pool.

    Args:
        length (int): The length of the random string to generate.
        pool (str or list): The pool of characters to choose from.

    Returns:
        str: A random string of the specified length.
    """
    return "".join(random.choice(pool) for _ in range(length))


def _random_string_helper(_, n=10, pool=None):
    """
    Helper function to calculate the assembly index of a random string.

    Args:
        _ (Any): Placeholder for an unused parameter.
        n (int, optional): The length of the random string to generate. Defaults to 10.
        pool (str or list, optional): The pool of characters to choose from. Defaults to None.

    Returns:
        int: The assembly index of the generated random string.
    """
    return att.calculate_string_assembly_index(random_string(n, pool), mode='cfg')[0]


def random_string_parallel(n_reps, n, pool):
    """
    Generate assembly indices for multiple random strings in parallel.

    Args:
        n_reps (int): The number of random strings to generate.
        n (int): The length of each random string.
        pool (str or list): The pool of characters to choose from.

    Returns:
        list: A list of assembly indices for the generated random strings.
    """
    func = partial(_random_string_helper, n=n, pool=pool)
    return att.mp_calc(func, range(n_reps))


if __name__ == "__main__":
    print(flush=True)
    pool_ss = 'guac'
    seq = 'guuggcuacuaugccagcugguggauugcucggcucaggcgcugaugaaggacgugccaagcugcgauaagcuguggggagccgcacggaggcgaagaaccacagauuuccgaaugagaaucucucuaacaauugcuucgcgcaaugaggaaccccgagaacugaaacaucucaguaucgggaggaacagaaaacgcaacgugaugucguuaguaaccgcgagugaacgcgauacagcccaaaccgaagcccucacgggcaauguggugucagggcuaccucucaucagccgaccgucuucacgaagucucuuggaauagagcgugauacagggugacaaccccguacugaagaccaguacgcugugcgguagugccagaguagcggggguuggauaucccucgcgaauaacgcaggcaucgacugcgaaggcuaaacacaaccugagaccgauagugaacaaguagugugaacgaacgcugcaaaguacccucagaagggaggcgaaauagagcaugaaaucaguuggcgaucgagcgacagggcauacaaggucccuugacgaaugaccgagacgcgagucuccaguaagacucacgggaagccgauguucugucguacguuuugaaaaacgagccagggagugugucuguauggcaagucuaaccggaguauccggggaggcacagggaaaccgacauggccgcagggcuuugcccgagggccgccgucuucaagggcggggagccauguggacacgacccgaauccggacgaucuacgcauggacaagaugaagcgugccgaaaggcacguggaagucuguuagaguugguguccuacaauacccucucgugaucuauguguaggggugaaaggcccaucgaguccggcaacagcugguuccaaucgaaacaugucgaagcaugaccuccgccgagguagucugugagguagagcgaccgauugguguguccgccuccgagaggagucggcacaccugucaaacuccaaacuuacagacgcuguuugacgcggggauuccggugcgcgggguaagccuguguaccaggaggggaacaacccagagauagguuaagguccccaaguguggauuaaguguaauccucugaagguggucucgagcccuagacagccgggaggugagcuuagaagcagcuacccucuaagaaaagcguaacagcuuaccggccgagguuugaggcgcccaaaaugaucgggacucaaauccaccaccgagaccuguccguaccacucauacugguaaucgaguagauuggcgcucuaauuggauggaagcaggggcgagagcuccuguggaccgauuagugacgaaaauccuggccauaguagcagcgauagucgggugagaaccccgacggccuaauggauaaggguuccucagcacugcugaucagcugaggguuagccgguccuaagucucaccgcaacucgacugagacgaaaugggaaacagguuaauauuccugugccaucaugcagugaaaguugacgcccuggggucgaucacgccgggcauucgcccggucgaaccguccaacuccguggaagccguaauggcaggaagcggacgaacggcggcauagggaaacgugauucaaccuggggcccaugaaaagacgagcaugauguccguaccgagaaccgacacagguguccauggcggcgaaagccaaggccugucgggagcaaccaacguuagggaauucggcaaguuagucccguaccuucggaagaagggaugccugcuccggaacggagcaggucgcagugacucggaagcucggacugucuaguaacaacauaggugaccgcaaauccgcaaggacucguacggucacugaauccugcccagugcagguaucugaacaccucguacaagaggacgaaggaccugucaacggcggggguaacuaugacccucuuaagguagcguaguaccuugccgcaucaguagcggcuugcaugaauggauuaaccagagcuucacugucccaacguugggcccggugaacuguacauuccagugcggagucuggagacacccagggggaagcgaagacccuauggagcuuuacugcaggcugucgcugagacguggucgccgaugugcagcauagguaggagucguuacagagguacccgcgcuagcgggccacccagacaacagugaaauacuacccgucggugacugcgacucucacuccgggaggaggacaccgauagccgggcaguuugacuggggcgguacgcgcucgaaaagauaucgagcgcgcccuauggucaucucagccgggacagagacccggcgaagagugcaagagcaaaagaugacuugacaguguucuucccaacgaggaacgcugacgcgaaagcguggucuagcgaaccaauuagccugcuugaugcgggcaauugaugacagaaaagcuacccuagggauaacagagucgucacucgcaagagcacauaucgaccgaguggcuugcuaccucgaugucgguucccuccauccugcccgugcagaagcgggcaagggugagguuguucgccuauuaaaggaggucgugagcuggguuuagaccgucgugagacaggucggcugcuaucuacuggguguguaauggugucugacaagaacgaccguauaguacgagaggaacuacgguugguggccacugguguaccgguuguucgagagagcacgugccggguagccacgccacacgggguaagagcugaacgcaucuaagcucgaaacccacuuggaaaagagacaccgccgaggucccgcguacaagacgcggucgauagacucggggugugcgcgucgagguaacgagacguuaagcccacgagcacuaacagaccaaagccaucau'

    print(f"Length of the sequence: {len(seq)}", flush=True)

    n_reps = 50_000
    data = random_string_parallel(n_reps, len(seq), pool_ss)

    ai, _, _ = att.calculate_string_assembly_index(seq, mode='cfg')
    print(f"Assembly Index: {ai}", flush=True)

    fontsize = 16
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = range(int(min(data)), int(max(data)) + 2)
    ax.hist(data,
            bins=bins,
            label='Randomised RNA',
            color='#264f70')
    ax.axvline(ai,
               color='red',
               linestyle='dashed',
               linewidth=2.0,
               label='3CC2 RNA')
    att.ax_plot(fig,
                ax,
                xlab='Assembly Index',
                ylab='Frequency',
                xs=fontsize,
                ys=fontsize)
    plt.legend()
    plt.savefig('rna_string.pdf', bbox_inches='tight')
    plt.show()
