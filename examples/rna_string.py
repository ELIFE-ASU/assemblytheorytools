import random
from functools import partial
from typing import Sequence, Union, Optional

import matplotlib.pyplot as plt

import assemblytheorytools as att


def random_string(
        length: int,
        pool: Union[str, Sequence[str]],
        seed: Optional[Union[int, str, bytes]] = None,
) -> str:
    if length < 0:
        raise ValueError("length must be >= 0")

    choices = list(pool)
    if not choices:
        raise ValueError("pool must not be empty")

    rng = random.Random(seed)
    chooser = rng.choice

    return "".join(chooser(choices) for _ in range(length))


def _random_string_helper(i, n=10, pool=None):
    tmp = random_string(n, pool)
    ai, _, _ = att.calculate_string_assembly_index(tmp, mode='cfg')
    return ai


def random_string_parallel(n_reps, n, pool):
    func = partial(_random_string_helper, n=n, pool=pool)
    return att.mp_calc(func, range(n_reps))


if __name__ == "__main__":
    print(flush=True)
    pool_ss = 'guac'
    seq = 'guuggcuacuaugccagcugguggauugcucggcucaggcgcugaugaaggacgugccaagcugcgauaagcuguggggagccgcacggaggcgaagaaccacagauuuccgaaugagaaucucucuaacaauugcuucgcgcaaugaggaaccccgagaacugaaacaucucaguaucgggaggaacagaaaacgcaacgugaugucguuaguaaccgcgagugaacgcgauacagcccaaaccgaagcccucacgggcaauguggugucagggcuaccucucaucagccgaccgucuucacgaagucucuuggaauagagcgugauacagggugacaaccccguacugaagaccaguacgcugugcgguagugccagaguagcggggguuggauaucccucgcgaauaacgcaggcaucgacugcgaaggcuaaacacaaccugagaccgauagugaacaaguagugugaacgaacgcugcaaaguacccucagaagggaggcgaaauagagcaugaaaucaguuggcgaucgagcgacagggcauacaaggucccuugacgaaugaccgagacgcgagucuccaguaagacucacgggaagccgauguucugucguacguuuugaaaaacgagccagggagugugucuguauggcaagucuaaccggaguauccggggaggcacagggaaaccgacauggccgcagggcuuugcccgagggccgccgucuucaagggcggggagccauguggacacgacccgaauccggacgaucuacgcauggacaagaugaagcgugccgaaaggcacguggaagucuguuagaguugguguccuacaauacccucucgugaucuauguguaggggugaaaggcccaucgaguccggcaacagcugguuccaaucgaaacaugucgaagcaugaccuccgccgagguagucugugagguagagcgaccgauugguguguccgccuccgagaggagucggcacaccugucaaacuccaaacuuacagacgcuguuugacgcggggauuccggugcgcgggguaagccuguguaccaggaggggaacaacccagagauagguuaagguccccaaguguggauuaaguguaauccucugaagguggucucgagcccuagacagccgggaggugagcuuagaagcagcuacccucuaagaaaagcguaacagcuuaccggccgagguuugaggcgcccaaaaugaucgggacucaaauccaccaccgagaccuguccguaccacucauacugguaaucgaguagauuggcgcucuaauuggauggaagcaggggcgagagcuccuguggaccgauuagugacgaaaauccuggccauaguagcagcgauagucgggugagaaccccgacggccuaauggauaaggguuccucagcacugcugaucagcugaggguuagccgguccuaagucucaccgcaacucgacugagacgaaaugggaaacagguuaauauuccugugccaucaugcagugaaaguugacgcccuggggucgaucacgccgggcauucgcccggucgaaccguccaacuccguggaagccguaauggcaggaagcggacgaacggcggcauagggaaacgugauucaaccuggggcccaugaaaagacgagcaugauguccguaccgagaaccgacacagguguccauggcggcgaaagccaaggccugucgggagcaaccaacguuagggaauucggcaaguuagucccguaccuucggaagaagggaugccugcuccggaacggagcaggucgcagugacucggaagcucggacugucuaguaacaacauaggugaccgcaaauccgcaaggacucguacggucacugaauccugcccagugcagguaucugaacaccucguacaagaggacgaaggaccugucaacggcggggguaacuaugacccucuuaagguagcguaguaccuugccgcaucaguagcggcuugcaugaauggauuaaccagagcuucacugucccaacguugggcccggugaacuguacauuccagugcggagucuggagacacccagggggaagcgaagacccuauggagcuuuacugcaggcugucgcugagacguggucgccgaugugcagcauagguaggagucguuacagagguacccgcgcuagcgggccacccagacaacagugaaauacuacccgucggugacugcgacucucacuccgggaggaggacaccgauagccgggcaguuugacuggggcgguacgcgcucgaaaagauaucgagcgcgcccuauggucaucucagccgggacagagacccggcgaagagugcaagagcaaaagaugacuugacaguguucuucccaacgaggaacgcugacgcgaaagcguggucuagcgaaccaauuagccugcuugaugcgggcaauugaugacagaaaagcuacccuagggauaacagagucgucacucgcaagagcacauaucgaccgaguggcuugcuaccucgaugucgguucccuccauccugcccgugcagaagcgggcaagggugagguuguucgccuauuaaaggaggucgugagcuggguuuagaccgucgugagacaggucggcugcuaucuacuggguguguaauggugucugacaagaacgaccguauaguacgagaggaacuacgguugguggccacugguguaccgguuguucgagagagcacgugccggguagccacgccacacgggguaagagcugaacgcaucuaagcucgaaacccacuuggaaaagagacaccgccgaggucccgcguacaagacgcggucgauagacucggggugugcgcgucgagguaacgagacguuaagcccacgagcacuaacagaccaaagccaucau'

    print(f"Length of the sequence: {len(seq)}", flush=True)

    n_reps = 50_000
    ai_list = random_string_parallel(n_reps, len(seq), pool_ss)

    ai, _, _ = att.calculate_string_assembly_index(seq, mode='cfg')
    print(f"Assembly Index: {ai}", flush=True)

    fontsize = 16
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(ai_list,
            bins=50,
            alpha=1.0,
            label='Randomised RNA',
            color='#264f70')
    ax.axvline(ai,
               color='red',
               linestyle='dashed',
               linewidth=2.0,
               label='3CC2 RNA')
    # ax.set_yscale('log')
    att.ax_plot(fig,
                ax,
                xlab='Assembly Index',
                ylab='Frequency',
                xs=fontsize,
                ys=fontsize)
    plt.legend()
    plt.savefig('rna_string.pdf', bbox_inches='tight')
    plt.show()
