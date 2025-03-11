import random

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

import reassembler as ra

RDLogger.DisableLog('rdApp.*')


def pick_two(to_be_combined):
    """Pick two fragments randomly. ToBeComb records fragments indices."""
    picked, remain = [], []
    if len(to_be_combined) <= 1:
        return picked, remain
    else:
        temp = np.random.choice(len(to_be_combined), 2, replace=False)
        picked = [to_be_combined[temp[0]], to_be_combined[temp[1]]]
        for i in range(len(to_be_combined)):
            if i != temp[0] and i != temp[1]:
                remain.append(to_be_combined[i])  # get what index remained
        return picked, remain


def get_num_atom(formula, elements):
    """Calculate the number of elements (ele) in the molecule (represented by its formula)"""
    idx1 = formula.find(elements)
    if idx1 == -1:  # no ele found
        return 0
    else:
        temp = formula[idx1 + len(elements):]
        if len(temp) == 0:  # e.g., C20H25NO, here #O = 1
            return 1
        else:
            try:
                idx2 = temp.find(next(filter(str.isalpha, temp)))
            except StopIteration:
                return int(temp)  # when temp is the last digit
            nstr = temp[: idx2]
            if len(nstr) == 0:  # e.g., C20H25NO, here #N = 1
                return 1
            else:
                return int(nstr)


def degree_unsaturation(mol):
    """Calculate Degree of Unsaturation of mol"""
    formu = rdMolDescriptors.CalcMolFormula(mol)
    nC = get_num_atom(formu, 'C')
    nN = get_num_atom(formu, 'N')
    nX = get_num_atom(formu, 'F') + get_num_atom(formu, 'Cl') + get_num_atom(formu, 'Br') + get_num_atom(formu, 'I')
    nH = get_num_atom(formu, 'H')
    dou = (2 * nC + 2 + nN - nX - nH) / 2  # a well-defined formula
    return dou


def generate_mols(
        n_mol_needed=1000,
        pool_file='Pool.txt',
        output_inchi_file='newMols.txt',
        output_fig_path='MolsFig',
        mw_min=281,
        mw_max=368,
        unsat_min=9,
        unsat_max=12,
        one_atom_weight=12,
        mw_delta=0.1
):
    """
    #    NmolNeeded = 1000
    #    PoolFile = 'Pool.txt'
    #    OutputFile = 'newMols.txt'
    #    OutputFigPath = 'MolsFig'
    #    mwMin = 281 # minimum molecular weight of the 6 natural opiates we used
    #    mwMax = 368 # maximum molecular weight of the 6 natural opiates we used
    #    DoUMin = 9 # minimum Degree of Unsaturation of the 6 natural opiates we used
    #    DoUMax = 12 # maximum Degree of Unsaturation of the 6 natural opiates we used
    #    oneAtomWeight = 12 # choose for Carbon atom
    #    mwDelta = 0.1


    The main function that generates new molecules from an assembly pool.
    Args:
        n_mol_needed (int): how many molecules needed to be generated.
        pool_file (str): path to assembly pool inchis.
        output_inchi_file (str): file name where the output inchis will be written into.
        output_fig_path (str): path of a folder where the pictures of the newly-generated molecules will be put into.
        mw_min (int): minimum molecular weight of newly-generated molecules.
        mw_max (int): maximum molecular weight of newly-generated molecules.
        unsat_min (int): minimum Degree of Unsaturation of newly-generated molecules.
        unsat_max (int): maximum Degree of Unsaturation of newly-generated molecules.
        one_atom_weight (int): an approximate molecular weight lost when an atom is thrown away when two fragments are combined.
        mw_delta (float): give a range of the molecular weight that could be relaxed.
    """
    frag = []
    with open(pool_file) as f:
        for inc in f:
            if inc[:5] == 'InChI':
                inc = inc.replace('\n', '')
                frag.append(Chem.MolFromInchi(inc))
    Nfrag = len(frag)
    print('# fragments in the assembly pool:', Nfrag, flush=True)

    mwMinTry = mw_min * (1 - mw_delta)
    mwMaxTry = mw_max * (1 + mw_delta)
    n = 0
    newMols = []
    nFiltered = 0

    while n < n_mol_needed:
        try:
            mw = one_atom_weight
            idxList = []
            idx = []
            while True:
                # to obtain idxList that contains a list of (list of fragments)
                # each (list of fragments) is within the molecular weight range [mwMinTry, mwMaxTry]
                # so later, we randomly choose one (list of fragments) is a proper set of fragments that when they're combined, the weight is within the requred range.
                idxThis = random.randint(0, Nfrag - 1)
                mw += rdMolDescriptors.CalcExactMolWt(frag[idxThis], onlyHeavy=True)
                if mw < mwMinTry:
                    idx.append(idxThis)
                    mw -= one_atom_weight
                else:
                    if mw <= mwMaxTry:
                        idx.append(idxThis)
                        idxList.append(idx.copy())
                        mw -= one_atom_weight
                    else:
                        break

            ContinueSearch = False
            if len(idxList) == 0:
                print('warning: mwMin and mwMax may be too close.', flush=True)
            else:
                ToBeComb = idxList[random.randint(0, len(idxList) - 1)]
                # randomly choose a (list of fragments)

                ifrag = Nfrag
                while len(ToBeComb) > 1:
                    # combine fragments until only a big one exists.
                    # randomly pick two fragments from ToBeComb and combine them, and then put it back into ToBeComb; Then randomly pick two from ToBeComb and repeat.
                    picked, remain = pick_two(ToBeComb)
                    M1 = frag[picked[0]]
                    M2 = frag[picked[1]]

                    if M1.GetNumBonds() == 1 or M2.GetNumBonds() == 1:
                        # when either has only one bond, one one atom can be joined ("overlapped")
                        newmol = ra.assemble(M1, M2, 1)
                    else:
                        # otherwise, decide whether overlap 1 or 2 atoms
                        newmol = ra.assemble(M1, M2, random.randint(1, 2))

                    if newmol is None:
                        # it could happen that combination fails
                        ContinueSearch = True
                        break
                    else:
                        if ifrag >= len(frag):
                            frag.append(newmol)
                        else:
                            frag[ifrag] = newmol
                        remain.append(ifrag)
                        ifrag += 1
                        ToBeComb = remain
                if ContinueSearch:
                    continue
                newMolecule = frag[ToBeComb[0]]

                # consider Degree of Unsaturation
                dou = degree_unsaturation(newmol)
                if dou > unsat_max:
                    print('>', end='', flush=True)
                    continue
                elif dou < unsat_min:
                    print('<', end='', flush=True)
                    # if DoU is smaller than DoUMin, then use operation Origami() to make rings (details in Reassembler.py)
                    # one Origami() operation, DoU is increased by 1.
                    nOrigami = random.randint(unsat_min - int(dou), int(unsat_max - dou))
                    # randomly decide how many Origami() operation will be done
                    try:
                        for i in range(nOrigami):
                            newMolecule = random.choice(ra.origami(newMolecule))
                    except:
                        print('-', end='', flush=True)
                        continue
                else:
                    pass

                # apply filterMol() which is commonly used to filter obviously impossible molecules
                newMolecule = ra.filterMol(newMolecule)
                if newMolecule is None:
                    print(':', end='', flush=True)
                    nFiltered += 1
                    continue

                # apply confilter() which filters molecules that do not have valid conformations
                newMolecule = ra.confilter(newMolecule)
                if newMolecule is None:
                    print(':', end='', flush=True)
                    nFiltered += 1
                    continue

                # then, a new molecule is found
                n += 1
                print(n, end=', ', flush=True)
                newMols.append(Chem.MolToInchi(newMolecule))
        except:
            continue

    f = open(output_inchi_file, 'w')
    i = 0
    for inc in newMols:
        i += 1
        f.write(inc + '\n')
        f.write('--- Molecule ' + str(i) + '\n')
    f.close()
    print('\n', 'nFiltered =', nFiltered, flush=True)

    # visualize, generate figures
    ra.printer(output_inchi_file, output_fig_path)
