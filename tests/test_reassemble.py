import os
import shutil

import networkx as nx
import numpy as np
import pytest
from ase.io import read
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
import assemblytheorytools as att
import matplotlib.pyplot as plt


def test_assemble():
    pass


def test_origami():
    print(flush=True)
    mol = Chem.MolFromSmiles('OCC(O)CO')
    mol = att.origami(mol)

    smi_list = [Chem.MolToSmiles(m) for m in mol]

    assert 'OCC1CO1' in smi_list
    assert 'OC1COC1' in smi_list

    mol_out = att.origami(mol[0])
    assert Chem.MolToSmiles(mol_out[0]) == Chem.MolToSmiles(mol[0])


def test_get_num_atom():
    print(flush=True)
    res = att.get_num_atom('CH3', 'C')
    assert res == 1
    res = att.get_num_atom('CH3', 'H')
    assert res == 3


def test_degree_unsaturation():
    print(flush=True)
    # Ethane
    sat = att.degree_unsaturation(Chem.MolFromSmiles('CC'))
    assert sat == 0.0
    # Ethene
    sat = att.degree_unsaturation(Chem.MolFromSmiles('C=C'))
    assert sat == 1.0
    # Cyclopropane
    sat = att.degree_unsaturation(Chem.MolFromSmiles('C1CC1'))
    assert sat == 1.0
    # Benzene
    sat = att.degree_unsaturation(Chem.MolFromSmiles('c1ccccc1'))
    assert sat == 4.0



def test_get_unique_mols():
    pass


def test_reassemble_mols():
    pass
