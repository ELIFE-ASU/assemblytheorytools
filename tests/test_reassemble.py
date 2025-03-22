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

from rdkit import Chem
from rdkit.Chem import AllChem
import random


def react_smiles(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if not mol1 or not mol2:
            return None

        heavy_atom_smarts = Chem.MolFromSmarts("[!#1]")
        heavy_atoms1 = mol1.GetSubstructMatches(heavy_atom_smarts)
        heavy_atoms2 = mol2.GetSubstructMatches(heavy_atom_smarts)

        if not heavy_atoms1 or not heavy_atoms2:
            return None

        idx1 = random.choice(heavy_atoms1)[0]
        idx2 = random.choice(heavy_atoms2)[0]

        # Randomize bond order (SINGLE, DOUBLE, or TRIPLE)
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE
        ]
        random_bond = random.choice(bond_types)

        emol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))
        emol.AddBond(idx1, mol1.GetNumAtoms() + idx2, order=random_bond)

        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)

        return Chem.MolToSmiles(new_mol)
    except:
        return None


def test_1():
    print(flush=True)
    smiles1 = "CCO"  # ethanol, for example
    smiles2 = "O"  # benzene
    product_smiles = react_smiles(smiles1, smiles2)
    print(smiles1)
    print(smiles2)
    print(product_smiles)


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


def test_assemble():
    print(flush=True)
    mol = att.assemble(Chem.MolFromSmiles('OCC(O)CO'), Chem.MolFromSmiles('C=C'), 1)
    if mol is not None:
        print(Chem.MolToSmiles(mol), flush=True)
    assert Chem.MolToSmiles(mol) == 'C=C(O)C(O)CO'


def test_reassemble_mols():
    pass
