from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def test_reassemble():
    print(flush=True)
    pool = ["CCO", "O"]
    product_smiles = att.reassemble(pool,
                                    n_mol_needed=5,
                                    recycle_to_pool=False,
                                    sterioisomers=True,
                                    tautomers=True,
                                    heterocycles=True)

    for product in product_smiles:
        print(product, flush=True)

    assert len(product_smiles) == 5


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


def test_reassemble_old():
    print(flush=True)
    molecules = ['[H]OC(=O)C([H])([H])N([H])[H]']
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    mols_out = att.convert_pathway_dict_to_list(virt_obj)

    re_mols = [Chem.MolToInchi(mol) for mol in mols_out]
    print(re_mols, flush=True)

    re_mols = att.reassemble_old(mols_out, n_mol_needed=2)
    re_mols = [Chem.MolToInchi(mol) for mol in re_mols]
    print(re_mols, flush=True)
