from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES string for glycine, alanine, serine, and proline

    smi = ['C(C(=O)O)N', 'C[C@@H](C(=O)O)N', 'C([C@@H](C(=O)O)N)O', 'C1C[C@H](NC1)C(=O)O']

    # Convert SMILES strings to RDKit Mol objects using att's helper
    mols = [att.smi_to_mol(s, add_hydrogens=True) for s in smi]

    # print the molecular weights
    for m in mols:
        print(Descriptors.MolWt(m))

    # Combine all molecules into one superstructure
    mol = att.combine_mols(mols)

    # Calculate the assembly index without removing hydrogens
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=False)

    print(virt_obj, flush=True)

    # Convert pathway molecules to InChI strings for easy comparison/printing
    mols_out = [att.smi_to_mol(smile, add_hydrogens=True) for smile in virt_obj]

    # Reassemble the original molecule(s) from substructures, multiple times
    re_mols = att.reassemble_old(mols_out, n_mol_needed=20, mw_min=75, mw_max=200)

    smiles = [Chem.MolToSmiles(mol) for mol in re_mols]
    # Get a list of unique smiles
    smiles = list(set(smiles))
    moles_out = [att.smi_to_mol(smile, add_hydrogens=True) for smile in smiles]

    # Convert to graph
    graphs = [att.mol_to_nx(mol) for mol in moles_out]
    # Remove the hydrogens
    graphs = [att.remove_hydrogen_from_graph(g) for g in graphs]
    graphs = list(set(graphs))

    moles_out = [att.nx_to_mol(g) for g in graphs]

    # α-amino-acid backbone:
    #  N — Cα — C(=O)O(H or -)
    # - allows primary/secondary amines (incl. proline) and protonated amines
    AA_BACKBONE = Chem.MolFromSmarts(
        '[$([NX3;H1,H2,H3;+0,+1]),$([NX4+])] - [CX4] - '
        '[$([CX3](=O)[OX1H]),$([CX3](=O)[O-])]'
    )

    # Variant that also accepts esters (e.g., methyl/ethyl esters of amino acids)
    AA_BACKBONE_OR_ESTER = Chem.MolFromSmarts(
        '[$([NX3;H1,H2,H3;+0,+1]),$([NX4+])] - [CX4] - '
        '[$([CX3](=O)[OX1H]),$([CX3](=O)[O-]),$([CX3](=O)[OX2H0][#6])]'
    )


    def keep_alpha_amino_acid_like(mols, include_esters=False):
        pat = AA_BACKBONE_OR_ESTER if include_esters else AA_BACKBONE
        return [m for m in mols if m is not None and m.HasSubstructMatch(pat)]


    moles_out = keep_alpha_amino_acid_like(moles_out, include_esters=True)

    smiles = [Chem.MolToSmiles(mol) for mol in re_mols]
    # Get a list of unique smiles
    smiles = list(set(smiles))
    moles_out = [att.smi_to_mol(smile, add_hydrogens=True) for smile in smiles]
    # remove molecules with lower mw than 75 or higher than 200
    moles_out = [mol for mol in moles_out if Descriptors.MolWt(mol) >= 75 and Descriptors.MolWt(mol) <= 200]

    img = Draw.MolsToGridImage(moles_out, molsPerRow=10)
    img.save('Mols.png')

    # Convert reassembled molecules to smiles for output
    print([Chem.MolToSmiles(mol) for mol in re_mols], flush=True)
