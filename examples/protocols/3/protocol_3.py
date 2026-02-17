import assemblytheorytools as att
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set the timeout duration for assembly index calculations (in seconds)
    timeout = 5.0 * 60.0

    # Define a list of molecule names to analyze
    mols_str = ["papaverine",
                "thebaine",
                "codeine",
                "morphine",
                "diamorphine",
                "fentanyl"]
    # Corresponding SMILES strings for the molecules
    smis = ['COc1ccc(cc1OC)Cc2c3cc(c(cc3ccn2)OC)OC',
            'COC1=CC=C2[C@@H](C3)N(C)CC[C@@]24C5=C3C=CC(OC)=C5O[C@@H]14',
            # 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O',
            # 'CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5',
            # 'CC(OC1=C(O[C@@H]2[C@]34CCN(C)[C@@H]([C@@H]4C=C[C@@H]2OC(C)=O)C5)C3=C5C=C1)=O',
            # 'O=C(CC)N(C1CCN(CC1)CCc2ccccc2)c3ccccc3',
            ]

    # Convert the SMILES strings to NetworkX graph representations
    graphs = [att.smi_to_nx(smi) for smi in smis]

    pathway = att.calculate_assembly_index_pairwise_joint(graphs,
                                                          settings={'strip_hydrogen': True,
                                                                    'timeout': timeout})
    att.plot_pathway(pathway,
                     show_icons=True,
                     fig_size=(14, 7),
                     layout_style='crossmin_long')
    plt.show()

    # Plot the directed graph representation of the assembly pathway. Display synonyms for virtual object names
    att.plot_digraph_metro(pathway)
