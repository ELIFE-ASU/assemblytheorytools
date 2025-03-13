from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdMolTransforms import (
    GetBondLength,
    GetAngleDeg,
    GetDihedralDeg,
)
import numpy as np
import os
import pandas as pd
import pickle


#------------------------CONFORMER_GENERATION------------------------
# APPLY EVERYTHING TO A BATCH
def mol_to_3d_confomer(
    molecule: Chem.Mol,
    num_conformers: int = 10,
    num_threads: int = 20,
):
    """Given a molecule as rdkit mol generate some conformers

    Args:
        smiles (str): _description_

    Returns:
        mol_H (Chem.Mol | bool): protonated molecule | False
            if failed to embed molecule or optimize conformers
            -> False
        converged (list[int]): True if not all conformers converged
    """
    mol_H = Chem.AddHs(molecule)
    ps = rdDistGeom.ETKDGv3()
    # ps.trackFailures = True
    ps.randomSeed = 0xF00D
    ps.enforceChirality = False
    ps.ignoreSmoothingFailures = True
    ps.numZeroFail = 10
    ps.NumThreads = 4
    ps.useRandomCoords = True

    # try to embed molecule
    embedded_mols = AllChem.EmbedMolecule(mol_H, ps)
    if embedded_mols == -1:
        #print("Failed to embed molecule")
        print(Chem.MolToSmiles(mol_H))
        return False, False

    ps = rdDistGeom.ETKDGv3()
    ps.enforceChirality = False
    ps.ignoreSmoothingFailures = True
    ps.numZeroFail = 10
    ps.NumThreads = num_threads
    ps.useRandomCoords = True
    # Generate conformers and optimize
    _ = AllChem.EmbedMultipleConfs(mol_H, num_conformers, ps)

    not_converged = AllChem.UFFOptimizeMoleculeConfs(
        mol_H, numThreads=12, maxIters=500
    )
    not_converged = [val[0] for val in not_converged]

    if not get_converged_ids(not_converged):
        #print("Failed to optimize conformers")
        return False, False
    #print("Conformers generated and optimized")
    return Chem.RemoveAllHs(mol_H), get_converged_ids(not_converged)


def get_converged_ids(not_converged):
    """
    Get the indices of conformers that converged
    """
    return [idx for idx, conv in enumerate(not_converged) if conv == 0]


def compute_conformer_geometry_parameters(molecule, confID: int):
    """Compute geometrical paramters for a given molecule and conformer ID

    Args:
        molecule (_type_): _description_
        confID (int): _description_

    Returns:
        _type_: _description_
    """
    m = MoleculeGeometrie(molecule, confID=confID)
    m.prepare_molecule()

    geometry_parameters = m.get_molecule_stats()
    return geometry_parameters









#------------------------GEOMETRY-------------------------------
# FETCH MOLECULE PARAMETERS
class MoleculeGeometrie:
    """
    Given a rd.Chem.Mol object, this class will find all bonds, angles and dihedral angles
    """

    bond_type_to_smiles = {
        "SINGLE": "-",
        "DOUBLE": "=",
        "TRIPLE": "#",
        "AROMATIC": "*",
    }

    def __init__(
        self,
        molecule: Chem.Mol,
        smiles: str = None,
        confID: int = 0,
    ):
        self.molecule: Chem.rdchem.Mol = molecule
        self.smiles: str = Chem.MolToSmiles(molecule)
        self.angles: list[Angle] = []
        self.dihedral_angles: list[Dihedral] = []
        self.bonds: list[Bond] = []
        self.confID: int = confID

        assert (
            smiles is not None or molecule is not None
        ), "Either smiles or molecule must be provided"

    def __str__(self) -> str:
        print(f"Bond Stats of {len(self.bonds)} bonds:", end="\n")
        for bond in self.bonds:
            print(
                f"Bond type: {bond.bond_type} of Valence {bond.bond_valence} Bond length: {round(bond.bond_length, 3)}"
            )
        print(f"Angle Stats of {len(self.angles)} bonds:", end="\n")
        for angle in self.angles:
            print(
                f"Angle type: {angle.angle_type} Angle length: {round(angle.angle, 3)}"
            )
        print(
            f"Dihedral Stats of {len(self.dihedral_angles)} bonds:", end="\n"
        )
        for dihedral in self.dihedral_angles:
            print(
                f"Dihedral type: {dihedral.angle_type} Dihedral length: {round(dihedral.angle, 3)}"
            )
        return ""

    def prepare_molecule(self) -> None:
        self.find_bonds()
        self.find_angles()
        self.find_dihedral_angles()
        return None

    def get_molecule_stats(self) -> dict:
        """_summary_"""
        data = {"Bonds": [], "Angles": [], "DihedralAngles": []}

        # format that is expected for datbase
        for bond in self.bonds:
            data["Bonds"].append(
                (bond.bond_type, bond.bond_valence, bond.bond_length)
            )
        for angle in self.angles:
            data["Angles"].append((angle.angle_type, angle.angle))
        # for dihedral in self.dihedral_angles:
        #    data["DihedralAngles"].append((dihedral.angle_type, dihedral.angle))
        return data

    def find_bonds(self) -> None:
        """Find all bonds in molecule
        and append them to self.bonds
        """
        confs = self.molecule.GetConformer(self.confID)
        for bond in self.molecule.GetBonds():
            # get bond atoms
            atom1 = bond.GetBeginAtom().GetSymbol()
            atom2 = bond.GetEndAtom().GetSymbol()
            rdkit_bond_type = str(bond.GetBondType())

            # construct bond-type
            bond_type = [atom1, atom2]

            # remove hydrogens
            if "H" in bond_type:
                continue

            bond_type.sort()
            bond_type = self.bond_type_to_smiles[rdkit_bond_type].join(
                bond_type
            )

            bond_length = GetBondLength(
                confs, bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            )
            bond_valence = str(bond.GetBondType())
            self.bonds.append(
                Bond(
                    bond_type=bond_type,
                    bond_length=bond_length,
                    bond_valence=bond_valence,
                )
            )
        return None

    def find_angles(self) -> None:
        """Find all angles in a molecules
        So a depth first search for depth=2
        """
        confs = self.molecule.GetConformer(self.confID)
        angles = []
        # iterate over atoms
        for atom in self.molecule.GetAtoms():
            neighbours = self._find_neighbours(atom)
            for neighbour in neighbours:
                second_neighbours = self._find_neighbours(neighbour)

                angles += self._construct_angles(
                    atom, neighbour, second_neighbours
                )
        # remove duplicates
        angles = [list(x) for x in set(tuple(angle) for angle in angles)]

        # get angles for all angles
        for angle in angles:
            angle_atoms = [angle[0], angle[2], angle[4]]

            # double check at sth like atom1, atom2, atom1 is not in the list
            if len(set(angle_atoms)) != len(angle_atoms):
                continue

            try:
                angle_type = [
                    self.molecule.GetAtomWithIdx(atom).GetSymbol()
                    if atom not in ["-", "=", "#", "*"]
                    else atom
                    for atom in angle
                ]

                # remove hydrogens
                if "H" in angle_type:
                    continue

                # make angles have uniform order
                if angle_type == angle_type[::-1]:
                    pass
                elif angle_type[0] > angle_type[-1]:
                    angle_type = angle_type[::-1]

                # get angle in degree
                angle_degree = GetAngleDeg(confs, angle[0], angle[2], angle[4])

                # create anlge type
                angle_type = "".join(angle_type)

                self.angles.append(Angle(angle_type, angle_degree))
            except:
                continue
        return None

    def find_dihedral_angles(self):
        """Find all torsion anlges in a molecule
        Essentially just a depth first search with degree 3
        """
        confs = self.molecule.GetConformer(self.confID)
        angles = []
        # iterate over atoms
        for atom in self.molecule.GetAtoms():
            neighbours = self._find_neighbours(atom)
            for neighbour in neighbours:
                second_neighbours = self._find_neighbours(neighbour)

                for second_neighbour in second_neighbours:
                    # no 'backwards' movement allowed
                    if atom.GetIdx() == second_neighbour.GetIdx():
                        continue

                    third_neighbours = self._find_neighbours(second_neighbour)
                    angles += self._construct_dihedral_angles(
                        atom, neighbour, second_neighbour, third_neighbours
                    )
        # remove duplicates
        angles = [list(x) for x in set(tuple(angle) for angle in angles)]

        # get angles for all angles
        for angle in angles:
            angle_atoms = [angle[0], angle[2], angle[4], angle[6]]
            if len(set(angle_atoms)) != len(angle_atoms):
                continue

            try:
                angle_type = [
                    self.molecule.GetAtomWithIdx(atom).GetSymbol()
                    if atom not in ["-", "=", "#", "*"]
                    else atom
                    for atom in angle
                ]
                # make angles have uniform order
                if angle_type == angle_type[::-1]:
                    pass
                elif angle_type[0] > angle_type[-1]:
                    angle_type = angle_type[::-1]

                angle_degree = GetDihedralDeg(
                    confs, angle[0], angle[2], angle[4], angle[6]
                )
                # create anlge type
                angle_type = "".join(angle_type)

                self.dihedral_angles.append(Dihedral(angle_type, angle_degree))
            except Exception as e:
                print("Exception:", e)
                continue

        return None

    def _construct_dihedral_angles(
        self,
        atom: Chem.rdchem.Atom,
        neighbours: Chem.rdchem.Atom,
        second_neighbours: Chem.rdchem.Atom,
        third_neighbours: list[Chem.rdchem.Atom],
    ) -> list[list[int]]:
        angles = [
            [
                atom.GetIdx(),
                self.bond_type_to_smiles[
                    str(
                        self.molecule.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbours.GetIdx()
                        ).GetBondType()
                    )
                ],
                neighbours.GetIdx(),
                self.bond_type_to_smiles[
                    str(
                        self.molecule.GetBondBetweenAtoms(
                            neighbours.GetIdx(), second_neighbours.GetIdx()
                        ).GetBondType()
                    )
                ],
                second_neighbours.GetIdx(),
                self.bond_type_to_smiles[
                    str(
                        self.molecule.GetBondBetweenAtoms(
                            second_neighbours.GetIdx(), n.GetIdx()
                        ).GetBondType()
                    )
                ],
                n.GetIdx(),
            ]
            for n in third_neighbours
            if neighbours.GetIdx() != n.GetIdx()
        ]
        return angles

    def _construct_angles(
        self,
        atom: Chem.rdchem.Atom,
        neighbours: Chem.rdchem.Atom,
        second_neighbours: list[Chem.rdchem.Atom],
    ) -> list[list[int]]:
        # angle has the form [atom, bond_type, neighbour, bond_type, second_neighbour]
        # for example C-C=C (so smiles format just that single bond is explicit)
        angles = [
            [
                atom.GetIdx(),
                self.bond_type_to_smiles[
                    str(
                        self.molecule.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbours.GetIdx()
                        ).GetBondType()
                    )
                ],
                neighbours.GetIdx(),
                self.bond_type_to_smiles[
                    str(
                        self.molecule.GetBondBetweenAtoms(
                            neighbours.GetIdx(), n.GetIdx()
                        ).GetBondType()
                    )
                ],
                n.GetIdx(),
            ]
            for n in second_neighbours
            if atom.GetIdx() != n.GetIdx()
        ]
        return angles

    def _find_neighbours(
        self, atom: Chem.rdchem.Atom
    ) -> list[Chem.rdchem.Atom]:
        return atom.GetNeighbors()


# essentially some dataclasses
class Angle:
    def __init__(self, angle_type, angle):
        self.angle_type: str = angle_type  # atom types forming the angle
        self.angle: float = angle


class Dihedral:
    def __init__(self, angle_type, angle):
        self.angle_type: str = angle_type  # atom types forming the angle
        self.angle: float = angle


class Bond:
    def __init__(self, bond_type, bond_length, bond_valence):
        self.bond_type = bond_type
        self.bond_valence = bond_valence
        self.bond_length = bond_length




#------------------------ CONFIG-----------------------------
current_dir = os.path.dirname(__file__)
DIST_path = f"{current_dir}/seb_required_data/distributions/"
DIST_file = "gaussians_100k.pickle"

BADLIST_path = f"{current_dir}/seb_required_data/bad_structures/"

MAX_BOND_LENGTH = 3.5  # Angstrom
MAX_ANGLE = 180.0

filter_stats = {
    "total_molecules": 0,
    "failed_substructure": 0,
    "failed_optimization": 0,
    "failed_geometrical": 0,
    "passed_filter": 0,
}

filter_info = {
    "passed_filter": False,
    "failure_reason": None,
}



#--------------------DB_CONFIG-----------------
database_structure = {
    "DihedralAngles": ["Type", "Angle", "File"],
    "Angles": ["Type", "Angle", "File"],
    "Bonds": ["Type", "Valence", "Length", "File"],
    "TetrahedraVolumes": ["Type", "Volume", "File"],
}


allowed_atoms = ["C", "H", "N", "O", "P", "S", "Cl", "Br", "Se", "F", "I", "B", "Si"]




#---------------- IO_HELPERS -------------------------
def write_dict(data, filename, storage_path):
    with open(storage_path + filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def read_dict(filename=DIST_file, storage_path=DIST_path):
    """
    Default behaviour is loading precomputed distributions
    """
    with open(storage_path + filename, "rb") as handle:
        b = pickle.load(handle)
    return b






# ------------------------MOLECULE_FILTER -------------------------
parameter_distributions = read_dict()


def reset_filter_info() -> None:
    filter_info["passed_filter"] = False
    filter_info["failure_reason"] = None
    return None

def get_filter_stats() -> dict:
    return filter_stats

def reset_filter_stats() -> None:
    filter_stats["total_molecules"] = 0
    filter_stats["failed_substructure"] = 0
    filter_stats["failed_optimization"] = 0
    filter_stats["failed_geometrical"] = 0
    filter_stats["passed_filter"] = 0
    return None

def is_point_in_distribution(
    mean: float, variance: float, point: float, tolerance: int = 4
) -> bool:
    """Given mean and variacne of a Gaussians, tests if a point is within tolerance stdev of that distribution

    Args:
        mean (float): _description_
        variance (float): _description_
        point (float): _description_

    Returns:
        bool: _description_
    """
    return (
        (mean - tolerance * np.sqrt(variance))
        <= abs(point)
        <= (mean + tolerance * np.sqrt(variance))
    )


def load_motif_filters(
    motif_filter_path=BADLIST_path,
) -> list[Chem.rdchem.Mol]:
    filters = []
    files = os.listdir(motif_filter_path)
    for file in files:
        if file.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(
                motif_filter_path + file, strictParsing=False
            )
        elif file.endswith(".csv"):
            df = pd.read_csv(motif_filter_path + file)
            smarts = df["smarts"]
            suppl = [Chem.MolFromSmarts(smart) for smart in smarts]
        for mol in suppl:
            if mol is None:
                continue
            filters.append(mol)
    return filters


def apply_filters(
    molecule, parameter_distributions: dict = parameter_distributions, substruc_only=False
):
    """Returns True if molecule passes all filters

    Args:
        molecule (Chem.Mol): rdkit molecule

    Returns:
        bool: True if molecule passes all filters, else False
    """
    bad_mol = None
    filter_stats["total_molecules"] += 1
    reset_filter_info()

    motifs = load_motif_filters()
    for motif_filter in motifs:
        if not apply_motif_filter(molecule, motif_filter):
            continue
        else:
            bad_mol = molecule
            filter_stats["failed_substructure"] += 1
            filter_info["failure_reason"] = "bad substructure"
            return filter_info, bad_mol
    if not filter_info["failure_reason"]:
        filter_info["passed_filter"] = True
        return filter_info, bad_mol

    if substruc_only:
        return filter_info, bad_mol
    passed = apply_geometrical_filters(molecule, parameter_distributions)
    if not passed:
        bad_mol = molecule
    return filter_info, bad_mol


def apply_motif_filter(molecule: Chem.Mol, filter) -> bool:
    # why this weird conversion works? I have no idea. But it does
    if not molecule:
        return False
    return molecule.HasSubstructMatch(
        Chem.MolFromSmarts(Chem.MolToSmiles(filter))
    )


def apply_geometrical_filters(
    molecule: Chem.Mol, parameter_distributions: dict
):
    passed_filters = check_conformer_geometry(
        molecule=molecule,
        distributions=parameter_distributions,
    )
    return passed_filters

def apply_geometrical_filter(
    parameter: float, filter: list[list[float]]
) -> bool:
    """Given a Parameter [Bond, Angle, Dihedral] test if that parameter is within 2stdev of
        known distirubtions for that parameter according to analysis of the CCDC DB

    Args:
        parameter (float): float value of the parameter provided (i.e. bond length or angle)
        filter (list[list[float]]): [mean, variance] of the gaussian fitted to that specification of
                                        gaussians
    Returns:
        bool: True if the parameter lies within at least one of the gaussians fitted to the data
                else False
    """
    for f in filter:
        if is_point_in_distribution(f[0], f[1], parameter):
            return True
        else:
            continue
    return False


def check_conformer_geometry(
    molecule: Chem.Mol,
    distributions,
    number_wildcards: int | None = None,
):
    """Given a molecules as a smiles
    computes 3d conformers and checks if the generated conformers (for)
    every method returned by  mol_to_3d_confomer pass the geometrical filters

    Args:
        molecule (Chem.Mol): rdkit molecule
        distributions (_type_): dict of parameter[i.e. CCC]: gaussian[mean,var] pairs
        number_wildcards (int, optional): How many parameters are allowed to not pass the fitlers.
    Returns:
        bool: True if any conformer passes all filters, else False
    """
    # set number wildcards as ((num_atom - min_atoms) // 5) + 1
    # meaning that for every 5 atoms we allow one wildcard
    MIN_ATOMS = 15
    if not number_wildcards:
        number_wildcards = max(
            ((molecule.GetNumAtoms() - MIN_ATOMS) // 5) + 1, 1
        )
    # dont check molecules with less than 5 atoms
    if molecule.GetNumAtoms() < 5:
        filter_info["passed_filter"] = True
        return True
        
    input_number_wildcards = number_wildcards
    passed_filters = True

    molecule, converged = mol_to_3d_confomer(molecule)
    if not molecule:
        filter_stats["failed_optimization"] += 1
        filter_info["failure_reason"] = "failed optimization"
        return False
    
    for cid in converged:
        number_wildcards = input_number_wildcards
        data = compute_conformer_geometry_parameters(molecule, confID=cid)
        # iterate over parameters; keys are Bond, Angle, Torsion
        for key in data.keys():
            for parameter in data[key]:
                if parameter[0] in distributions[key]:
                    gaussians = distributions[key][parameter[0]]
                    # check if there are any gaussians for this parameter
                    # is the case if the number of examples for that parameter is too low in the CCDC database
                    if not gaussians:
                        continue
                # if no data for the parameter exists in the database
                # we just continue with the next parameter. Most relevant should be precomputed
                else:
                    continue
                passed_filters = apply_geometrical_filter(
                    parameter[-1], filter=gaussians
                )
                if not passed_filters:
                    number_wildcards -= 1
                    if number_wildcards >= 0:
                        continue
                    else:
                        break
                else:
                    continue
            # break here is number_wildcards has reached 0 meaning that 3 (default for number_wildcards)
            if number_wildcards < 0:
                # print("Too many bad parameters")
                break
        # if number_wildcards >= 0 all parameters have passed (or atleast fewer not passed than number_wildcards)
        if number_wildcards >= 0:
            filter_stats["passed_filter"] += 1
            filter_info["passed_filter"] = True
            return passed_filters
    # if we reach this point no conformer passed all tests
    filter_stats["failed_geometrical"] += 1
    filter_info["failure_reason"] = "bad geometry"
    return passed_filters
