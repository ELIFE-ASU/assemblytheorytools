from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import csv

DATA_PATH = Path(__file__).with_name("test_molecule_data.csv")

@dataclass(frozen=True)
class Molecule:
    name: str
    category: str
    smiles: str
    inchi: str | None
    assembly_index: int | None
    test_include: bool

def _load_molecules() -> dict[str, Molecule]:
    mols: dict[str, Molecule] = {}
    with DATA_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            category = row["category"].strip()
            smiles = row["smiles"].strip()
            inchi = (row.get("inchi") or "").strip() or None

            ai_str = (row.get("assembly_index") or "").strip()
            assembly_index = int(ai_str) if ai_str else None

            test_include_str = (row.get("test_include") or "").strip()
            # handles 'True'/'False' as in your CSV
            test_include = test_include_str.lower() == "true"

            mols[name] = Molecule(
                name=name,
                category=category,
                smiles=smiles,
                inchi=inchi,
                assembly_index=assembly_index,
                test_include=test_include,
            )
    return mols


MOLS_BY_NAME: dict[str, Molecule] = _load_molecules()

SMILES = SimpleNamespace(
    **{name: mol.smiles for name, mol in MOLS_BY_NAME.items()}
)
ASSEMBLY_INDEX = SimpleNamespace(
    **{name: mol.assembly_index for name, mol in MOLS_BY_NAME.items()}
)