from dataclasses import dataclass
from pathlib import Path
import csv

DATA_PATH = Path(__file__).with_name("tests/data/test_molecule_data.csv")


@dataclass(frozen=True)
class Molecule:
    """
    Container for molecule metadata.

    Attributes
    ----------
    name : str
        Molecule name (typically stored lowercased by the loader).
    category : str
        Category or type of the molecule.
    smiles : str
        SMILES (Simplified Molecular Input Line Entry System) string.
    inchi : str or None
        InChI (International Chemical Identifier) string, or ``None`` if not provided.
    assembly_index : int or None
        Optional assembly index parsed from the CSV; ``None`` if absent.
    test_include : bool
        Flag indicating whether this molecule should be included in tests.
    """
    name: str
    category: str
    smiles: str
    inchi: str | None
    assembly_index: int | None
    test_include: bool


def _load_molecules() -> dict[str, Molecule]:
    """
    Load molecule records from the CSV at ``DATA_PATH`` and return a mapping.

    The CSV is expected to contain the following columns (whitespace is trimmed):
    - ``name``: molecule name (used as the dictionary key, lowercased)
    - ``category``: category/type
    - ``smiles``: SMILES string
    - ``inchi``: optional InChI string
    - ``assembly_index``: optional integer
    - ``test_include``: optional boolean represented as 'True'/'False'

    Returns
    -------
    dict[str, Molecule]
        Mapping from lowercase molecule name to corresponding ``Molecule`` instance.

    Raises
    ------
    ValueError
        If a non-empty ``assembly_index`` field cannot be converted to an integer.
    """
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


test_mols: dict[str, Molecule] = _load_molecules()
