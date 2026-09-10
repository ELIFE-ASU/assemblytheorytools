"""
Sampling of virtual objects found in shortest assembly calculations.

The historical :func:`all_shortest_paths` name is retained for compatibility,
but it does not return or exhaustively enumerate pathways. It repeats the
default calculation after random atom renumberings and collects the unique
virtual-object SMILES strings encountered.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol

from .assembly import calculate_assembly_index


def _get_atom_order(mol: Mol) -> List[int]:
    """Return atom indices by canonical rank, accounting for chirality."""
    ranks = Chem.CanonicalRankAtoms(mol, includeChirality=True)
    return sorted(range(len(ranks)), key=ranks.__getitem__)


def _scramble_list(lst: list) -> list:
    """Shuffle a copy of ``lst`` using NumPy's global random state."""
    shuffled = lst.copy()
    np.random.shuffle(shuffled)
    return shuffled


def all_shortest_paths(
    mol: Mol,
    settings: Optional[Dict[str, Any]] = None,
    f_graph_care: bool = False,
    max_attempts: int = 3,
) -> List[str]:
    """
    Sample unique virtual objects by scrambling a molecule's atom indices.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The input RDKit molecule object.
    settings : dict, optional
        Settings passed to the assembly index calculation, by default None.
        Canonicalization is disabled; a nonempty dictionary is updated in
        place.
    f_graph_care : bool, optional
        Whether to kekulize the molecule, by default False.
    max_attempts : int, optional
        Maximum number of consecutive attempts without finding new virtual
        objects (VOs) before terminating the search, by default 3.

    Returns
    -------
    List[str]
        Unique virtual-object (VO) SMILES strings encountered across sampled
        calculations, in no particular order. These are not pathways.

    Raises
    ------
    ValueError
        If the input is not an RDKit molecule object.

    Notes
    -----
    This stochastic sampler uses atom-index scrambling to expose different
    calculator results and collect their virtual objects. The attempt budget
    is four times the number of bonds, with early termination after
    ``max_attempts`` consecutive iterations find no new VO.
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Input must be an RDKit molecule object.")

    settings = settings or {}
    settings["canonicalize"] = False

    atom_order = _get_atom_order(mol)
    seen: set[str] = set()
    stale_attempts = 0

    for _ in range(4 * mol.GetNumBonds()):
        if stale_attempts >= max_attempts:
            break

        renumbered_mol = Chem.RenumberAtoms(mol, _scramble_list(atom_order))
        if f_graph_care:
            Chem.Kekulize(renumbered_mol)

        _, virtual_objects, _ = calculate_assembly_index(renumbered_mol, **settings)
        previous_count = len(seen)
        seen.update(virtual_objects)
        stale_attempts = stale_attempts + 1 if len(seen) == previous_count else 0

    return list(seen)
