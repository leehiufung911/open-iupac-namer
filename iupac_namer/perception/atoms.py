"""
iupac_namer/perception/atoms.py

AtomAnalysis subsystem — Subsystem 1 of the Perception layer.

Computes per-atom structural information from an RDKit mol.  Element-agnostic
by design: no assumptions about which elements are present.  This is the
foundation that all other subsystems build upon.

Computed once and cached on construction.  All public data is immutable.

See ARCHITECTURE_PERCEPTION.md §Subsystem 1 for the full spec.
"""

from __future__ import annotations

from iupac_namer.types import AtomInfo


# ---------------------------------------------------------------------------
# Bond-type mapping
# ---------------------------------------------------------------------------

# RDKit's BondType string ends with the bond name after the last dot.
# e.g. "rdkit.Chem.rdchem.BondType.SINGLE" -> "SINGLE"
_BOND_TYPE_MAP: dict[str, str] = {
    "SINGLE": "single",
    "DOUBLE": "double",
    "TRIPLE": "triple",
    "AROMATIC": "aromatic",
    "ONEANDAHALF": "oneandahalf",   # dative/partial
    "TWOANDAHALF": "twoandahalf",
    "THREEANDAHALF": "threeandahalf",
    "FOURANDAHALF": "fourandahalf",
    "FIVEANDAHALF": "fiveandahalf",
    "IONIC": "ionic",
    "HYDROGEN": "hydrogen",
    "THREECENTER": "threecenter",
    "DATIVEONE": "dativeone",
    "DATIVE": "dative",
    "DATIVEL": "dativel",
    "DATIVER": "dativer",
    "OTHER": "other",
    "ZERO": "zero",
}


def _rdkit_bond_type_to_str(bond_type: object) -> str:
    """Convert an RDKit BondType enum to a lowercase short name."""
    key = str(bond_type).split(".")[-1]
    return _BOND_TYPE_MAP.get(key, key.lower())


# ---------------------------------------------------------------------------
# AtomAnalysis
# ---------------------------------------------------------------------------


class AtomAnalysis:
    """Per-atom structural analysis for an RDKit molecule.

    Computed once on construction.  All returned data is immutable.

    Parameters
    ----------
    mol:
        An RDKit ``Mol`` object (after sanitisation).

    Attributes
    ----------
    all_atoms:
        Immutable tuple of :class:`~iupac_namer.types.AtomInfo`, one per atom,
        in atom-index order.
    """

    def __init__(self, mol: object) -> None:
        self._mol = mol
        self._atom_infos: tuple[AtomInfo, ...] = self._analyze()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _analyze(self) -> tuple[AtomInfo, ...]:
        """Build an :class:`~iupac_namer.types.AtomInfo` for every atom."""
        infos: list[AtomInfo] = []

        for atom in self._mol.GetAtoms():  # type: ignore[attr-defined]
            idx: int = atom.GetIdx()

            # Neighbors (indices only)
            neighbors: tuple[int, ...] = tuple(
                n.GetIdx() for n in atom.GetNeighbors()
            )

            # Bond types keyed by other-atom index
            bond_types: list[tuple[int, str]] = []
            for bond in atom.GetBonds():
                other_idx: int = bond.GetOtherAtomIdx(idx)
                bt_str: str = _rdkit_bond_type_to_str(bond.GetBondType())
                bond_types.append((other_idx, bt_str))

            infos.append(
                AtomInfo(
                    idx=idx,
                    element=atom.GetSymbol(),
                    atomic_num=atom.GetAtomicNum(),
                    valence=atom.GetTotalValence(),
                    charge=atom.GetFormalCharge(),
                    degree=atom.GetDegree(),
                    in_ring=atom.IsInRing(),
                    aromatic=atom.GetIsAromatic(),
                    neighbors=neighbors,
                    coordination_number=atom.GetDegree(),
                    bond_types=tuple(bond_types),
                    isotope=atom.GetIsotope(),
                )
            )

        return tuple(infos)

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> AtomInfo:
        """Return the :class:`~iupac_namer.types.AtomInfo` for atom *idx*.

        Parameters
        ----------
        idx:
            RDKit atom index (0-based).

        Raises
        ------
        IndexError
            If *idx* is out of range.
        """
        return self._atom_infos[idx]

    def __len__(self) -> int:
        """Number of atoms in the molecule."""
        return len(self._atom_infos)

    def __iter__(self):
        """Iterate over all :class:`~iupac_namer.types.AtomInfo` in index order."""
        return iter(self._atom_infos)

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    @property
    def all_atoms(self) -> tuple[AtomInfo, ...]:
        """All :class:`~iupac_namer.types.AtomInfo`, one per atom, in index order."""
        return self._atom_infos

    def atoms_by_element(self, element: str) -> tuple[AtomInfo, ...]:
        """Return all atoms with the given element symbol.

        Parameters
        ----------
        element:
            Element symbol string, e.g. ``"C"``, ``"N"``, ``"O"``.
            Case-sensitive (use standard element symbols).

        Returns
        -------
        tuple[AtomInfo, ...]
            May be empty if no atoms of this element are present.
        """
        return tuple(a for a in self._atom_infos if a.element == element)

    def ring_atoms(self) -> frozenset[int]:
        """Return the set of atom indices that are in a ring."""
        return frozenset(a.idx for a in self._atom_infos if a.in_ring)

    def non_ring_atoms(self) -> frozenset[int]:
        """Return the set of atom indices that are NOT in any ring."""
        return frozenset(a.idx for a in self._atom_infos if not a.in_ring)

    def heavy_atom_count(self) -> int:
        """Count non-hydrogen atoms.

        Hydrogen atoms (element == ``"H"``) are excluded.  This works correctly
        for explicit-hydrogen mol objects as well as implicit-H mol objects.
        """
        return sum(1 for a in self._atom_infos if a.element != "H")

    # ------------------------------------------------------------------
    # Bond queries
    # ------------------------------------------------------------------

    def get_bond_type(self, idx1: int, idx2: int) -> str | None:
        """Return the bond type string between atoms *idx1* and *idx2*.

        Parameters
        ----------
        idx1, idx2:
            RDKit atom indices.

        Returns
        -------
        str | None
            One of ``"single"``, ``"double"``, ``"triple"``, ``"aromatic"``, etc.
            ``None`` if no bond exists between the two atoms.
        """
        bond = self._mol.GetBondBetweenAtoms(idx1, idx2)  # type: ignore[attr-defined]
        if bond is None:
            return None
        return _rdkit_bond_type_to_str(bond.GetBondType())

    def has_bond(self, idx1: int, idx2: int) -> bool:
        """Return ``True`` if there is any bond between atoms *idx1* and *idx2*."""
        return self._mol.GetBondBetweenAtoms(idx1, idx2) is not None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    def atoms_with_charge(self) -> tuple[AtomInfo, ...]:
        """Return all atoms carrying a non-zero formal charge."""
        return tuple(a for a in self._atom_infos if a.charge != 0)

    def aromatic_atoms(self) -> frozenset[int]:
        """Return the set of atom indices flagged as aromatic by RDKit."""
        return frozenset(a.idx for a in self._atom_infos if a.aromatic)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AtomAnalysis({len(self._atom_infos)} atoms, "
            f"{self.heavy_atom_count()} heavy)"
        )
