"""
iupac_namer/perception/fragments.py

FragmentAnalysis subsystem — Subsystem 3 of the Perception layer.

Detects disconnected fragments (salts, multi-component species) using RDKit's
``GetMolFrags``.  This subsystem is independent of AtomAnalysis — it works
directly from the RDKit mol.

Fragment detection is consumed by the engine for salt handling.  Salts are
handled BEFORE interpretation generation (preprocessor step 3).

See ARCHITECTURE_PERCEPTION.md §Subsystem 3 for the full spec.
"""

from __future__ import annotations

from rdkit import Chem  # type: ignore[import]

from iupac_namer.types import Fragment


# ---------------------------------------------------------------------------
# FragmentAnalysis
# ---------------------------------------------------------------------------


class FragmentAnalysis:
    """Disconnected-fragment detection for an RDKit molecule.

    Computed once on construction.  All returned data is immutable.

    For a single-component molecule (no disconnected fragments), this returns
    a one-element tuple wrapping the whole molecule.

    For salts or multi-component species, each disconnected fragment is
    returned as a separate :class:`~iupac_namer.types.Fragment` with its own
    sub-mol and net charge.

    Parameters
    ----------
    mol:
        An RDKit ``Mol`` object (after sanitisation).
    """

    def __init__(self, mol: object) -> None:
        self._mol = mol
        self._fragments: tuple[Fragment, ...] = self._analyze()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _analyze(self) -> tuple[Fragment, ...]:
        """Detect disconnected fragments.

        Uses :func:`rdkit.Chem.GetMolFrags` to split the molecule.  For a
        single-component molecule the whole mol is wrapped directly without
        any atom-list surgery.
        """
        # Get atom-index lists for each fragment (tuple of tuples)
        frag_atom_lists: tuple[tuple[int, ...], ...] = Chem.GetMolFrags(self._mol)

        if len(frag_atom_lists) == 1:
            # Single fragment — return the whole molecule as-is.
            charge = sum(
                atom.GetFormalCharge() for atom in self._mol.GetAtoms()
            )
            return (
                Fragment(
                    atom_indices=frozenset(range(self._mol.GetNumAtoms())),
                    mol=self._mol,
                    charge=charge,
                ),
            )

        # Multiple fragments: build a sub-mol for each.
        # GetMolFrags(asMols=True) returns sanitised RDKit Mol objects.
        frag_mols: list[object] = list(
            Chem.GetMolFrags(self._mol, asMols=True, sanitizeFrags=True)
        )

        fragments: list[Fragment] = []
        for atom_indices, frag_mol in zip(frag_atom_lists, frag_mols):
            charge = sum(
                atom.GetFormalCharge()
                for atom in frag_mol.GetAtoms()  # type: ignore[attr-defined]
            )
            fragments.append(
                Fragment(
                    atom_indices=frozenset(atom_indices),
                    mol=frag_mol,
                    charge=charge,
                )
            )

        return tuple(fragments)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fragments(self) -> tuple[Fragment, ...]:
        """All detected fragments, in RDKit's canonical fragment order."""
        return self._fragments

    @property
    def is_salt(self) -> bool:
        """``True`` if the molecule has two or more disconnected fragments."""
        return len(self._fragments) > 1

    @property
    def fragment_count(self) -> int:
        """Number of disconnected fragments (always >= 1)."""
        return len(self._fragments)

    def largest_fragment(self) -> Fragment:
        """Return the fragment with the most atoms.

        For tie-breaking (equal atom count), the fragment appearing first in
        RDKit's canonical order is returned.
        """
        return max(self._fragments, key=lambda f: len(f.atom_indices))

    def fragments_by_charge(self, charge: int) -> tuple[Fragment, ...]:
        """Return all fragments whose net charge equals *charge*.

        Parameters
        ----------
        charge:
            Net formal charge, e.g. ``+1``, ``-1``, ``0``.
        """
        return tuple(f for f in self._fragments if f.charge == charge)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FragmentAnalysis({self.fragment_count} fragment(s), "
            f"is_salt={self.is_salt})"
        )
