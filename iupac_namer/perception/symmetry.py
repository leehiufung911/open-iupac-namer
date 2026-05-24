"""
iupac_namer/perception/symmetry.py

SymmetryAnalysis subsystem — Subsystem 6 of the Perception layer.

Detects identical substructures for multiplicative and ring-assembly
nomenclature.

Two categories are detected:

  1. **Ring assembly** — two or more identical ring systems connected by a
     direct bond (no intervening atoms), e.g. biphenyl.  These become
     ``SymmetryGroup`` objects with ``linking_type == "direct_bond"``.

  2. **Multiplicative** — two or more identical subunits connected via an
     intervening linking group (e.g. -O-, -NH-, -CH2-).  These become
     ``SymmetryGroup`` objects with ``linking_type == "linking_group"``.

Results are computed once on construction and cached.  All public data is
immutable.

See ARCHITECTURE_PERCEPTION.md §Subsystem 6 for the full spec.
See ARCHITECTURE_DATA_STRUCTURES.md for the SymmetryGroup dataclass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.types import SymmetryGroup

if TYPE_CHECKING:
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom counts for small linking groups (multiplicative nomenclature).
# A linking group must be small (1-4 heavy atoms) to qualify.
# ---------------------------------------------------------------------------

_MAX_LINKING_ATOMS = 4


# ---------------------------------------------------------------------------
# Single-substituent locant omission by parent symmetry (P-14.3.4.4)
# ---------------------------------------------------------------------------

def single_substituent_locant_forced_by_symmetry(
    mol: object,
    parent_atom_indices: "frozenset[int] | tuple[int, ...] | list[int]",
    attach_atom_idx: int,
    substituent_atom_indices: "frozenset[int] | tuple[int, ...] | list[int]",
) -> bool:
    """Return True when a single substituent's locant is forced by symmetry.

    Implements IUPAC 2013 P-14.3.4.4 (omission of locants that are unique by
    symmetry): when *every* parent position at which the single substituent
    could attach is in one symmetry class, the substituent can only produce one
    distinct compound regardless of which equivalent position it occupies, so
    the locant is redundant and omitted from the PIN.  This generalises the
    long-standing all-carbon-monocyclic special case (``chlorobenzene``,
    ``methylcyclohexane``) to *fused* (``chlorocoronene``) and *heterocyclic*
    (``pyrazinecarboxylic acid``) parents.

    The single substituent may be a prefix (``Cl`` on coronene) **or** the
    sole principal-characteristic-group suffix (``-carboxylic acid`` on
    pyrazine).  In both cases ``substituent_atom_indices`` are the atoms of that
    one substituent and ``attach_atom_idx`` is the parent backbone atom it is
    bonded to.

    Algorithm (graph automorphism via RDKit canonical ranks):
      1. Build the *parent skeleton* = the molecule with the single substituent's
         atoms deleted and the freed valences capped with explicit hydrogen.
         This removes the symmetry-breaking influence of the substituent itself
         while preserving every other structural feature (a double bond, a
         heteroatom, a charge centre, or a second substituent that breaks ring
         symmetry all survive and correctly make the result False — cf.
         3-bromocyclohex-1-ene, where positions 3/4/5 are inequivalent).
      2. Compute ``CanonicalRankAtoms(breakTies=False)`` on the skeleton; atoms
         that share a rank are graph-symmetry-equivalent.
      3. The *candidate* positions are **all** parent atoms that have a free
         hydrogen in the skeleton — i.e. every position OPSIN could attach an
         unlocanted substituent to, regardless of element.  The locant is
         forced iff every such candidate is in the attach atom's rank class.
         Including heteroatoms is essential for round-trip safety: barbituric
         acid (1,3-diazinane-2,4,6-trione) has two free N-H positions and one
         free C-H (C5); the C5-amino is graph-unique among *carbons* but OPSIN
         defaults an unlocanted "amino" to N1, so the locant is load-bearing and
         the rule must (and does) return False here.

    Returns False (keep the locant) on any structural ambiguity or on internal
    failure — the rule is strictly conservative.
    """
    from rdkit import Chem

    parent_set = set(parent_atom_indices)
    remove = set(substituent_atom_indices)
    if attach_atom_idx in remove:
        # The attach atom must survive in the skeleton.
        return False
    if attach_atom_idx not in parent_set:
        return False

    keep = [i for i in range(mol.GetNumAtoms()) if i not in remove]
    rw = Chem.RWMol()
    old2new: dict[int, int] = {}
    for i in keep:
        a = mol.GetAtomWithIdx(i)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        # Pin the original hydrogen count so removing the substituent does not
        # silently re-perceive valence; freed valences are re-added below.
        na.SetNumExplicitHs(a.GetTotalNumHs())
        na.SetNoImplicit(True)
        old2new[i] = rw.AddAtom(na)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in old2new and j in old2new:
            rw.AddBond(old2new[i], old2new[j], b.GetBondType())
    # Cap each freed valence (every bond that crossed into the removed
    # substituent) with an explicit hydrogen so the skeleton is a valid,
    # neutral parent hydride.
    for i in keep:
        orig = mol.GetAtomWithIdx(i)
        lost = sum(1 for n in orig.GetNeighbors() if n.GetIdx() in remove)
        if lost:
            na = rw.GetAtomWithIdx(old2new[i])
            na.SetNumExplicitHs(na.GetNumExplicitHs() + lost)

    skel = rw.GetMol()
    try:
        Chem.SanitizeMol(skel)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "single_substituent_locant_forced_by_symmetry: skeleton "
            "sanitise failed (%s) — keeping locant", exc,
        )
        return False

    try:
        ranks = list(Chem.CanonicalRankAtoms(skel, breakTies=False))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "single_substituent_locant_forced_by_symmetry: rank failed "
            "(%s) — keeping locant", exc,
        )
        return False

    candidates = [
        i for i in parent_atom_indices
        if i in old2new
        and skel.GetAtomWithIdx(old2new[i]).GetTotalNumHs() > 0
    ]
    if not candidates:
        return False
    attach_rank = ranks[old2new[attach_atom_idx]]
    return all(ranks[old2new[i]] == attach_rank for i in candidates)


class SymmetryAnalysis:
    """Detect identical substructures in an RDKit molecule.

    Parameters
    ----------
    mol:
        An RDKit ``Mol`` object (after sanitisation).
    atom_analysis:
        The :class:`~iupac_namer.perception.atoms.AtomAnalysis` for this mol.
    ring_analysis:
        The :class:`~iupac_namer.perception.rings.RingAnalysis` for this mol.
    """

    def __init__(
        self,
        mol: object,
        atom_analysis: "AtomAnalysis",
        ring_analysis: "RingAnalysis",
    ) -> None:
        self._mol = mol
        self._atoms = atom_analysis
        self._rings = ring_analysis
        self._symmetry_groups: tuple[SymmetryGroup, ...] = self._analyze()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def symmetry_groups(self) -> tuple[SymmetryGroup, ...]:
        """All detected symmetry groups (immutable tuple)."""
        return self._symmetry_groups

    @property
    def has_symmetry(self) -> bool:
        """True if at least one symmetry group was detected."""
        return len(self._symmetry_groups) > 0

    @property
    def ring_assembly_candidates(self) -> tuple[SymmetryGroup, ...]:
        """Symmetry groups that are ring-assembly candidates (direct bond linking)."""
        return tuple(
            sg for sg in self._symmetry_groups if sg.linking_type == "direct_bond"
        )

    @property
    def multiplicative_candidates(self) -> tuple[SymmetryGroup, ...]:
        """Symmetry groups that are multiplicative candidates (linking group)."""
        return tuple(
            sg for sg in self._symmetry_groups if sg.linking_type == "linking_group"
        )

    # ------------------------------------------------------------------
    # Internal analysis
    # ------------------------------------------------------------------

    def _analyze(self) -> tuple[SymmetryGroup, ...]:
        """Detect all symmetry groups; return an immutable tuple."""
        groups: list[SymmetryGroup] = []

        groups.extend(self._detect_ring_assemblies())
        groups.extend(self._detect_multiplicative())

        return tuple(groups)

    # ------------------------------------------------------------------
    # Ring assembly detection
    # ------------------------------------------------------------------

    def _detect_ring_assemblies(self) -> list[SymmetryGroup]:
        """Find pairs (or larger groups) of identical directly-bonded ring systems."""
        from rdkit import Chem

        ring_systems = self._rings.ring_systems
        if len(ring_systems) < 2:
            return []

        # Compute canonical SMILES for each ring system to identify identical ones.
        ring_smiles: dict[str, list] = {}
        for rs in ring_systems:
            sub_mol = self._extract_submol(rs.atom_indices)
            if sub_mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(sub_mol)  # type: ignore[attr-defined]
            except Exception:
                logger.debug(
                    "Could not compute canonical SMILES for ring system atoms=%s",
                    rs.atom_indices,
                )
                continue
            ring_smiles.setdefault(smi, []).append(rs)

        groups: list[SymmetryGroup] = []

        for smi, rs_list in ring_smiles.items():
            if len(rs_list) < 2:
                continue

            # For each pair of identical ring systems, check direct bonding.
            for i, rs1 in enumerate(rs_list):
                for rs2 in rs_list[i + 1 :]:
                    if not self._are_directly_bonded(rs1.atom_indices, rs2.atom_indices):
                        continue

                    linking = self._find_linking_atoms(rs1.atom_indices, rs2.atom_indices)
                    sub_mol_obj = self._extract_submol(rs1.atom_indices)

                    sg = SymmetryGroup(
                        subunit_atoms=(rs1.atom_indices, rs2.atom_indices),
                        subunit_mol=sub_mol_obj,
                        linking_atoms=linking,
                        linking_type="direct_bond",
                        linking_group_mol=None,
                        multiplicity=2,
                    )
                    groups.append(sg)
                    logger.debug(
                        "Ring assembly candidate: smi=%r, atoms1=%s, atoms2=%s",
                        smi,
                        rs1.atom_indices,
                        rs2.atom_indices,
                    )

        return groups

    # ------------------------------------------------------------------
    # Multiplicative detection
    # ------------------------------------------------------------------

    def _detect_multiplicative(self) -> list[SymmetryGroup]:
        """Find pairs of identical subunits connected by a small linking group.

        Strategy:
          1. Find all bonds where removing them disconnects the molecule into
             at least two large fragments (not just a hydrogen/leaf atom).
          2. For each such bond, check whether the two "sides" contain identical
             ring systems (or identical acyclic fragments).
          3. The atoms NOT in either subunit form the linking group.

        This is a lightweight heuristic adequate for Phase 1.  Full support for
        arbitrary multiplicative assemblies can be added in later phases.
        """
        from rdkit import Chem

        groups: list[SymmetryGroup] = []

        ring_systems = self._rings.ring_systems
        if len(ring_systems) < 2:
            return groups

        # Build a lookup: atom_idx -> ring system (if the atom is in a ring system)
        atom_to_rs: dict[int, object] = {}
        for rs in ring_systems:
            for idx in rs.atom_indices:
                atom_to_rs[idx] = rs

        # Compute canonical SMILES for each ring system
        rs_to_smi: dict[int, str] = {}  # id(rs) -> canonical_smi
        for rs in ring_systems:
            sub_mol = self._extract_submol(rs.atom_indices)
            if sub_mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(sub_mol)  # type: ignore[attr-defined]
                rs_to_smi[id(rs)] = smi
            except Exception:
                pass

        # Find pairs of ring systems connected by a small linking group
        for i, rs1 in enumerate(ring_systems):
            smi1 = rs_to_smi.get(id(rs1))
            if smi1 is None:
                continue
            for rs2 in ring_systems[i + 1 :]:
                smi2 = rs_to_smi.get(id(rs2))
                if smi2 is None or smi1 != smi2:
                    continue

                # They are identical ring systems — check for a linking group.
                linking_atoms = self._find_linking_group_atoms(
                    rs1.atom_indices, rs2.atom_indices
                )
                if linking_atoms is None:
                    continue  # Not connected through a small linker

                # Extract the linking group mol
                linking_mol = self._extract_submol(linking_atoms)
                sub_mol_obj = self._extract_submol(rs1.atom_indices)

                sg = SymmetryGroup(
                    subunit_atoms=(rs1.atom_indices, rs2.atom_indices),
                    subunit_mol=sub_mol_obj,
                    linking_atoms=linking_atoms,
                    linking_type="linking_group",
                    linking_group_mol=linking_mol,
                    multiplicity=2,
                )
                groups.append(sg)
                logger.debug(
                    "Multiplicative candidate: smi=%r, linker_atoms=%s",
                    smi1,
                    linking_atoms,
                )

        return groups

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _are_directly_bonded(
        self, atoms1: frozenset[int], atoms2: frozenset[int]
    ) -> bool:
        """Return True if at least one atom in atoms1 is directly bonded to
        an atom in atoms2 (i.e. there is a bond crossing the boundary)."""
        for a1 in atoms1:
            atom_info = self._atoms[a1]
            for nb in atom_info.neighbors:
                if nb in atoms2:
                    return True
        return False

    def _find_linking_atoms(
        self, atoms1: frozenset[int], atoms2: frozenset[int]
    ) -> frozenset[int]:
        """Return the set of atoms (from atoms1 and atoms2) that lie on cross-bonds."""
        linking: set[int] = set()
        for a1 in atoms1:
            atom_info = self._atoms[a1]
            for nb in atom_info.neighbors:
                if nb in atoms2:
                    linking.add(a1)
                    linking.add(nb)
        return frozenset(linking)

    def _find_linking_group_atoms(
        self, atoms1: frozenset[int], atoms2: frozenset[int]
    ) -> frozenset[int] | None:
        """Return the set of linking-group atoms between two non-adjacent ring systems.

        The linking group consists of atoms that are:
          - NOT in atoms1 or atoms2
          - Reachable from atoms1 via non-ring atoms without entering atoms2

        Returns None if no such linking group exists (atoms1 and atoms2 are
        directly bonded or not connected at all) or if the linking group is
        too large (> _MAX_LINKING_ATOMS atoms).
        """
        # Collect all non-ring, non-subunit atoms as possible linker candidates.
        all_indices = {a.idx for a in self._atoms.all_atoms}
        excluded = atoms1 | atoms2
        candidate_linker = all_indices - excluded

        # BFS from boundary atoms of atoms1 through candidate_linker atoms.
        # Boundary atoms of atoms1 are those with at least one neighbour not in atoms1.
        boundary1 = frozenset(
            a for a in atoms1
            if any(nb not in atoms1 for nb in self._atoms[a].neighbors)
        )

        visited: set[int] = set()
        frontier = set()

        for b in boundary1:
            for nb in self._atoms[b].neighbors:
                if nb in candidate_linker:
                    frontier.add(nb)

        if not frontier:
            # atoms1 and atoms2 may be directly bonded or disconnected — not a
            # multiplicative pattern.
            return None

        # BFS through candidate_linker
        while frontier:
            nxt: set[int] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for nb in self._atoms[node].neighbors:
                    if nb in candidate_linker and nb not in visited:
                        nxt.add(nb)
                    # Do not cross into atoms2 during BFS
            frontier = nxt

        linker = frozenset(visited)

        if not linker:
            return None

        # Verify linker actually bridges to atoms2
        linker_borders_atoms2 = any(
            nb in atoms2
            for a in linker
            for nb in self._atoms[a].neighbors
        )
        if not linker_borders_atoms2:
            return None

        # Reject oversized linking groups
        if len(linker) > _MAX_LINKING_ATOMS:
            logger.debug(
                "Linking group too large (%d atoms) — skipping multiplicative candidate",
                len(linker),
            )
            return None

        return linker

    def _extract_submol(self, atom_indices: frozenset[int]) -> object | None:
        """Extract a sub-molecule for the given atom indices using RDKit."""
        from rdkit import Chem
        from rdkit.Chem import AllChem  # noqa: F401

        try:
            indices = sorted(atom_indices)
            rw_mol = Chem.RWMol(self._mol)  # type: ignore[attr-defined]
            atoms_to_keep = set(indices)
            # Remove atoms NOT in atoms_to_keep (reverse order to preserve indices)
            atoms_to_remove = sorted(
                (a.GetIdx() for a in rw_mol.GetAtoms() if a.GetIdx() not in atoms_to_keep),
                reverse=True,
            )
            for idx in atoms_to_remove:
                rw_mol.RemoveAtom(idx)
            Chem.SanitizeMol(rw_mol)  # type: ignore[attr-defined]
            return rw_mol.GetMol()
        except Exception as exc:
            logger.debug("_extract_submol failed for atoms=%s: %s", atom_indices, exc)
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SymmetryAnalysis("
            f"n_ring_assembly={len(self.ring_assembly_candidates)}, "
            f"n_multiplicative={len(self.multiplicative_candidates)})"
        )

    def __len__(self) -> int:
        return len(self._symmetry_groups)
