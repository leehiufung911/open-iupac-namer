"""
iupac_namer/ring_naming/methylenedioxy_bridge.py

Stage 5 — methylenedioxy-bridge naming on retained polycyclic bases.

Background
----------
Some polycyclic ring systems have a 5-membered ``O-CH2-O`` dioxolane whose
two oxygen atoms attach to two *adjacent* atoms of an otherwise retained
polycyclic base (e.g. cyclopenta[a]phenanthrene / the steroid kernel).

For these systems IUPAC prefers a *methylenedioxy-bridge* construction over
a fused-dioxolo construction:

    <locant1>,<locant2>-methylenedioxy-<base-name>

rather than e.g. ``[1,3]dioxolo[4,5-X]<base>``.  The Stage 4 investigation
(commit f2f2b9d) demonstrated that OPSIN does not accept any
``[1,3]dioxolo[4,5-X]cyclopenta[a]phenanthrene`` (letters a..n) which
round-trips to the steroid D-ring 16,17-dioxolane topology — the canonical
IUPAC form is a methylenedioxy bridge on the saturated steroid kernel:

    16,17-methylenedioxy-hexadecahydro-1H-cyclopenta[a]phenanthrene

This module implements that construction with a TIGHT guard that avoids
collisions with existing naming paths:

  (a) The ring system must be "fused" (not "spiro" or "bridged") and contain
      at least 2 rings.
  (b) Exactly one constituent ring is a 5-ring with atoms
      ``O - CH2 - O - C - C`` where the CH2 has ``TotalNumHs() == 2``, both
      oxygens are neutral and non-aromatic, and the ring carbon has no exo
      substituents other than H's (``GetDegree() == 2``).
  (c) The two non-O, non-CH2 atoms of the 5-ring are adjacent atoms of the
      retained polycyclic base AND their connecting bond is NOT the 5-ring's
      internal bond — i.e. the bridge attaches to an edge already present
      in the base.
  (d) The remaining (N-1) rings, when carved to a sub-mol and looked up via
      the existing retained-name lookup, produce a curated retained name
      with a valid ``atom_locants`` dictionary that covers both attachment
      atoms.

If ALL those conditions hold, emit the methylenedioxy-bridge name.

Architectural notes
-------------------
* Amcinonide (FDA-0054) has an ACETONIDE (O-C(cyclopentyl)-O), NOT a
  methylenedioxy bridge: the ketal carbon carries the spiro-cyclopentane
  and has ``GetTotalNumHs() == 0``.  The guard above therefore does NOT
  fire on amcinonide; its existing polyspiro-articulation name is
  preserved unchanged (guaranteed by test_amcinonide_not_matched_as_
  methylenedioxy in tests/test_methylenedioxy_bridge.py).

* The FDA eval set does NOT currently contain any compound whose topology
  matches the tight guard (O-CH2-O on two adjacent atoms of a retained
  polycyclic base that cannot otherwise be named via benzodioxol / fused
  dioxolo retained paths).  All three O-CH2-O-containing eval compounds
  (ZT-2506, ZT-2621, FDA-1088) are on MONOCYCLIC benzene bases and are
  correctly named via the existing retained fused ``1,3-benzodioxol`` /
  ``[1,3]dioxol`` path.  Consequently this module cannot change the eval
  score; its value is purely architectural, enabling correct naming for
  methylenedioxy-bridged steroids / phenanthrenes / anthracenes / etc.
  that may appear in future datasets.

* The naming method rank is ``methylenedioxy_bridge = 45.0`` in
  ``strategy.py`` — beats ``spiro_polycyclic`` (40.0) so a genuine
  methylenedioxy-bridged steroid wins over an articulation-split polyspiro
  form, but loses to ``retained`` (100.0) so benzodioxol / [1,3]dioxol
  fused retained names always win for monocyclic bases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from iupac_namer.data_loader import _RING_CURATED_SMILES
from iupac_namer.types import Locant, NamedParent, Numbering

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard: detect a genuine O-CH2-O bridge on a retained polycyclic base
# ---------------------------------------------------------------------------


def _find_methylenedioxy_ring(
    ring_system: "RingSystem", mol
) -> tuple[frozenset[int], int, int, int] | None:
    """Return (bridge_ring, o1, ch2, o2) if the ring system contains exactly
    one 5-ring matching the O-CH2-O dioxolane pattern, or None.

    Guard details:
      * ring has exactly 5 atoms
      * exactly two of those atoms are neutral, non-aromatic oxygens whose
        only neighbors are the CH2 and one ring carbon of the base
      * exactly one atom between the two oxygens is a sp3 carbon with
        ``TotalNumHs() == 2`` AND ``GetDegree() == 2`` (no exo substituents)
      * the two remaining atoms (the base-side attachment pair) are both
        carbons in the ring system but NOT members of any OTHER ring of the
        same ring system with only 5 atoms that also matches (to keep the
        match unique)
    """
    candidates: list[tuple[frozenset[int], int, int, int]] = []
    for ring in ring_system.rings:
        if len(ring) != 5:
            continue
        oxygens: list[int] = []
        carbons: list[int] = []
        for a in ring:
            atom = mol.GetAtomWithIdx(a)
            num = atom.GetAtomicNum()
            if num == 8:
                if atom.GetFormalCharge() != 0:
                    break
                if atom.GetIsAromatic():
                    break
                oxygens.append(a)
            elif num == 6:
                carbons.append(a)
            else:
                break
        else:
            if len(oxygens) != 2 or len(carbons) != 3:
                continue
            # Identify the CH2 (must bond to both oxygens)
            o_set = set(oxygens)
            ch2_candidates: list[int] = []
            for c in carbons:
                catom = mol.GetAtomWithIdx(c)
                nbr_idx = {n.GetIdx() for n in catom.GetNeighbors()}
                if not o_set.issubset(nbr_idx):
                    continue
                if catom.GetTotalNumHs() != 2:
                    continue
                if catom.GetDegree() != 2:
                    continue
                if catom.GetFormalCharge() != 0:
                    continue
                if catom.GetIsAromatic():
                    continue
                # No double/triple bonds from this C
                if any(
                    b.GetBondTypeAsDouble() != 1.0 for b in catom.GetBonds()
                ):
                    continue
                ch2_candidates.append(c)
            if len(ch2_candidates) != 1:
                continue
            ch2 = ch2_candidates[0]
            base_attach = [c for c in carbons if c != ch2]
            if len(base_attach) != 2:
                continue
            # The two attach carbons must be directly bonded (adjacent) —
            # otherwise this is a peri-fused dioxole, not a bridge.
            a1, a2 = base_attach
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue
            # Validate each O has EXACTLY two neighbors: CH2 and one base C.
            o1, o2 = oxygens
            o1_nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(o1).GetNeighbors()}
            o2_nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(o2).GetNeighbors()}
            if len(o1_nbrs) != 2 or len(o2_nbrs) != 2:
                continue
            if ch2 not in o1_nbrs or ch2 not in o2_nbrs:
                continue
            # O1 must bond to exactly one of {a1, a2}, and O2 to the other
            o1_base = (o1_nbrs - {ch2}).pop()
            o2_base = (o2_nbrs - {ch2}).pop()
            if {o1_base, o2_base} != {a1, a2}:
                continue
            candidates.append((frozenset(ring), o1, ch2, o2))

    if len(candidates) != 1:
        return None
    return candidates[0]


def _base_ring_system_and_locants(
    ring_system: "RingSystem",
    mol,
    bridge_ring: frozenset[int],
    o1: int,
    o2: int,
) -> tuple[str, dict[int, Locant], int, int] | None:
    """Given the full ring_system and the identified O-CH2-O bridge ring,
    carve the remaining (N-1) rings as a sub-ring-system, try a retained-
    name lookup, and extract the IUPAC locants for the two base atoms that
    the oxygens attach to.

    Returns ``(base_name, atom_to_locant, base_loc1, base_loc2)`` on success
    or None if the base does not resolve to a retained polycyclic name with
    complete atom_locants that cover both attachment atoms.

    ``atom_to_locant`` maps full-molecule atom indices of the BASE ring
    atoms → IUPAC locants.
    """
    from iupac_namer.ring_naming.common import extract_ring_mol

    # The base atoms = ring_system atoms minus the three unique dioxolane
    # atoms (CH2 + both oxygens).  The two attach carbons (a1, a2) remain
    # base atoms; the bridge ring's set ∩ base = {a1, a2}.
    ch2_and_o = set()
    for a in bridge_ring:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetAtomicNum() == 8:
            ch2_and_o.add(a)
        elif atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2:
            ch2_and_o.add(a)
    base_atoms = frozenset(ring_system.atom_indices) - ch2_and_o
    if not base_atoms:
        return None

    # Build a synthetic RingSystem for the base so we can reuse
    # extract_ring_mol's carving + canonical-SMILES machinery.  We only
    # need the atom_indices and type — no ring-by-ring decomposition is
    # needed here because the retained-name curated table is keyed on
    # a canonical SMILES of the carved sub-mol.
    from iupac_namer.types import RingSystem as RS

    base_rs = RS(
        atom_indices=base_atoms,
        rings=tuple(r for r in ring_system.rings if r != bridge_ring),
        type=ring_system.type,   # typically "fused"
        aromatic=ring_system.aromatic,
        bridge_sizes=None,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=None,
        ring_size=len(base_atoms),
    )
    base_mol = extract_ring_mol(base_rs, mol)
    if base_mol is None:
        return None
    try:
        base_smiles = Chem.MolToSmiles(base_mol)
    except Exception:
        return None
    # Stereo-stripped canonical (steroid fragments carry @H descriptors
    # that cause exact-string miss against the stereo-free curated keys).
    base_no_stereo: str | None = None
    try:
        ns = Chem.Mol(base_mol)
        Chem.RemoveStereochemistry(ns)
        ns_smi = Chem.MolToSmiles(ns)
        if ns_smi and ns_smi != base_smiles:
            base_no_stereo = ns_smi
    except Exception:
        base_no_stereo = None

    record = _RING_CURATED_SMILES.get(base_smiles)
    curated_key = base_smiles if record is not None else None
    if record is None and base_no_stereo:
        record = _RING_CURATED_SMILES.get(base_no_stereo)
        if record is not None:
            curated_key = base_no_stereo
    if record is None:
        return None
    atom_locants_raw = record.get("atom_locants")
    if not atom_locants_raw:
        return None
    base_name_str = record["name"]

    # Must be a LARGE polycyclic retained base (>= 4 rings).
    #
    # Rationale (Stage 4 investigation, commit f2f2b9d):
    #   * Monocyclic bases (benzene, pyridine) → use existing retained fused
    #     ``1,3-benzodioxol`` / ``[1,3]dioxol`` path.  OPSIN accepts both the
    #     fused and the 3,4-methylenedioxybenzene forms; the retained fused
    #     name is IUPAC-preferred.
    #   * 2-ring bases (naphthalene, quinoline) → existing Stage 2B
    #     ``name_fused`` emits ``[1,3]dioxolo[4,5-b]naphthalene`` etc.,
    #     which OPSIN accepts and IUPAC prefers over a methylenedioxy
    #     bridge on naphthalene / decalin.  Firing Stage 5 here causes
    #     regressions (test_stage2_fused_ring_name).
    #   * 3-ring bases (anthracene, phenanthrene, chrysene) → OPSIN still
    #     accepts the fused dioxolo form (see
    #     test_opsin_accepts_fused_dioxolo_on_other_4ring_bases), so defer
    #     Stage 5 to larger bases.
    #   * 4+ ring bases (cyclopenta[a]phenanthrene / steroid kernel) →
    #     OPSIN rejects every fused ``[1,3]dioxolo[4,5-X]`` form that
    #     round-trips, so the methylenedioxy-bridge form is the ONLY
    #     canonical IUPAC name that OPSIN accepts.  Stage 5 fires here.
    if len(base_rs.rings) < 4:
        return None

    # Substructure-match the curated-key mol into the full mol so we can
    # assign IUPAC locants to base atoms.
    try:
        rebuilt = Chem.MolFromSmiles(curated_key)
        if rebuilt is None or rebuilt.GetNumAtoms() != base_mol.GetNumAtoms():
            rebuilt = base_mol
    except Exception:
        rebuilt = base_mol
    try:
        matches = list(mol.GetSubstructMatches(rebuilt, uniquify=False))
    except Exception:
        matches = []
    if not matches:
        return None

    # For each candidate match, build a {full_mol_idx: Locant} map and
    # check it covers both attach atoms (o1's and o2's base-side neighbor).
    o1_base = next(
        (n.GetIdx() for n in mol.GetAtomWithIdx(o1).GetNeighbors()
         if n.GetIdx() in base_atoms),
        None,
    )
    o2_base = next(
        (n.GetIdx() for n in mol.GetAtomWithIdx(o2).GetNeighbors()
         if n.GetIdx() in base_atoms),
        None,
    )
    if o1_base is None or o2_base is None:
        return None

    def _parse_locant(raw) -> Locant:
        if isinstance(raw, int):
            return Locant.numeric(raw)
        s = str(raw)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num = int(s[:i]) if s[:i] else 0
        suf = s[i:]
        if suf:
            return Locant.hetero(num, suf)
        return Locant.numeric(num)

    def _locant_key(loc: Locant) -> tuple[int, str]:
        return (loc._numeric_value or 0, getattr(loc, "suffix", "") or "")

    best: tuple[
        str, dict[int, Locant], int, int, tuple[int, int]
    ] | None = None
    for match in matches:
        if match is None:
            continue
        atom_to_loc: dict[int, Locant] = {}
        ok = True
        for ring_idx, iupac_loc in atom_locants_raw.items():
            if ring_idx >= len(match):
                ok = False
                break
            full_idx = match[ring_idx]
            atom_to_loc[full_idx] = _parse_locant(iupac_loc)
        if not ok:
            continue
        loc1 = atom_to_loc.get(o1_base)
        loc2 = atom_to_loc.get(o2_base)
        if loc1 is None or loc2 is None:
            continue
        # IUPAC: cite the lower-locant pair.  Order (a, b) with a < b.
        pair = tuple(sorted(
            ((loc1._numeric_value or 0), (loc2._numeric_value or 0))
        ))
        if best is None or pair < best[4]:
            best = (base_name_str, atom_to_loc, loc1._numeric_value or 0,
                    loc2._numeric_value or 0, pair)

    if best is None:
        return None
    base_name_out, atom_to_loc, locA, locB, _pair = best
    return base_name_out, atom_to_loc, locA, locB


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def name_methylenedioxy_bridge(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Generate a methylenedioxy-bridge-on-retained-polycyclic-base name.

    Returns a 1-element list ``[NamedParent]`` on success, or ``[]`` when
    any guard condition fails.  Refuses silently on failure — the caller
    falls back to other naming pathways.
    """
    # Guard 0: must be a multi-ring fused system.
    if ring_system.type != "fused":
        return []
    if len(ring_system.rings) < 2:
        return []

    # Guard 1: find the O-CH2-O bridge ring.
    found = _find_methylenedioxy_ring(ring_system, mol)
    if found is None:
        return []
    bridge_ring, o1, ch2, o2 = found

    # Guard 2: resolve the base to a retained polycyclic name + locants.
    base_info = _base_ring_system_and_locants(
        ring_system, mol, bridge_ring, o1, o2
    )
    if base_info is None:
        return []
    base_name_str, atom_to_loc, locA, locB = base_info

    if locA == 0 or locB == 0 or locA == locB:
        return []

    # IUPAC: lower locant cited first.
    low = min(locA, locB)
    high = max(locA, locB)

    name_str = f"{low},{high}-methylenedioxy-{base_name_str}"
    stem_str = name_str[:-1] if name_str.endswith("e") else name_str

    # Build a Numbering covering base atoms + the three bridge atoms.  The
    # base atoms get their IUPAC locants from the retained lookup; the
    # bridge atoms (CH2 + both Os) are NOT locanted in the canonical name
    # — they're encoded by the "methylenedioxy" prefix.  We omit them from
    # the numbering so downstream substituent scoring does not try to
    # assign locants there.
    assignments = tuple(sorted(atom_to_loc.items(), key=lambda kv: (
        kv[1]._numeric_value or 0, getattr(kv[1], "suffix", "") or ""
    )))
    locant_set = tuple(loc for _, loc in assignments)
    numbering = Numbering(
        _assignments=assignments,
        locant_set=locant_set,
    )

    # The NamedParent's candidate must claim ALL ring-system atoms (base +
    # bridge), matching the original CandidateParent.  We keep the caller's
    # candidate unchanged.
    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem_str,
        alkyl_stem=None,
        naming_method="methylenedioxy_bridge",
        indicated_hydrogen=None,
        numbering_options=(numbering,),
    )]


__all__ = ["name_methylenedioxy_bridge"]
