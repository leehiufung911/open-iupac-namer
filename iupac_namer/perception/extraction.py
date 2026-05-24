"""
iupac_namer/perception/extraction.py

Fragment carving utilities — used by path handler execute() methods during the
execution phase to produce standalone sub-molecules for recursive naming.

Functions
---------
carve_substituent          — extract a single-attachment substituent as a mol
carve_bridging_substituent — extract a bridging substituent (2+ attachment points)
carve_fc_fragments         — split a molecule at functional-class boundaries (stub)
strip_additive_atoms       — remove additive atoms (N-oxide O, P-oxide O) from mol

v13 changes
-----------
G1  Canonical index normalisation: after FragmentOnBonds + dummy replacement the
    fragment is renumbered using RDKit canonical atom ranking so that any two
    carving operations that produce the same canonical SMILES also produce the
    same attachment-atom index.
C1  strip_additive_atoms: remove N-oxide and P-oxide atoms, adjust formal charge.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # RDKit imported lazily to keep import cost low at module level

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_renumber(mol: object, attachment_idxs: list[int]) -> tuple:
    """Renumber *mol* according to RDKit canonical atom ordering.

    Parameters
    ----------
    mol:
        An RDKit mol (may be an RWMol; will be treated as a Mol).
    attachment_idxs:
        Atom indices (in *mol*'s current numbering) that are "special" and
        whose new indices should be returned.

    Returns
    -------
    (reordered_mol, new_attachment_idxs)
        *reordered_mol* has atoms renumbered in canonical order.
        *new_attachment_idxs* maps each entry in *attachment_idxs* to its new
        index in *reordered_mol*.
    """
    from rdkit import Chem

    n = mol.GetNumAtoms()
    canonical_ranks = list(Chem.CanonicalRankAtoms(mol))
    # new_to_old[new_idx] = old_idx, sorted ascending by canonical rank
    new_to_old = sorted(range(n), key=lambda i: canonical_ranks[i])
    old_to_new = {old: new for new, old in enumerate(new_to_old)}

    reordered = Chem.RenumberAtoms(mol, new_to_old)
    new_attachment = [old_to_new[idx] for idx in attachment_idxs]
    return reordered, new_attachment


def _replace_dummy_with_hydrogen(rw: object) -> None:
    """In-place: replace all dummy atoms (atomic number 0) with hydrogen."""
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)


def _clear_nonring_aromaticity(rw: object) -> None:
    """In-place: clear the aromatic flag from any atom/bond not in a ring.

    After carving an aromatic ring atom into an acyclic fragment, RDKit retains
    the aromatic flag on that atom but SanitizeMol then rejects the fragment
    ("non-ring atom X marked aromatic").  The aromatic flag must be cleared
    BEFORE re-sanitization so that SMARTS matchers see normal sp2/sp3 atoms.
    """
    # RingInfo may be stale after FragmentOnBonds; recompute.
    from rdkit import Chem
    try:
        Chem.GetSSSR(rw)
    except Exception:
        pass
    ring_info = rw.GetRingInfo()
    for atom in rw.GetAtoms():
        if atom.GetIsAromatic() and not ring_info.NumAtomRings(atom.GetIdx()):
            atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        if bond.GetIsAromatic() and not ring_info.NumBondRings(bond.GetIdx()):
            bond.SetIsAromatic(False)
            # If the bond was formerly aromatic and now severed from its ring,
            # demote it to SINGLE; sanitization will re-derive the correct
            # order from explicit bond types / valence.
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.SINGLE)


def _find_fragment_containing(frag_atom_lists: tuple,
                               frag_mols: tuple,
                               target_orig_idx: int) -> tuple[int, int]:
    """Return (frag_idx, atom_idx_within_frag) for *target_orig_idx*.

    After ``Chem.GetMolFrags(fragmented, asMols=False)`` the i-th entry of
    *frag_atom_lists* gives a tuple of original-molecule indices for the atoms
    in fragment *i*.  This maps back to the original index to find which
    fragment a given atom ended up in.
    """
    for frag_idx, orig_indices in enumerate(frag_atom_lists):
        for local_idx, orig_idx in enumerate(orig_indices):
            if orig_idx == target_orig_idx:
                return frag_idx, local_idx
    raise ValueError(
        f"Atom with original index {target_orig_idx} not found in any fragment"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def carve_substituent(
    mol: object,
    substituent_atoms: "frozenset[int]",
    attachment_bond: "tuple[int, int]",
) -> "tuple[object, int, int]":
    """Extract a substituent fragment as a standalone mol.

    Parameters
    ----------
    mol:
        RDKit mol of the parent molecule.
    substituent_atoms:
        Atom indices belonging to the substituent (not used for the cut — the
        attachment bond determines the cut; this parameter is kept for API
        symmetry and future validation).
    attachment_bond:
        ``(parent_atom_idx, substituent_atom_idx)`` — the bond to cut.

    Returns
    -------
    (fragment_mol, attachment_atom_in_fragment, attachment_bond_order)
        *fragment_mol* is a sanitised RDKit mol representing the substituent
        with the cut bond replaced by H.
        *attachment_atom_in_fragment* is the canonical index of the atom in
        *fragment_mol* that was bonded to the parent.
        *attachment_bond_order* is the integer bond order (1 = single,
        2 = double, 3 = triple).

    Raises
    ------
    ValueError
        If the attachment bond does not exist or a fragment cannot be found.

    v13 G1
        After FragmentOnBonds + dummy replacement the fragment is canonically
        renumbered so that equivalent substituents always have the same
        attachment index.
    """
    from rdkit import Chem

    parent_atom_idx, sub_atom_idx = attachment_bond

    bond = mol.GetBondBetweenAtoms(parent_atom_idx, sub_atom_idx)
    if bond is None:
        raise ValueError(
            f"No bond between atoms {parent_atom_idx} and {sub_atom_idx}"
        )
    bond_order = int(bond.GetBondTypeAsDouble())
    bond_idx = bond.GetIdx()

    # Capture the parent-mol CIP descriptor at the substituent-side attachment
    # atom BEFORE carving.  After carving, the dummy→H replacement removes a
    # heavy neighbour and the attachment atom may no longer appear stereogenic
    # in isolation (e.g. an α-C of an amino-acid residue carved through the
    # amide N: -NH-[C@H](CH3)-... loses its CIP code because the N is replaced
    # by H, leaving two identical H neighbours).  The atom is still
    # stereogenic in context, however, and IUPAC P-91.5.4 requires reporting
    # the parent-derived descriptor.  Stash it as a property to be re-emitted
    # by StereoAnalysis when the carved fragment is named.
    #
    # We capture the parent CIP code for EVERY atom in the parent mol that
    # carries one — not just the attachment atom.  Internal stereocenters of
    # the substituent can have their CIP priorities flipped after carving
    # because the parent-side branch (now an H) loses neighbours that
    # contributed to CIP ranking.  Per IUPAC P-92.1.4.3, the free valence on
    # a substituent is treated as a "phantom" atom of the parent's identity,
    # i.e. CIP within a substituent should be evaluated *as if* the
    # substituent were still attached to its parent.  The simplest faithful
    # implementation: inherit the parent's CIP descriptor for every atom in
    # the carved fragment.  StereoAnalysis prefers _ParentCIPCode over the
    # locally-recomputed _CIPCode.
    parent_cip = None
    parent_cip_map: "dict[int, str]" = {}  # parent_atom_idx → CIP code
    try:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=False)  # type: ignore[attr-defined]
        # Refresh CIP labels using the modern labeller so that
        # pseudoasymmetric (lowercase ``r`` / ``s``) and meso descriptors are
        # correct.  The legacy ``AssignStereochemistry`` above seeds chiral
        # tags / bond stereo flags that ``rdCIPLabeler`` reads; the modern
        # call overwrites the ``_CIPCode`` properties with authoritative
        # values.
        try:
            from rdkit.Chem import rdCIPLabeler  # type: ignore[attr-defined]
            rdCIPLabeler.AssignCIPLabels(mol)
        except Exception:  # pragma: no cover — defensive only
            pass
        sub_atom = mol.GetAtomWithIdx(sub_atom_idx)
        if sub_atom.HasProp("_CIPCode"):
            parent_cip = sub_atom.GetProp("_CIPCode")
        for parent_atom in mol.GetAtoms():
            if parent_atom.HasProp("_CIPCode"):
                parent_cip_map[parent_atom.GetIdx()] = parent_atom.GetProp("_CIPCode")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to read parent CIP code at attachment: %s", exc)

    # FragmentOnBonds inserts dummy atoms (*) at the cut points.
    fragmented = Chem.FragmentOnBonds(
        mol,
        [bond_idx],
        addDummies=True,
        dummyLabels=[(0, 0)],
    )

    # GetMolFrags returns:
    #   asMols=False → tuple of tuples of ORIGINAL atom indices
    #   asMols=True  → tuple of RDKit mol objects
    frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
    frag_mols = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

    # Locate the fragment that contains the substituent attachment atom.
    # Note: FragmentOnBonds preserves the original atom indices for non-dummy
    # atoms; dummy atoms are appended with new indices.  Because
    # frag_atom_lists contains the ORIGINAL indices (including the dummy
    # atom's index in the *fragmented* mol), we search for sub_atom_idx.
    target_frag_idx, local_sub_idx = _find_fragment_containing(
        frag_atom_lists, frag_mols, sub_atom_idx
    )

    # Build mapping: parent-mol atom idx → local idx in raw_frag (frag_mols).
    # frag_atom_lists[i][j] gives the parent-mol idx of the j-th atom in
    # frag_mols[i] (for non-dummies; dummy atoms have indices >= parent N).
    parent_n_atoms = mol.GetNumAtoms()
    raw_local_to_parent: "dict[int, int]" = {}
    for local_idx, orig_idx in enumerate(frag_atom_lists[target_frag_idx]):
        if orig_idx < parent_n_atoms:
            raw_local_to_parent[local_idx] = orig_idx

    raw_frag = frag_mols[target_frag_idx]

    # Replace dummy atom with H and sanitize.
    rw = Chem.RWMol(raw_frag)
    # Stamp parent CIP codes onto fragment atoms BEFORE the dummy→H replacement
    # and sanitize step.  Properties survive sanitize, RemoveHs (for heavy
    # atoms), and canonical renumber.  StereoAnalysis prefers _ParentCIPCode
    # over the locally-recomputed _CIPCode on the carved fragment so internal
    # stereocenters whose CIP priorities flip after carving still get the
    # correct (parent-context) descriptor.
    if parent_cip_map:
        for local_idx, orig_idx in raw_local_to_parent.items():
            cip = parent_cip_map.get(orig_idx)
            if cip is not None:
                rw.GetAtomWithIdx(local_idx).SetProp("_ParentCIPCode", cip)
    _replace_dummy_with_hydrogen(rw)
    try:
        Chem.SanitizeMol(rw)
    except Exception as exc:
        logger.debug("SanitizeMol after dummy replacement: %s", exc)
        # Carving can leave atoms flagged aromatic that are no longer part
        # of any ring (e.g. a C=O substituent carved off a heteroaromatic
        # pyridone keeps its aromatic C flag).  Such atoms break downstream
        # SMARTS (amide/ester/etc. patterns use capital-C atoms) and cause
        # silent degradation to an atom-by-atom prefix (-C(=O)NH2 rendered
        # as "aminooxomethyl" instead of "carbamoyl").  Scrub the aromatic
        # flag from any non-ring atom and re-sanitize.
        _clear_nonring_aromaticity(rw)
        try:
            Chem.SanitizeMol(rw)
        except Exception as exc2:
            logger.debug("SanitizeMol after aromaticity scrub: %s", exc2)

    sanitized = rw.GetMol()

    # Strip explicit H atoms left by dummy→H replacement.  These are
    # redundant — the attachment point is tracked by index.  Use a
    # temporary atom-map number to survive the index renumbering that
    # RemoveHs may cause.
    _MAP_TAG = 9999
    rw2 = Chem.RWMol(sanitized)
    rw2.GetAtomWithIdx(local_sub_idx).SetAtomMapNum(_MAP_TAG)
    try:
        no_h = Chem.RemoveHs(rw2.GetMol())
    except Exception as exc:
        logger.debug("RemoveHs after dummy replacement: %s", exc)
        no_h = rw2.GetMol()
    # Find the attachment atom in the H-stripped mol via the tag.
    new_local_sub_idx = next(
        a.GetIdx() for a in no_h.GetAtoms()
        if a.GetAtomMapNum() == _MAP_TAG
    )
    # Clear the temporary tag.
    rw3 = Chem.RWMol(no_h)
    rw3.GetAtomWithIdx(new_local_sub_idx).SetAtomMapNum(0)
    sanitized = rw3.GetMol()
    local_sub_idx = new_local_sub_idx

    # Stash the parent's CIP code on the attachment atom (see comment above
    # the parent_cip capture).  Apply BEFORE canonical renumber so the
    # property survives the renumbering (RDKit copies atom props).
    if parent_cip is not None:
        rw_cip = Chem.RWMol(sanitized)
        rw_cip.GetAtomWithIdx(local_sub_idx).SetProp("_ParentCIPCode", parent_cip)
        sanitized = rw_cip.GetMol()

    # v13 G1: canonical renumbering.
    fragment_mol, [canonical_attachment] = _canonical_renumber(
        sanitized, [local_sub_idx]
    )

    return (fragment_mol, canonical_attachment, bond_order)


def carve_bridging_substituent(
    mol: object,
    substituent_atoms: "frozenset[int]",
    attachment_bonds: "tuple[tuple[int, int], ...]",
) -> "tuple[object, list[int], list[int]]":
    """Extract a bridging substituent (-CH2-, -O-, -NH-).

    Parameters
    ----------
    mol:
        RDKit mol.
    substituent_atoms:
        Atom indices of the bridging group.
    attachment_bonds:
        ``((parent1, sub1), (parent2, sub2), ...)`` — all bonds to cut.

    Returns
    -------
    (fragment_mol, attachment_atoms_in_fragment, bond_orders)
        *fragment_mol* is a sanitised RDKit mol representing the bridging
        substituent with all cut bonds replaced by H.
        *attachment_atoms_in_fragment* lists the canonical indices (one per
        entry in *attachment_bonds*) of the substituent-side atoms in
        *fragment_mol*.
        *bond_orders* lists the integer bond orders in the same order as
        *attachment_bonds*.

    v13 G1
        Same canonical renumbering as ``carve_substituent``.
    """
    from rdkit import Chem

    if not attachment_bonds:
        raise ValueError("attachment_bonds must not be empty")

    bond_idxs = []
    bond_orders = []
    sub_atom_idxs = []  # substituent-side atom for each cut

    for parent_atom_idx, sub_atom_idx in attachment_bonds:
        bond = mol.GetBondBetweenAtoms(parent_atom_idx, sub_atom_idx)
        if bond is None:
            raise ValueError(
                f"No bond between atoms {parent_atom_idx} and {sub_atom_idx}"
            )
        bond_idxs.append(bond.GetIdx())
        bond_orders.append(int(bond.GetBondTypeAsDouble()))
        sub_atom_idxs.append(sub_atom_idx)

    # Build dummy labels — one label pair per cut bond.
    dummy_labels = [(0, 0)] * len(bond_idxs)

    fragmented = Chem.FragmentOnBonds(
        mol,
        bond_idxs,
        addDummies=True,
        dummyLabels=dummy_labels,
    )

    frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
    frag_mols = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

    # The bridging substituent fragment contains ALL sub_atom_idxs.  Find the
    # fragment that contains the first substituent atom; verify the others are
    # also in the same fragment (sanity check).
    target_frag_idx, _ = _find_fragment_containing(
        frag_atom_lists, frag_mols, sub_atom_idxs[0]
    )

    local_sub_idxs = []
    for sub_atom_idx in sub_atom_idxs:
        frag_idx, local_idx = _find_fragment_containing(
            frag_atom_lists, frag_mols, sub_atom_idx
        )
        if frag_idx != target_frag_idx:
            raise ValueError(
                f"Substituent atom {sub_atom_idx} is in fragment {frag_idx}, "
                f"but atom {sub_atom_idxs[0]} is in fragment {target_frag_idx}. "
                "All substituent attachment atoms must be in the same fragment."
            )
        local_sub_idxs.append(local_idx)

    raw_frag = frag_mols[target_frag_idx]

    rw = Chem.RWMol(raw_frag)
    _replace_dummy_with_hydrogen(rw)
    try:
        Chem.SanitizeMol(rw)
    except Exception as exc:
        logger.debug("SanitizeMol after dummy replacement (bridging): %s", exc)

    sanitized = rw.GetMol()

    # Strip explicit H atoms left by dummy→H replacement.  Tag each
    # unique attachment atom with a unique map number so indices survive
    # RemoveHs.  Multiple entries in local_sub_idxs may refer to the same
    # atom (e.g. single-atom bridge -O- has both attachment points at O).
    _MAP_BASE = 9000
    rw2 = Chem.RWMol(sanitized)
    unique_idxs = list(dict.fromkeys(local_sub_idxs))  # deduplicated, ordered
    for tag_offset, idx in enumerate(unique_idxs):
        rw2.GetAtomWithIdx(idx).SetAtomMapNum(_MAP_BASE + tag_offset)
    try:
        no_h = Chem.RemoveHs(rw2.GetMol())
    except Exception as exc:
        logger.debug("RemoveHs after dummy replacement (bridging): %s", exc)
        no_h = rw2.GetMol()
    # Recover attachment indices from the tags.
    old_to_new_idx = {
        unique_idxs[tag - _MAP_BASE]: a.GetIdx()
        for a in no_h.GetAtoms()
        for tag in [a.GetAtomMapNum()]
        if tag != 0
    }
    new_local_sub_idxs = [old_to_new_idx[old_idx] for old_idx in local_sub_idxs]
    # Clear temporary tags.
    rw3 = Chem.RWMol(no_h)
    for idx in dict.fromkeys(new_local_sub_idxs):
        rw3.GetAtomWithIdx(idx).SetAtomMapNum(0)
    sanitized = rw3.GetMol()
    local_sub_idxs = new_local_sub_idxs

    # v13 G1: canonical renumbering — maps all attachment indices at once.
    fragment_mol, canonical_attachments = _canonical_renumber(
        sanitized, local_sub_idxs
    )

    return (fragment_mol, canonical_attachments, bond_orders)


def _carve_polyester(mol: object, decomposition: object) -> dict:
    """Carve a fully-esterified poly-acid into parent acid + N alkyl groups.

    Returns a dict:
        "acid"        -> (acid_mol, None)
        "alcohol_0".. -> (r_mol, attach_idx)  one per ester
        "_polyester_acyl" -> {alcohol_role: acyl_c_original_idx}

    The acid_mol's atoms carry the integer property ``_orig_idx`` recording
    their index in the ORIGINAL molecule, so that execute() can map each ester's
    acyl carbon to the locant the acid-naming machinery assigns it.  This lets
    assembly attach a locant to the alkyl word when the parent acid positions
    are not symmetry-equivalent (P-65.6.3.3.2 "when necessary, locants ...").
    """
    from rdkit import Chem

    root_atoms = getattr(decomposition, "root_atoms", None) or frozenset()
    pieces = getattr(decomposition, "pieces", None)
    if not pieces or len(pieces) < 1:
        return {}
    acid_piece = pieces[0]
    acid_atoms_set = set(acid_piece.atom_indices)

    # Rebuild (acyl_c, alkyl_o, alkyl_c) triples from root_atoms.
    # alkyl_o: O in root that is on the acid side.
    # alkyl_c: C in root NOT on the acid side (the R-group attachment carbon).
    # acyl_c : the acid-side C bonded to an alkyl_o that also bears a =O.
    alkyl_o_list: list[int] = []
    alkyl_c_list: list[int] = []
    for a in root_atoms:
        atom_obj = mol.GetAtomWithIdx(a)
        if a in acid_atoms_set and atom_obj.GetAtomicNum() == 8:
            alkyl_o_list.append(a)
        elif a not in acid_atoms_set and atom_obj.GetAtomicNum() == 6:
            alkyl_c_list.append(a)

    if len(alkyl_o_list) < 2 or len(alkyl_o_list) != len(alkyl_c_list):
        return {}

    # Pair each alkyl_o with its alkyl_c (bonded) and its acyl_c (the other
    # heavy neighbour of the O, which carries the =O on the acid side).
    triples: list[tuple[int, int, int]] = []  # (acyl_c, alkyl_o, alkyl_c)
    cut_bond_idxs: list[int] = []
    for ao in alkyl_o_list:
        ao_atom = mol.GetAtomWithIdx(ao)
        this_alkyl_c = None
        this_acyl_c = None
        for nb in ao_atom.GetNeighbors():
            if nb.GetAtomicNum() == 1:
                continue
            if nb.GetIdx() in alkyl_c_list:
                this_alkyl_c = nb.GetIdx()
            elif nb.GetIdx() in acid_atoms_set:
                this_acyl_c = nb.GetIdx()
        if this_alkyl_c is None or this_acyl_c is None:
            return {}
        bond = mol.GetBondBetweenAtoms(ao, this_alkyl_c)
        if bond is None:
            return {}
        triples.append((this_acyl_c, ao, this_alkyl_c))
        cut_bond_idxs.append(bond.GetIdx())

    # --- Build the parent-acid mol by cutting all ester bonds at once and
    #     replacing the dummy stubs (former alkyl-C) with H -> -C(=O)-OH. ---
    # Tag every original atom with an atom-map number = (orig_idx + 1) so we can
    # recover the original index after FragmentOnBonds/canonicalisation.
    work = Chem.RWMol(mol)
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    work_mol = work.GetMol()

    fragmented = Chem.FragmentOnBonds(
        work_mol, cut_bond_idxs, addDummies=True,
        dummyLabels=[(0, 0)] * len(cut_bond_idxs),
    )
    frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
    frag_mols_list = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

    # The acid fragment contains the first acyl carbon's atom-map tag.
    acyl0_tag = triples[0][0] + 1
    acid_frag_idx = None
    for fi, frag in enumerate(frag_mols_list):
        if any(a.GetAtomMapNum() == acyl0_tag for a in frag.GetAtoms()):
            acid_frag_idx = fi
            break
    if acid_frag_idx is None:
        return {}

    raw_acid = frag_mols_list[acid_frag_idx]
    rw = Chem.RWMol(raw_acid)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            # dummy (former alkyl-C attachment) -> H, restoring -OH on ester O
            atom.SetAtomicNum(1)
            atom.SetNoImplicit(False)
            atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception as exc:
        logger.debug("SanitizeMol after polyester acid split: %s", exc)
    acid_mol = rw.GetMol()
    # Stash _orig_idx from the atom-map tag, then clear the map numbers so they
    # do not leak into the naming machinery / SMILES.
    for atom in acid_mol.GetAtoms():
        tag = atom.GetAtomMapNum()
        if tag > 0:
            atom.SetIntProp("_orig_idx", tag - 1)
        atom.SetAtomMapNum(0)
    try:
        acid_mol = Chem.RemoveHs(acid_mol)
    except Exception as exc:
        logger.debug("RemoveHs after polyester acid split: %s", exc)

    # --- Carve each R group. ---
    result: dict = {"acid": (acid_mol, None)}
    acyl_by_role: dict[str, int] = {}
    for i, (acyl_c, ao, ac) in enumerate(triples):
        try:
            r_sub_mol, r_att_idx, _bo = carve_substituent(
                mol, frozenset(pieces[1].atom_indices), (ao, ac)
            )
        except Exception as exc:
            logger.debug("polyester R carve failed: %s", exc)
            return {}
        role = f"alcohol_{i}"
        result[role] = (r_sub_mol, r_att_idx)
        acyl_by_role[role] = acyl_c

    result["_polyester_acyl"] = acyl_by_role
    return result


def carve_fc_fragments(mol: object, decomposition: object) -> dict:
    """Split a molecule at functional-class boundaries.

    Parameters
    ----------
    mol:
        RDKit mol.
    decomposition:
        Decomposition object with FC split info. Currently supports:
        - subtype="ester":
            Returns {
                "acid":             (mol, None),   # carboxylic acid mol
                "alcohol":          (mol, attachment_idx),  # substituent mol
            }

    Returns
    -------
    dict mapping role names to (RDKit_mol, attachment_atom_idx_or_None).

    Notes
    -----
    For an intermolecular ester, the split happens at the alkyl-C to alkyl-O
    bond.

    - The acid side (carbonyl C + carbonyl O + ester O) is capped with H on
      the ester O → canonical carboxylic acid SMILES. The attachment index
      is None because the acid side is named standalone.

    - The alcohol side (alkyl C plus its R group) is capped with H on the
      alkyl C; the alkyl C itself is preserved and its canonical index in
      the fragment is returned as the attachment atom. The caller renames
      the alcohol fragment with OutputForm.SUBSTITUENT + a FreeValenceInfo
      pointing at that atom, producing "methyl"/"ethyl"/"phenyl".

    Phase 2d: only intermolecular esters are supported. Intramolecular
    (lactone) decompositions are rejected by strategy.accept_plan before
    reaching here.
    """
    from rdkit import Chem

    subtype = getattr(decomposition, "subtype", None)
    if subtype not in (
        "ester", "carbamate", "acyl_isothiocyanate",
        "thioester", "thionoester", "dithioester",
        "thionocarbamate", "dithiocarbamate",
        "carbamothioate",
        "symmetric_diester", "polyester",
    ):
        return {}
    if getattr(decomposition, "intramolecular", False):
        return {}

    # --- Poly-ester / mixed ester: cut every ester C-O bond -> parent acid +
    #     N alkyl substituents (P-65.6.3.3.2). ---
    if subtype == "polyester":
        result = _carve_polyester(mol, decomposition)
        return result or {}

    # --- Symmetric diester: two alkyl_c/alkyl_o cuts → diacid + one R group ---
    if subtype == "symmetric_diester":
        pieces = getattr(decomposition, "pieces", None)
        if not pieces or len(pieces) != 2:
            return {}
        acid_piece, r_piece = pieces
        root_atoms = getattr(decomposition, "root_atoms", None) or frozenset()

        # root_atoms = {ac1, ao1, rc1, ac2, ao2, rc2}
        # Identify the two (alkyl_o, alkyl_c) pairs from root_atoms.
        # alkyl_o atoms: in acid piece AND symbol O
        # alkyl_c atoms: in r piece AND symbol C (not in acid piece)
        acid_atoms_set = set(acid_piece.atom_indices)
        r_atoms_set = set(r_piece.atom_indices)

        alkyl_o_list: list[int] = []
        alkyl_c_list: list[int] = []
        for a in root_atoms:
            atom_obj = mol.GetAtomWithIdx(a)
            if a in acid_atoms_set and atom_obj.GetSymbol() == "O":
                alkyl_o_list.append(a)
            elif a not in acid_atoms_set and atom_obj.GetAtomicNum() == 6:
                alkyl_c_list.append(a)

        if len(alkyl_o_list) != 2 or len(alkyl_c_list) != 2:
            return {}

        # Find pairing: each alkyl_c bonds to its alkyl_o
        pairs: list[tuple[int, int]] = []  # (alkyl_o, alkyl_c) in bond order
        for ao in alkyl_o_list:
            for ac in alkyl_c_list:
                if mol.GetBondBetweenAtoms(ao, ac) is not None:
                    pairs.append((ao, ac))
        if len(pairs) != 2:
            return {}

        # Cut both alkyl_c -- alkyl_o bonds simultaneously to get acid mol.
        bond1 = mol.GetBondBetweenAtoms(pairs[0][0], pairs[0][1])
        bond2 = mol.GetBondBetweenAtoms(pairs[1][0], pairs[1][1])
        if bond1 is None or bond2 is None:
            return {}

        fragmented = Chem.FragmentOnBonds(
            mol,
            [bond1.GetIdx(), bond2.GetIdx()],
            addDummies=True,
            dummyLabels=[(0, 0), (0, 0)],
        )
        frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
        frag_mols_list = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

        # Identify the acid fragment (contains both acyl_c atoms)
        ao1 = pairs[0][0]
        acid_frag_idx = None
        for fi, orig_indices in enumerate(frag_atom_lists):
            if ao1 in orig_indices:
                acid_frag_idx = fi
                break
        if acid_frag_idx is None:
            return {}

        # Build acid mol: replace dummy atoms with OH
        def _finalize_diacid(raw_frag):
            rw = Chem.RWMol(raw_frag)
            for atom in rw.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    # Replace dummy (was an alkyl C attachment point) with H
                    # so the ester O becomes an -OH (restoring -COOH)
                    atom.SetAtomicNum(1)
                    atom.SetNoImplicit(False)
            try:
                Chem.SanitizeMol(rw)
            except Exception as exc:
                logger.debug("SanitizeMol after symmetric_diester acid split: %s", exc)
            clean = rw.GetMol()
            try:
                clean = Chem.RemoveHs(clean)
            except Exception as exc:
                logger.debug("RemoveHs after symmetric_diester acid split: %s", exc)
            return clean

        acid_mol = _finalize_diacid(frag_mols_list[acid_frag_idx])

        # Carve ONE R group (the piece indexed to pairs[0][1] = alkyl_c 1)
        # Both R groups are identical (symmetry verified upstream).
        rc1 = pairs[0][1]
        ao1 = pairs[0][0]
        r_sub_mol, r_att_idx, _ = carve_substituent(
            mol,
            frozenset(r_atoms_set),
            (ao1, rc1),  # parent-side = alkyl_o, sub-side = alkyl_c
        )

        return {
            "acid": (acid_mol, None),
            "alcohol": (r_sub_mol, r_att_idx),
        }

    # --- Acyl isothiocyanate: one piece (acid side), class word is fixed ---
    if subtype == "acyl_isothiocyanate":
        pieces = getattr(decomposition, "pieces", None)
        if not pieces or len(pieces) < 1:
            return {}
        acid_piece = pieces[0]
        root_atoms = getattr(decomposition, "root_atoms", None) or frozenset()
        acid_atoms_set = set(acid_piece.atom_indices)

        # Find acyl_c and n_atom from root_atoms
        acyl_c = None
        n_atom = None
        for a in root_atoms:
            atom_obj = mol.GetAtomWithIdx(a)
            if a in acid_atoms_set and atom_obj.GetSymbol() == "C":
                acyl_c = a
            elif atom_obj.GetSymbol() == "N":
                n_atom = a

        if acyl_c is None or n_atom is None:
            return {}

        # Cut the acyl_C -- N bond
        bond = mol.GetBondBetweenAtoms(acyl_c, n_atom)
        if bond is None:
            return {}
        bond_idx = bond.GetIdx()

        fragmented = Chem.FragmentOnBonds(
            mol, [bond_idx], addDummies=True, dummyLabels=[(0, 0)]
        )
        frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
        frag_mols = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

        # Find acid fragment (contains acyl_c)
        acid_frag_idx = None
        for frag_i, orig_indices in enumerate(frag_atom_lists):
            if acyl_c in orig_indices:
                acid_frag_idx = frag_i
                break

        if acid_frag_idx is None:
            return {}

        raw_acid = frag_mols[acid_frag_idx]
        # Replace the dummy (at cut N position) with OH to produce a
        # carboxylic acid fragment: Ph-C(=O)-N* -> Ph-C(=O)-OH = benzoic acid
        # This allows ACID_STEM naming to produce "benzoate" / ACYL to produce "benzoyl".
        rw = Chem.RWMol(raw_acid)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(8)   # O
                atom.SetIsotope(0)
                atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(rw)
        except Exception as exc:
            logger.debug("SanitizeMol after acyl_isothiocyanate acid split: %s", exc)
        try:
            acid_mol = Chem.RemoveHs(rw.GetMol())
        except Exception:
            acid_mol = rw.GetMol()

        return {"acid": (acid_mol, None)}

    pieces = getattr(decomposition, "pieces", None)
    if not pieces or len(pieces) != 2:
        return {}
    acid_piece, alcohol_piece = pieces
    acid_atoms = set(acid_piece.atom_indices)
    alcohol_atoms = set(alcohol_piece.atom_indices)

    root_atoms = getattr(decomposition, "root_atoms", None) or frozenset()

    # Bridge atom symbol varies by subtype:
    #   ester / thionoester / carbamate / thionocarbamate:  O
    #   thioester / dithioester / dithiocarbamate:          S
    if subtype in (
        "thioester", "dithioester", "dithiocarbamate", "carbamothioate",
    ):
        _bridge_sym = "S"
    else:
        _bridge_sym = "O"

    # Identify alkyl_c (in alcohol piece) and alkyl_o (in acid piece) from
    # root atoms — these are the two atoms that had their bond cut.
    # ``alkyl_o`` is legacy naming from the ester path; for the thio variants
    # it refers to the bridge chalcogen atom (O or S).
    alkyl_c = None
    alkyl_o = None
    for a in root_atoms:
        if a in alcohol_atoms:
            alkyl_c = a
        elif a in acid_atoms:
            atom = mol.GetAtomWithIdx(a)
            if atom.GetSymbol() == _bridge_sym:
                alkyl_o = a

    if alkyl_c is None or alkyl_o is None:
        return {}

    # Cut the alkyl_c--alkyl_o bond and produce two mol fragments.
    bond = mol.GetBondBetweenAtoms(alkyl_c, alkyl_o)
    if bond is None:
        return {}
    bond_idx = bond.GetIdx()

    fragmented = Chem.FragmentOnBonds(
        mol, [bond_idx], addDummies=True, dummyLabels=[(0, 0)]
    )
    frag_atom_lists = Chem.GetMolFrags(fragmented, asMols=False)
    frag_mols = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)

    # Figure out which fragment is the acid side and which is the alcohol side
    # by checking which fragment contains alkyl_o vs alkyl_c.
    acid_idx = None
    alcohol_idx = None
    for frag_i, orig_indices in enumerate(frag_atom_lists):
        if alkyl_o in orig_indices:
            acid_idx = frag_i
        if alkyl_c in orig_indices:
            alcohol_idx = frag_i

    if acid_idx is None or alcohol_idx is None or acid_idx == alcohol_idx:
        return {}

    def _finalize_acid(raw_frag):
        # Replace the dummy H on the ester O → canonical acid mol.
        rw = Chem.RWMol(raw_frag)
        _replace_dummy_with_hydrogen(rw)
        try:
            Chem.SanitizeMol(rw)
        except Exception as exc:
            logger.debug("SanitizeMol after ester acid split: %s", exc)
        clean = rw.GetMol()
        try:
            clean = Chem.RemoveHs(clean)
        except Exception as exc:
            logger.debug("RemoveHs after ester acid split: %s", exc)
        return clean

    acid_mol = _finalize_acid(frag_mols[acid_idx])

    # For the alcohol side, we use carve_substituent on the original mol.
    # This gives us a sanitised fragment with the alkyl_c as the attachment
    # atom in canonical renumbering — ready for OutputForm.SUBSTITUENT.
    alcohol_sub_mol, alcohol_att_idx, _ = carve_substituent(
        mol,
        frozenset(alcohol_atoms),
        (alkyl_o, alkyl_c),  # parent-side atom first, sub-side second
    )

    if subtype in (
        "carbamate", "thionocarbamate", "dithiocarbamate", "carbamothioate",
    ):
        # For carbamate (and its thiono- / dithio- variants) we carve each
        # N-substituent individually.
        # root_atoms = {acyl_c, bridge_chalc, alkyl_c, n_atom}
        # Find acyl_c and N from root_atoms.
        acyl_c = None
        n_in_root = None
        for a in root_atoms:
            if a in acid_atoms and a != alkyl_o:
                atom_a = mol.GetAtomWithIdx(a)
                if atom_a.GetSymbol() == "C":
                    acyl_c = a
                elif atom_a.GetSymbol() == "N":
                    n_in_root = a

        if acyl_c is None:
            return {}

        # Find N if not in root_atoms directly
        if n_in_root is None:
            acyl_c_atom = mol.GetAtomWithIdx(acyl_c)
            for nb in acyl_c_atom.GetNeighbors():
                if nb.GetSymbol() == "N" and nb.GetIdx() in acid_atoms:
                    n_in_root = nb.GetIdx()
                    break

        if n_in_root is None:
            return {}

        # Collect N-substituents: heavy neighbors of N that are not acyl_c.
        n_atom_obj = mol.GetAtomWithIdx(n_in_root)
        result = {"alcohol": (alcohol_sub_mol, alcohol_att_idx)}

        for i, nb in enumerate(n_atom_obj.GetNeighbors()):
            nb_idx = nb.GetIdx()
            if nb.GetAtomicNum() == 1:
                continue  # skip explicit H
            if nb_idx == acyl_c:
                continue  # skip the carbonyl C side

            # BFS from nb_idx staying in acid_atoms (which covers N + all N-subs)
            # but excluding the N itself (since we cut the N--nb bond).
            n_sub_atoms = set(acid_atoms) - {n_in_root, acyl_c}
            # Flood-fill from nb_idx through n_sub_atoms
            comp: set[int] = set()
            stack = [nb_idx]
            while stack:
                cur = stack.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                atom_cur = mol.GetAtomWithIdx(cur)
                for b in atom_cur.GetBonds():
                    other_idx = b.GetOtherAtomIdx(cur)
                    if other_idx not in comp and other_idx in n_sub_atoms:
                        stack.append(other_idx)

            try:
                sub_mol, sub_att_idx, _ = carve_substituent(
                    mol,
                    frozenset(comp),
                    (n_in_root, nb_idx),  # parent-side=N, sub-side=first carbon
                )
                result[f"n_sub_{i}"] = (sub_mol, sub_att_idx)
            except Exception as exc:
                logger.debug("Carbamate N-substituent carve failed: %s", exc)

        return result

    return {
        "acid": (acid_mol, None),
        "alcohol": (alcohol_sub_mol, alcohol_att_idx),
    }


def strip_additive_atoms(
    mol: object,
    additive_groups: list,
) -> "tuple[object, dict[int, int]]":
    """Remove additive atoms from a molecule, producing the parent molecule.

    Handles:
    - N-oxide ``[N+]([O-])``: remove the O atom, reduce N formal charge by 1.
    - P-oxide ``P(=O)``: remove the O atom; RDKit adjusts P valence on
      sanitisation.

    Parameters
    ----------
    mol:
        RDKit mol (read-only; an RWMol copy is made internally).
    additive_groups:
        List of dicts, each with keys:
            ``added_atom``      — index of the atom to remove
            ``center_atom``     — index of the atom whose charge/valence is adjusted
            ``center_element``  — element symbol of the center atom (e.g. ``'N'``)

        Pass an empty list to return the molecule unchanged with an identity map.

    Returns
    -------
    (parent_mol, atom_map)
        *parent_mol* is the molecule with additive atoms removed.
        *atom_map* maps ``{new_idx_in_parent_mol: old_idx_in_original_mol}``.

    v13 C1
        Implementation: RWMol editing — remove atoms in reverse index order to
        keep earlier indices stable during removal.
    """
    from rdkit import Chem

    if not additive_groups:
        identity_map = {i: i for i in range(mol.GetNumAtoms())}
        return (mol, identity_map)

    rw = Chem.RWMol(mol)
    atoms_to_remove: set[int] = set()

    for group in additive_groups:
        added_atom_idx: int = group["added_atom"]
        center_atom_idx: int = group["center_atom"]
        center_element: str = group.get("center_element", "")

        atoms_to_remove.add(added_atom_idx)

        # Adjust formal charge on the center atom where needed.
        if center_element == "N":
            # N-oxide: [N+][O-] → N; reduce charge by 1.
            center = rw.GetAtomWithIdx(center_atom_idx)
            center.SetFormalCharge(center.GetFormalCharge() - 1)
        # For P-oxide (P=O) no explicit charge adjustment is needed —
        # removing the double-bonded O and then sanitising allows RDKit to
        # infer the correct valence.

    # Build the atom map BEFORE removal so we know old index → new index.
    n_original = rw.GetNumAtoms()
    kept_old_idxs = sorted(
        set(range(n_original)) - atoms_to_remove
    )
    # atom_map[new_idx] = old_idx
    atom_map: dict[int, int] = {
        new_idx: old_idx
        for new_idx, old_idx in enumerate(kept_old_idxs)
    }

    # Remove atoms in reverse index order so earlier indices are not
    # invalidated by the removal of later ones.
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        rw.RemoveAtom(atom_idx)

    try:
        Chem.SanitizeMol(rw)
    except Exception as exc:
        logger.debug("SanitizeMol after strip_additive_atoms: %s", exc)

    return (rw.GetMol(), atom_map)
