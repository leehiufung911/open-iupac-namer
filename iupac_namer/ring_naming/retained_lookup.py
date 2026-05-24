"""
iupac_namer/ring_naming/retained_lookup.py

Retained ring name matching.

Strategy:
1. Extract the ring system into an isolated molecule.
2. Convert to canonical SMILES.
3. Look up in retained_rings.json and rings_from_opsin.json.
4. Also maintain a hand-curated SMILES->name map for the most common rings
   (benzene, pyridine, naphthalene, etc.) to guarantee correct lookup
   independent of how data files are keyed.

Returns list[NamedParent] (0 or 1 items).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from iupac_namer.data_loader import (
    get_retained_rings,
    get_rings_from_opsin,
    _RING_CURATED_SMILES,
)
from iupac_namer.ring_naming.kekule_store import maybe_rewrite_for_kekule
from iupac_namer.types import Locant, Numbering, NamedParent

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hand-curated canonical-SMILES -> (name, substituent_form, alkyl_stem_ok)
# ---------------------------------------------------------------------------
# _CURATED is now derived from the single source of truth in data_loader.py.
# DO NOT add NEW name entries here — add them to data_loader._RING_CURATED_SMILES.
#
# Atom-locant *augmentations* for curated entries that lack atom_locants live
# in _CURATED_ATOM_LOCANTS_AUGMENT below — same role as the
# _OPSIN_RING_ATOM_LOCANTS table for OPSIN-data entries.


# Atom-locant augmentations for curated entries that ship without atom_locants.
# Key: canonical SMILES (must already exist as a key in
# data_loader._RING_CURATED_SMILES — this table NEVER introduces a new name
# mapping, only pins ring-atom -> IUPAC-locant for an existing curated entry).
# Value: {ring_mol_atom_idx: iupac_locant (int or str like "3a", "7a")}.
#
# Use this table when:
#   * the curated entry has no atom_locants and the default sorted-atom-index
#     numbering misnumbers substituted forms, AND
#   * the curated entry's name is already correct so we only need to align
#     atom indices to IUPAC locants.
# Each entry MUST be derived from OPSIN methyl-probing on every numeric locant
# accepted by OPSIN, with a regression test in tests/.
_CURATED_ATOM_LOCANTS_AUGMENT: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Stage 7 follow-up: 2,3-dihydro-1-benzofuran (coumaran).  Curated
    # entry was missing atom_locants, causing 4/5/6/7-methyl substituted
    # forms to get the default sorted-index numbering and round-trip to
    # the wrong locant.  Mapping derived from OPSIN methyl-probing on
    # 2-/3-/4-/5-/6-/7-methyl-2,3-dihydrobenzofuran with bond-generic
    # SubstructMatch onto the canonical ring SMILES.
    "c1ccc2c(c1)CCO2": {
        # 5-ring: O-2-3-3a-...-7a
        8: 1,        # O
        7: 2,        # CH2 next to O
        6: 3,        # CH2
        4: "3a",     # aromatic fusion C bonded to CH2 (atom 6)
        # benzene ring: 4-5-6-7
        5: 4,
        0: 5,
        1: 6,
        2: 7,
        3: "7a",     # aromatic fusion C bonded to O (atom 8)
    },
    # Phase 11 biotin core: hexahydrothieno[3,4-d]imidazole (C1NC2CSCC2N1).
    # The curated entry in data_loader.py ships without atom_locants, so the
    # engine emitted nonsense locants (5-oxo-...-8-yl) for the biotin
    # carbonyl + chain attachment.  The ring is C2-symmetric across the
    # S→middle-C axis: locants 1↔3, 3a↔6a, 4↔6 are paired equivalents.  The
    # SMARTS substructure match yields both orientations (uniquify=False),
    # and the strategy layer selects the orientation giving the lowest
    # principal-group + substituent locants per P-14.5.  For biotin,
    # numbering picks attach-point=4 (chain) and carbonyl=2 (C between
    # the two N's), yielding the OPSIN-canonical
    # ``2-oxohexahydrothieno[3,4-d]imidazol-4-yl`` substituent form.
    # Atom-locants verified by OPSIN methyl-probing every locant
    # 1, 2, 3, 3a, 4, 5, 6, 6a (see Phase 11 commit message).
    "C1NC2CSCC2N1": {
        0: 2,        # middle C between the two imidazole N's (locant 2)
        1: 1,        # imidazole N (locant 1)
        2: "6a",     # fusion C bonded to N1 and thiophene-C(6)
        3: 6,        # thiophene C adj to 6a
        4: 5,        # S
        5: 4,        # thiophene C adj to 3a
        6: "3a",     # fusion C bonded to N3 and thiophene-C(4)
        7: 3,        # imidazole N (locant 3)
    },
    # ------------------------------------------------------------------
    # Corrin (vitamin-B12 macrocyclic core), retained name per P-25.3.1.3.
    # Fixed IUPAC numbering (the corrinoid/cobalamin convention): the carbon
    # framework runs 1..19 around the macrocycle; C1-C19 is the DIRECT ring
    # A-D bond that distinguishes corrin from corrole; C5/C10/C15 are the meso
    # methine bridges; C4/C6/C9/C11/C14/C16 are non-substitutable imine
    # junction carbons (no H -- OPSIN rejects substituents there); the four
    # ring N atoms are 21/22/23/24 (ring A->21, B->22, C->23, D->24).
    #
    # Derived by OPSIN chloro-probing every accepted carbon locant (1,2,3,5,7,
    # 8,10,12,13,15,17,18,19 round-trip to ring atom indices via bond-generic
    # SubstructMatch; the H-free junction carbons 4,6,9,11,14,16 are closed by
    # the carbon-backbone walk since OPSIN rejects chloro there) plus N-locant
    # probing (21H/22H/23H/24H-corrin and 21..24-methylcorrin confirm the four
    # N's occupy 21-24, none lower/higher).  Verified consistent: all 13 chloro
    # anchors agree with the macrocycle walk; see commit message.
    #
    # Corrin is perceived as a bridged ring system, so a systematic von Baeyer
    # polycycle plan competes.  The retained-ring plan is senior (P-31.1.4.3 /
    # ring seniority) and is preferred via the retained>von_baeyer band in
    # strategy._numbering_score so this fixed numbering wins.
    "C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2": {
        # carbon backbone 1..19 (macrocycle walk verified against OPSIN probes)
        12: 1, 13: 2, 14: 3, 15: 4, 17: 5, 18: 6, 19: 7, 20: 8,
        21: 9, 0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 6: 15, 7: 16,
        8: 17, 9: 18, 10: 19,
        # ring N atoms 21..24 (ring A->21, B->22, C->23, D->24)
        16: 21,  # N bridging C1(idx12) & C4(idx15)  -- ring A
        22: 22,  # N bridging C6(idx18) & C9(idx21)  -- ring B
        5: 23,   # N bridging C11(idx1) & C14(idx4)  -- ring C
        11: 24,  # N bridging C16(idx7) & C19(idx10) -- ring D
    },
}


# ---------------------------------------------------------------------------
# Mancude-parent atom_locants used ONLY by the oxo+dihydro re-orientation path
# (P-31.1.4.1.1 / P-31.1.4.3.4).  Kept SEPARATE from
# _CURATED_ATOM_LOCANTS_AUGMENT so that adding them does NOT change the
# direct-curated-match numbering path for ordinary substituted furans /
# thiophenes (those keep the symmetric default-sorted numbering).  They are
# consumed exclusively by ``_try_derive_hydro_retained`` when re-deriving the
# ``<locs>-dihydro<parent>`` form for a furanone / thiophenone whose principal
# characteristic group forces a numbering that conflicts with the curated
# fixed-orientation dihydro name (e.g. ``2,3-dihydrofuran`` ->
# ``4,5-dihydrofuran`` once a 2-COOH is present).
#
# Each ring is Cs-symmetric across the heteroatom (locants 2<->5, 3<->4); the
# bond-generic substructure match recovers the mirror orientation at runtime
# and the strategy picks the lowest principal-group locant per P-14.5.  The
# tables pin ONE self-consistent perimeter walk (heteroatom = locant 1).
_OXO_DIHYDRO_MANCUDE_PARENT_LOCANTS: dict[str, dict] = {
    # furan: canonical ``c1ccoc1`` -> O at idx3.
    "c1ccoc1": {3: 1, 2: 2, 1: 3, 0: 4, 4: 5},
    # thiophene: canonical ``c1ccsc1`` -> S at idx3.
    "c1ccsc1": {3: 1, 2: 2, 1: 3, 0: 4, 4: 5},
}


def _build_curated_from_data_loader() -> tuple[
    dict[str, tuple[str, str | None, bool, dict | None]],
    set[str],
]:
    """Convert data_loader._RING_CURATED_SMILES (dict-of-dicts) into the
    tuple form ``(name, substituent_form, alkyl_stem_ok, atom_locants)`` expected by the
    consuming code in this module.

    atom_locants is a dict mapping ring-mol atom index -> IUPAC locant (int),
    or None if not provided (the numbering algorithm will compute locants).

    For curated entries that ship without atom_locants, fill them in from
    _CURATED_ATOM_LOCANTS_AUGMENT (a side table local to this module).  This
    preserves data_loader.py as the single source of truth for the
    SMILES->name mapping while still letting us pin atom-locant alignment
    for retained-name entries that need it.

    Returns a 2-tuple ``(curated, stage2_fusion_base_optout)`` where
    ``stage2_fusion_base_optout`` is the set of canonical SMILES whose
    curated record sets ``stage2_fusion_base: False``.  Membership in this
    set causes Stage 2B's multi-ring fusion-base lookup to skip the entry
    even when the entry has atom_locants.  Default behaviour (flag absent
    or True) leaves Stage 2B eligibility unchanged for backwards
    compatibility — every existing curated entry that participates as a
    Stage 2B base (naphthalene, biphenylene, 1,4-dihydronaphthalene, etc.)
    keeps that role.
    """
    result: dict[str, tuple[str, str | None, bool, dict | None]] = {}
    optout: set[str] = set()
    for smiles, record in _RING_CURATED_SMILES.items():
        # PIN-eligibility alias swap: retained names flagged pin_eligible=False
        # in data_loader.py (e.g. tetraline, indane, chroman, isochroman) are
        # general-nomenclature only.  When the record supplies pin_name /
        # pin_substituent_form, swap those into name / substituent_form so the
        # engine emits the systematic PIN (1,2,3,4-tetrahydronaphthalene,
        # 2,3-dihydro-1H-indene, etc.).  Atom_locants stay valid because the
        # retained-name and systematic spellings share the same ring numbering.
        if record.get("pin_eligible", True) is False:
            name = record.get("pin_name", record["name"])
            sub_form = record.get("pin_substituent_form",
                                  record.get("substituent_form"))
        else:
            name = record["name"]
            sub_form = record.get("substituent_form")
        alkyl_ok = bool(record.get("alkyl_stem_ok", False))
        atom_locants = record.get("atom_locants")  # {ring_mol_atom_idx: iupac_locant}
        # Augment with side-table atom_locants ONLY when the curated record
        # itself does not carry them.  This guards against silently
        # overriding an existing curated mapping.
        if atom_locants is None and smiles in _CURATED_ATOM_LOCANTS_AUGMENT:
            atom_locants = _CURATED_ATOM_LOCANTS_AUGMENT[smiles]
        result[smiles] = (name, sub_form, alkyl_ok, atom_locants)
        # stage2_fusion_base flag.  Default True (eligible) when absent so
        # the introduction of this field is fully backwards-compatible.
        if record.get("stage2_fusion_base", True) is False:
            optout.add(smiles)
    return result, optout


_CURATED: dict[str, tuple[str, str | None, bool, dict | None]]
_CURATED_STAGE2_FUSION_BASE_OPTOUT: set[str]
_CURATED, _CURATED_STAGE2_FUSION_BASE_OPTOUT = _build_curated_from_data_loader()


def _build_pin_ineligible_set() -> set[str]:
    """Return the set of canonical-SMILES keys whose curated record sets
    ``pin_eligible: False`` — i.e. retained names that are general-nomenclature
    only and must NOT be emitted as the IUPAC PIN.

    Per P-25.3.1.3 / P-31.1.4.2.4 / P-32.4 / P-53 / P-54.4.3.2, retained names
    such as tetraline, indane, chroman, and isochroman are general-nomenclature
    only.  Their PINs are the systematic hydro-derived forms
    (1,2,3,4-tetrahydronaphthalene; 2,3-dihydro-1H-indene;
    3,4-dihydro-2H-1-benzopyran; 3,4-dihydro-1H-2-benzopyran).  The substituent
    forms (tetralin-N-yl, indan-N-yl, chroman-N-yl, isochroman-N-yl) are
    likewise general-nomenclature only — the PIN substituents are the
    systematic dihydro/tetrahydro-X-yl forms.

    ``try_retained_name`` consults this set after a curated match is found and
    treats the entry as if it were absent, letting the systematic-naming path
    in ``ring_naming.__init__.try_name_ring`` fall through.
    """
    ineligible: set[str] = set()
    for smiles, record in _RING_CURATED_SMILES.items():
        if record.get("pin_eligible", True) is False:
            ineligible.add(smiles)
    return ineligible


_CURATED_PIN_INELIGIBLE: set[str] = _build_pin_ineligible_set()


# Names emitted by the OPSIN-extracted data-file fallback (rings_from_opsin.json
# etc.) that mirror the curated pin_eligible=False entries above.  These are
# stem forms of the retained general-nomenclature-only names; the data-file
# fallback path matches by canonical SMILES and would otherwise still emit
# "indan", "tetralin", "chroman", "isochroman" via the OPSIN-extracted table
# even after the curated lookup is gated.  Match by NAME because the data-file
# canonical-SMILES keys may differ slightly from the curated keys (different
# RDKit canonicalization rounds) yet point at the same retained stem.
_DATAFILE_PIN_INELIGIBLE_NAMES: frozenset[str] = frozenset({
    "tetralin", "tetraline",
    "indan", "indane",
    "chroman", "chromane",
    "isochroman", "isochromane",
})


def is_stage2_fusion_base_eligible(smiles: str | None) -> bool:
    """Return False iff the curated entry for ``smiles`` (or its alias) opts
    out of Stage 2B multi-ring fusion-base eligibility via the
    ``stage2_fusion_base: False`` field on its data_loader record.

    Returns True for any SMILES that is not in the curated table at all
    (entries from OPSIN data files implicitly retain prior Stage 2B
    behaviour) and for curated entries that omit the field or set it to
    True.

    This gate is consulted by ``_try_multiring_base_name_and_numberings``
    in ``iupac_namer/ring_naming/fused.py`` to keep Stage 2B's ≤3-ring
    invariant while still letting curated parents (e.g. anthracene) carry
    atom_locants for the substituent-locant rendering path.
    """
    if not smiles:
        return True
    if smiles in _CURATED_STAGE2_FUSION_BASE_OPTOUT:
        return False
    resolved = _CURATED_ALIASES.get(smiles)
    if resolved is not None and resolved in _CURATED_STAGE2_FUSION_BASE_OPTOUT:
        return False
    return True


# ---------------------------------------------------------------------------
# Atom-locant augmentations for OPSIN-extracted data entries.
#
# The OPSIN arylGroups.xml / retainedNames.xml tables supply a SMILES->name
# mapping for hundreds of ring systems, but carry no atom_locants.  Without
# locants, _compute_simple_numbering falls back to sorted-atom-index order,
# which silently mis-assigns substituent locants for rings whose IUPAC
# numbering does NOT coincide with RDKit's canonical atom traversal (spiro
# ring assemblies, PAHs with specific locant conventions, etc.).
#
# This table supplements OPSIN entries with proper atom_locants keyed by the
# canonical-form ring SMILES (what _build_lookup stores in _smiles_to_record).
# Entries here do NOT add new name mappings — they only pin down the
# ring-atom -> IUPAC-locant correspondence for entries that are already in
# the OPSIN data.  ``_build_lookup`` attaches these to the lookup record so
# downstream uses the same atom_locants path as curated entries.
#
# Key: canonical ring SMILES (result of Chem.MolToSmiles on the OPSIN smiles).
# Value: {ring_mol_atom_idx: iupac_locant (int or str like "4a", "9a'")}.
# ---------------------------------------------------------------------------
_OPSIN_RING_ATOM_LOCANTS: dict[str, dict] = {
    # spiro-9,9'-bifluorene — OPSIN arylGroups.xml entry.
    # Two fluorenes sharing the sp3 C9 atom; one fluorene gets unprimed
    # locants 1-8 + 4a, 4b, 8a, 9, 9a, the other gets primed equivalents
    # with 9 == 9' at the spiro atom.  Atom indices below refer to the
    # canonical SMILES "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc21" (the
    # RDKit canonical form of the OPSIN-stored SMILES).
    "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc21": {
        # first (unprimed) fluorene half
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: "4b",
        7: 5, 8: 6, 9: 7, 10: 8, 11: "8a", 12: 9, 3: "9a",
        # second (primed) fluorene half; spiro atom (12) carries the
        # unprimed locant 9 (9 == 9').
        14: "1'", 15: "2'", 16: "3'", 17: "4'",
        18: "4a'", 19: "4b'", 20: "5'", 21: "6'", 22: "7'", 23: "8'",
        24: "8a'", 13: "9a'",
    },

    # ------------------------------------------------------------------
    # Stage 4 unit 14: top Stage 2 LOCANT_WRONG offender rings from
    # eval/stage2_raw.csv.  All atom_locants derived from OPSIN methyl-
    # probing (bond-generic SubstructMatch) + topology closure verified.
    # Entries cover perimeter CH positions and junction atoms where
    # OPSIN accepts a letter locant; for aceanthrylene/acephenanthrylene
    # OPSIN rejects 2 interior junctions each so those 2 atoms carry no
    # locant (they are non-substitutable so this does not affect naming).
    # ------------------------------------------------------------------

    # indene: 1H-indene, 9 atoms
    "C1=Cc2ccccc2C1": {
        8: 1, 0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: 7, 7: "7a",
    },
    # phenalene: 1H-phenalene, 13 atoms; idx 11 is the central interior
    # bridgehead (9b).
    "C1=Cc2cccc3cccc(c23)C1": {
        12: 1, 0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: "6a",
        7: 7, 8: 8, 9: 9, 10: "9a", 11: "9b",
    },
    # trindene (15 atoms; three indene-type 5-rings fused to a central benzene).
    # The OPSIN-stored / data-key SMILES is the 7H tautomer: the sp3 CH2
    # (indicated hydrogen) sits on ONE specific 5-ring (idx 12, H2), which
    # breaks the apparent 3-fold symmetry and FIXES the numbering — so a
    # STRICT (bond-order-respecting) substructure match cleanly distinguishes
    # the 1- and 3-positions (no bond-generic all-aromatic gate fires, because
    # the sp3 CH2 makes the ring non-aromatic, exactly as intended).  Locants
    # verified by OPSIN chloro-probing every accepted position of 7H-trindene:
    # numeric 1..9 plus junction letters 3a/3b/6a/6b/9a/9b (9c/2a/8a rejected
    # -> they do not exist).  The CH2 (idx 12) is locant 7, hence the
    # ``7H-trindene`` indicated-hydrogen prefix the assembly layer emits.
    # Perimeter walk (7H): 1=idx1 -> 2=idx0 -> 9b=idx2 -> 9a=idx3 (these two
    # close back through the central ring) ... numeric run 3=idx8 -> 3a=idx7
    # -> 3b=idx6 -> 4=idx9 -> 5=idx10 -> 6=idx11 -> 6a=idx5 -> 6b=idx4 ->
    # 7(CH2)=idx12 -> 8=idx13 -> 9=idx14.  Junctions closed by topology
    # (OPSIN rejects chloro on a ring-fusion carbon because it would force an
    # sp3 quaternary centre, so the strict probe returns empty there).
    "C1=Cc2c3c(c4c(c2=C1)=CC=C4)CC=C3": {
        1: 1, 0: 2, 8: 3, 7: "3a", 6: "3b", 9: 4, 10: 5, 11: 6,
        5: "6a", 4: "6b", 12: 7, 13: 8, 14: 9, 3: "9a", 2: "9b",
    },
    # acridan (9,10-dihydroacridine), 14 atoms
    "c1ccc2c(c1)Cc1ccccc1N2": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 13: 10, 12: "10a",
        11: 5, 10: 6, 9: 7, 8: 8, 7: "8a", 6: 9, 4: "9a",
    },
    # cholanthrene (1,2-dihydrocholanthrene), 20 atoms (C20H14).
    # 2a/5a/6a/6b/10a/12a/12b/12c are ring junctions.
    "c1cc2c3c(c4ccc5ccccc5c4cc3c1)CC2": {
        18: 1, 19: 2, 2: "2a", 1: 3, 0: 4, 17: 5, 16: "5a",
        15: 6, 14: "6a", 13: "6b", 12: 7, 11: 8, 10: 9, 9: 10,
        8: "10a", 7: 11, 6: 12, 5: "12a", 4: "12b", 3: "12c",
    },
    # aceanthrene (1,2-dihydroaceanthrylene), 16 atoms
    "c1ccc2c3c4c(cccc4cc2c1)CC3": {
        15: 1, 14: 2, 6: "2a", 7: 3, 8: 4, 9: 5, 10: "5a",
        11: 6, 12: "6a", 13: 7, 0: 8, 1: 9, 2: 10,
        3: "10a", 4: "10b", 5: "10c",
    },
    # acephenanthrene (4,5-dihydroacephenanthrylene), 16 atoms
    "c1ccc2c(c1)cc1c3c(cccc32)CC1": {
        12: 1, 11: 2, 10: 3, 9: "3a", 14: 4, 15: 5, 7: "5a",
        6: 6, 4: "6a", 5: 7, 0: 8, 1: 9, 2: 10,
        3: "10a", 13: "10b", 8: "10c",
    },
    # chrysene, 18 atoms, C2h symmetry.  Walk pinned to one consistent
    # isomorphism (start idx 5 = locant 1).
    "c1ccc2c(c1)ccc1c3ccccc3ccc21": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 17: "4b", 16: 5, 15: 6,
        14: "6a", 13: 7, 12: 8, 11: 9, 10: 10, 9: "10a",
        8: "10b", 7: 11, 6: 12, 4: "12a",
    },
    # picene, 22 atoms, C2h symmetry.  Walk pinned to one consistent
    # isomorphism (start idx 2 = locant 1).
    "c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: 6, 8: "6a",
        21: "6b", 20: 7, 19: 8, 18: "8a", 17: 9, 16: 10, 15: 11,
        14: 12, 13: "12a", 12: "12b", 11: 13, 10: 14,
        9: "14a", 3: "14b",
    },
    # aceanthrylene, 16 atoms; OPSIN rejects 2a and 10b letter locants
    # so idx 14 and idx 2 have no locant (interior junctions).
    "C1=Cc2c3ccccc3cc3cccc1c23": {
        1: 1, 0: 2, 13: 3, 12: 4, 11: 5, 10: "5a",
        9: 6, 8: "6a", 7: 7, 6: 8, 5: 9, 4: 10,
        3: "10a", 15: "10c",
    },
    # acephenanthrylene, 16 atoms; OPSIN rejects 2a and 10b letter
    # locants so 2 interior junction atoms carry no locant.
    "C1=Cc2cc3ccccc3c3cccc1c23": {
        11: 1, 12: 2, 13: 3, 14: "3a", 0: 4, 1: 5, 3: 6,
        4: "6a", 5: 7, 6: 8, 7: 9, 8: 10,
        9: "10a", 15: "10c",
    },

    # ------------------------------------------------------------------
    # Stage 4 unit 15: heteroatom-analog ring systems.  All mappings
    # derived from OPSIN chloro-probing (numeric + letter locants) with
    # bond-generic RDKit substructure matching; C2/C2v-symmetric rings
    # pinned to a single orbit so each iupac locant maps to a unique
    # atom index.  See probe_unit15_final.py / probe_unit15_letters.py /
    # probe_unit15_raw.py.
    # ------------------------------------------------------------------

    # phosphanthridine (5-phospha analogue of phenanthridine), 14 atoms
    "c1ccc2c(c1)cpc1ccccc12": {
        12: 1, 11: 2, 10: 3, 9: 4, 8: "4a", 7: 5, 6: 6,
        4: "6a", 5: 7, 0: 8, 1: 9, 2: 10, 3: "10a", 13: "10b",
    },
    # arsinolizine (As-bridgehead 6-6 fused; quinolizine pattern), 10 atoms
    "C1=CC[As]2C=CC=CC2=C1": {
        9: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: "9a",
    },
    # arsindolizidine (saturated indolizidine with As bridgehead), 9 atoms
    "C1CC[As]2CCCC2C1": {
        6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a",
    },
    # phosphindolizidine (saturated indolizidine with P bridgehead), 9 atoms
    "C1CCP2CCCC2C1": {
        6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a",
    },
    # arsindolizine (unsaturated; indolizine-pattern with As bridgehead), 9 atoms
    "C1=CC2=CC=C[As]2C=C1": {
        3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 0: 7, 1: 8, 2: "8a",
    },
    # phosphindolizine (aromatic indolizine with P bridgehead), 9 atoms
    "c1ccp2cccc2c1": {
        6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a",
    },
    # phosphinolizine (quinolizine with P bridgehead), 10 atoms
    "C1=CCP2C=CC=CC2=C1": {
        9: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: "9a",
    },
    # benzanthrone (7H-benz[de]anthracen-7-one), 17 ring atoms + =O (idx 0
    # is the carbonyl oxygen, no ring locant).
    "O=C1c2ccccc2-c2cccc3cccc1c23": {
        9: 1, 10: 2, 11: 3, 12: "3a", 13: 4, 14: 5, 15: 6, 16: "6a",
        1: 7, 2: "7a", 3: 8, 4: 9, 5: 10, 6: 11, 7: "11a",
        8: "11b", 17: "11c",
    },
    # indolizidine (saturated), 9 atoms
    "C1CCN2CCCC2C1": {
        6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a",
    },
    # isoarsinoline (As-analogue of isoquinoline), 10 atoms
    "C1=Cc2ccccc2C=[As]1": {
        8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a",
    },
    # coumestan (6H-[1]benzofuro[3,2-c]chromen-6-one parent ring system),
    # 17 atoms.  OPSIN rejects chloro at the two ring O atoms (idx 6 pos 5
    # and idx 10 pos 11) and at the interior triple junction (idx 16), so
    # those three atoms carry no locant in this mapping.
    "c1ccc2c(c1)OCc1c-2oc2ccccc12": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 7: 6, 8: "6a",
        15: 7, 14: 8, 13: 9, 12: 10, 11: "10a",
        9: "11a", 3: "11b",
    },
    # acridophosphine (phenanthridine-style P analogue; P at pos 5,
    # central C at pos 10), 14 atoms.  C2-symmetric; pinned so pos 1 =
    # idx 13, pos 9 = idx 9.
    "c1ccc2pc3ccccc3cc2c1": {
        13: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: "5a",
        6: 6, 7: 7, 8: 8, 9: 9, 10: "9a", 11: 10, 12: "10a",
    },
    # indolizine (aromatic N-bridgehead), 9 atoms.  Pos 4 is the
    # bridgehead N (idx 3) but OPSIN rejects 4-chloro there (no H).
    "c1ccn2cccc2c1": {
        6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a",
    },
    # isophosphinoline (P-analogue of isoquinoline, P at pos 2), 10 atoms
    "c1ccc2cpccc2c1": {
        4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a",
    },
    # perimidine (1H-perimidine, peri-fused naphthalene with NH-CH=N),
    # 13 atoms.  9b is the interior triple junction (idx 11).
    "C1=Nc2cccc3cccc(c23)N1": {
        1: 1, 0: 2, 12: 3, 10: "3a", 9: 4, 8: 5, 7: 6, 6: "6a",
        5: 7, 4: 8, 3: 9, 2: "9a", 11: "9b",
    },
    # phenoselenazine (10H-phenoselenazine), 14 atoms.  C2v-symmetric
    # with NH (pos 10 = idx 6) and Se (pos 5 = idx 13); pinned so pos 1
    # = idx 5, pos 9 = idx 8.
    "c1ccc2c(c1)Nc1ccccc1[Se]2": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 13: 5, 12: "5a",
        11: 6, 10: 7, 9: 8, 8: 9, 7: "9a", 6: 10, 4: "10a",
    },
    # phenotellurazine (10H-phenotellurazine), 14 atoms.  Same
    # topology/numbering as phenoselenazine (Se -> Te).
    "c1ccc2c(c1)Nc1ccccc1[Te]2": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 13: 5, 12: "5a",
        11: 6, 10: 7, 9: 8, 8: 9, 7: "9a", 6: 10, 4: "10a",
    },
    # quinolizine (2H-quinolizine aromatic), 10 atoms.  Pos 5 = N
    # bridgehead (idx 3) — no H for chloro probe but placed via topology.
    "C1=CCN2C=CC=CC2=C1": {
        9: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: "9a",
    },
    # rubicene (dibenzo[g,p]chrysene), 26 atoms.  C2h-symmetric; pinned
    # so pos 1 = idx 9, pos 8 = idx 21.  Interior triple junctions
    # (idx 24, 25) have no IUPAC locant in this mapping (OPSIN accepts
    # 14d/14e for both, so they are symmetry-equivalent and
    # non-substitutable).
    "c1ccc2c(c1)c1cccc3c4c5ccccc5c5cccc(c2c13)c54": {
        9: 1, 8: 2, 7: 3, 6: "3a", 4: "3b", 5: 4, 0: 5, 1: 6, 2: 7,
        3: "7a", 11: "7b", 10: "7c",
        21: 8, 20: 9, 19: 10, 18: "10a", 17: "10b",
        16: 11, 15: 12, 14: 13, 13: 14,
        12: "14a", 23: "14b", 22: "14c",
    },
    # selenoxanthene (9H-selenoxanthene), 14 atoms.  C2v-symmetric with
    # Se at pos 10 (idx 13) and CH2 at pos 9 (idx 6); pinned pos 1 = idx 5.
    "c1ccc2c(c1)Cc1ccccc1[Se]2": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 13: 10, 12: "10a",
        11: 5, 10: 6, 9: 7, 8: 8, 7: "8a", 6: 9, 4: "9a",
    },
    # telluroxanthene (9H-telluroxanthene), 14 atoms.  Same numbering
    # as selenoxanthene with Te replacing Se.
    "c1ccc2c(c1)Cc1ccccc1[Te]2": {
        5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 13: 10, 12: "10a",
        11: 5, 10: 6, 9: 7, 8: 8, 7: "8a", 6: 9, 4: "9a",
    },
    # isothiochroman (3,4-dihydro-1H-isothiochromene), 10 atoms.
    # Pos 2 = S (idx 8).
    "c1ccc2c(c1)CCSC2": {
        9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a",
    },
    # isoselenochroman, 10 atoms.  Same numbering (S -> Se at idx 8).
    "c1ccc2c(c1)CC[Se]C2": {
        9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a",
    },
    # isotellurochroman, 10 atoms.  Same numbering (S -> Te at idx 8).
    "c1ccc2c(c1)CC[Te]C2": {
        9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a",
    },
    # xanthylium (xanthylium cation, pos 10 = O+), 14 atoms.
    # C2v-symmetric; pinned pos 1 = idx 13, pos 8 = idx 9.
    "c1ccc2[o+]c3ccccc3cc2c1": {
        13: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 10, 5: "10a",
        6: 5, 7: 6, 8: 7, 9: 8, 10: "8a", 11: 9, 12: "9a",
    },

    # ------------------------------------------------------------------
    # Stage 4 unit 16: ~34 more Stage-2 LOCANT_WRONG ring systems.  All
    # mappings derived from batched OPSIN chloro-probing (numeric + letter
    # locants) with bond-generic RDKit substructure matching.  C2/C2v-
    # symmetric rings are pinned to a single orbit (pin choice documented
    # per-entry).  See tmp_probes/probe_unit16.py and build_mappings.py.
    # ------------------------------------------------------------------

    # phenanthrene (OPSIN "phenanthroline" record maps to the phenanthrene
    # SMILES).  C2v; pinned pos 1 = idx 5 (the 2,3,4-chain nearest to a
    # bridgehead with idx < 13).
    "c1ccc2c(c1)ccc1ccccc12": {
        0: 2, 1: 3, 2: 4, 3: "4a", 4: "10a", 5: 1,
        6: 10, 7: 9, 8: "8a", 9: 8, 10: 7, 11: 6, 12: 5,
    },
    # azulene, 10 atoms.  C2v-symmetric; pinned pos 1 = idx 4.
    "c1ccc2cccc-2cc1": {
        0: 6, 1: 7, 2: 8, 3: "8a", 4: 1, 5: 2, 6: 3, 7: "3a",
        8: 4, 9: 5,
    },
    # 9H-fluorene, 13 atoms.  C2v-symmetric; pinned pos 1 = idx 5 and
    # pos 9 (sp3 CH2) = idx 6.
    "c1ccc2c(c1)Cc1ccccc1-2": {
        0: 2, 1: 3, 2: 4, 3: "4a", 4: "9a", 5: 1, 6: 9,
        7: "8a", 8: 8, 9: 7, 10: 6, 11: 5,
    },
    # phenarsazine (10H-phenarsazine), 14 atoms.  C2v; pinned pos 1 = idx 2.
    "c1ccc2c(c1)N=c1ccccc1=[As]2": {
        0: 3, 1: 2, 2: 1, 3: "10a", 4: "4a", 5: 4, 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10,
    },
    # phenomercurazine (10H-phenomercurazine), 14 atoms.  C2v; pinned
    # pos 1 = idx 2.  Bridgeheads at idx 4 and 7 are interior aromatic
    # carbons OPSIN rejects for chloro (no H), so they carry no locant.
    "c1cc[c]2c(c1)Nc1cccc[c]1[Hg]2": {
        0: 3, 1: 2, 2: 1, 3: "10a", 5: 4, 6: 5,
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10,
    },
    # phenophosphazine, 14 atoms.  Pos 10 = P (idx 4), pos 5 = N (idx 11);
    # pin pos 1 = idx 2.
    "c1ccc2pc3ccccc3nc2c1": {
        0: 3, 1: 2, 2: 1, 3: "10a", 4: 10, 5: "9a", 6: 9,
        7: 8, 8: 7, 9: 6, 10: "5a", 11: 5, 12: "4a", 13: 4,
    },
    # phenazasiline, 14 atoms.  Same topology (Si at idx 4).
    "c1ccc2[siH]c3ccccc3nc2c1": {
        0: 3, 1: 2, 2: 1, 3: "10a", 4: 10, 5: "9a", 6: 9,
        7: 8, 8: 7, 9: 6, 10: "5a", 11: 5, 12: "4a", 13: 4,
    },
    # phenoxaphosphine, 14 atoms.  Pos 5 = O (idx 6) and pos 10 = P (idx 13);
    # OPSIN rejects chloro at O, so pos 5 carries no RDKit mapping here
    # (the ring oxygen is at idx 6 and pos 5 by topology, already consistent).
    # Pin pos 1 = idx 2.
    "c1ccc2c(c1)Oc1ccccc1P2": {
        0: 3, 1: 2, 2: 1, 4: "4a", 5: 4, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 13: 10,
    },
    # phenothiarsine, 14 atoms.  Pos 5 = S (idx 6), pos 10 = As (idx 13).
    # Pin pos 1 = idx 2.
    "c1ccc2c(c1)Sc1ccccc1[AsH]2": {
        0: 3, 1: 2, 2: 1, 4: "4a", 5: 4, 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 13: 10,
    },
    # pleiadene (7H-pleiaden), 18 atoms.  C2v; pin pos 1 = idx 9.  Pos 3a
    # is idx 12 (one interior triple junction), 12b is idx 17 (the other).
    "C1=c2ccccc2=Cc2cccc3cccc1c23": {
        0: 7, 1: "7a", 2: 8, 3: 9, 4: 10, 5: 11,
        7: 12, 8: "12a", 9: 1, 10: 2, 11: 3, 12: "3a",
        13: 4, 14: 5, 15: 6, 16: "6a", 17: "12b",
    },
    # pyrrolizine, 8 atoms.  Pin pos 1 = idx 7 (the sp3 CH2 of 1H-pyrrolizine).
    "C1=Cn2cccc2C1": {
        0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: "7a", 7: 1,
    },
    # acenaphthylene, 12 atoms.  C2v (naphthalene + cyclopenta on peri).
    # Pos 5a = idx 6, pos 8b = idx 11.  Pin pos 1 = idx 0.  Pos 2a and 8a
    # are interior bridgeheads OPSIN rejects for chloro probes but are
    # pinned via topology (pos 2a adj to pos 2, pos 8a adj to pos 1).
    "C1=Cc2cccc3cccc1c23": {
        0: 1, 1: 2, 2: "2a", 3: 3, 4: 4, 5: 5, 6: "5a",
        7: 6, 8: 7, 9: 8, 10: "8a", 11: "8b",
    },
    # anthyridine, 14 atoms.  C2v with 3 N (pos 1, 9, 10).  Pin pos 1 = idx 2.
    "c1cnc2nc3ncccc3cc2c1": {
        0: 3, 1: 2, 2: 1, 3: "10a", 4: 10, 5: "9a", 6: 9,
        7: 8, 8: 7, 9: 6, 10: "5a", 11: 5, 12: "4a", 13: 4,
    },
    # as-indacene (1H-as-indacene), 12 atoms.  Pin pos 1 = idx 1.
    "C1=Cc2c3c(ccc2=C1)=CC=C3": {
        0: 2, 1: 1, 3: "8a", 4: "5a", 5: 5, 6: 4,
        7: "3a", 8: 3, 9: 6, 10: 7, 11: 8,
    },
    # s-indacene (1H-s-indacene), 12 atoms.  4-fold symmetric on most
    # positions; pin pos 2 = idx 0, pos 1 = idx 1.
    "C1=Cc2cc3c(cc2=C1)C=CC=3": {
        0: 2, 1: 1, 2: "8a", 3: 8, 4: "7a", 5: "4a",
        6: 4, 7: "3a", 8: 3, 9: 5, 10: 6, 11: 7,
    },
    # triphenodioxazine, 22 atoms.  C2h; pin pos 1 = idx 5, pos 14 = idx 14
    # (N), pos 6 = idx 11 (N).  Ring Os (idx 13, 21) and two interior
    # junctions OPSIN rejects, so they carry no locant here.
    "c1ccc2c(c1)N=c1cc3c(cc1O2)=Nc1ccccc1O3": {
        0: 2, 1: 3, 2: 4, 3: "4a", 5: 1, 8: 13, 9: "12a",
        11: 6, 12: "5a", 14: 14, 15: "14a",
        16: 8, 17: 9, 18: 10, 19: 11, 20: "11a",
    },
    # triphenodithiazine, 22 atoms.  Same topology (O -> S).  S at idx 13,
    # 21 are pos 5 and 12 respectively.
    "c1ccc2c(c1)N=c1cc3c(cc1S2)=Nc1ccccc1S3": {
        0: 2, 1: 3, 2: 4, 3: "4a", 5: 1, 8: 13, 9: "12a",
        11: 6, 12: "5a", 13: 5, 14: 14, 15: "14a",
        16: 8, 17: 9, 18: 10, 19: 11, 20: "11a", 21: 12,
    },
    # anthrazine, 30 atoms.  C2v-like double-quinoxaline.  Pin pos 1 =
    # idx 18, pos 10 = idx 29 (breaks 2-fold).  Pos 15 = N (idx 25) is
    # rejected by OPSIN chloro (no H); pos 15a is the corresponding
    # bridgehead (idx 24) captured via topology.
    "c1ccc2cc3c(ccc4nc5c(ccc6cc7ccccc7cc65)nc43)cc2c1": {
        0: 11, 1: 12, 2: 13, 3: "13a", 4: 14, 5: "14a",
        6: "8a", 7: 8, 8: 7, 9: "6a", 10: 6, 11: "5b",
        12: "15a", 13: 16, 14: 17, 15: "17a", 16: 18, 17: "18a",
        18: 1, 19: 2, 20: 3, 21: 4, 22: "4a", 23: 5, 24: "5a",
        27: 9, 28: "9a", 29: 10,
    },

    # 1H-isothiochromene, 10 atoms.  Pos 2 = S (idx 9).  Atom idx 2 is a
    # bridgehead OPSIN rejects for chloro (no H).
    "C1=Cc2ccccc2CS1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },
    # 1H-isoselenochromene (S -> Se at idx 9)
    "C1=Cc2ccccc2C[Se]1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },
    # 1H-isotellurochromene (S -> Te at idx 9)
    "C1=Cc2ccccc2C[Te]1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },

    # thebenidine, 16 atoms.  Peri-fused tri/tetracyclic; clean (no
    # symmetry).  Idx 15 is interior triple junction (OPSIN rejects chloro).
    "c1cc2ccc3cccc4ncc(c1)c2c34": {
        0: 7, 1: 8, 2: "8a", 3: 9, 4: 10, 5: "10a",
        6: 1, 7: 2, 8: 3, 9: "3a", 10: 4, 11: 5, 12: "5a",
        13: 6, 14: "10c",
    },
    # arsanthridine, 14 atoms.  Clean.
    "C1=c2ccccc2=c2ccccc2=[As]1": {
        0: 6, 1: "6a", 2: 7, 3: 8, 4: 9, 5: 10, 6: "10a",
        7: "10b", 8: 1, 9: 2, 10: 3, 11: 4, 12: "4a", 13: 5,
    },
    # 2H-arsindole, 9 atoms.  Clean.
    "C1=Cc2ccccc2[AsH]1": {
        0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: 7, 7: "7a", 8: 1,
    },
    # arsinoline, 10 atoms.  Clean.  Pos 1 = As.
    "C1=Cc2ccccc2[As]=C1": {
        0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a",
        8: 1, 9: 2,
    },
    # 2H-thiochromene, 10 atoms.  Pos 1 = S.  Atom idx 2 is bridgehead.
    "C1=Cc2ccccc2SC1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },
    # 2H-selenochromene
    "C1=Cc2ccccc2[Se]C1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },
    # 2H-tellurochromene
    "C1=Cc2ccccc2[Te]C1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1, 9: 2,
    },
    # phosphinoline, 10 atoms.  Clean.  Pos 1 = P.
    "c1ccc2pcccc2c1": {
        0: 6, 1: 7, 2: 8, 3: "8a", 4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5,
    },
    # 2,1-benzisoxazole (anthranil), 9 atoms.  Pos 2 = O (idx 5), OPSIN
    # rejects chloro at O so it carries no locant here.
    "c1ccc2nocc2c1": {
        0: 5, 1: 6, 2: 7, 3: "7a", 4: 1, 6: 3, 7: "3a", 8: 4,
    },
    # arsindoline, 9 atoms.  Clean.  Pos 1 = As.
    "c1ccc2c(c1)CC[AsH]2": {
        0: 5, 1: 6, 2: 7, 3: "7a", 4: "3a", 5: 4, 6: 3, 7: 2, 8: 1,
    },
    # thiochroman, 10 atoms.  Clean.
    "c1ccc2c(c1)CCCS2": {
        0: 6, 1: 7, 2: 8, 3: "8a", 4: "4a", 5: 5, 6: 4, 7: 3, 8: 2, 9: 1,
    },
    # selenochroman, 10 atoms.
    "c1ccc2c(c1)CCC[Se]2": {
        0: 6, 1: 7, 2: 8, 3: "8a", 4: "4a", 5: 5, 6: 4, 7: 3, 8: 2, 9: 1,
    },
    # tellurochroman, 10 atoms.
    "c1ccc2c(c1)CCC[Te]2": {
        0: 6, 1: 7, 2: 8, 3: "8a", 4: "4a", 5: 5, 6: 4, 7: 3, 8: 2, 9: 1,
    },
    # 1H-isochromene, 10 atoms.  Pos 2 = O (idx 9), OPSIN rejects chloro.
    "C1=Cc2ccccc2CO1": {
        0: 3, 1: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a", 8: 1,
    },

    # ------------------------------------------------------------------
    # Stage 4 unit 17: additional Stage-2 LOCANT_WRONG ring systems.
    # Derived from OPSIN chloro-probing (numeric + letter locants) with
    # bond-generic RDKit substructure matching; dibenzo-C2v-symmetric
    # rings pinned to one orbit.  Skipped mappings explained inline.
    # See tmp_probes/probe_unit17_batched.py.
    # ------------------------------------------------------------------

    # phenylium (benzenium cation, 6 atoms), C+ at idx 0 = position 1.
    "[C+]1=CC=CC=C1": {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,
    },

    # phenoxathiine, 14 atoms; O (idx 6) = pos 5, S (idx 13) = pos 10.
    # Same atom-walk shared by all phenoxa-X-ine analogs below.
    "c1ccc2c(c1)Oc1ccccc1S2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },
    # phenoxaselenine (O + Se), same topology as phenoxathiine.
    "c1ccc2c(c1)Oc1ccccc1[Se]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },
    # phenoxasiline (O + SiH2).
    "c1ccc2c(c1)Oc1ccccc1[SiH2]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },
    # phenoxatellurine (O + Te).
    "c1ccc2c(c1)Oc1ccccc1[Te]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },
    # phenoxarsinine (O + AsH).
    "c1ccc2c(c1)Oc1ccccc1[AsH]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },

    # phosphindole (1H-phosphindole, P-analog of 1H-indole), 9 atoms.
    # P (idx 4) = pos 1.
    "c1ccc2[pH]ccc2c1": {
        0: 5, 1: 6, 2: 7, 3: "7a", 4: 1, 5: 2, 6: 3, 7: "3a", 8: 4,
    },

    # isophosphindole (2H-isophosphindole, P-analog of 2H-isoindole), 9 atoms.
    # P (idx 5) = pos 2; C2v-symmetric across the P→fused-bond axis (2 graph
    # automorphisms), so all six numeric positions pair up (1↔3, 4↔7, 5↔6) and
    # the two junctions (3a↔7a) are equivalent.  Pinned to a single orientation
    # (idx 4 = pos 1); the bond-generic match in
    # _build_numbering_from_atom_locants supplies the mirror orientation at
    # runtime so the strategy picks the lowest substituent locant per P-14.5.
    # Mapping verified by OPSIN chloro-probing every accepted locant
    # (1,2,3,3a,4,5,6,7,7a).
    "c1ccc2c[pH]cc2c1": {
        4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a",
    },

    # isobenzothiofuran (2-benzothiophene / benzo[c]thiophene), 9 atoms.
    # S (idx 5) = pos 2; C2-symmetric like isophosphindole (1↔3, 4↔7, 5↔6,
    # 3a↔7a).  Pinned to one orientation (idx 4 = pos 1); the bond-generic
    # match supplies the mirror orientation at runtime.  Verified by OPSIN
    # chloro-probing every accepted locant (1,2,3,3a,4,5,6,7,7a).
    "c1ccc2cscc2c1": {
        4: 1, 5: 2, 6: 3, 3: "7a", 2: 7, 1: 6, 0: 5, 8: 4, 7: "3a",
    },

    # isophosphindoline (2,3-dihydro-1H-isophosphindole), 9 atoms.  Saturated
    # 5-ring (C1-P2-C3) fused to benzene; P (idx 7) = pos 2.  C2-symmetric
    # (2 automorphisms): 1↔3, 4↔7, 5↔6, 3a↔7a.  Fully saturated, so the
    # uniquify=False ring-mol fallback already yields both orientations; the
    # pin here just fixes one self-consistent labeling (idx 6 = pos 1).
    # Verified by OPSIN chloro-probing every accepted locant.
    "c1ccc2c(c1)CPC2": {
        6: 1, 7: 2, 8: 3, 4: "7a", 3: "3a", 5: 7, 0: 6, 1: 5, 2: 4,
    },

    # acenaphthene (2,3-dihydroacenaphthylene), 12 atoms.  sp3 CH2 bridge
    # = positions 1,2; 5a (idx 8) and 8b (idx 3) are interior junctions.
    "c1cc2c3c(cccc3c1)CC2": {
        10: 1, 11: 2, 2: "2a", 1: 3, 0: 4, 9: 5, 8: "5a",
        7: 6, 6: 7, 5: 8, 4: "8a", 3: "8b",
    },

    # coumaran (2,3-dihydrobenzofuran), 9 atoms.  Pos 1 = O (idx 8), OPSIN
    # rejects chloro on O so idx 8 has no probe hit but is position 1.
    "c1ccc2c(c1)CCO2": {
        8: 1, 7: 2, 6: 3, 4: "3a", 5: 4, 0: 5, 1: 6, 2: 7, 3: "7a",
    },

    # flavylium (2-phenylchromen-1-ium), 16 atoms.  Core chromenylium:
    # O+ (idx 13) = 1; 2-phenyl gets primed locants (2'..6').  Idx 3
    # is the ipso attachment point (phenyl C1' bonded to chromene C2);
    # OPSIN accepts "1'-chloroflavylium" only as a cyclohexadienyl
    # tautomer, so we leave idx 3 unmapped.  idx 7 is 4a (OPSIN accepts
    # both "4a" and "10" aliases for this junction).
    "c1ccc(-c2ccc3ccccc3[o+]2)cc1": {
        13: 1, 4: 2, 5: 3, 6: 4, 7: "4a", 8: 5, 9: 6, 10: 7, 11: 8,
        12: "8a",
        2: "2'", 1: "3'", 0: "4'", 15: "5'", 14: "6'",
    },

    # isoarsindoline (2,3-dihydro-2H-isoarsindole, As-analog of
    # isoindoline), 9 atoms.  As (idx 7) = pos 2.
    "c1ccc2c(c1)C[AsH]C2": {
        6: 1, 7: 2, 8: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 5: 7, 4: "7a",
    },

    # phosphindoline (2,3-dihydro-1H-phosphindole), 9 atoms.  P (idx 8)
    # = pos 1.
    "c1ccc2c(c1)CCP2": {
        0: 5, 1: 6, 2: 7, 3: "7a", 4: "3a", 5: 4, 6: 3, 7: 2, 8: 1,
    },

    # acridarsine (As-analog of acridine, bridge at pos 5), 14 atoms.
    # As (idx 7) = pos 5; meso CH (idx 0) = pos 10.
    "C1=c2ccccc2=[As]c2ccccc21": {
        2: 1, 3: 2, 4: 3, 5: 4, 6: "4a", 7: 5, 8: "5a",
        9: 6, 10: 7, 11: 8, 12: 9, 13: "9a", 0: 10, 1: "10a",
    },

    # anthracene, 14 atoms, D2h symmetry; pinned to one orbit.
    # Meso CHs (9,10) = idx 4, 11.
    "c1ccc2cc3ccccc3cc2c1": {
        2: 1, 1: 2, 0: 3, 13: 4, 12: "4a", 9: 5, 8: 6, 7: 7, 6: 8,
        5: "8a", 4: 9, 3: "9a", 11: 10, 10: "10a",
    },

    # trinaphthylene, 30 atoms.  Three naphthalene units arranged with C3 +
    # mirror symmetry (6 graph automorphisms) — a fully-aromatic all-carbon
    # PAH.  Every numeric position pairs into a 6-membered orbit, so a single
    # consistent labeling fully determines the numbering; the bond-generic
    # match (all-aromatic gate) recovers the other 5 orientations at runtime
    # and the strategy picks the lowest substituent locant per P-14.5.
    # Labeling solved by adjacency-preserving backtracking over the OPSIN-
    # chloro-probed locant orbits (every accepted locant 1-18 + junction
    # letters 4a/5a/5b/6a/10a/11a/11b/12a/16a/17a/17b/18a).
    "c1ccc2cc3c(cc2c1)c1cc2ccccc2cc1c1cc2ccccc2cc31": {
        2: 1, 0: 2, 1: 3, 9: 4, 3: "4a", 4: 5, 5: "5a", 6: "5b", 7: 6,
        8: "6a", 13: 7, 14: 8, 15: 9, 16: 10, 12: "10a", 11: 11,
        10: "11a", 19: "11b", 18: 12, 17: "12a", 23: 13, 24: 14, 25: 15,
        26: 16, 22: "16a", 21: 17, 20: "17a", 29: "17b", 28: 18, 27: "18a",
    },

    # phenanthrazine, 30 atoms (fully-aromatic aza-PAH, 2 ring N's at locants
    # 9 and 18; 4 graph automorphisms).  Each numeric carbon position pairs
    # into a 4-membered orbit, the two N's into a 2-membered orbit.  Single
    # consistent labeling solved by adjacency-preserving backtracking over the
    # OPSIN-chloro-probed locant orbits (numeric 1-18 + junctions 4a/4b/8a/8b/
    # 9a/9b/13a/13b/17a/17b/18a/18b; N positions confirmed via
    # 9,18-dihydrophenanthrazine).  Bond-generic match (aromatic gate) supplies
    # the other orientations; the strategy picks the lowest substituent locant.
    "c1ccc2c(c1)c1ccccc1c1nc3c4ccccc4c4ccccc4c3nc21": {
        2: 1, 1: 2, 0: 3, 4: "4a", 5: 4, 6: "4b", 7: 5, 8: 6, 9: 7,
        10: 8, 13: 9, 14: "9a", 12: "8b", 11: "9b", 3: "8a",
        16: 10, 17: 11, 18: 12, 19: 13, 22: 14, 23: 15, 24: 16, 25: 17,
        20: "13a", 21: "13b", 15: "17a", 26: "18b", 27: "17b", 28: 18,
        29: "18a",
    },

    # boranthrene (9,10-diboraanthracene), 14 atoms.  Pinned to one
    # orbit; B atoms (idx 0, 7) = meso positions 5 and 10 (OPSIN rejects
    # chloro there due to B valency).  Junction letter locants rejected
    # by OPSIN but included for numbering-consistency.
    "b1c2ccccc2bc2ccccc12": {
        5: 1, 4: 2, 3: 3, 2: 4, 1: "4a", 0: 5, 13: "5a",
        12: 6, 11: 7, 10: 8, 9: 9, 8: "10a", 7: 10, 6: "9a",
    },

    # fluoranthene, 16 atoms.  Interior triple-junctions 3a (idx 10) and
    # 10c (idx 15) pinned unique by probe; D/C2v mirror pinned to one
    # orbit for the rest.
    "c1ccc2c(c1)-c1cccc3cccc-2c13": {
        13: 1, 12: 2, 11: 3, 10: "3a", 9: 4, 8: 5, 7: 6, 6: "6a",
        5: 7, 0: 8, 1: 9, 2: 10, 3: "10a", 14: "10b", 4: "6b",
        15: "10c",
    },

    # isoarsindole (2H-isoarsindole, As-analog of 2H-isoindole), 9 atoms.
    # As (idx 8) = pos 2.
    "C1=c2ccccc2=C[AsH]1": {
        0: 1, 8: 2, 7: 3, 6: "3a", 5: 4, 4: 5, 3: 6, 2: 7, 1: "7a",
    },

    # naphthacene (tetracene), 18 atoms, D2h; pinned to one orbit.
    # Peri CHs 5,6,11,12 = idx 15, 13, 6, 4.
    "c1ccc2cc3cc4ccccc4cc3cc2c1": {
        2: 1, 1: 2, 0: 3, 17: 4, 16: "4a", 15: 5, 14: "5a",
        13: 6, 12: "6a", 11: 7, 10: 8, 9: 9, 8: 10, 7: "10a",
        6: 11, 5: "11a", 4: 12, 3: "12a",
    },

    # oxanthrene (dibenzo-1,4-dioxine), 14 atoms.  Both bridges = O
    # (idx 6, 13); pinned so idx 6 = pos 5, idx 13 = pos 10.  Same atom
    # walk as phenoxathiine.
    "c1ccc2c(c1)Oc1ccccc1O2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },

    # phosphanthrene (9,10-diphospha-anthracene), 14 atoms.  Topology
    # matches silanthrene (not oxanthrene) -- hetero atoms at idx 4, 11
    # bridging ring A (idx 0-3, 12-13) and ring B (idx 5-10).
    "c1ccc2pc3ccccc3pc2c1": {
        13: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: "5a",
        6: 6, 7: 7, 8: 8, 9: 9, 10: "9a", 11: 10, 12: "10a",
    },

    # arsanthrene (9,10-diarsa-anthracene; OPSIN spelling has the two As at
    # the meso positions 5 and 10), 14 atoms.  Unlike phosphanthrene/
    # silanthrene the OPSIN-stored SMILES carries an explicit mancude
    # As=C double-bond pattern (canonical
    # ``c1ccc2c(c1)[As]=c1ccccc1=[As]2``), so the ring is NOT all-aromatic
    # (the As atoms are non-aromatic) and the bond-generic all-aromatic gate
    # does NOT fire.  The fixed As=C kekulé makes peri positions {1,4} and
    # {6,9} chemically DISTINCT despite the carbon framework's D2h symmetry
    # (OPSIN chloro-probe: 1-/4-chloro round-trip to one isomer, 6-/9-chloro
    # to the other), so the atom_locants MUST be aligned to the kekulé via a
    # STRICT (bond-order-respecting) substructure match — exactly what the
    # default bond-order SMARTS path does.  Each strict match yields 2 of the
    # 4 graph automorphisms (the pair that preserves the As=C placement); for
    # any given substituent that is the {1,4} or {6,9} pair, and the strategy
    # picks the lower per P-14.5.  Labeling derived by OPSIN chloro-probing
    # every accepted numeric locant (1,2,3,4,6,7,8,9 — strict-matched back to
    # ring-mol indices) with 4a/5/5a/9a/10/10a closed by topology (OPSIN
    # rejects chloro at junctions and at the As atoms; 10b and 11 do not
    # exist — the absent locant 11 is why the old sorted-index default's
    # "11-chloroarsanthrene" did not round-trip).  Walk: pos1=idx8 ->
    # 2=idx9 -> 3=idx10 -> 4=idx11 -> 4a=idx12 -> 5(As)=idx13 -> 5a=idx3 ->
    # 6=idx2 -> 7=idx1 -> 8=idx0 -> 9=idx5 -> 9a=idx4 -> 10(As)=idx6 ->
    # 10a=idx7.
    "c1ccc2c(c1)[As]=c1ccccc1=[As]2": {
        8: 1, 9: 2, 10: 3, 11: 4, 12: "4a", 13: 5, 3: "5a",
        2: 6, 1: 7, 0: 8, 5: 9, 4: "9a", 6: 10, 7: "10a",
    },

    # selenanthrene, 14 atoms.  Dibenzo-1,4-diselenine topology; bridges
    # Se at idx 6, 13 (same atom walk as oxanthrene/phenoxathiine).
    "c1ccc2c(c1)[Se]c1ccccc1[Se]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },

    # silanthrene, 14 atoms.  Same topology as phosphanthrene (bridges
    # Si at idx 4, 11).
    "c1ccc2[siH]c3ccccc3[siH]c2c1": {
        13: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: "5a",
        6: 6, 7: 7, 8: 8, 9: 9, 10: "9a", 11: 10, 12: "10a",
    },

    # telluranthrene, 14 atoms.  Dibenzo-1,4-ditellurine; same topology
    # as oxanthrene (bridges Te at idx 6, 13).
    "c1ccc2c(c1)[Te]c1ccccc1[Te]2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },

    # thianthrene, 14 atoms.  Dibenzo-1,4-dithiine; same atom walk as
    # oxanthrene (bridges S at idx 6, 13).
    "c1ccc2c(c1)Sc1ccccc1S2": {
        2: 1, 1: 2, 0: 3, 5: 4, 4: "4a", 6: 5, 7: "5a",
        8: 6, 9: 7, 10: 8, 11: 9, 12: "9a", 13: 10, 3: "10a",
    },

    # ------------------------------------------------------------------
    # phthalhydrazide (OPSIN simpleCyclicGroups.xml retained name; canonical
    # ring SMILES carries the two exocyclic C=O of the cyclic hydrazide).
    # OPSIN's phthalhydrazide numbering is NON-STANDARD: it does NOT follow
    # phthalazine numbering (where the benzo carbon adjacent to a ring-fusion
    # carbon is locant 5).  Instead OPSIN labels the two ring-fusion carbons
    # 1,2 and the four benzo CH carbons 3,4,5,6 (the hydrazide C=O carbons and
    # the two N atoms are NOT separately numbered — they are implicit in the
    # ``-hydrazide`` stem).  Verified by OPSIN chloro-probing every accepted
    # locant of phthalhydrazide:
    #   3-chloro == 6-chloro -> O=c1[nH][nH]c(=O)c2c(Cl)cccc12  (benzo C adj
    #       to a fusion carbon)
    #   4-chloro == 5-chloro -> O=c1[nH][nH]c(=O)c2cc(Cl)ccc12  (the meta
    #       benzo C)
    #   7,8 are REJECTED (do not exist — only four benzo positions, 3-6).
    # Because the molecule is C2-symmetric the perimeter has two mirror-image
    # substructure matches; the strategy's lowest-locant rule (P-14.5) then
    # picks 3 over 6 and 4 over 5 on the canonical input.  This entry replaces
    # the previous generic fused-numbering fallback, which numbered the benzo
    # perimeter in the wrong direction and emitted the mirrored locant
    # (5-chloro where OPSIN wants 3/4-chloro, etc.).  P-25.3.3.
    # Canonical key atom indices (O=c1[nH][nH]c(=O)c2ccccc12):
    #   1,4 = C=O carbons; 2,3 = N-H; 0,5 = exocyclic =O; 6,11 = fusion C;
    #   7,8,9,10 = benzo CH.
    "O=c1[nH][nH]c(=O)c2ccccc12": {
        11: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6,
    },
}


# Secondary SMILES lookups (alternate canonicalizations that RDKit might produce)
# All target values MUST be keys in _CURATED (i.e. in _RING_CURATED_SMILES).
_CURATED_ALIASES: dict[str, str] = {
    # Sometimes RDKit canonicalizes starting from different atoms
    "n1ccccc1":           "c1ccncc1",   # pyridine rotated
    "c1cc[nH]cc1":        "c1cc[nH]c1", # pyrrole rotated (if this ever appears)
    "C1CCCO1":            "C1CCOC1",    # THF rotated
    "O1CCCC1":            "C1CCOC1",    # THF O-first
    "N1CCCCC1":           "C1CCNCC1",   # piperidine N-first
    "N1CCNCC1":           "C1CNCCN1",   # piperazine
    "O1CCNCC1":           "C1COCCN1",   # morpholine (canonical is C1COCCN1)
    "C1CNCCO1":           "C1COCCN1",   # morpholine alt
    # Pyrimidine: old non-canonical key -> canonical key
    "c1ccncn1":           "c1cncnc1",   # pyrimidine (canonical is c1cncnc1)
    # Isoquinoline: old non-canonical key -> canonical key
    "c1cnc2ccccc2c1":     "c1ccc2cnccc2c1",  # isoquinoline old key
    # Old invalid entries from _CURATED that were aliases for proper canonical forms
    "c1cnco1":            "c1cocn1",    # 1,3-oxazole old key
    "c1cncs1":            "c1cscn1",    # 1,3-thiazole old key
    # Tetrazole tautomers: 1H- and 2H- are STRUCTURALLY DISTINCT (different
    # NH atom; RDKit canonical SMILES differ; OPSIN treats them as separate
    # molecules).  Each tautomer has its own curated entry in
    # _RING_CURATED_SMILES; do not collapse them via this alias.
    # (Substituent-locant pinning for 2H-tetrazol-N-yl lives in
    # engine._TAUTOMER_NH_RING_SUBSTITUENT_DATA.)
    # Pteridine: alternate canonical form produced when extracting from N-substituted
    # or keto-tautomer pteridine compounds (after [nH] stripping in _normalize_nh_fragment)
    "c1cnc2ncncc2n1":     "c1cnc2nccnc2n1",  # pteridine alternate
    # Purine: alternate canonical produced from keto-tautomer purines (xanthine etc.)
    "c1ncc2[nH]cnc2n1":   "c1nc2c[nH]cnc-2n1",  # 9H-purine alternate
}


# ---------------------------------------------------------------------------
# Build canonical SMILES lookup table from data files (lazy, module-level)
# ---------------------------------------------------------------------------

_smiles_to_record: dict[str, dict] = {}
_lookup_built = False


def _build_lookup() -> None:
    """Build a canonical-SMILES -> record dict from retained_rings + rings_from_opsin."""
    global _smiles_to_record, _lookup_built
    if _lookup_built:
        return

    # retained_rings.json: nested dict by category
    try:
        rings = get_retained_rings()
        for category_data in rings.values():
            if isinstance(category_data, dict):
                for ring_name, record in category_data.items():
                    smiles_raw = None
                    sub_form = None
                    if isinstance(record, dict):
                        smiles_raw = record.get("smiles") or record.get("canonical_smiles")
                        sub_form = record.get("substituent_form")
                    elif isinstance(record, str):
                        smiles_raw = record
                    if smiles_raw:
                        # Canonicalize the stored SMILES
                        mol = Chem.MolFromSmiles(smiles_raw)
                        if mol is not None:
                            can = Chem.MolToSmiles(mol)
                            if can not in _smiles_to_record:
                                _smiles_to_record[can] = {
                                    "name": ring_name,
                                    "substituent_form": sub_form,
                                }
    except Exception as e:
        logger.debug("retained_rings lookup build error: %s", e)

    # rings_from_opsin.json: list of {name, smiles, source}
    # NOTE: arylGroups.xml entries are OPSIN *substituent stems* (e.g. "pyrrolidin",
    # "anthracen") — they drop the terminal 'e' so "-yl" attaches cleanly to form
    # substituent names ("pyrrolidin-1-yl", "anthracen-9-yl"). They are NOT
    # parent-name forms. For saturated heterocycles with canonical HW saturated-
    # ring endings we re-append 'e' to recover the parent name. For everything
    # else we keep the stem as-is (some names like "furan", "indol", "coumarin"
    # are real non-'e' parent names).
    _SAT_HW_STEM_ENDINGS = (
        "iridin", "etidin", "olidin", "olan", "iran", "etan",
        "inan", "epan", "ocan", "onan", "ecan",
        "morpholin",
    )
    # PAHs, metallocenes, and other unsaturated hydrocarbons/heterocycles stored
    # in arylGroups.xml as "-en" stems (anthracen, pleiaden, trinden, aceanthrylen,
    # fluoren, xanthen, thiophen, …) all have canonical parent forms ending in
    # "-ene".  The parent form is what we want here; the substituent form is
    # derived separately by attaching "-yl" to the stem.  Without this
    # rehydration, the engine emits broken names like "pliaden" (after elide()
    # munches the interior "ei") or non-parseable "trinden".
    _UNSAT_EN_STEM_TAIL = "en"
    # OPSIN-extracted name overrides: the arylGroups.xml entry uses a stem
    # that is broken when emitted as a parent name.  Map the OPSIN stem to
    # the spec PIN form so the engine emits a valid IUPAC name.
    #
    # Per P-54.4.4, the PIN is "1,2-didehydrobenzene" but the retained name
    # "benzyne" is the historical / general-nomenclature form OPSIN parses;
    # the engine emits the retained spelling because the systematic
    # didehydro path is not yet implemented (a 5-aromatic-bond + 1 triple-
    # bond ring is not handled by the cyclo-polyene branch).  Choosing
    # "benzyne" preserves OPSIN round-trip equivalence.
    _OPSIN_NAME_PIN_OVERRIDE: dict[str, str] = {
        "benzyn": "1,2-didehydrobenzene",
    }
    try:
        opsin_rings = get_rings_from_opsin()
        for entry in opsin_rings:
            if not isinstance(entry, dict):
                continue
            smiles_raw = entry.get("smiles")
            name_raw = entry.get("name", "")
            source = entry.get("source", "")
            if not smiles_raw or not name_raw:
                continue
            # Skip pipe-variant fusion component entries (they're not standalone names)
            if source == "fusionComponents.xml":
                continue
            mol = Chem.MolFromSmiles(smiles_raw)
            if mol is not None:
                can = Chem.MolToSmiles(mol)
                # Only record if not already in the lookup (retained_rings takes priority)
                if can not in _smiles_to_record:
                    # Strip pipe variants: "1H-pyrrole|pyrrol" -> "1H-pyrrole" (first variant)
                    primary_name = name_raw.split("|")[0].strip()
                    # Rehydrate saturated heterocycle stems from arylGroups.xml:
                    # "pyrrolidin" -> "pyrrolidine", "imidazolidin" -> "imidazolidine", etc.
                    if (
                        source == "arylGroups.xml"
                        and not primary_name.endswith("e")
                        and any(primary_name.endswith(s) for s in _SAT_HW_STEM_ENDINGS)
                    ):
                        primary_name = primary_name + "e"
                    # Rehydrate unsaturated "-en" stems from arylGroups.xml to
                    # "-ene" parent form (anthracen -> anthracene, pleiaden ->
                    # pleiadene, trinden -> trindene, aceanthrylen ->
                    # aceanthrylene, fluoren -> fluorene, thiophen -> thiophene).
                    # Covers PAHs, metallocenes, and unsaturated heterocycles.
                    elif (
                        source == "arylGroups.xml"
                        and primary_name.endswith(_UNSAT_EN_STEM_TAIL)
                    ):
                        primary_name = primary_name + "e"
                    # Apply spec-PIN overrides (P-54.4.4 et al.).
                    if primary_name in _OPSIN_NAME_PIN_OVERRIDE:
                        primary_name = _OPSIN_NAME_PIN_OVERRIDE[primary_name]
                    _smiles_to_record[can] = {"name": primary_name, "substituent_form": None}
    except Exception as e:
        logger.debug("rings_from_opsin lookup build error: %s", e)

    # Attach atom_locants augmentations for specific OPSIN entries (see
    # _OPSIN_RING_ATOM_LOCANTS docstring).  The key in that table is the
    # canonical SMILES produced by _build_lookup above; we only attach when
    # the entry is actually present in _smiles_to_record so the side table
    # never introduces a new name mapping.
    for can_smi, al in _OPSIN_RING_ATOM_LOCANTS.items():
        if can_smi in _smiles_to_record:
            _smiles_to_record[can_smi]["atom_locants"] = al

    _lookup_built = True


# ---------------------------------------------------------------------------
# Non-substitutable retained ring names (P-25.3 / P-31)
# ---------------------------------------------------------------------------
# A handful of retained ring names accepted by OPSIN as bare scaffolds are NOT
# substitutable in OPSIN's vocabulary: OPSIN refuses to attach a substituent
# locant to the retained stem (it parses "xanthine" but rejects
# "8-chloroxanthine").  Per P-25.3 / P-31, the systematic ring name is the PIN
# and is fully substitutable, so a substituted instance must fall back to the
# systematic parent.  For the xanthine family the systematic parent is the
# mancude purine carrying a 2,6-dione (``3,7-dihydro-1H-purine-2,6-dione``),
# which the oxo-on-mancude derivation (_try_derive_oxo_aromatic_retained, with
# the over-NH partial-sanitize fallback) reconstructs and the downstream
# characteristic-group + indicated-H layer renders.
#
# The bare scaffolds themselves (xanthine, hypoxanthine, caffeine, theobromine,
# theophylline, guanine, adenine) are resolved by the WHOLE-MOLECULE retained
# table in data_loader/retained_names_expanded.json BEFORE the ring-naming path
# is reached, so they keep their retained names; only forms that reach
# ``try_retained_name`` with an unexpressible ring substituent fall back here.
#
# GENERAL GAP: ideally this lives as a ``substitutable: false`` flag on the
# affected entries in data_loader._RING_CURATED_SMILES and on the
# rings_from_opsin data entries, so any non-substitutable retained ring (not
# just xanthine) auto-falls-back.  data_loader.py is outside this change's
# scope; pending that data-side flag, the set below enumerates the
# verified-non-substitutable names this module gates on.
_NONSUBSTITUTABLE_RETAINED_NAMES: frozenset[str] = frozenset({
    "xanthine",
    "xanthin",  # rings_from_opsin stem form (no trailing 'e')
})


def _ring_has_unexpressible_substituent(ring_system, mol) -> bool:
    """Return True iff a ring atom carries a heavy-atom substituent that a
    non-substitutable retained ring name cannot express.

    "Expressible" exocyclic groups are the ring-carbon double-bond
    characteristic groups (=O / =S / =Se / =N-) that the systematic
    fall-back parent renders as a suffix/prefix (``-dione`` / ``-dioxo-``);
    those are already implied by the retained name itself (e.g. the two oxo
    groups of xanthine).  Any OTHER heavy neighbour of a ring atom — a
    single-bonded halogen, alkyl/aryl carbon, hetero substituent, or an
    N-substituent on a ring nitrogen — is a genuine ring substituent the
    non-substitutable retained stem cannot carry, so the molecule must be
    named on the systematic substitutable parent instead.
    """
    try:
        ring_atoms = frozenset(ring_system.atom_indices)
    except Exception:
        return False
    if not ring_atoms:
        return False
    # N (7), O (8), S (16), Se (34) — the suffix-eligible exocyclic
    # double-bond partners that the systematic dione/thione/imine renders.
    _SUFFIX_DB_ATOMS = {7, 8, 16, 34}
    for atom_idx in ring_atoms:
        a = mol.GetAtomWithIdx(atom_idx)
        for b in a.GetBonds():
            other = b.GetOtherAtom(a)
            o_idx = other.GetIdx()
            if o_idx in ring_atoms:
                continue
            if other.GetAtomicNum() == 1:
                continue  # explicit H is not a substituent
            # Exocyclic chalcogen/imine double bond off a ring carbon: this is
            # the dione/thione/imine the systematic parent expresses, NOT a
            # substituent.  (Ring-N -> exocyclic =O, an N-oxide, is left to the
            # general path; it is rare for these scaffolds.)
            if (
                b.GetBondType() == Chem.BondType.DOUBLE
                and a.GetAtomicNum() == 6
                and other.GetAtomicNum() in _SUFFIX_DB_ATOMS
            ):
                continue
            return True
    return False


# ---------------------------------------------------------------------------
# Main lookup function
# ---------------------------------------------------------------------------

def try_retained_name(
    ring_system: "RingSystem", mol
) -> "list[NamedParent]":
    """Try to find a retained name for a ring system.

    Returns a list with 0 or 1 NamedParent objects.
    """
    from iupac_namer.ring_naming.common import extract_ring_mol, extract_ring_mol_with_exo_oxo

    ring_mol = extract_ring_mol(ring_system, mol)
    ring_smiles: str | None = None
    if ring_mol is not None:
        try:
            ring_smiles = Chem.MolToSmiles(ring_mol)
        except Exception:
            ring_smiles = None

    # Stereo-stripped fallback canonical SMILES.  When a ring is carved from a
    # stereo-bearing molecule (e.g. an FDA drug with @H descriptors), the
    # extracted ring SMILES carries those stereo markers, which prevents an
    # exact-string match against the curated/data tables (whose keys are
    # stereo-free).  The ring system's IUPAC parent identity is independent
    # of stereochemistry, so we also try the no-stereo canonical form.
    no_stereo_smiles: str | None = None
    if ring_mol is not None and ring_smiles:
        try:
            ring_mol_ns = Chem.Mol(ring_mol)  # copy
            Chem.RemoveStereochemistry(ring_mol_ns)
            candidate = Chem.MolToSmiles(ring_mol_ns)
            if candidate and candidate != ring_smiles:
                no_stereo_smiles = candidate
        except Exception:
            no_stereo_smiles = None

    # 1. Check curated table first (highest accuracy)
    match_name = match_sub = match_atom_locants = None
    match_alkyl_ok = False
    matched_curated_key: str | None = None
    extra_atom_indices: frozenset[int] | None = None

    # 1a (priority).  Try the with-=O canonical FIRST if the ring carries a
    # ring-carbon C=O.  Many lactone / lactam / cyclic-ketone tautomers have
    # both a curated retained PIN keyed under the with-=O canonical (e.g.
    # ``O=C1C=CCCO1`` → 5,6-dihydro-2H-pyran-2-one) AND a curated indicated-H
    # parent under the no-=O canonical (``C1=CCOCC1`` → 3,6-dihydro-2H-pyran).
    # The with-=O entry is the more specific match — it pins the C=O at L2
    # and re-routes the ring numbering accordingly per P-31.1.4.3.4 (lowest
    # locant for principal characteristic group), so prefer it over the
    # bare-skeleton + exo-oxo decoration which would force the wrong locants.
    # Only fires when the with-=O canonical is genuinely different from the
    # bare ring canonical (i.e. there IS a ring-carbon C=O to lift in).
    if ring_smiles:
        oxo_result_priority = extract_ring_mol_with_exo_oxo(ring_system, mol)
        if oxo_result_priority is not None:
            oxo_ring_mol_p, oxo_extra_p = oxo_result_priority
            try:
                oxo_smiles_p = Chem.MolToSmiles(oxo_ring_mol_p)
            except Exception:
                oxo_smiles_p = None
            if oxo_smiles_p and oxo_smiles_p != ring_smiles:
                m_name, m_sub, m_alkyl_ok, m_atom_locants = _lookup_curated(oxo_smiles_p)
                if m_name is not None:
                    match_name = m_name
                    match_sub = m_sub
                    match_alkyl_ok = m_alkyl_ok
                    match_atom_locants = m_atom_locants
                    matched_curated_key = _lookup_curated_key(oxo_smiles_p)
                    ring_mol = oxo_ring_mol_p
                    extra_atom_indices = frozenset(oxo_extra_p)

    if match_name is None and ring_smiles:
        match_name, match_sub, match_alkyl_ok, match_atom_locants = _lookup_curated(ring_smiles)
        if match_name is not None:
            matched_curated_key = _lookup_curated_key(ring_smiles)

    if match_name is None and no_stereo_smiles:
        # 1b. Try stereo-stripped lookup against curated table.
        match_name, match_sub, match_alkyl_ok, match_atom_locants = _lookup_curated(no_stereo_smiles)
        if match_name is not None:
            matched_curated_key = _lookup_curated_key(no_stereo_smiles)

    if match_name is None and ring_smiles:
        # 2. Check data-file table.
        #
        # Guard: for BRIDGED ring systems (von Baeyer polycycles like
        # norbornene, bicyclo[2.2.1]hept-5-ene), skip the OPSIN-extracted
        # trivial-name table.  Those entries carry no atom_locants, so the
        # engine cannot align the retained-name's canonical numbering
        # (e.g. "norbornen-2-yl" implies a fixed 1,4-bridgehead numbering
        # with the double bond at 5,6) with the actual input attachment
        # point.  Result: we emit "norbornen-2-yl" but OPSIN parses it to
        # a different attachment isomer, breaking SMILES round-trip.
        # IUPAC P-23.2.6.2 prefers systematic von Baeyer for these scaffolds
        # as PIN anyway.  Fused/monocyclic retained names are unaffected.
        if getattr(ring_system, "type", None) != "bridged":
            _build_lookup()
            record = _smiles_to_record.get(ring_smiles)
            matched_record_key: str | None = None
            if record is not None:
                matched_record_key = ring_smiles
            elif no_stereo_smiles:
                record = _smiles_to_record.get(no_stereo_smiles)
                if record is not None:
                    matched_record_key = no_stereo_smiles
            if record is not None:
                # PIN-eligibility gate (data-file path): for retained names
                # that the OPSIN-extracted data tables also list (indan,
                # tetralin, chroman, isochroman) the curated table's
                # pin_name / pin_substituent_form alias takes precedence so
                # the systematic PIN (2,3-dihydro-1H-indene,
                # 1,2,3,4-tetrahydronaphthalene, etc.) is emitted per
                # P-25.3.1.3 / P-31.1.4.2.4 / P-32.4 / P-53 / P-54.4.3.2.
                if record["name"] in _DATAFILE_PIN_INELIGIBLE_NAMES:
                    record = None
                    matched_record_key = None
            if record is not None:
                match_name = record["name"]
                match_sub = record.get("substituent_form")
                match_alkyl_ok = False  # conservative default for data-file entries
                # Most OPSIN-extracted entries have no atom_locants; a small
                # curated side table (_OPSIN_RING_ATOM_LOCANTS) supplements
                # specific entries whose IUPAC numbering does not coincide
                # with RDKit's canonical atom order (e.g. spiro-9,9'-bifluorene).
                match_atom_locants = record.get("atom_locants")
                if match_atom_locants is not None:
                    # Atom-locants are indexed against the canonical-SMILES
                    # record key; the downstream rebuild path uses this to
                    # align ring_mol atom ordering with atom_locants.
                    matched_curated_key = matched_record_key

    # 3. Exocyclic-oxo fallback.  Many curated lactam / cyclic-ketone entries
    # (e.g. ``O=C1CCc2ccccc2N1`` for 3,4-dihydroquinolin-2(1H)-one) are keyed
    # with the exocyclic =O included.  ``extract_ring_mol`` strips that O,
    # so the substitutive lookup path (where the ring is carved out of a
    # larger molecule and the whole-molecule retained-name path doesn't fire)
    # never matches.  Re-add exocyclic =O on sp2 ring carbons and retry.
    # The substructure-match in _build_numbering_from_atom_locants uses this
    # same enriched ring_mol, so atom_locants stay consistent.  We also track
    # the full-mol indices of those =O atoms so the retained-name parent can
    # claim them (preventing downstream from emitting a redundant "-N-oxo-"
    # prefix for a carbonyl already encoded in the retained-name stem).
    # NOTE: ``extra_atom_indices`` is declared earlier (in section 1a) so the
    # priority with-=O path can populate it; this section only fills it when
    # 1a did not match.
    # Phase 8 (P-31.1 pyrazolone family): set True only on the data-file
    # fallthrough below for retained stems that lexically embed a suffix-form
    # ending (``5-pyrazolone`` → ``-one``, ``urazol`` → ``-zol(e)``,
    # ``phthalhydrazide`` → ``-hydrazide``).  OPSIN's parser refuses to glue
    # a separable PCG suffix onto such stems.  Curated retained entries
    # (cephem, sulfolene, …) leave the flag False because their atom-locant
    # metadata supports separable suffixes (``cephem-4-carboxylate``).
    precomposed_no_separable_suffix: bool = False
    if match_name is None:
        oxo_result = extract_ring_mol_with_exo_oxo(ring_system, mol)
        if oxo_result is not None:
            oxo_ring_mol, oxo_extra = oxo_result
            try:
                oxo_smiles = Chem.MolToSmiles(oxo_ring_mol)
            except Exception:
                oxo_smiles = None
            oxo_no_stereo: str | None = None
            if oxo_smiles:
                try:
                    oxo_ns = Chem.Mol(oxo_ring_mol)
                    Chem.RemoveStereochemistry(oxo_ns)
                    cand = Chem.MolToSmiles(oxo_ns)
                    if cand and cand != oxo_smiles:
                        oxo_no_stereo = cand
                except Exception:
                    oxo_no_stereo = None
            # Try the curated table first (highest accuracy).
            if oxo_smiles:
                match_name, match_sub, match_alkyl_ok, match_atom_locants = _lookup_curated(oxo_smiles)
                if match_name is not None:
                    matched_curated_key = _lookup_curated_key(oxo_smiles)
                if match_name is None and oxo_no_stereo:
                    match_name, match_sub, match_alkyl_ok, match_atom_locants = _lookup_curated(oxo_no_stereo)
                    if match_name is not None:
                        matched_curated_key = _lookup_curated_key(oxo_no_stereo)
                if match_name is not None:
                    # Use the enriched ring_mol so atom_locants (keyed against
                    # the with-=O canonical SMILES) line up with substructure
                    # matching against the parent mol.
                    ring_mol = oxo_ring_mol
                    extra_atom_indices = frozenset(oxo_extra)

            # Fall back to the OPSIN data-file table for entries that keep
            # the exocyclic =O (and/or =S, or SO2 S) in the canonical form,
            # e.g. "4-pyrazolone" (O=C1C=NNC1), "5-pyrazolone" (O=C1CC=NN1),
            # "urazol" (O=c1[nH][nH]c(=O)[nH]1), "sulfolene" / "sulfol-3-ene"
            # (O=S1(=O)CC=CC1).  These are pre-composed retained names: the
            # name itself encodes the carbonyl / sulfonyl locants, so we
            # claim the exo-=O atoms via ``extra_atom_indices`` and no further
            # ``-N-one`` / ``-N,M-dione`` / ``-dioxide`` suffix is attached.
            # Only accept when the matched name does NOT end in a separable
            # suffix that downstream would re-emit (e.g. "-one", "-dione",
            # "-quinone"): those need atom_locants alignment, which the
            # OPSIN data usually lacks.
            if match_name is None and oxo_smiles:
                _build_lookup()
                record = _smiles_to_record.get(oxo_smiles)
                if record is None and oxo_no_stereo:
                    record = _smiles_to_record.get(oxo_no_stereo)
                if record is not None:
                    candidate_name = record["name"]
                    # Only accept pre-composed retained names whose stem
                    # already embeds the =O locants.  Names ending in bare
                    # "-one"/"-dione"/"-quinone" need explicit locant
                    # alignment via atom_locants (absent here); names like
                    # "4-pyrazolone", "urazol", "sulfolene" do not.
                    _emits_separate_oxo_suffix = any(
                        candidate_name.endswith(suf)
                        for suf in ("-one", "-dione", "-trione", "-quinone")
                    )
                    if not _emits_separate_oxo_suffix:
                        match_name = candidate_name
                        match_sub = record.get("substituent_form")
                        match_alkyl_ok = False
                        match_atom_locants = record.get("atom_locants")
                        ring_mol = oxo_ring_mol
                        extra_atom_indices = frozenset(oxo_extra)
                        # OPSIN data-table entries reach this fallthrough
                        # because ``extract_ring_mol_with_exo_oxo`` enriched
                        # the ring with =O / =S / SO2 to match the canonical
                        # SMILES key.  These names have NO atom_locants in
                        # the OPSIN data and the stem itself encodes the
                        # carbonyl/sulfonyl as a suffix-form ending.  Mark
                        # them so the engine routes any separate PCG to the
                        # prefix slot instead of trying to glue a suffix
                        # onto the lexically-frozen stem.
                        precomposed_no_separable_suffix = True

    # 3b. Non-substitutable retained-name fallback (P-25.3 / P-31).
    # A few retained ring names (xanthine and family) parse in OPSIN only as
    # bare scaffolds — OPSIN refuses a substituent locant on the retained stem
    # ("xanthine" parses, "8-chloroxanthine" does not).  When sections 1-3
    # matched such a name BUT the molecule carries a ring substituent the stem
    # cannot express, discard the match so the systematic substitutable parent
    # (e.g. the mancude purine with a 2,6-dione) is derived instead via the
    # oxo-on-mancude path (section 5-oxo).  Bare scaffolds are unaffected: they
    # resolve via the whole-molecule retained table before reaching this code,
    # and even if one reached here it has no unexpressible substituent so the
    # gate does not fire.
    if (
        match_name is not None
        and match_name in _NONSUBSTITUTABLE_RETAINED_NAMES
        and _ring_has_unexpressible_substituent(ring_system, mol)
    ):
        match_name = None
        match_sub = None
        match_alkyl_ok = False
        match_atom_locants = None
        matched_curated_key = None
        extra_atom_indices = None
        precomposed_no_separable_suffix = False
        # NOTE: ``ring_mol`` is intentionally left as-is (it may be the with-=O
        # enriched mol that section 1a swapped in).  The systematic fall-back
        # used here — _try_derive_oxo_aromatic_retained (section 5-oxo) — takes
        # only ``ring_system`` and ``mol``; it ignores ``ring_mol`` and
        # re-derives the mancude parent from the full molecule.  The hydro
        # derivation (section 5) runs first but returns None for these
        # dione-on-mancude scaffolds (no curated partial-saturation skeleton
        # matches), so it does not interfere.

    # 4. Strip-oxo retained-parent derivation.  When the curated table holds
    # the base "indicated-H" retained parent (e.g. ``C1=COC=CC1`` → 4H-pyran)
    # but NOT the decorated ketone form (e.g. ``O=c1ccocc1`` → 4H-pyran-4-one),
    # strip the exocyclic =O on sp2 ring C, look up the resulting canonical,
    # and — if the match's indicated-H locant aligns with the former-C=O
    # position — emit the base retained parent with NO =O claim.  The engine's
    # ketone-FG layer then appends ``-<N>-one`` naturally, yielding the full
    # "<NH>-pyran-<N>-one" form.  This is an architectural derivation: any
    # retained ring parent R whose curated key pins the indicated-H at locant
    # N automatically becomes a source for ``R-N-one`` names without a
    # dedicated curated ketone entry.
    if match_name is None:
        from iupac_namer.ring_naming.common import extract_ring_mol_stripping_exo_oxo
        stripped_result = extract_ring_mol_stripping_exo_oxo(ring_system, mol)
        if stripped_result is not None:
            stripped_mol, stripped_exo_indices, stripped_co_indices = stripped_result
            try:
                stripped_smiles = Chem.MolToSmiles(stripped_mol)
            except Exception:
                stripped_smiles = None
            if stripped_smiles:
                s_name, s_sub, s_alkyl_ok, s_atom_locants = _lookup_curated(
                    stripped_smiles
                )
                s_curated_key = _lookup_curated_key(stripped_smiles) if s_name else None
                # Accept the stripped match only if:
                #   (a) the matched name begins with a single "<digit>H-"
                #       indicated-H prefix (no embedded "(NH)" or locant list),
                #   (b) we can resolve the indicated-H locant to a ring-mol
                #       atom via atom_locants, AND
                #   (c) that atom is exactly one of the former-C=O ring atoms.
                # Together these ensure the retained parent's canonical
                # numbering places the =O on the correct locant, so downstream
                # ketone-FG detection emits "-<N>-one" coherently.
                if s_name is not None and s_atom_locants is not None:
                    m_ih = _re.match(r"^(\d+)H-", s_name)
                    if m_ih is not None:
                        ih_locant = int(m_ih.group(1))
                        # atom_locants maps {ring_mol_atom_idx: iupac_locant}.
                        # Find the ring-mol atom idx whose locant equals ih_locant.
                        ih_ring_mol_idx = None
                        for rm_idx, loc in s_atom_locants.items():
                            if isinstance(loc, int) and loc == ih_locant:
                                ih_ring_mol_idx = rm_idx
                                break
                        if ih_ring_mol_idx is not None and s_curated_key:
                            # Substructure-match the curated key mol into the
                            # stripped ring mol to map ring-mol indices to
                            # full-mol indices.  If the mapped atom is one of
                            # the former-C=O atoms, the derivation is valid.
                            try:
                                key_mol = Chem.MolFromSmiles(s_curated_key)
                                key_matches = stripped_mol.GetSubstructMatches(
                                    key_mol, uniquify=False
                                ) if key_mol is not None else ()
                            except Exception:
                                key_matches = ()
                            # Also need to map stripped_mol atom indices to
                            # full-mol indices.  When we built stripped_mol we
                            # preserved kept-atom relative ordering, so the
                            # ring-set atoms appear in sorted order from the
                            # parent.  Re-derive via substructure match of
                            # stripped_mol into the parent (without the =O).
                            sorted_ring_atoms = sorted(ring_system.atom_indices)
                            # Build a mapping: stripped_mol atom idx -> full-mol atom idx.
                            # Kept atoms were ring atoms (sorted) + exo-O; exo-O
                            # was removed afterwards, so stripped_mol's atom
                            # indices 0..n_ring-1 correspond to sorted ring atoms.
                            n_ring_stripped = stripped_mol.GetNumAtoms()
                            if n_ring_stripped == len(sorted_ring_atoms):
                                stripped_to_full = {
                                    i: sorted_ring_atoms[i]
                                    for i in range(n_ring_stripped)
                                }
                                # For each curated-key match, check whether
                                # the key's indicated-H atom (ih_ring_mol_idx)
                                # maps to one of the former-C=O full-mol atoms.
                                former_co_full_set = set(stripped_co_indices)
                                valid = False
                                for km in key_matches:
                                    if ih_ring_mol_idx >= len(km):
                                        continue
                                    stripped_idx = km[ih_ring_mol_idx]
                                    full_idx = stripped_to_full.get(stripped_idx)
                                    if full_idx in former_co_full_set:
                                        valid = True
                                        break
                                if valid:
                                    match_name = s_name
                                    match_sub = s_sub
                                    match_alkyl_ok = s_alkyl_ok
                                    match_atom_locants = s_atom_locants
                                    matched_curated_key = s_curated_key
                                    # Use the stripped ring mol for the
                                    # downstream substructure-match step so
                                    # locants align with the base retained
                                    # parent (not the with-=O shape).  Do NOT
                                    # set extra_atom_indices — we want the
                                    # exocyclic =O to fall through to the
                                    # ketone-FG layer.
                                    ring_mol = stripped_mol

    # 5. Dihydro / tetrahydro / hexahydro derivation.  When the curated table
    # holds the fully-aromatic retained parent (e.g. naphthalene → c1ccc2ccccc2c1)
    # but NOT the partly-hydrogenated form (e.g. 1,4-dihydronaphthalene,
    # 1,2-dihydronaphthalene), "re-aromatize" the ring by flagging every ring
    # atom and ring bond as aromatic and look up the resulting canonical
    # SMILES.  If it matches a fully-aromatic retained entry, the sp3 ring
    # atoms in the input become the hydrogenation locants and we emit
    # ``<sorted-locants>-<multiplier>hydro-<retained_name>``.  This is a
    # general pass — any retained aromatic parent with known atom_locants
    # automatically becomes a source for its partly-hydrogenated derivatives.
    derived_dihydro: dict | None = None
    derived_orientations: list[dict] = []
    if match_name is None and ring_mol is not None:
        derived_dihydro = _try_derive_hydro_retained(
            ring_system=ring_system,
            ring_mol=ring_mol,
            mol=mol,
        )
        if derived_dihydro is not None:
            match_name = derived_dihydro["name"]
            match_sub = derived_dihydro.get("substituent_form")
            match_alkyl_ok = False
            match_atom_locants = None  # numbering built directly below
            matched_curated_key = None

    # 5-oxo. Oxo-to-mancude retained-parent derivation (P-31.1.4.3.4 / P-66.6).
    # When a fused ring carries ring-carbon exocyclic =O (or =S/=Se/=NR) groups
    # such that STRIPPING those exocyclic doublets and re-aromatizing the ring
    # recovers a fully-aromatic retained parent (e.g. naphthalene-1,2-dione →
    # naphthalene + two -one suffixes), emit the mancude retained parent and
    # leave the exocyclic =O for the downstream characteristic-group layer to
    # express as ``-dione`` / ``-one`` etc.  This is the dione analogue of the
    # count==1 added-IH path in ``_try_derive_hydro_retained`` (which already
    # handles a SINGLE in-ring ketone, e.g. naphthalen-1(2H)-one) and of the
    # dihydro path (which handles a curated dihydro skeleton, e.g. the para
    # 1,4-dione that lands on the curated 1,4-dihydronaphthalene key).  The
    # adjacent (ortho) dione has no curated dihydro skeleton, so this
    # generative pass is required to name it on the mancude parent.
    if match_name is None and ring_mol is not None:
        derived_oxo = _try_derive_oxo_aromatic_retained(
            ring_system=ring_system,
            mol=mol,
        )
        if derived_oxo is not None:
            derived_dihydro = derived_oxo  # reuse numbering plumbing below
            match_name = derived_oxo["name"]
            match_sub = derived_oxo.get("substituent_form")
            match_alkyl_ok = False
            match_atom_locants = None  # numbering carried in derived dict
            matched_curated_key = None

    # 5b. Partial-saturation orientation enumeration.  When the curated entry
    # IS itself a partial-saturation form (its name carries a leading
    # "<digits>-<mult>hydro" prefix), the curated hydro-locants are the
    # CONVENTIONAL labelling for the unsubstituted parent — but a substituted
    # variant may demand a different ring numbering (e.g. lowest locant for
    # a principal characteristic group per P-31.1.4.3.4) which shifts the
    # hydro-prefix to a structurally-equivalent alternative
    # (e.g. 1,2,3,6-tetrahydropyridine ↔ 1,2,5,6-tetrahydropyridine).
    #
    # Stage 22 R22-A: invoke the same derive-hydro machinery against the
    # underlying aromatic parent for these entries, returning ALL valid
    # orientations as separate NamedParent candidates.  The strategy layer
    # picks the orientation whose chosen numbering minimizes
    # principal-characteristic-group locants (then prefix-substituent
    # locants), avoiding the failure mode where a hard-coded curated
    # hydro-prefix forces a substituent onto a structurally-incompatible
    # locant (e.g. sp2 C carrying the COOH but rendered with a name whose
    # locant 3 is sp3 in the curated form).
    if (
        match_name is not None
        and ring_mol is not None
        and _re.match(r"^[\d,]+-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)hydro", match_name)
    ):
        all_orientations = _try_derive_hydro_retained(
            ring_system=ring_system,
            ring_mol=ring_mol,
            mol=mol,
            return_all_orientations=True,
        )
        # Only emit derived orientations when the underlying aromatic
        # parent has true symmetry-based ambiguity (>=2 valid match
        # orientations).  For asymmetric parents (e.g. benzofuran), the
        # single derive-output would simply duplicate the curated form
        # with a less-canonical hydro-locant rendering — leave the curated
        # entry as the sole option in that case.
        if isinstance(all_orientations, list) and len(all_orientations) >= 2:
            derived_orientations = all_orientations

    # 5c. Oxo+dihydro re-orientation (P-31.1.4.1.1 / P-31.1.4.3.4).  The
    # curated bare-skeleton match above pins a FIXED dihydro orientation in its
    # name (e.g. ``2,3-dihydrofuran``), but a furanone / thiophenone whose ring
    # carbonyl is NOT the principal characteristic group (a higher-seniority
    # group such as -COOH is present) must renumber so that group gets the
    # lowest locant — which moves the actually-saturated ring atoms to
    # different locants and forces the ring carbonyl to be cited as an ``oxo``
    # PREFIX.  In that decomposition the carbonyl carbon is itself a saturation
    # (added-H) position, so the dihydro locant set must INCLUDE it (e.g.
    # ``2,3-dihydrofuran`` -> ``5-oxo-4,5-dihydrofuran-2-carboxylic acid``, not
    # the structurally-wrong ``5-oxo-2,3-dihydrofuran-...``).
    #
    # Derive these orientations from the fully-mancude parent (furan /
    # thiophene) with the ring carbonyl counted as a saturation position, and
    # offer them as additional candidates.  The strategy layer keeps the
    # curated form for the no-higher-PCG case (where the carbonyl is the
    # suffix) and selects an oxo+dihydro orientation only when it gives the
    # lower principal-group locant.  Gated on the curated entry being a
    # 5-ring dihydro form whose mancude parent we carry oxo-dihydro locants for
    # (so 6-ring pyranones, already handled via the 2H-/4H-pyran partial-
    # saturation parents + plain 5b, are untouched).
    #
    # CRITICAL: only ADD orientations whose dihydro locant SET differs from the
    # curated ``match_name``'s set.  When the curated entry is the SYMMETRIC
    # saturation form (e.g. ``2,5-dihydrofuran``, saturation flanking the
    # heteroatom), every re-orientation reproduces the same locant set and the
    # ring carbonyl is correctly handled as the suffix at its lowest locant by
    # the curated path — adding equal-set orientations would only inject a
    # higher-locant oxo numbering (``...-5-one`` instead of ``...-2-one``).
    # The re-orientation is needed ONLY for the ASYMMETRIC adjacent-saturation
    # form (``2,3-dihydrofuran``), where a higher-seniority PCG flips the
    # numbering and the saturation genuinely moves (2,3 -> 4,5).
    if (
        match_name is not None
        and ring_mol is not None
        and _re.match(r"^[\d,]+-dihydro", match_name)
    ):
        oxo_dihydro_orientations = _try_derive_hydro_retained(
            ring_system=ring_system,
            ring_mol=ring_mol,
            mol=mol,
            return_all_orientations=True,
            include_oxo_carbons_as_saturation=True,
        )
        if isinstance(oxo_dihydro_orientations, list) and oxo_dihydro_orientations:
            _curated_loc_match = _re.match(r"^([\d,]+)-dihydro", match_name)
            _curated_set = (
                frozenset(_curated_loc_match.group(1).split(","))
                if _curated_loc_match else frozenset()
            )

            def _orient_loc_set(orient_name: str | None) -> frozenset[str]:
                if not orient_name:
                    return frozenset()
                m = _re.match(r"^([\d,]+)-dihydro", orient_name)
                return frozenset(m.group(1).split(",")) if m else frozenset()

            _filtered = [
                o for o in oxo_dihydro_orientations
                if _orient_loc_set(o.get("name")) != _curated_set
            ]
            if _filtered:
                derived_orientations = derived_orientations + _filtered

    if match_name is None:
        return []

    # Kekulé-disambiguation rewrite.  Some canonical SMILES key multiple
    # retained kekulé tautomers that each carry a distinct IUPAC numbering
    # (e.g. 1H-indene vs ind-1-ene: identical canonical, mirror-image
    # locants).  The atom_locants in ``_OPSIN_RING_ATOM_LOCANTS`` are pinned
    # to one specific tautomer's numbering; the default emitted name may be
    # a sibling whose convention rotates the locants — producing a round-
    # trip mirror mismatch on substituted probes.  The kekule_store
    # rewrites the name/sub_form to the tautomer matching the atom_locants,
    # leaving numbering and atom claims unchanged.  See
    # ``iupac_namer/ring_naming/kekule_store.py`` docstring.
    _rewrite_key = matched_curated_key
    if _rewrite_key is None and ring_mol is not None:
        try:
            _rewrite_key = Chem.MolToSmiles(ring_mol)
        except Exception:
            _rewrite_key = None
    if _rewrite_key:
        _rewrite = maybe_rewrite_for_kekule(_rewrite_key)
        if _rewrite is not None:
            match_name = _rewrite.name
            if _rewrite.substituent_form is not None:
                match_sub = _rewrite.substituent_form

    # If we have atom_locants (ring-mol-idx -> IUPAC locant), convert to
    # full-mol numbering by substructure matching ring_mol into mol.
    #
    # Critical: atom_locants are indexed against the atoms of the RDKit mol
    # built from the curated KEY SMILES.  When a ring is carved from a
    # stereo-bearing parent, ``ring_mol`` may have a different canonical atom
    # ordering than the curated key (stereo markers change the canonical
    # ranking).  Using the carved ring_mol's indices directly produces a
    # permuted locant assignment (bug seen on steroid scaffolds: OHs at
    # locants 2,4 instead of 3,7).  Rebuild ring_mol from the curated key
    # SMILES so atom indices align with atom_locants.
    numbering_options: tuple[Numbering, ...] = ()
    if derived_dihydro is not None:
        numbering_options = derived_dihydro["numbering_options"]
    elif match_atom_locants and ring_mol is not None:
        ring_mol_for_match = ring_mol
        if matched_curated_key is not None:
            try:
                rebuilt = Chem.MolFromSmiles(matched_curated_key)
                if rebuilt is not None and rebuilt.GetNumAtoms() == ring_mol.GetNumAtoms():
                    ring_mol_for_match = rebuilt
            except Exception:
                pass
        numbering_options = _build_numbering_from_atom_locants(
            ring_mol=ring_mol_for_match,
            mol=mol,
            ring_system=ring_system,
            atom_locants=match_atom_locants,
        )

    # P-31.1.4.2.4 / IUPAC indicated-hydrogen tautomer correction.
    # Curated retained names like "9H-purine" hard-code one tautomer's
    # indicated-H locant.  If the input molecule actually has NH at a
    # different ring N (e.g. hypoxanthine = 1H,7H form -> "1H-purin-...";
    # quinolin-4(1H)-one with [nH] at N1; xanthine derivatives with NH at
    # N1 vs N9), retag the indicated-H prefix from the actual NH atoms in
    # ``mol`` mapped through the numbering.  This is a generic fix: any
    # retained name beginning with "<digit>H-" gets re-prefixed to match
    # the input tautomer.
    #
    # Per-numbering split: when distinct numbering orientations yield
    # different retagged names (e.g. for symmetric 6-ring N-heterocycles
    # like pyrimidine where N1 vs N3 indicated-H positioning depends on
    # which orientation gives the lowest substituent locants), emit a
    # separate NamedParent for each unique (name, numbering) pair.  The
    # strategy layer then scores each candidate and selects the one with
    # the lowest principal-group + substituent locants per P-14.5 +
    # P-31.1.4.2.4.  Without this split, the retagged name was locked to
    # ``numbering_options[0]`` and a different numbering chosen later by
    # scoring would carry a stale indicated-H prefix that mis-locates the
    # NH (OPSIN re-parses it as a different lactam tautomer).
    per_numbering_named_parents: "list[tuple[str, tuple[Numbering, ...]]]" = []
    # Pre-composed retained names (``phthalhydrazide``, ``urazol``,
    # ``5-pyrazolone`` …) already encode their ring N-H / saturation positions
    # in the lexically-frozen stem (the ``-hydrazide`` / ``-zol`` / ``-one``
    # ending IS the indicated-H specification).  Prepending a fresh ``NH-`` /
    # carbocyclic-``H-`` prefix here would duplicate that information and OPSIN
    # rejects the doubled marker (e.g. ``2H,3H-phthalhydrazide`` is parsed as a
    # partly-saturated, non-aromatic ring — round-trip failure).  Skip both
    # retag passes for such names; their numbering is supplied by the curated
    # atom_locants and needs no tautomer correction.
    if numbering_options and match_name and not precomposed_no_separable_suffix:
        seen_names: set[str] = set()
        for nb in numbering_options:
            tagged_name = _retag_indicated_h(
                name=match_name,
                mol=mol,
                ring_system=ring_system,
                numbering=nb,
            )
            # Carbocyclic indicated-hydrogen (P-31.1.4.2): mancude polycyclic
            # retained parents whose sp3 ring CH/CH2 sits at a topologically
            # ambiguous position (e.g. trindene's 7H/9H tautomers) need that
            # position cited so OPSIN reconstructs the right saturation isomer.
            # Runs after the N-H retag and is a no-op for names that already
            # carry an indicated-H / hydro marker.
            tagged_name = _maybe_add_carbocyclic_indicated_h(
                name=tagged_name,
                mol=mol,
                ring_mol=ring_mol,
                ring_system=ring_system,
                numbering=nb,
            )
            if tagged_name not in seen_names:
                seen_names.add(tagged_name)
                per_numbering_named_parents.append((tagged_name, (nb,)))
        if len(per_numbering_named_parents) == 1:
            # Single retagged variant — collapse back to the original
            # multi-numbering layout so symmetry-equivalent matches still
            # let the strategy enumerate locant assignments.
            match_name = per_numbering_named_parents[0][0]
            per_numbering_named_parents = []
        else:
            # Use the first variant's name as the primary match_name; the
            # additional variants are emitted as extra NamedParents below.
            match_name = per_numbering_named_parents[0][0]
            numbering_options = per_numbering_named_parents[0][1]

    # Stage 23 R23-A: cis/trans-decalin functional-class override.
    # When the curated ``decalin`` retained name is matched against a
    # stereo-bearing input where the molecule IS exactly decalin (no
    # substituents), upgrade the bare retained name to the P-23.5.5
    # functional-class form ``cis-decalin`` / ``trans-decalin``.  The bare
    # word silently drops ring-junction stereo; OPSIN does NOT parse the
    # systematic ``(4aR,8aR)-decahydronaphthalene`` form for any letter-
    # suffix R/S, so the retained-functional-class shortcut is the only
    # round-trip-safe way to encode the cis/trans distinction.  We gate
    # narrowly: only fires when the molecule consists solely of the
    # decalin ring system AND both junction atoms carry stereo tags.
    # Match by curated-SMILES key rather than by match_name because the
    # pin-eligibility alias-swap (decalin -> decahydronaphthalene PIN) makes
    # match_name == "decahydronaphthalene" for the unstereoed case.  The
    # curated key is the load-bearing identity that pins this branch to the
    # actual decalin ring system.
    if matched_curated_key == "C1CCC2CCCCC2C1":
        cis_trans = _cis_trans_decalin_override(mol, ring_system)
        if cis_trans is not None:
            match_name = cis_trans

    # Carry through added-indicated-H atoms from the dihydro derivation when
    # the count==1 path produced an "added-IH" form (P-31.1.4.2.4 / P-58.2.2).
    _added_ih_atoms: tuple[int, ...] | None = None
    if derived_dihydro is not None:
        _added_ih_atoms = derived_dihydro.get("added_indicated_h_atoms")

    results: list[NamedParent] = [_build_named_parent(
        ring_system=ring_system,
        name_str=match_name,
        sub_form=match_sub,
        alkyl_ok=match_alkyl_ok,
        naming_method="retained",
        numbering_options=numbering_options,
        extra_atom_indices=extra_atom_indices,
        added_indicated_h_atoms=_added_ih_atoms,
        precomposed_retained_no_suffix=precomposed_no_separable_suffix,
    )]

    # Emit per-numbering retagged variants as additional candidates (see the
    # retag block above).  Skip the first entry — already emitted as the
    # primary NamedParent above.  Each variant carries exactly one numbering
    # so the strategy layer's scoring picks the indicated-H + locant set
    # combination with the lowest overall locants.
    for variant_name, variant_nbs in per_numbering_named_parents[1:]:
        results.append(_build_named_parent(
            ring_system=ring_system,
            name_str=variant_name,
            sub_form=match_sub,
            alkyl_ok=match_alkyl_ok,
            naming_method="retained",
            numbering_options=variant_nbs,
            extra_atom_indices=extra_atom_indices,
            added_indicated_h_atoms=_added_ih_atoms,
            precomposed_retained_no_suffix=precomposed_no_separable_suffix,
        ))

    # Append per-orientation NamedParents from the partial-saturation
    # enumeration (Stage 22 R22-A).  Each orientation has its own
    # rewritten hydro-prefix + Numbering.  The downstream dedup in
    # name_ring_system collapses duplicates by name string; the strategy
    # layer scores the rest and picks lowest-locant-for-principal-group.
    for orient in derived_orientations:
        orient_name = orient.get("name")
        if not orient_name:
            continue
        orient_sub = orient.get("substituent_form")
        orient_nbs = orient.get("numbering_options", ())
        orient_added_ih = orient.get("added_indicated_h_atoms")
        # Apply indicated-H tautomer correction to orientation names too,
        # mirroring the primary-name path above.
        if orient_nbs:
            orient_name = _retag_indicated_h(
                name=orient_name,
                mol=mol,
                ring_system=ring_system,
                numbering=orient_nbs[0],
            )
        results.append(_build_named_parent(
            ring_system=ring_system,
            name_str=orient_name,
            sub_form=orient_sub,
            alkyl_ok=False,
            naming_method="retained",
            numbering_options=orient_nbs,
            extra_atom_indices=extra_atom_indices,
            added_indicated_h_atoms=orient_added_ih,
            precomposed_retained_no_suffix=precomposed_no_separable_suffix,
        ))

    return results


# ---------------------------------------------------------------------------
# Stage 23 R23-A: cis/trans-decalin retained-functional-class override
# ---------------------------------------------------------------------------


def _cis_trans_decalin_override(mol, ring_system) -> str | None:
    """Return ``"cis-decalin"`` / ``"trans-decalin"`` (or None) for the
    decalin retained match.

    Returns a name only when ALL of:
      * the molecule consists solely of the decalin ring atoms
        (no exocyclic substituents — the bare-parent case),
      * exactly two ring atoms carry tetrahedral chirality, AND
      * those two atoms are the ring-junction (degree-3 within the
        ring system) carbons.

    Detection:
      * cis-decalin — both junction chiral_tags equal (same face, both
        H atoms on the same side of the ring plane)
      * trans-decalin — junction chiral_tags differ (opposite faces)

    OPSIN does not parse a systematic ``(4aR,8aR)-decahydronaphthalene``
    name (any letter-suffix R/S on this parent returns empty), so the
    P-23.5.5 functional-class names ``cis-decalin`` / ``trans-decalin``
    are the only round-trip-safe encoding of the cis/trans distinction.

    Substituted-cis-decalin variants (e.g. ``2-methyl-cis-decalin``) are
    NOT round-trip-parseable through OPSIN, so we deliberately gate on
    the bare-parent case only — substituted decalins fall through to the
    bare ``decalin`` name with stereo dropped.  This matches OPSIN's own
    coverage envelope for the functional-class form.
    """
    if mol is None or ring_system is None:
        return None
    ring_atoms = ring_system.atom_indices
    # Bare-parent gate: the molecule must consist solely of the ring atoms.
    # An exocyclic substituent (or any atom outside the ring system) would
    # produce e.g. "2-methyl-cis-decalin", which OPSIN does not parse.
    if mol.GetNumAtoms() != len(ring_atoms):
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    # Canonicalize first so chiral_tag CW/CCW comparisons are stable across
    # input SMILES variants (different atom orderings can flip the per-atom
    # tag while preserving parity; the equality-of-tags test below is
    # parity-invariant only on a single canonical neighbour ordering).
    try:
        canon_smiles = Chem.MolToSmiles(mol)
        canon_mol = Chem.MolFromSmiles(canon_smiles)
    except Exception:
        return None
    if canon_mol is None or canon_mol.GetNumAtoms() != mol.GetNumAtoms():
        return None
    chi_unspec = Chem.ChiralType.CHI_UNSPECIFIED
    chiral_atoms: list[int] = []
    for atom in canon_mol.GetAtoms():
        if atom.GetChiralTag() != chi_unspec:
            chiral_atoms.append(atom.GetIdx())
    # Decalin has exactly two stereogenic centres (the two junctions).
    # Anything else (0, 1, or >2) is not a clean cis/trans-decalin.
    if len(chiral_atoms) != 2:
        return None
    # Both chiral atoms must be the ring-junction carbons (degree 3 within
    # the molecule — decalin is bare here, so the molecular graph is the
    # ring system itself).  In decalin the only degree-3 atoms ARE the
    # junctions, so this also guards against accidentally matching a
    # stereo-bearing non-junction carbon.
    for idx in chiral_atoms:
        if canon_mol.GetAtomWithIdx(idx).GetDegree() != 3:
            return None
    tag_a = canon_mol.GetAtomWithIdx(chiral_atoms[0]).GetChiralTag()
    tag_b = canon_mol.GetAtomWithIdx(chiral_atoms[1]).GetChiralTag()
    return "cis-decalin" if tag_a == tag_b else "trans-decalin"


# ---------------------------------------------------------------------------
# Indicated-H tautomer correction
# ---------------------------------------------------------------------------

import re as _re

_INDICATED_H_PREFIX_RE = _re.compile(
    r"^(?:(\d+[a-z]?H)(?:,(\d+[a-z]?H))*-)"
)


def _retag_indicated_h(
    *,
    name: str,
    mol,
    ring_system: "RingSystem",
    numbering: "Numbering",
) -> str:
    """Replace the leading "NH-" indicated-hydrogen prefix in ``name`` with
    the locants of ring atoms that actually carry an NH in ``mol``.

    Generic mechanism for retained names whose tautomer-bearing form is
    keyed under a single canonical (e.g. "9H-purine") but whose input
    tautomer has H on a different ring N.  Examples:
      - hypoxanthine input ``O=c1[nH]cnc2[nH]cnc12`` is matched as a
        purine derivative with NH on N1 and N7 -> "1H,7H-purin-..."
        rather than the hard-coded "9H-".
      - 4-quinolone input ``O=c1cc[nH]c2ccccc12`` is matched as quinoline
        with NH on N1 -> emit "1H-quinolin-4-one" instead of bare
        "quinolin-4-one" (which OPSIN parses as a different tautomer).

    Only acts when the existing name starts with one or more "NH-" tokens.
    For names without such a prefix, only ADD a prefix when the input
    requires one to disambiguate from a non-NH tautomer (currently:
    quinoline / isoquinoline / quinazoline-class with ring NH).

    Returns the (possibly retagged) name.
    """
    try:
        atom_to_locant = numbering.atom_to_locant
    except Exception:
        return name

    # Find ring atoms that actually have an explicit hydrogen on a ring N
    # (aromatic [nH] or sp3 NH that is part of the ring system).  Only
    # consider atoms that belong to this ring_system AND appear in the
    # numbering (so we have a locant for them).
    nh_locants: list[Locant] = []
    for atom_idx in sorted(ring_system.atom_indices):
        if atom_idx not in atom_to_locant:
            continue
        try:
            atom = mol.GetAtomWithIdx(atom_idx)
        except Exception:
            continue
        if atom.GetAtomicNum() != 7:
            continue
        # Indicated-H disambiguation only applies to aromatic tautomers
        # (e.g. 1H/3H/7H/9H-purine, 1H/3H-pyrrole).  A saturated ring N-H
        # in retained rings like isoindoline, pyrrolidine, piperidine is
        # mandatory by saturation, not a tautomer choice — prepending
        # "NH-" there is spurious.  RDKit marks ring aromatic atoms with
        # GetIsAromatic(); skip non-aromatic N here.
        if not atom.GetIsAromatic():
            continue
        # Treat aromatic [n-] as an indicated-H position: the anion occupies
        # the same tautomer slot as [nH] in the neutral scaffold (ring-anion
        # nomenclature emits the charge via a separate -N-ide suffix, see
        # ring_anion_locants; the indicated-H marker still needs to point
        # at this N so OPSIN can round-trip the neutralized parent name).
        is_ring_anion = (
            atom.GetFormalCharge() == -1
            and atom.GetTotalNumHs(includeNeighbors=False) == 0
        )
        if not is_ring_anion:
            if atom.GetTotalNumHs(includeNeighbors=False) < 1:
                continue
            # Skip other charged N atoms: an H on [nH+] is a protonation
            # (already encoded by the "-ium" suffix), not a tautomer
            # indicated-H.
            if atom.GetFormalCharge() != 0:
                continue
        nh_locants.append(atom_to_locant[atom_idx])

    # Sort numerically by locant value, lowest first, to get IUPAC-style
    # "1H,7H-" rather than "7H,1H-".  Defined here so both the no-NH branch
    # below and the with-NH branch further down can use it.
    def _loc_key(loc):
        s = str(loc)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num = int(s[:i]) if s[:i] else 0
        return (num, s[i:])

    if not nh_locants:
        # No ring N carries an explicit H.  Two sub-cases:
        #
        # (a) The indicated-H locant atom HAS an exocyclic substituent (e.g.
        #     N9 of "9H-purine" used as parent for a 9-alkyl purine, where
        #     the [nH] was replaced by the alkyl).  The original "9H-" prefix
        #     is still semantically correct: it marks the tautomer slot that
        #     became N-substituted.  Leave the name alone.
        #
        # (b) The indicated-H locant atom has NO H AND NO exocyclic
        #     substituent — it is a bare pyridine-type =N-.  An indicated-H
        #     prefix at such an atom is a contradiction: OPSIN tries to place
        #     an H there, dearomatizing the ring (e.g. "9H-purin-8-yl" with
        #     N9 sp2 and N7 substituted is parsed by OPSIN as a 5-ring with
        #     C8 sp3, breaking the imidazole double bond).  Retag the
        #     indicated-H to a substituted ring N instead so OPSIN can place
        #     the substituent at that locant and leave the bare =N- at its
        #     correct sp2 position.
        m_ih = _INDICATED_H_PREFIX_RE.match(name)
        if not m_ih:
            return name
        try:
            ih_locant_str = m_ih.group(1)[:-1]  # strip trailing "H"
            # Parse leading number
            ih_num_str = ""
            for ch in ih_locant_str:
                if ch.isdigit():
                    ih_num_str += ch
                else:
                    break
            ih_locant_letter = ih_locant_str[len(ih_num_str):]
            ih_locant_num = int(ih_num_str)
        except (ValueError, IndexError):
            return name
        # Build target Locant matching the indicated-H prefix.
        # Locant.numeric takes a positional `suffix` arg for ring junction
        # letters ("4a", etc.); pass the optional letter (e.g. "a") if the
        # indicated-H prefix is on a junction atom like "8aH".
        ih_target = Locant.numeric(ih_locant_num, ih_locant_letter)
        # Find the ring atom at that locant (in the full mol).  Compare by
        # locant label string to avoid object-identity issues.
        ih_atom_idx: int | None = None
        ih_target_label = str(ih_target)
        for atom_idx, loc in atom_to_locant.items():
            if str(loc) == ih_target_label and atom_idx in ring_system.atom_indices:
                ih_atom_idx = atom_idx
                break
        if ih_atom_idx is None:
            return name
        try:
            ih_atom = mol.GetAtomWithIdx(ih_atom_idx)
        except Exception:
            return name
        # If the indicated-H atom is not aromatic N, leave the name alone
        # (this branch is only meaningful for aromatic-tautomer indicated-H).
        if ih_atom.GetAtomicNum() != 7 or not ih_atom.GetIsAromatic():
            return name
        # Has exocyclic substituent? -> case (a), keep the name.
        has_exo_sub = any(
            nbr.GetIdx() not in ring_system.atom_indices
            for nbr in ih_atom.GetNeighbors()
        )
        if has_exo_sub:
            return name
        # Case (b): bare pyridine-type =N- at the indicated-H slot.
        # Find a substituted ring N to retag the indicated-H to.  Prefer the
        # lowest IUPAC locant to give a deterministic, IUPAC-friendly result.
        sub_n_locants: list[Locant] = []
        for atom_idx in sorted(ring_system.atom_indices):
            if atom_idx not in atom_to_locant:
                continue
            try:
                a = mol.GetAtomWithIdx(atom_idx)
            except Exception:
                continue
            if a.GetAtomicNum() != 7 or not a.GetIsAromatic():
                continue
            if a.GetTotalNumHs(includeNeighbors=False) >= 1:
                continue  # would have been picked up above
            if a.GetFormalCharge() != 0:
                continue
            # Must have an exocyclic substituent (so OPSIN can place a sub at
            # this locant, satisfying the indicated-H slot's tautomer role).
            has_exo = any(
                nbr.GetIdx() not in ring_system.atom_indices
                for nbr in a.GetNeighbors()
            )
            if not has_exo:
                continue
            sub_n_locants.append(atom_to_locant[atom_idx])
        if not sub_n_locants:
            return name
        sub_n_sorted = sorted(set(sub_n_locants), key=_loc_key)
        new_ih_prefix = f"{sub_n_sorted[0]}H-"
        return new_ih_prefix + name[m_ih.end():]

    nh_locants_sorted = sorted(set(nh_locants), key=_loc_key)
    new_prefix = ",".join(f"{loc}H" for loc in nh_locants_sorted) + "-"

    # If the existing name has a leading "NH-" (or "NH,MH-") prefix, replace it.
    m = _INDICATED_H_PREFIX_RE.match(name)
    if m:
        # Guard: when the curated name carries BOTH a leading "NH-" prefix AND
        # an inline "(NH)" marker (e.g. ``1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one``,
        # ``1H-thieno[2,3-e][1,4]diazepin-2(3H)-one``), the leading prefix and
        # the inline marker EACH pin a different ring-N tautomer slot.  In a
        # mobile-H lactam tautomer system, RDKit's canonical perception may
        # place [nH] at one slot, two slots, or neither (when one or both NHs
        # are substituted), and InChI considers all such perceptions
        # equivalent.  Naively replacing the leading "NH-" with the input's
        # explicit-H locants breaks the curated IUPAC PIN tautomer when:
        #   - the input has [nH] at a NEW locant (other than the two pinned
        #     slots) — retag emits e.g. "4H-pyrazolo[4,3-d]pyrimidin-7(6H)-one"
        #     which OPSIN rejects (inline "(6H)" still claims N6-H but the
        #     leading "4H-" claims N4-H, contradicting the lactam topology),
        #   - one of the curated-name slots is satisfied by an N-alkyl
        #     substituent (sildenafil's N1-methyl, vardenafil's N3-substituent)
        #     and the OTHER slot has an explicit [nH] in the input — retag
        #     would rewrite the leading prefix to point at the explicit-H
        #     locant, dropping the structural N-substituent slot from the
        #     name.
        # Preserve the curated name in this case: the retained-name's
        # combined "NH-...(MH)-" form is the IUPAC PIN for the entire
        # tautomer class.
        if "(" in name and _re.search(r"\(\d+[a-z]?H\)", name[m.end():]):
            return name
        return new_prefix + name[m.end():]

    # Otherwise, only prepend a fresh "NH-" prefix when the parent name has
    # no other indicated-hydrogen marker that already accounts for the ring
    # NH.  Skip prepending when:
    #   - the name embeds an indicated H inline (e.g. "(1H)-", "(2H)-"),
    #   - the name contains a dihydro/tetrahydro/hexahydro saturation prefix
    #     that already encodes added H positions,
    #   - the name contains a digit-H pattern anywhere (e.g. "2H-indol-2-one"
    #     stem already used by the assembly pipeline).
    # These cases would yield "1H-3,4-dihydroquinolin-2(1H)-one" — duplicate
    # / contradictory indicated-H markers that OPSIN rejects.
    if "(" in name and _re.search(r"\(\d+[a-z]?H\)", name):
        return name
    # Saturation-multiplier prefixes in retained-ring names follow the
    # Greek numerical series: di-, tetra-, hexa-, octa-, deca-, dodeca-,
    # tetradeca-, hexadeca-, octadeca-, icosa-/eicosa-.  A locant-list
    # already enumerates every added-H position, so prepending a fresh
    # "NH-" here would duplicate that information and break OPSIN parsing.
    if _re.search(r"\b(?:di|tetra|hexa|octa|nona|deca|dodeca|tetradeca|hexadeca|octadeca|icosa|eicosa)hydro", name):
        return name
    if _re.search(r"\b\d+[a-z]?H[-,]", name):
        return name

    return new_prefix + name


def _carbocyclic_ih_orbit_ambiguous(ring_mol, sp3_ring_mol_idx: int) -> bool:
    """Return True iff the carbocyclic indicated-hydrogen position is NOT
    topologically unique within ``ring_mol`` — i.e. the ring system's graph
    automorphism group maps the sp3 ring carbon at ``sp3_ring_mol_idx`` onto
    at least one other atom.

    When the indicated-H position is ambiguous, the IUPAC name MUST cite it
    with a ``<locant>H-`` prefix (e.g. ``7H-trindene``) so OPSIN reconstructs
    the correct saturation tautomer; omitting it lets OPSIN default to a
    different sp3 ring (round-trip failure).  When the position is unique
    (orbit size 1, e.g. fluorene's lone bridge CH2), the prefix is optional
    and we leave it off to match the bare retained spelling already in wide
    use.

    The orbit is computed on the BOND-GENERIC skeleton (all ring bonds treated
    as plain single bonds) so the test reflects pure ring topology, not the
    arbitrary kekulé placement of the mancude double bonds.  This is the
    correct notion: indicated-H ambiguity is a property of the ring graph
    (how many symmetry-distinct positions the saturated atom could occupy),
    independent of which kekulé the SMILES happens to encode.
    """
    try:
        rw = Chem.RWMol()
        idxmap: dict[int, int] = {}
        for a in ring_mol.GetAtoms():
            na = Chem.Atom(a.GetAtomicNum())
            na.SetNoImplicit(True)
            idxmap[a.GetIdx()] = rw.AddAtom(na)
        for b in ring_mol.GetBonds():
            rw.AddBond(
                idxmap[b.GetBeginAtomIdx()],
                idxmap[b.GetEndAtomIdx()],
                Chem.BondType.SINGLE,
            )
        g = rw.GetMol()
        Chem.SanitizeMol(
            g,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
        ranks = list(Chem.CanonicalRankAtoms(g, breakTies=False))
    except Exception:
        # On any failure, conservatively report "unique" (no prefix added) so
        # we never regress a ring that currently round-trips without one.
        return False
    target = idxmap.get(sp3_ring_mol_idx)
    if target is None or target >= len(ranks):
        return False
    r = ranks[target]
    orbit = sum(1 for rk in ranks if rk == r)
    return orbit > 1


def _maybe_add_carbocyclic_indicated_h(
    *,
    name: str,
    mol,
    ring_mol,
    ring_system: "RingSystem",
    numbering: "Numbering",
) -> str:
    """Prepend a CARBOCYCLIC indicated-hydrogen prefix (e.g. ``7H-``) to a
    retained mancude-ring name when the saturated ring carbon's position is
    topologically ambiguous and the name does not already carry one.

    Mancude polycyclic retained names with an isolated sp3 ring CH/CH2 (the
    "indicated hydrogen", P-31.1.4.2) require that position to be cited when
    the ring graph admits more than one symmetry-distinct location for it —
    otherwise OPSIN reconstructs a different saturation tautomer.  Trindene is
    the canonical case: its 7H / 9H tautomers are distinct molecules, so
    ``trindene`` alone is under-specified and ``9-chlorotrindene`` round-trips
    to the wrong sp3 ring.  Fluorene (one possible bridge CH2) is unambiguous
    and keeps its bare spelling.

    This is the carbocyclic analogue of ``_retag_indicated_h`` (which handles
    only ring N-H tautomers).  It is GATED to fire only when:
      * the name has no leading ``<n>H-`` prefix and no inline ``(<n>H)`` /
        hydro saturation marker (those already encode the added H),
      * exactly the sp3 ring CH/CH2 carbons that have a locant are considered,
      * and each such position is topologically ambiguous (orbit > 1 in the
        bond-generic ring graph).
    When the gate does not fire the name is returned unchanged, so existing
    round-tripping rings (fluorene, indene via kekulé rewrite, etc.) are
    untouched.
    """
    try:
        atom_to_locant = numbering.atom_to_locant
    except Exception:
        return name
    if ring_mol is None:
        return name
    # Skip when an indicated-H / saturation marker already accounts for the H.
    if _INDICATED_H_PREFIX_RE.match(name):
        return name
    if "(" in name and _re.search(r"\(\d+[a-z]?H\)", name):
        return name
    if _re.search(
        r"\b(?:di|tetra|hexa|octa|nona|deca|dodeca|tetradeca|hexadeca|"
        r"octadeca|icosa|eicosa)hydro",
        name,
    ):
        return name
    if _re.search(r"\b\d+[a-z]?H[-,]", name):
        return name

    # Skip when any ring carbon bears an exocyclic double bond to a non-carbon
    # heteroatom (O/S/Se/Te/N) — i.e. a cyclic ketone/thione/selone/tellone/
    # imine.  Such a ring expresses its saturated ("added hydrogen") position
    # through the -one/-thione/...-(nH) SUFFIX, so prepending a STANDALONE
    # carbocyclic indicated-H here double-marks the same hydrogen: e.g.
    # naphthalen-1(2H)-one would become the spurious, OPSIN-rejected
    # "2H-naphthalen-1(2H)-one".  Let the suffix's added-hydrogen carry it.
    for _ai in ring_system.atom_indices:
        try:
            _a = mol.GetAtomWithIdx(_ai)
        except Exception:
            continue
        if _a.GetAtomicNum() != 6:
            continue
        for _b in _a.GetBonds():
            if _b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            _o = _b.GetOtherAtom(_a)
            if (_o.GetIdx() not in ring_system.atom_indices
                    and _o.GetAtomicNum() in (7, 8, 16, 34, 52)):
                return name

    # MANCUDE GATE.  Indicated hydrogen (P-31.1.4.2) is a property of *mancude*
    # ring systems — rings carrying the maximum number of non-cumulative double
    # bonds, where the lone sp3 CH/CH2 is the "extra" saturated position.  It
    # does NOT apply to fully-saturated von Baeyer polycycles (cubane,
    # adamantane, tricyclo[2.2.1]heptane, …) where EVERY ring carbon is sp3 by
    # construction; prepending "2H,3H,4H,…" there is spurious and OPSIN rejects
    # it.  Two structural conditions distinguish the two:
    #   (1) the ring system must contain at least one aromatic ring atom (a
    #       genuine mancude ring always retains an aromatic/sp2 core; saturated
    #       cages have none), AND
    #   (2) the sp3 indicated-H carbons must be a small minority of the ring
    #       (a mancude ring has at most a couple of indicated-H positions; a
    #       cage is all-sp3).
    ring_atom_objs = []
    for ai in ring_system.atom_indices:
        try:
            ring_atom_objs.append(mol.GetAtomWithIdx(ai))
        except Exception:
            return name
    n_ring = len(ring_atom_objs)
    if n_ring == 0:
        return name
    n_aromatic = sum(1 for a in ring_atom_objs if a.GetIsAromatic())
    if n_aromatic == 0:
        # No aromatic core -> not a mancude ring (saturated cage or alicyclic).
        return name
    n_sp3_ch = sum(
        1 for a in ring_atom_objs
        if a.GetAtomicNum() == 6
        and not a.GetIsAromatic()
        and a.GetHybridization() == Chem.HybridizationType.SP3
        and a.GetTotalNumHs(includeNeighbors=False) >= 1
    )
    # A mancude ring's indicated hydrogen is the isolated saturated position;
    # if a large fraction of the ring is sp3-CH the system is a hydro-derivative
    # (handled by the dihydro/hydro derivation path) or a saturated cage, not a
    # mancude parent — leave its name to those paths.
    if n_sp3_ch * 3 > n_ring:
        return name

    # Map full-mol ring atom indices to ring_mol indices (for the orbit test).
    try:
        ring_canon = Chem.MolToSmiles(ring_mol)
        ring_query = Chem.MolFromSmarts(ring_canon)
    except Exception:
        ring_query = None
    full_to_ringmol: dict[int, int] = {}
    if ring_query is not None:
        rs_set = set(ring_system.atom_indices)
        for match in mol.GetSubstructMatches(ring_query, uniquify=False):
            if all(a in rs_set for a in match):
                # match[ring_mol_idx] = full_mol_idx
                full_to_ringmol = {full: rmi for rmi, full in enumerate(match)}
                break

    ih_locants: list[Locant] = []
    for atom_idx in sorted(ring_system.atom_indices):
        if atom_idx not in atom_to_locant:
            continue
        try:
            a = mol.GetAtomWithIdx(atom_idx)
        except Exception:
            continue
        # Carbocyclic indicated-H position: an sp3 ring CARBON bearing >=1 H
        # that is NOT aromatic and carries no exocyclic substituent that would
        # itself force the saturation (a substituent makes the position fixed
        # by the substituent, not a free indicated-H choice).
        if a.GetAtomicNum() != 6:
            continue
        if a.GetIsAromatic():
            continue
        if a.GetTotalNumHs(includeNeighbors=False) < 1:
            continue
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        # Must be a genuine ring-internal sp3 (degree-2 within the ring for a
        # CH2, or part of the ring skeleton) — exclude exocyclic CH groups.
        ring_neighbors = [
            n for n in a.GetNeighbors()
            if n.GetIdx() in ring_system.atom_indices
        ]
        if len(ring_neighbors) < 2:
            continue
        rmi = full_to_ringmol.get(atom_idx)
        if rmi is None:
            continue
        if _carbocyclic_ih_orbit_ambiguous(ring_mol, rmi):
            ih_locants.append(atom_to_locant[atom_idx])

    if not ih_locants:
        return name

    def _loc_key(loc):
        s = str(loc)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        return (int(s[:i]) if s[:i] else 0, s[i:])

    ordered = sorted(set(ih_locants), key=_loc_key)
    # Genuine carbocyclic indicated hydrogen is a SINGLE sp3 position in an
    # otherwise-mancude system (trindene 7H, fluorene 9H, phenalene 1H).  Two or
    # more sp3 ring CH2 means the system is a RETAINED REDUCED form whose
    # saturation is already encoded by the name itself — acenaphthene is
    # 1,2-dihydroacenaphthylene; isoindoline is 2,3-dihydro-1H-isoindole — so
    # prepending "1H,2H-"/"1H,3H-" double-marks the saturation and OPSIN rejects
    # the result (regression observed in the reverse-coverage audit).  Restrict
    # firing to exactly one indicated-hydrogen position.
    if len(ordered) != 1:
        return name
    prefix = ",".join(f"{loc}H" for loc in ordered) + "-"
    return prefix + name


def _lookup_curated(smiles: str) -> tuple[str | None, str | None, bool, dict | None]:
    """Look up a SMILES in the curated table, returning (name, sub_form, alkyl_ok, atom_locants)."""
    # Direct lookup
    entry = _CURATED.get(smiles)
    if entry:
        return entry

    # Alias lookup (resolve alias, then look up)
    resolved = _CURATED_ALIASES.get(smiles)
    if resolved:
        entry = _CURATED.get(resolved)
        if entry:
            return entry

    return None, None, False, None


def _lookup_curated_key(smiles: str) -> str | None:
    """Return the canonical curated key that matches *smiles* (possibly via alias).

    atom_locants in the curated table are indexed against the atoms of the
    RDKit mol built from the *key* SMILES.  When a ring is carved from a
    stereo-bearing parent, the carved ring_mol has a different atom ordering
    than the curated key (canonical atom indices are permuted).  Callers use
    this key to rebuild ring_mol with atom indices that align with
    atom_locants.
    """
    if smiles in _CURATED:
        return smiles
    resolved = _CURATED_ALIASES.get(smiles)
    if resolved and resolved in _CURATED:
        return resolved
    return None


# ---------------------------------------------------------------------------
# Oxo-to-mancude derivation from a retained aromatic parent
# ---------------------------------------------------------------------------

def _try_derive_oxo_aromatic_retained(
    *,
    ring_system: "RingSystem",
    mol,
) -> "dict | None":
    """Derive a fully-aromatic retained parent for a ring system whose only
    departure from the mancude form is one or more ring-carbon exocyclic =O
    (or =S/=Se/=NR) doublets, i.e. a fused di/poly-one such as
    naphthalene-1,2-dione.

    Procedure (P-31.1.4.3.4 lowest-locant principal characteristic group;
    P-66.6 ketone/thione/imine suffix on a mancude ring):

      1. Locate every ring carbon that bears exactly one exocyclic double bond
         to O/S/Se/N (the "oxo" carbons).  Bail if none.
      2. Build an aromatized copy of the ring system with those exocyclic
         doublets removed and every ring atom/bond flagged aromatic.  Bail if
         the result will not sanitize (i.e. the skeleton is not a genuine
         mancude ring).
      3. Require that, IGNORING the oxo carbons, every remaining ring atom is
         already aromatic in the input.  This keeps the derivation tight: a
         ring that also has saturated CH2 positions is a ``dihydro…dione`` case
         handled by the curated dihydro skeleton + suffix path, not here.
      4. Look up the aromatized canonical in the curated table.  It must carry
         ``atom_locants`` for a deterministic numbering.
      5. Build the full-mol Numbering from the curated key's atom_locants via
         the shared ``_build_numbering_from_atom_locants`` helper, leaving the
         exocyclic =O for the downstream characteristic-group layer.

    Returns a dict ``{"name", "substituent_form", "numbering_options"}`` (same
    shape as ``_try_derive_hydro_retained``), or ``None`` when inapplicable.
    """
    try:
        ring_atoms_full = frozenset(ring_system.atom_indices)
    except Exception:
        return None
    if not ring_atoms_full:
        return None

    # P-66.6 suffix-eligible exocyclic doublets: =O (one), =S (thione),
    # =Se (selone), =Te (tellone), =NR (imine).  These are the ring-carbon
    # characteristic groups that can be lifted off to recover the mancude parent.
    _SUFFIX_DB_ATOMS = {7, 8, 16, 34, 52}  # N, O, S, Se, Te
    oxo_carbons: set[int] = set()
    for atom_idx in ring_atoms_full:
        a = mol.GetAtomWithIdx(atom_idx)
        if a.GetAtomicNum() != 6:
            continue
        if a.GetFormalCharge() != 0:
            return None  # charged ring carbons are out of scope here
        exo_db = 0
        for b in a.GetBonds():
            other = b.GetOtherAtom(a)
            if other.GetIdx() in ring_atoms_full:
                continue
            if b.GetBondType() == Chem.BondType.DOUBLE and \
                    other.GetAtomicNum() in _SUFFIX_DB_ATOMS:
                exo_db += 1
        if exo_db >= 1:
            oxo_carbons.add(atom_idx)

    if not oxo_carbons:
        return None

    # Tightness gate: every non-oxo ring atom must be mancude-compatible — i.e.
    # either already aromatic, or engaged in an (endocyclic) double bond so it
    # is an sp2 member of the maximally-unsaturated skeleton (e.g. the isolated
    # ``C=C`` between positions 3 and 4 of naphthalene-1,2-dione).  A genuinely
    # saturated ring atom (no double bond at all — a CH2) would require a
    # ``dihydro`` prefix and is the province of the curated dihydro skeleton +
    # suffix path, NOT this oxo-on-mancude derivation.  Intrinsically-divalent
    # ring chalcogens (O/S/Se/Te at valence 2) are divalent in both the mancude
    # and saturated forms, so they are mancude-compatible by definition.
    _CHALCOGEN_DIVALENT = (8, 16, 34, 52)
    for atom_idx in ring_atoms_full:
        if atom_idx in oxo_carbons:
            continue
        a = mol.GetAtomWithIdx(atom_idx)
        if a.GetIsAromatic():
            continue
        if a.GetAtomicNum() in _CHALCOGEN_DIVALENT and a.GetTotalValence() == 2:
            continue
        if any(b.GetBondTypeAsDouble() >= 2.0 for b in a.GetBonds()):
            continue  # sp2/sp mancude member
        return None  # genuine saturated (hydro) position — out of scope here

    # Build an aromatized ring-only mol: copy the parent, kekulize, delete all
    # non-ring atoms (including the exocyclic chalcogen/imine) and force every
    # remaining ring atom/bond aromatic, then sanitize.  Preserve ring-atom
    # ordering so we can map back to full-mol indices.
    sorted_ring = sorted(ring_atoms_full)
    ring_set = set(sorted_ring)
    rw = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        return None
    n_total = rw.GetNumAtoms()
    new_to_old: list[int] = [i for i in range(n_total) if i in ring_set]
    for old in range(n_total - 1, -1, -1):
        if old not in ring_set:
            rw.RemoveAtom(old)
    # new_to_old[new_idx] = old(full-mol) idx, in ascending old order.
    for a in rw.GetAtoms():
        a.SetIsAromatic(True)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.AROMATIC)
        b.SetIsAromatic(True)
        b.SetStereo(Chem.BondStereo.STEREONONE)
    aromatized = rw.GetMol()
    arom_smi: str | None = None
    try:
        Chem.SanitizeMol(aromatized)
        arom_smi = Chem.MolToSmiles(aromatized)
    except Exception:
        # Over-NH mancude skeletons (e.g. the purine core of a 2,6-dione:
        # stripping the two C=O leaves THREE ring N-H, one more than a neutral
        # aromatic purine can carry, so full kekulization fails).  These rings
        # still have a well-defined mancude parent whose retained name is keyed
        # under the bare-aromatic skeleton SMILES (the same canonical that
        # _normalize_nh_fragment Strategy 4 produces, e.g. ``c1ncc2ncnc2n1`` =
        # 9H-purine).  Retry with partial sanitization (skip kekulization) to
        # recover that skeleton; the curated lookup carries a dedicated entry
        # for it.  The added indicated hydrogens implied by the dione are
        # supplied downstream by _retag_indicated_h, yielding the
        # ``N,M-dioxo-...H-<parent>`` PIN (P-31.1.4.2.4 / P-66.6).
        try:
            partial = Chem.RWMol(aromatized)
            Chem.SanitizeMol(
                partial,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            arom_smi = Chem.MolToSmiles(partial.GetMol())
        except Exception:
            return None
    if not arom_smi:
        return None

    # Look up the aromatized canonical in the curated retained table.
    name, sub_form, _alkyl_ok, atom_locants = _lookup_curated(arom_smi)
    if name is None or atom_locants is None:
        return None
    curated_key = _lookup_curated_key(arom_smi)
    if curated_key is None:
        return None

    # Build a synthetic RingSystem standing in for the aromatic parent so the
    # shared numbering helper can substructure-match the curated key into the
    # FULL molecule and assign locants.  We reuse the real ring_system's
    # atom_indices (identical atom set) and pass the curated-key ring_mol so
    # atom_locants line up with the key's atom indices.
    key_mol = Chem.MolFromSmiles(curated_key)
    if key_mol is None:
        # Bare-aromatic over-NH skeleton keys (e.g. the 9H-purine key
        # ``c1ncc2ncnc2n1``) do not fully kekulize on their own, so the normal
        # parse returns None.  Re-parse with partial sanitization (skip
        # kekulization) — the resulting mol still substructure-matches the
        # full molecule (bond-generic) and carries the same atom indices the
        # curated atom_locants are keyed against.
        try:
            key_mol = Chem.MolFromSmiles(curated_key, sanitize=False)
            if key_mol is not None:
                Chem.SanitizeMol(
                    key_mol,
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
        except Exception:
            key_mol = None
    if key_mol is None or key_mol.GetNumAtoms() != len(sorted_ring):
        return None

    numbering_options = _build_numbering_from_atom_locants(
        ring_mol=key_mol,
        mol=mol,
        ring_system=ring_system,
        atom_locants=atom_locants,
    )
    if not numbering_options:
        return None

    return {
        "name": name,
        "substituent_form": sub_form,
        "numbering_options": numbering_options,
    }


# ---------------------------------------------------------------------------
# Dihydro / tetrahydro derivation from a retained aromatic parent
# ---------------------------------------------------------------------------

def _try_derive_hydro_retained(
    *,
    ring_system: "RingSystem",
    ring_mol,
    mol,
    return_all_orientations: bool = False,
    include_oxo_carbons_as_saturation: bool = False,
) -> "dict | list[dict] | None":
    """Derive an ``<locants>-<multiplier>hydro-<retained_name>`` name when the
    input ring system is a partly-hydrogenated form of an aromatic retained
    parent.

    Procedure:
        1. Build an "aromatized" copy of ``ring_mol`` — every ring atom is
           flagged aromatic and every ring bond is set to aromatic.
        2. Canonicalize the aromatized form and look it up in ``_CURATED``.
           If absent, bail.
        3. If the matched retained entry carries ``atom_locants`` (required
           for a deterministic locant assignment), build a relaxed-bond
           SMARTS query from the curated key and substructure-match it into
           ``mol`` to map curated-key atom indices to full-mol atom indices.
        4. Collect the sp3 (non-aromatic, with one or more implicit Hs) ring
           atoms in the input ``mol``; these are the hydrogenation positions.
           Translate their full-mol indices to IUPAC locants via the
           atom_locants mapping (picking the match orientation that yields
           the lowest-locant sp3 set).
        5. Build the hydro-prefix and the synthesised name; build a Numbering
           from the chosen match.

    When ``return_all_orientations`` is True, returns a *list* of dicts —
    one per substruct match orientation — instead of just the lowest-sp3
    one.  This lets the caller offer multiple NamedParent candidates so the
    strategy layer can pick the orientation that minimizes principal
    characteristic group locants (P-31.1.4.3.4) rather than always
    minimizing the hydro-prefix locants.

    Returns a dict with keys ``name``, ``substituent_form``, and
    ``numbering_options`` (a tuple of Numbering); a list of such dicts when
    ``return_all_orientations`` is True; or ``None`` when the derivation
    doesn't apply.

    When ``include_oxo_carbons_as_saturation`` is True (the oxo+dihydro
    re-orientation path, P-31.1.4.1.1 / P-31.1.4.3.4), ring carbons bearing
    exactly one exocyclic suffix-eligible double bond (=O / =S / =Se / =Te /
    =NR) are ALSO counted as saturation (added-H) positions.  Lifting that
    double bond off as an ``oxo`` (etc.) prefix turns the carbon into a
    saturated ring member, so it contributes to the dihydro count.  The
    exocyclic =O is left UNCLAIMED (not added to ``extra_atom_indices``) so the
    downstream characteristic-group layer renders it as the ``oxo-`` prefix.
    In this mode the parent atom_locants are sourced from
    ``_OXO_DIHYDRO_MANCUDE_PARENT_LOCANTS`` first (so adding them does not
    perturb the ordinary direct-curated-match numbering), falling back to the
    curated table.  This mode is invoked ONLY from section 5b for curated
    partial-saturation entries, so it cannot affect the count==1 added-IH
    path used by mancude ketones such as ``naphthalen-1(2H)-one``.
    """
    from iupac_namer.data_loader import get_multiplier

    # Only carbocycle-style retained aromatic bases are considered here; the
    # algorithm generalises to hetero parents but the first stable target is
    # naphthalene (and relatives).  Rings with charged or radical atoms are
    # out of scope for this derivation.
    try:
        ring_atoms_full = frozenset(ring_system.atom_indices)
    except Exception:
        return None
    if not ring_atoms_full:
        return None

    # Identify hydrogenation (added-H) positions: ring atoms saturated relative
    # to the aromatic skeleton.  A position qualifies when it is non-aromatic
    # AND not engaged in any double/triple bond (endocyclic or exocyclic).
    #
    # The criterion is bond-order membership, NOT RDKit hybridisation: a
    # conjugated ring N-H — e.g. the N of 1,2-dihydropyridine (``C1=CC=CNC1``)
    # — is perceived SP2 by RDKit even though it carries an "added" hydrogen
    # relative to pyridine, so a hybridisation test silently drops it and the
    # ring collapses to the fully-saturated HW name ("azinane").  Excluding
    # only atoms that sit in a real double/triple bond keeps the genuine sp2
    # ring members out (e.g. the ``C=C`` of 1,4-dihydronaphthalene) while
    # admitting saturated CH2 carbons and conjugated saturated heteroatoms.
    # Intrinsically-divalent chalcogens (O/S/Se/Te at valence 2) are divalent in
    # BOTH the mancude and the saturated ring, so they are never added-H
    # positions — counting them inflates the hydro multiplier (e.g. turning
    # 2,5-dihydrothiazole's 2 positions into 3) and breaks the locant mapping.
    _CHALCOGEN_DIVALENT = (8, 16, 34, 52)  # O, S, Se, Te
    # Suffix-eligible exocyclic doublets (P-66.6): =O (one), =S (thione),
    # =Se (selone), =Te (tellone), =NR (imine).  When
    # ``include_oxo_carbons_as_saturation`` is set, a ring carbon bearing
    # exactly such a doublet is treated as a saturation position (the doublet
    # lifts off as an oxo-class prefix, leaving a saturated ring carbon).
    _EXO_SUFFIX_DB_ATOMS = {7, 8, 16, 34, 52}  # N, O, S, Se, Te

    def _exocyclic_oxo_count(atom) -> int:
        n = 0
        for b in atom.GetBonds():
            other = b.GetOtherAtom(atom)
            if other.GetIdx() in ring_atoms_full:
                continue
            if (b.GetBondType() == Chem.BondType.DOUBLE
                    and other.GetAtomicNum() in _EXO_SUFFIX_DB_ATOMS):
                n += 1
        return n

    sp3_full: set[int] = set()
    oxo_saturation_full: set[int] = set()
    for atom_idx in ring_atoms_full:
        try:
            a = mol.GetAtomWithIdx(atom_idx)
        except Exception:
            return None
        if a.GetIsAromatic():
            continue
        if a.GetFormalCharge() != 0:
            return None
        if (a.GetAtomicNum() in _CHALCOGEN_DIVALENT
                and a.GetTotalValence() == 2):
            continue
        if any(b.GetBondTypeAsDouble() >= 2.0 for b in a.GetBonds()):
            # Genuine sp2/sp ring member.  In the oxo+dihydro re-orientation
            # mode a ring CARBON whose only multiple bond is exactly ONE
            # exocyclic suffix-eligible doublet (the ring carbonyl) is a
            # saturation position once that doublet becomes an oxo prefix.
            if (include_oxo_carbons_as_saturation
                    and a.GetAtomicNum() == 6
                    and _exocyclic_oxo_count(a) == 1
                    and not any(
                        b.GetBondTypeAsDouble() >= 2.0
                        and b.GetOtherAtom(a).GetIdx() in ring_atoms_full
                        for b in a.GetBonds()
                    )):
                oxo_saturation_full.add(atom_idx)
            continue  # genuine sp2/sp ring member — retains aromatic-form H count
        sp3_full.add(atom_idx)

    # In oxo+dihydro mode the oxo-bearing carbons join the saturation set.
    if include_oxo_carbons_as_saturation:
        if not oxo_saturation_full:
            # No ring carbonyl to lift — this mode adds nothing over the plain
            # hydro derivation; defer to that path (the caller invokes the
            # plain mode separately).
            return None
        sp3_full = sp3_full | oxo_saturation_full

    if not sp3_full:
        return None  # fully aromatic — the direct lookup path handled it (or
                     # would have); don't override here.

    # Build an aromatized copy of ring_mol: all ring atoms aromatic, ring
    # bonds aromatic.  Bail if the resulting molecule cannot be sanitized
    # (e.g. a 5-membered carbocyclic all-aromatic attempt like indane's
    # 5-ring -> cyclopentadiene fails Hückel).
    #
    # For heterocycles where the saturated form's heteroatom carries an
    # explicit/implicit H that the aromatic parent does NOT carry (e.g.
    # tetrahydropyridine NH → pyridine n with no H), the first sanitize
    # pass fails ("Can't kekulize").  Retry clearing per-atom explicit-H
    # counts, letting RDKit recompute aromatic-form H counts from valence
    # (e.g. pyridinic n=0 H, pyrrolic [nH]=1 H).
    #
    # ``reset`` modes, tried in order:
    #   "none"            keep all H as given (works for all-carbon saturated
    #                     rings fused to an aromatic heterocycle, e.g.
    #                     4,5,6,7-tetrahydroindole — the pyrrole N-H is kept)
    #   "saturated_only"  clear H only on atoms that are NOT already aromatic
    #                     in the input — i.e. the saturated ring members that
    #                     are being promoted.  This preserves an already-
    #                     aromatic pyrrolic N-H while letting a saturated ring
    #                     N drop to pyridinic.  Required for fused systems with
    #                     a heteroatom IN the saturated ring fused to an
    #                     aromatic heterocycle (e.g. a tetrahydro-pyrrolo-
    #                     pyridine whose saturated ring carries its own N-H):
    #                     clearing ALL H would wrongly de-protonate the
    #                     pyrrole N and the whole system fails to kekulize.
    #   "all"             clear H on every ring atom (last resort)
    def _try_aromatize(reset):
        rm_copy = Chem.RWMol(ring_mol)
        for atom in rm_copy.GetAtoms():
            if atom.IsInRing():
                was_aromatic = atom.GetIsAromatic()
                atom.SetIsAromatic(True)
                if reset == "all" or (reset == "saturated_only" and not was_aromatic):
                    atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(False)
        for bond in rm_copy.GetBonds():
            if bond.GetBeginAtom().IsInRing() and bond.GetEndAtom().IsInRing():
                bond.SetBondType(Chem.BondType.AROMATIC)
                bond.SetIsAromatic(True)
        aromatized = rm_copy.GetMol()
        Chem.SanitizeMol(aromatized)
        return Chem.MolToSmiles(aromatized)

    arom_smi = None
    for _mode in ("none", "saturated_only", "all"):
        try:
            arom_smi = _try_aromatize(_mode)
            if arom_smi:
                break
        except Exception:
            arom_smi = None
    if not arom_smi:
        return None

    # Look up the aromatic canonical in the curated table.
    name, sub_form, _alkyl_ok, atom_locants = _lookup_curated(arom_smi)
    if name is None:
        return None
    curated_key = _lookup_curated_key(arom_smi)
    if curated_key is None:
        return None
    # In oxo+dihydro mode, source the mancude-parent atom_locants from the
    # dedicated side table FIRST (kept separate so they do not perturb the
    # ordinary direct-curated-match numbering for plain substituted furans /
    # thiophenes).  Fall back to the curated atom_locants when the parent
    # already carries them (e.g. the pyran partial-saturation parents).
    if include_oxo_carbons_as_saturation:
        parent_locants = _OXO_DIHYDRO_MANCUDE_PARENT_LOCANTS.get(curated_key)
        if parent_locants is None and curated_key != arom_smi:
            parent_locants = _OXO_DIHYDRO_MANCUDE_PARENT_LOCANTS.get(arom_smi)
        if parent_locants is not None:
            atom_locants = parent_locants
    if atom_locants is None:
        return None

    # Build a relaxed-bond SMARTS query from the curated key so it matches
    # partly-saturated ring systems.  The aromatic SMARTS ':' becomes '~'
    # (any bond) and the explicit aromatic atom specs fall back to [#6]/[#7]
    # via MolToSmarts.  We only need a topology match — not aromaticity.
    try:
        key_mol = Chem.MolFromSmiles(curated_key)
        if key_mol is None:
            return None
        key_smarts = Chem.MolToSmarts(key_mol)
    except Exception:
        return None
    relaxed_smarts = key_smarts.replace(":", "~")
    query = Chem.MolFromSmarts(relaxed_smarts)
    if query is None:
        return None

    try:
        all_matches = list(mol.GetSubstructMatches(query, uniquify=False))
    except Exception:
        all_matches = []
    if not all_matches:
        return None

    # Among matches whose atom set equals our ring_system.atom_indices,
    # collect all valid (sp3-locant-set, match) pairs.  Each one is a
    # candidate orientation; the lowest-sp3 wins for the single-result API,
    # but the multi-result API exposes them all so the strategy layer can
    # pick the orientation that minimizes principal-characteristic-group
    # locants per P-31.1.4.3.4 (which outranks lowest-hydro-locant).
    def _loc_sort_key(loc_val):
        s = str(loc_val)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num = int(s[:i]) if s[:i] else 0
        return (num, s[i:])

    valid_orientations: list[tuple[list, tuple]] = []  # list of (sp3_locants, match)
    for match in all_matches:
        match_atom_set = frozenset(match)
        if match_atom_set != ring_atoms_full:
            continue
        # match[key_idx] = full_mol_idx.  Translate sp3 full-mol atoms to
        # IUPAC locants via atom_locants.
        try:
            full_to_key = {full_idx: key_idx for key_idx, full_idx in enumerate(match)}
        except Exception:
            continue
        sp3_locants_here: list = []
        ok = True
        for full_idx in sp3_full:
            key_idx = full_to_key.get(full_idx)
            if key_idx is None:
                ok = False
                break
            loc = atom_locants.get(key_idx)
            if loc is None:
                ok = False
                break
            sp3_locants_here.append(loc)
        if not ok:
            continue
        sp3_locants_here.sort(key=_loc_sort_key)
        valid_orientations.append((sp3_locants_here, match))

    if not valid_orientations:
        return None

    # Sort by lowest-sp3 locant set so the single-result API still picks
    # the same orientation as before (preserving prior behaviour for the
    # carbocycle path).
    valid_orientations.sort(
        key=lambda pm: tuple(_loc_sort_key(l) for l in pm[0])
    )
    best_locs, best_match = valid_orientations[0]

    # Build the hydro-prefix.  IUPAC uses the numerical multiplier
    # matching the count of hydrogenated positions: 2 → di, 4 → tetra,
    # 6 → hexa.  A lone sp3 atom (count 1) has no simple 'monohydro'
    # form — defer to the fully-saturated path / curated entries.
    count = len(best_locs)
    mult = get_multiplier(count)
    if mult is None and count != 1:
        return None

    # Helper: identify sp3 atoms adjacent (in-ring) to a ring atom that carries
    # an exocyclic double-bond to a chalcogen or imino-class atom — i.e. atoms
    # eligible to be absorbed into added-indicated-H form (P-31.1.4.2.4 / P-58.2.2).
    # P-66.6 extends the added-IH form beyond ketones to thione (=S), selone (=Se),
    # tellone (=Te), and imine (=NR).  E.g. ``pyridin-3(4H)-imine`` for a ring
    # with sp3 at position 4 and exocyclic =NH at position 3.
    def _sp3_eligible_for_added_ih(sp3_full_idx: int) -> bool:
        from rdkit import Chem as _Chem
        sp3_atom = mol.GetAtomWithIdx(sp3_full_idx)
        _PCG_DB_ATOMS = {7, 8, 16, 34, 52}  # N, O, S, Se, Te for =NR/=O/=S/=Se/=Te
        for nb in sp3_atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in ring_atoms_full:
                continue
            for sub_nb in nb.GetNeighbors():
                if sub_nb.GetIdx() in ring_atoms_full:
                    continue
                if sub_nb.GetAtomicNum() not in _PCG_DB_ATOMS:
                    continue
                bond = mol.GetBondBetweenAtoms(nb_idx, sub_nb.GetIdx())
                if bond is None:
                    continue
                if bond.GetBondType() == _Chem.BondType.DOUBLE:
                    return True
        return False

    if count == 1:
        # Added-indicated-H (P-31.1.4.2.4 / P-58.2.2): a lone sp3 atom in an
        # otherwise mancude retained parent has no "monohydro" prefix form.
        # Per spec it is cited inline as ``(NH)`` between the suffix locant
        # and the suffix tail (e.g. ``naphthalen-1(2H)-one``).  This requires
        # a ring-anchor sp2 atom carrying an exocyclic double-bonded chalcogen-
        # or imino-class atom (=O, =S, =Se, =NH) adjacent to the sp3 atom;
        # without it, no ``(NH)`` form is possible — bail and let the upstream
        # path decide.
        sp3_full_idx = next(iter(sp3_full))
        if not _sp3_eligible_for_added_ih(sp3_full_idx):
            return None
        # Single-orientation result with the mancude retained name and
        # ``added_indicated_h_atoms`` carrying the sp3 full-mol atom idx.
        return _finalize_added_ih_orientation(
            name=name,
            sub_form=sub_form,
            atom_locants=atom_locants,
            mol=mol,
            best_match=best_match,
            added_ih_atoms=(sp3_full_idx,),
        )

    # Per-orientation builder: takes a (sp3_locants, match) pair from
    # valid_orientations and returns a result dict for that orientation.
    # Used for both the single-result and multi-result APIs.
    #
    # P-31.1.4.2.4 / P-58.2.2: when count is ODD (3, 5, ...), the hydro prefix
    # alone cannot describe the saturation — saturation comes in even pairs
    # (dihydro, tetrahydro, hexahydro).  The odd sp3 atom must be absorbed
    # into the added-indicated-H "(NH)" form, leaving an even-count hydro
    # prefix.  E.g. 1-tetralone (3 sp3 atoms at 2,3,4) → "3,4-dihydro" (2 sp3)
    # + "(2H)" added-IH (the sp3 atom adjacent to the ring-ketone C1).  The
    # absorbed sp3 atom must be adjacent to a ring-PCG-bearing atom (exocyclic
    # C=O / C=S / C=Se / C=NR).
    def _build_orientation_result(
        cur_locs: list, cur_match: tuple
    ) -> "dict | None":
        cur_count = len(cur_locs)
        if cur_count % 2 == 1:
            full_to_key = {full_idx: key_idx for key_idx, full_idx in enumerate(cur_match)}
            sp3_pairs: list[tuple] = []  # (loc, full_idx)
            for full_idx in sp3_full:
                key_idx = full_to_key.get(full_idx)
                if key_idx is None:
                    return None
                loc = atom_locants.get(key_idx)
                if loc is None:
                    return None
                sp3_pairs.append((loc, full_idx))
            sp3_pairs.sort(key=lambda pair: _loc_sort_key(pair[0]))
            absorb_idx: int | None = None
            for _loc, full_idx in sp3_pairs:
                if _sp3_eligible_for_added_ih(full_idx):
                    absorb_idx = full_idx
                    break
            if absorb_idx is None:
                # No sp3 atom adjacent to a ring-PCG → cannot form a valid
                # odd-count name.  Bail rather than emit a malformed
                # "<n>hydro" prefix.
                return None
            absorbed_loc = atom_locants.get(full_to_key[absorb_idx])
            new_locs = [l for l in cur_locs if str(l) != str(absorbed_loc)]
            new_count = len(new_locs)
            new_mult = get_multiplier(new_count)
            if new_mult is None:
                return None
            new_loc_str = ",".join(str(l) for l in new_locs)
            result = _finalize_hydro_orientation(
                name=name,
                sub_form=sub_form,
                atom_locants=atom_locants,
                mol=mol,
                best_locs=new_locs,
                best_match=cur_match,
                loc_str=new_loc_str,
                mult=new_mult,
            )
            if result is None:
                return None
            result["added_indicated_h_atoms"] = (absorb_idx,)
            return result
        cur_mult = get_multiplier(cur_count)
        if cur_mult is None:
            return None
        cur_loc_str = ",".join(str(l) for l in cur_locs)
        return _finalize_hydro_orientation(
            name=name,
            sub_form=sub_form,
            atom_locants=atom_locants,
            mol=mol,
            best_locs=cur_locs,
            best_match=cur_match,
            loc_str=cur_loc_str,
            mult=cur_mult,
        )

    if return_all_orientations:
        results: list[dict] = []
        seen_keys: set[tuple] = set()
        for cur_locs, cur_match in valid_orientations:
            res = _build_orientation_result(cur_locs, cur_match)
            if res is None:
                continue
            # Dedup by (name, ordered assignments).  Symmetric rings produce
            # multiple substruct matches that yield identical Numberings;
            # collapse those.
            try:
                nbs = res.get("numbering_options", ())
                key = (res["name"], tuple(
                    tuple(nb._assignments) for nb in nbs
                ))
            except Exception:
                key = None
            if key is not None:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            results.append(res)
        return results if results else None

    return _build_orientation_result(best_locs, best_match)


def _finalize_hydro_orientation(
    *,
    name: str,
    sub_form: str | None,
    atom_locants: dict,
    mol,
    best_locs: list,
    best_match: tuple,
    loc_str: str,
    mult: str,
) -> "dict | None":
    """Helper for ``_try_derive_hydro_retained``: compose the final name +
    Numbering for a single match orientation."""
    # Indicated-H recompute.  When the aromatic-parent name carries a leading
    # "<digit>H-" marker (e.g. "1H-pyrrole", "2H-isoindole"), that marker pins
    # the indicated-H on one ring atom.  After partial saturation:
    #   (a) if that locant is now one of the hydro-locants (sp3), the marker
    #       is meaningless / contradictory — strip it.  If there is still a
    #       non-sp3 ring atom that needs an indicated-H to balance valence,
    #       re-emit the marker at the lowest-locant such atom.
    #   (b) if the locant remains non-sp3 AND still needs the indicated-H,
    #       keep the parent's marker unchanged.
    # This is the generic mechanism: ANY retained aromatic parent with an
    # "NH-" prefix automatically gets the correct indicated-H for its
    # partly-saturated derivatives — no per-entry curation required.
    effective_name = name
    effective_sub = sub_form
    m_ih = _re.match(r"^(\d+[a-z]?)H-", name)
    if m_ih is not None:
        parent_ih_locant_str = m_ih.group(1)
        sp3_loc_strs = {str(l) for l in best_locs}
        parent_ih_is_sp3 = parent_ih_locant_str in sp3_loc_strs
        if parent_ih_is_sp3:
            # The parent's indicated-H locant atom is now sp3 — the "NH-"
            # marker must be moved (or dropped).  Find ring atoms that are
            # still non-sp3 AND have at least one free valence slot (explicit
            # H or room to carry one in the saturated form).  Map locants via
            # atom_locants + best_match.
            key_to_loc = atom_locants  # ring_mol-key-idx -> IUPAC locant
            # Build locant -> full_idx mapping via the chosen match.
            loc_to_full: dict[str, int] = {}
            for key_idx, full_idx in enumerate(best_match):
                loc = key_to_loc.get(key_idx)
                if loc is not None:
                    loc_to_full[str(loc)] = full_idx

            # Candidate indicated-H locants: ring atoms that are
            #   (i) not in the hydro (sp3) set,
            #   (ii) carry >=1 explicit H in the input mol (the saturated slot
            #        that the indicated-H is literally marking).
            # Sort by numeric locant value; pick the lowest.
            def _loc_num(s: str) -> tuple[int, str]:
                i = 0
                while i < len(s) and s[i].isdigit():
                    i += 1
                return (int(s[:i]) if s[:i] else 0, s[i:])

            candidates: list[str] = []
            for loc_str_candidate, full_idx in loc_to_full.items():
                if loc_str_candidate in sp3_loc_strs:
                    continue
                try:
                    a = mol.GetAtomWithIdx(full_idx)
                except Exception:
                    continue
                if a.GetTotalNumHs(includeNeighbors=False) < 1:
                    continue
                candidates.append(loc_str_candidate)

            new_ih: str | None = None
            if candidates:
                candidates.sort(key=_loc_num)
                new_ih = candidates[0]

            # Strip the parent's "<N>H-" marker from name / sub_form, then
            # re-prepend the new marker (or leave stripped if no candidate).
            stripped_name = name[m_ih.end():]
            if new_ih is not None:
                effective_name = f"{new_ih}H-{stripped_name}"
            else:
                effective_name = stripped_name
            if sub_form is not None:
                m_sub = _re.match(r"^(\d+[a-z]?)H-", sub_form)
                if m_sub is not None:
                    stripped_sub = sub_form[m_sub.end():]
                    if new_ih is not None:
                        effective_sub = f"{new_ih}H-{stripped_sub}"
                    else:
                        effective_sub = stripped_sub

    # IUPAC writes the hydro-prefix directly fused to the retained name with
    # no separating hyphen when the name begins with a letter, e.g.
    # "1,4-dihydronaphthalene" (not "1,4-dihydro-naphthalene").
    #
    # A separating hyphen IS required (P-16.3.3) whenever the retained name
    # begins with a locant — both the indicated-hydrogen marker case
    # ("1H-indole" -> "5,6,7-trihydro-1H-indole") and the bare locant-prefix
    # case ("1,3-thiazole" -> "4,5-dihydro-1,3-thiazole"): a hyphen always
    # separates the trailing "hydro" from a following locant digit.
    _starts_with_locant = bool(_re.match(r"^\d", effective_name))
    if _starts_with_locant:
        derived_name = f"{loc_str}-{mult}hydro-{effective_name}"
    else:
        derived_name = f"{loc_str}-{mult}hydro{effective_name}"

    # Build a substituent form mirroring the base name's substituent form.
    # For retained aromatic parents we typically have e.g. "naphthalenyl";
    # stash the analogous dihydro substituent form.  OPSIN accepts forms
    # like "1,4-dihydronaphthalen-2-yl" so we emit "<loc>-<mult>hydro<sub>".
    derived_sub: str | None = None
    if effective_sub is not None:
        if _re.match(r"^\d", effective_sub):
            derived_sub = f"{loc_str}-{mult}hydro-{effective_sub}"
        else:
            derived_sub = f"{loc_str}-{mult}hydro{effective_sub}"

    # Build a Numbering directly from best_match + atom_locants.  We already
    # have the full→key atom mapping (best_match is indexed by curated-key
    # atom position); combine with atom_locants to produce the (full_idx,
    # Locant) pairs.  Using _build_numbering_from_atom_locants here would
    # attempt an aromatic-vs-saturated substructure match that fails — we
    # short-circuit by constructing the Numbering ourselves.
    def _parse_locant(loc) -> Locant:
        if isinstance(loc, int):
            return Locant.numeric(loc)
        s = str(loc)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num_part = s[:i]
        suf_part = s[i:]
        if num_part:
            return Locant.numeric(int(num_part), suf_part)
        return Locant.numeric(1)

    assignments: list[tuple[int, Locant]] = []
    for key_idx, full_idx in enumerate(best_match):
        loc_val = atom_locants.get(key_idx)
        if loc_val is None:
            continue
        assignments.append((full_idx, _parse_locant(loc_val)))
    if not assignments:
        return None

    def _loc_order(loc: Locant):
        s = loc.label
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num = int(s[:i]) if s[:i] else 0
        return (num, s[i:])

    assignments.sort(key=lambda pair: _loc_order(pair[1]))
    locant_set = tuple(loc for _, loc in assignments)
    numbering = Numbering(
        _assignments=tuple(assignments),
        locant_set=locant_set,
    )
    numbering_options = (numbering,)

    return {
        "name": derived_name,
        "substituent_form": derived_sub,
        "numbering_options": numbering_options,
    }


def _finalize_added_ih_orientation(
    *,
    name: str,
    sub_form: str | None,
    atom_locants: dict,
    mol,
    best_match: tuple,
    added_ih_atoms: tuple[int, ...],
) -> "dict | None":
    """Helper for ``_try_derive_hydro_retained``: compose a NamedParent-style
    result for the count==1 added-indicated-H form (P-31.1.4.2.4 / P-58.2.2).

    Unlike ``_finalize_hydro_orientation``, this path keeps the mancude
    retained name unchanged (``naphthalene`` rather than ``X-monohydro-…``)
    and returns the sp3 atom indices via ``added_indicated_h_atoms`` so the
    downstream NamedParent → SuffixGroup chain can render them as ``(NH)``
    inline with the suffix locant (e.g. ``naphthalen-1(2H)-one``).
    """
    # Build a Numbering directly from best_match + atom_locants.  The match
    # was produced by a relaxed-bond SMARTS query against the curated key,
    # so best_match[key_idx] = full_mol_idx.
    def _parse_locant(loc) -> Locant:
        if isinstance(loc, int):
            return Locant.numeric(loc)
        s = str(loc)
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num_part = s[:i]
        suf_part = s[i:]
        if num_part:
            return Locant.numeric(int(num_part), suf_part)
        return Locant.numeric(1)

    assignments: list[tuple[int, Locant]] = []
    for key_idx, full_idx in enumerate(best_match):
        loc_val = atom_locants.get(key_idx)
        if loc_val is None:
            continue
        assignments.append((full_idx, _parse_locant(loc_val)))
    if not assignments:
        return None

    def _loc_order(loc: Locant):
        s = loc.label
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        num = int(s[:i]) if s[:i] else 0
        return (num, s[i:])

    assignments.sort(key=lambda pair: _loc_order(pair[1]))
    locant_set = tuple(loc for _, loc in assignments)
    numbering = Numbering(
        _assignments=tuple(assignments),
        locant_set=locant_set,
    )

    return {
        "name": name,
        "substituent_form": sub_form,
        "numbering_options": (numbering,),
        "added_indicated_h_atoms": added_ih_atoms,
    }


# ---------------------------------------------------------------------------
# Build a Numbering from ring-mol atom_locants + substructure match
# ---------------------------------------------------------------------------

def _build_numbering_from_atom_locants(
    ring_mol,
    mol,
    ring_system: "RingSystem",
    atom_locants: dict,
) -> "tuple[Numbering, ...]":
    """Convert ring-mol-indexed atom_locants to a full-mol Numbering.

    atom_locants maps {ring_mol_atom_idx: IUPAC_locant (int or str like "4a")}.
    We use GetSubstructMatch to map ring_mol indices to full_mol indices,
    then build a Numbering with the full-mol atom indices.

    Returns a 1-tuple with the Numbering, or () on failure.
    """
    try:
        # Map ring_mol atom indices -> full_mol atom indices.
        # Use SMARTS-based matching to enforce explicit H counts (e.g. [nH] vs n).
        # Standard GetSubstructMatch with a mol object does NOT enforce aromatic N-H
        # counts, leading to wrong locant assignments for benzimidazole, indole, etc.
        try:
            ring_smi = Chem.MolToSmiles(ring_mol)
            ring_query = Chem.MolFromSmarts(ring_smi)
        except Exception:
            ring_query = None

        # For symmetric all-carbon fused rings (naphthalene, anthracene),
        # we need uniquify=False to get ALL match orientations so the strategy
        # can pick the one that assigns substituents the lowest locants (P-14.5).
        # For heterocyclic rings WITH explicit [nH], the SMARTS pins the NH
        # placement and a single match is sufficient.  For heterocyclic rings
        # WITHOUT [nH] (pyrimidine `c1cncnc1`, pyrazine `c1cnccn1`, pyridazine
        # `c1ccnnc1`), the ring is C2-symmetric and uniquify=True would
        # arbitrarily pick one orientation — yielding e.g. "6-methylpyrimidine"
        # when "4-methylpyrimidine" is the correct lowest-locant choice.  Use
        # uniquify=False whenever the SMARTS query lacks an [nH] anchor.
        ring_has_heteroatoms = any(
            a.GetAtomicNum() not in (1, 6) for a in ring_mol.GetAtoms()
        )
        ring_has_explicit_nh = "[nH]" in ring_smi if ring_smi else False
        # uniquify=True only when heteroatoms AND an [nH] anchor pin the match
        use_uniquify = ring_has_heteroatoms and ring_has_explicit_nh

        # Tautomer-tolerant fallback query: replace [nH] with n in the canonical
        # SMILES so the SMARTS doesn't enforce a specific NH location on aromatic
        # nitrogens.  Required for retained heterocycles (e.g. purine) where the
        # curated key fixes one tautomer but the input molecule may have NH on a
        # different ring N (xanthine vs hypoxanthine vs guanine vs N9-alkylated
        # purines, etc.).  The atom_locants table is constructed so the topology-
        # based locant assignment is correct regardless of which N actually
        # carries the H, so dropping the H-strictness is safe here.
        ring_query_taut = None
        try:
            ring_smi_taut = ring_smi.replace("[nH]", "n")
            # Drop forced single-bond ring closures ('c-2', 'c-1' etc.).  After
            # stripping [nH], the ring's aromatic system is fully closed by
            # implicit aromatic bonds; leaving the explicit '-' (single bond)
            # makes the SMARTS reject aromatic ring-closure bonds in the input.
            import re as _re_local
            ring_smi_taut = _re_local.sub(r"-(\d)", r"\1", ring_smi_taut)
            ring_smi_taut = _re_local.sub(r"-(%\d\d)", r"\1", ring_smi_taut)
            if ring_smi_taut != ring_smi:
                ring_query_taut = Chem.MolFromSmarts(ring_smi_taut)
        except Exception:
            ring_query_taut = None

        # Bond-generic query: a copy of ``ring_mol`` whose bond orders and
        # aromaticity are wildcarded (atoms still match by element + degree).
        # Required for FULLY-AROMATIC retained rings whose IUPAC numbering is
        # FIXED but whose kekulé pattern SHIFTS when a substituent is introduced
        # (the PAH / aza-PAH family: phenanthrazine, trinaphthylene,
        # isobenzothiofuran, isophosphindole, …).  A substituent on an sp2 ring
        # carbon migrates an adjacent double bond, so the bond-order-strict
        # SMARTS built from the curated key either fails to match or — worse —
        # matches a *single* orientation whose bonds happen to line up at the
        # wrong locant.  Bond-generic matching recovers ALL graph-automorphism-
        # equivalent orientations of the aromatic skeleton; because every kekulé
        # of a fully-aromatic ring is equivalent and every automorphism is a
        # valid renumbering direction, the strategy layer can then pick the
        # orientation giving the lowest substituent locants per P-14.5.
        #
        # GATED to fully-aromatic rings only.  For PARTIALLY-saturated rings the
        # in-ring double-bond *position* IS the structural identity (3,4- vs
        # 3,6-dihydropyran; 1,2- vs 1,4-dihydronaphthalene; tetraline) and even
        # for mancude indicated-H rings (indene, trindene) the sp3 CH2 makes the
        # 1- and 3-positions inequivalent — bond-generic matching would conflate
        # them and mis-place substituents.  The all-aromatic gate excludes every
        # such case (any sp3 ring atom makes the ring non-aromatic).
        ring_all_aromatic = all(
            a.GetIsAromatic() for a in ring_mol.GetAtoms()
        ) and ring_mol.GetNumAtoms() > 0
        ring_query_generic = None
        if ring_all_aromatic:
            try:
                _gmol = Chem.Mol(ring_mol)
                for _b in _gmol.GetBonds():
                    _b.SetBondType(Chem.BondType.SINGLE)
                    _b.SetIsAromatic(False)
                for _a in _gmol.GetAtoms():
                    _a.SetIsAromatic(False)
                Chem.SanitizeMol(_gmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
                _gp = Chem.AdjustQueryParameters.NoAdjustments()
                _gp.makeBondsGeneric = True
                ring_query_generic = Chem.AdjustQueryProperties(_gmol, _gp)
            except Exception:
                ring_query_generic = None

        try:
            if ring_query is not None:
                all_matches: list[tuple[int, ...]] = list(
                    mol.GetSubstructMatches(ring_query, uniquify=use_uniquify)
                )
            else:
                all_matches = []
            # Supplement with bond-generic matches whenever they reveal
            # additional orientations the bond-order-strict query missed (the
            # kekulé-shift case described above).  Merge + dedup so existing
            # strict matches are preserved and only genuinely new orientations
            # are added.  Skipped when uniquify pinned a single NH-anchored
            # orientation (heterocycles with explicit [nH] have a unique
            # tautomer numbering and must not be loosened).
            if ring_query_generic is not None and not use_uniquify:
                try:
                    generic_matches = list(
                        mol.GetSubstructMatches(ring_query_generic, uniquify=False)
                    )
                except Exception:
                    generic_matches = []
                if len(generic_matches) > len(all_matches):
                    seen_m = {tuple(m) for m in all_matches}
                    for gm in generic_matches:
                        tgm = tuple(gm)
                        if tgm not in seen_m:
                            seen_m.add(tgm)
                            all_matches.append(gm)
            if not all_matches:
                # Fallback to tautomer-tolerant matching via the ring_mol object.
                # Use uniquify=False so rings with symmetric N positions (e.g.
                # 1H-1,2,4-triazole: either adjacent N can be N1) yield all
                # orientations; the strategy layer will pick the one giving
                # lowest substituent locants per P-14.5.
                all_matches = list(mol.GetSubstructMatches(ring_mol, uniquify=False))
            if not all_matches and ring_query_taut is not None:
                all_matches = list(
                    mol.GetSubstructMatches(ring_query_taut, uniquify=False)
                )
        except TypeError:
            # Older RDKit versions: uniquify is positional
            if ring_query is not None:
                all_matches = list(mol.GetSubstructMatches(ring_query, use_uniquify))
            else:
                all_matches = []
            if not all_matches:
                all_matches = list(mol.GetSubstructMatches(ring_mol, False))
            if not all_matches and ring_query_taut is not None:
                all_matches = list(mol.GetSubstructMatches(ring_query_taut, False))
        if not all_matches:
            logger.debug("_build_numbering_from_atom_locants: no substructure match")
            return ()

        # Restrict matches to the ring_system being numbered.  When the curated
        # SMILES (e.g. pyrimidine `c1cncnc1`) appears in MULTIPLE ring systems
        # of the parent molecule (e.g. bosentan-style central + substituent
        # pyrimidines), uniquify=False yields matches across all of them.
        # Without this filter, downstream numbering picks an alignment whose
        # atoms belong to a sibling ring, leaving substituent atoms on the
        # actual ring without a locant ("pyrimidin-yl" with no attachment
        # number, prefix locants dropped).
        #
        # Allow exocyclic atoms that are bonded to ring atoms (e.g. =O for
        # lactam tautomers keyed with the with-=O canonical: each curated key
        # like ``O=C1CN2Cc3ccccc3N=C2N1`` includes the exocyclic O).  These
        # match-ring-extras are valid even though they aren't in
        # ``ring_system.atom_indices``.  Use the union of ring atoms +
        # neighbours-of-ring-atoms as the membership test.
        rs_atoms = ring_system.atom_indices
        allowed_atoms = set(rs_atoms)
        try:
            for ri in rs_atoms:
                for nb in mol.GetAtomWithIdx(ri).GetNeighbors():
                    allowed_atoms.add(nb.GetIdx())
        except Exception:
            allowed_atoms = set(rs_atoms)
        all_matches = [m for m in all_matches if all(a in allowed_atoms for a in m)]
        if not all_matches:
            logger.debug(
                "_build_numbering_from_atom_locants: no substructure match within ring_system"
            )
            return ()

        def _parse_locant(loc) -> Locant:
            """Parse int or string locant into a Locant object.

            Compound locants like ``"4a"`` (fused ring junction), ``"2'"`` (primed
            ring-assembly locant), or ``"4a'"`` are parsed as numeric locants with
            a suffix so the strategy layer can still score them by their numeric
            value (e.g. ``2' < 7'`` at equal priority).  Using Locant.hetero here
            would silently set ``_numeric_value=None`` and the scorer would treat
            all compound locants as equally-zero, breaking lowest-locant selection.
            """
            if isinstance(loc, int):
                return Locant.numeric(loc)
            # String like "4a", "8a", "4b", "2'", "4a'", etc.
            s = str(loc)
            # Extract numeric prefix and letter/prime suffix
            i = 0
            while i < len(s) and s[i].isdigit():
                i += 1
            num_part = s[:i]
            suf_part = s[i:]
            if num_part:
                return Locant.numeric(int(num_part), suf_part)
            return Locant.numeric(1)  # fallback

        # match[ring_mol_idx] = full_mol_idx
        # Build assignments: (full_mol_idx, Locant)
        # Sort by locant for consistent ordering
        def _locant_sort_key(loc):
            if isinstance(loc, int):
                return (loc, '')
            s = str(loc)
            i = 0
            while i < len(s) and s[i].isdigit():
                i += 1
            num = int(s[:i]) if s[:i] else 0
            return (num, s[i:])

        locant_set = tuple(
            _parse_locant(loc)
            for loc in sorted(atom_locants.values(), key=_locant_sort_key)
        )

        numberings: list[Numbering] = []
        for match in all_matches:
            assignments = []
            ok = True
            for ring_idx, iupac_loc in sorted(atom_locants.items(),
                                               key=lambda x: _locant_sort_key(x[1])):
                if ring_idx >= len(match):
                    logger.debug("atom_locants ring_idx %d out of range (match len %d)", ring_idx, len(match))
                    ok = False
                    break
                full_idx = match[ring_idx]
                assignments.append((full_idx, _parse_locant(iupac_loc)))
            if ok and assignments:
                numberings.append(Numbering(
                    _assignments=tuple(assignments),
                    locant_set=locant_set,
                ))

        return tuple(numberings) if numberings else ()
    except Exception as e:
        logger.debug("_build_numbering_from_atom_locants failed: %s", e)
        return ()


# ---------------------------------------------------------------------------
# Build a NamedParent from a retained name match
# ---------------------------------------------------------------------------

def _build_named_parent(
    ring_system: "RingSystem",
    name_str: str,
    sub_form: str | None,
    alkyl_ok: bool,
    naming_method: str = "retained",
    numbering_options: "tuple[Numbering, ...]" = (),
    extra_atom_indices: "frozenset[int] | None" = None,
    added_indicated_h_atoms: "tuple[int, ...] | None" = None,
    precomposed_retained_no_suffix: bool = False,
) -> "NamedParent":
    """Build a NamedParent from a retained name.

    ``extra_atom_indices`` lets the caller mark additional atoms (beyond
    the ring atoms themselves) as claimed by the retained name — used by
    the exocyclic-=O fallback so downstream assembly does not re-emit a
    redundant ``-N-oxo-`` prefix for the carbonyl O that is already
    encoded in the retained-name stem (e.g. ``-2(1H)-one``).

    ``added_indicated_h_atoms`` carries full-mol atom indices of ring atoms
    whose locants must be cited inline as ``(NH)`` after the suffix locant
    (P-31.1.4.2.4 / P-58.2.2 added indicated hydrogen — e.g.
    ``naphthalen-1(2H)-one``).

    ``precomposed_retained_no_suffix`` (Phase 8 P-31.1) marks retained
    names whose stem already lexically embeds a suffix-form ending
    (``5-pyrazolone``, ``urazol``, ``phthalhydrazide``).  OPSIN's parser
    refuses to glue a separable PCG suffix onto these stems; the engine
    routes any FG to the prefix slot instead.
    """
    from iupac_namer.types import CandidateParent

    if extra_atom_indices:
        atom_indices = ring_system.atom_indices | extra_atom_indices
    else:
        atom_indices = ring_system.atom_indices

    # Build the CandidateParent if not already wrapping one
    candidate = CandidateParent(
        atom_indices=atom_indices,
        type=ring_system.type,
        length=ring_system.ring_size,
        ring_system=ring_system,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )

    return _build_named_parent_from_candidate(
        candidate=candidate,
        name_str=name_str,
        sub_form=sub_form,
        alkyl_ok=alkyl_ok,
        naming_method=naming_method,
        numbering_options=numbering_options,
        added_indicated_h_atoms=added_indicated_h_atoms,
        precomposed_retained_no_suffix=precomposed_retained_no_suffix,
    )


def _build_named_parent_from_candidate(
    candidate: "CandidateParent",
    name_str: str,
    sub_form: str | None,
    alkyl_ok: bool,
    naming_method: str = "retained",
    numbering_options: "tuple[Numbering, ...]" = (),
    added_indicated_h_atoms: "tuple[int, ...] | None" = None,
    precomposed_retained_no_suffix: bool = False,
) -> "NamedParent":
    """Build NamedParent from an existing CandidateParent and name string."""
    # stem = name without trailing 'e' (for Method 2 suffix attachment)
    # E.g. "benzene" -> "benzen", "pyridine" -> "pyridin", "furan" -> "furan" (no 'e')
    if name_str.endswith("e"):
        stem = name_str[:-1]
    else:
        stem = name_str

    # alkyl_stem: for Method (1), strip "-ane"/"-ene" entirely.
    # Only applicable for saturated monocyclic carbocycles.
    alkyl_stem: str | None = None
    if alkyl_ok:
        if name_str.endswith("ane"):
            alkyl_stem = name_str[:-3]
        elif name_str.endswith("ene"):
            alkyl_stem = name_str[:-3]

    return NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=alkyl_stem,
        naming_method=naming_method,
        indicated_hydrogen=None,
        numbering_options=numbering_options,
        added_indicated_h_atoms=added_indicated_h_atoms,
        precomposed_retained_no_suffix=precomposed_retained_no_suffix,
    )
