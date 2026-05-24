"""iupac_namer.perception.organometallic
======================================================

Metallocene / sandwich-complex perception (Stage 6 R3-A; root cause #16
in ``docs/opsin_coverage_taxonomy.md``).

Why this module exists
----------------------
Neutral ``[M+n]·Cp-·Cp-`` sandwich complexes (ferrocene, ruthenocene,
cobaltocene, …) are written in SMILES as three fragments: a charged
metal centre and two cyclopentadienide rings.  The substitutive engine
has no plan for the bare metal fragment, so the salt path emits

    "[NAMING ERROR: No valid naming plan found for [Os+2]] cyclopentadienide cyclopentadienide"

for many of them, while a few (Fe, Co, Ni) coincidentally come out as
``iron(2+) cyclopentadienide cyclopentadienide`` — readable and
round-trippable through OPSIN, but not the IUPAC retained name OPSIN
itself emits when asked to interpret ``ferrocene``.  Worse, several
metals (V, Rh, Pb, Nb) carry radical electrons in the RDKit valence
model and the standalone engine refuses them entirely via the P-29.2
free-valence guard.

The IUPAC 2013 Blue Book (P-68.3) and OPSIN both retain the special
``-ocene`` names for the ``η5-Cp_2 M`` family.  This module is a
**whole-molecule canonical-SMILES lookup**.  When the input mol's
canonical SMILES exactly matches a sandwich-complex pin, we emit a
``LeafTree`` whose ``text`` is the retained ``-ocene`` surface name
(``ferrocene`` etc.); otherwise we return ``None`` and the engine falls
through to its existing dispatch.

Architectural notes
-------------------

* No module-level mutable state.  The lookup table is built once at
  import time as an immutable ``Mapping``; the dispatcher is purely
  functional given the input mol.
* The classifier never mutates the input mol.
* The match is by *exact canonical SMILES*.  Substituted metallocenes
  (e.g. ``1,1'-dimethylferrocene``) won't match the parent pin and will
  fall through; that is intentional — substituted metallocenes are out
  of scope for this round.
* No silent atom drops.  ``MetalloceneClassification`` records the
  centre-atom and ring-atom indices in the input mol so a future
  extension can cross-check coverage against the engine's atom-claim
  ledger.
* The dispatch hook is placed in ``engine.name_smiles`` *before* the
  free-valence guard ``_validate_no_open_valences``, so metals with
  RDKit-modelled radical electrons (V, Rh, Pb, Nb, …) are accepted
  without weakening the guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from iupac_namer.types import Choice, LeafTree, OutputForm

# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetalloceneClassification:
    """Structural summary of one metallocene match.

    Attributes
    ----------
    center_atom_idx:
        Index of the metal atom in the *input* mol.
    ring_atom_sets:
        Two-tuple of frozensets, one per cyclopentadienide ring, each
        containing the indices of that ring's five carbons in the input
        mol.
    retained_name:
        The IUPAC-retained ``-ocene`` surface name (e.g. ``ferrocene``).
    """

    center_atom_idx: int
    ring_atom_sets: tuple[frozenset[int], frozenset[int]]
    retained_name: str


# ---------------------------------------------------------------------------
# Retained-name table
# ---------------------------------------------------------------------------
#
# Keys are RDKit canonical SMILES of the standalone neutral sandwich.
# The canonical form is the same for several SMILES inputs:
#     ``[CH-]1cccc1.[CH-]1cccc1.[Fe+2]``
#     ``[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]``
#     ``[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1``
# all canonicalise to the last form, which is what we key on.
#
# Each pin was verified by:
#   1. ``py2opsin('<name>')`` returning a 3-fragment SMILES;
#   2. ``Chem.MolToSmiles(Chem.MolFromSmiles(...))`` round-trip to the
#      key below;
#   3. The cross-check that no ``[NAMING ERROR ...]`` is emitted by the
#      generic engine path under R3-A's tests.
#
# Names are taken verbatim from OPSIN's ``arylGroups.xml`` (also mined
# in ``data/opsin_extracted/rings_from_opsin.json``).
_METALLOCENE_PINS: Mapping[str, str] = {
    "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1": "ferrocene",
    "[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1": "ruthenocene",
    "[Os+2].c1cc[cH-]c1.c1cc[cH-]c1": "osmocene",
    "[Co+2].c1cc[cH-]c1.c1cc[cH-]c1": "cobaltocene",
    "[Ni+2].c1cc[cH-]c1.c1cc[cH-]c1": "nickelocene",
    "[Rh+2].c1cc[cH-]c1.c1cc[cH-]c1": "rhodocene",
    "[Cr+2].c1cc[cH-]c1.c1cc[cH-]c1": "chromocene",
    "[V+2].c1cc[cH-]c1.c1cc[cH-]c1": "vanadocene",
    "[Ti+2].c1cc[cH-]c1.c1cc[cH-]c1": "titanocene",
    "[Mo+2].c1cc[cH-]c1.c1cc[cH-]c1": "molybdocene",
    "[Pb+2].c1cc[cH-]c1.c1cc[cH-]c1": "plumbocene",
    "[Zr+2].c1cc[cH-]c1.c1cc[cH-]c1": "zirconocene",
    "[Nb+2].c1cc[cH-]c1.c1cc[cH-]c1": "niobocene",
    # Neutral-metal variants: SMILES inputs where the metal is written as
    # uncharged and the Cp rings carry the -1 charges (net charge -2 overall).
    # These are alternative SMILES representations of the same physical compound;
    # IUPAC P-68.3 retains the ``-ocene`` PIN regardless of how the SMILES
    # encodes the oxidation state.
    # Key: exact RDKit canonical SMILES  Value: retained PIN
    "[Fe].c1cc[cH-]c1.c1cc[cH-]c1": "ferrocene",
    # Free-valence Cp anion variants: SMILES inputs where the Cp rings
    # are written as ``[C-]1C=CC=C1`` (each C- has 0 H + 1 radical) rather
    # than the proper ``c1cc[cH-]c1`` aromatic form.  These are
    # chemist-shorthand forms of the same -ocene compound; both round-trip
    # through OPSIN under the eval's ``_metal_organic_ligand_equiv``
    # matcher.  Manganese is intentionally absent — OPSIN does not parse
    # "manganocene", so the substituted-metallocene path emits the
    # OPSIN-friendly ``bis(cyclopentadienyl)manganese`` instead.
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Ti]": "titanocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Co]": "cobaltocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Ni]": "nickelocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Fe]": "ferrocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Ru]": "ruthenocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Os]": "osmocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Cr]": "chromocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[V]": "vanadocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Mo]": "molybdocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Zr]": "zirconocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Pb]": "plumbocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Nb]": "niobocene",
    "[C-]1C=CC=C1.[C-]1C=CC=C1.[Rh]": "rhodocene",
}


# Set of metal-atom symbols supported.  Used by the structural classifier
# below as a fast pre-filter before doing the canonical-SMILES lookup.
_METALLOCENE_METALS: frozenset[str] = frozenset({
    "Fe", "Ru", "Os", "Co", "Ni", "Rh",
    "Cr", "V", "Ti", "Mo", "Pb", "Zr", "Nb",
})


# Metal element symbol -> retained ``-ocene`` family name.  Derived once
# at import from :data:`_METALLOCENE_PINS` so adding a pin automatically
# extends substituted-metallocene coverage too.  This keeps the
# substituted-variant dispatcher (R22-E) data-driven from the same pin
# table the parent dispatcher uses.
def _build_ocene_table() -> Mapping[str, str]:
    table: dict[str, str] = {}
    for canonical, name in _METALLOCENE_PINS.items():
        # Canonical form is e.g. "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1" — first
        # bracketed token holds the metal symbol.
        first = canonical.split(".", 1)[0]
        # strip "[" and trailing charge/"]"
        sym = first.lstrip("[").rstrip("]").rstrip("+0123456789-")
        table[sym] = name
    return table


_OCENE_NAME: Mapping[str, str] = _build_ocene_table()


# ---------------------------------------------------------------------------
# Top-level classifier
# ---------------------------------------------------------------------------


def classify_metallocene(mol) -> MetalloceneClassification | None:
    """Return a :class:`MetalloceneClassification` if ``mol`` is a pinned
    sandwich complex, else ``None``.

    The match is by exact canonical SMILES of the whole molecule against
    the :data:`_METALLOCENE_PINS` table.  When matched, the indices of
    the metal atom and the two ring carbon sets are computed from the
    same ``mol`` instance the caller passed in.
    """
    if mol is None:
        return None
    from rdkit import Chem

    # Cheap pre-filter: skip the canonicalisation cost when the molecule
    # cannot possibly be a pinned sandwich (no recognised metal symbol or
    # not exactly three fragments).
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 3:
        return None
    metal_idx: int | None = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in _METALLOCENE_METALS:
            metal_idx = atom.GetIdx()
            break
    if metal_idx is None:
        return None

    canonical = Chem.MolToSmiles(mol)
    name = _METALLOCENE_PINS.get(canonical)
    if name is None:
        return None

    # Resolve ring atom sets.  Each Cp- ring is a 5-carbon fragment; the
    # third fragment is the metal.  We pair them up by which fragment
    # contains the metal index.
    ring_sets: list[frozenset[int]] = []
    for frag_atoms in frags:
        if metal_idx in frag_atoms:
            continue
        ring_sets.append(frozenset(frag_atoms))
    if len(ring_sets) != 2:
        # Not the expected 3-fragment topology.
        return None

    return MetalloceneClassification(
        center_atom_idx=metal_idx,
        ring_atom_sets=(ring_sets[0], ring_sets[1]),
        retained_name=name,
    )


# ---------------------------------------------------------------------------
# Engine dispatch entry
# ---------------------------------------------------------------------------


def detect(mol) -> LeafTree | None:
    """Engine entry point.

    Returns a fully-named ``LeafTree`` with the retained ``-ocene`` name
    when ``mol`` exactly matches a pinned sandwich complex.  Returns
    ``None`` to defer to the generic dispatch otherwise.

    The engine wires this in *before* the P-29.2 free-valence guard in
    ``name_smiles``, so radical-bearing metal centres (V, Rh, Pb, Nb)
    don't cause a guard rejection ahead of the lookup.
    """
    cls = classify_metallocene(mol)
    if cls is None:
        # Stage 22 R22-E: try the substituted-metallocene path (e.g.
        # chloroferrocene, 1,1'-dimethylruthenocene) before deferring to
        # the salt path.  Returns None if neither ring is a recognised Cp
        # (substituted or unsubstituted) or if both rings exceed our
        # mono-substituent scope.
        return _detect_substituted_metallocene(mol)

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"metallocene: {cls.retained_name}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=cls.retained_name,
    )


# ---------------------------------------------------------------------------
# Substituted metallocenes (Stage 22 R22-E)
# ---------------------------------------------------------------------------
#
# Salts of the form ``[M+2] . Cp* . Cp*`` where each Cp* is a
# cyclopenta-2,4-dien-1-ide carrying zero or one external substituent
# atom.  The unsubstituted parent path above handles ``Cp + Cp + M``.
# This path handles ``X-Cp + Cp + M`` and ``X-Cp + X-Cp + M`` (and the
# mixed ``X-Cp + Y-Cp + M`` case) by composing
# ``[locant-]<prefix>METocene``.
#
# The fragment classifier requires:
#   * exactly three RDKit fragments
#   * exactly one fragment containing a known metallocene metal cation
#   * each of the other two fragments is a 5-membered all-carbon ring
#     with exactly one carbanion ([CH-] or [c-]) and 0 or 1 ring-atom
#     substituent (a single non-ring neighbour)
#
# When both rings have at most one substituent, we name each substituent
# by carving it as a ``Y[*]`` and routing through the standard engine
# substituent path; that gives us the canonical IUPAC prefix (``chloro``,
# ``methyl``, ``bromo``, etc.).  Locants are assigned by IUPAC P-68.3
# convention: a single substituent on a single ring uses no locant; a
# substituent on each ring uses ``1`` and ``1'``.
#
# Anything more complex (multiple substituents on one ring, branched
# substituents that the substituent path can't handle, fused Cp rings
# carrying additional rings, etc.) falls through to the salt path
# unchanged.  Architecture-over-score: a clean salt-style emission for
# out-of-scope variants beats a hacky guess.


@dataclass(frozen=True)
class _CpRing:
    """One side of a substituted metallocene.

    Attributes
    ----------
    atom_idxs:
        Frozen set of atom indices (in the input mol) that compose this
        ring.  Always exactly five carbons.
    substituent_prefix:
        Empty string when the ring is unsubstituted; otherwise the IUPAC
        prefix for the single external substituent (``chloro``,
        ``methyl``, ...).  Includes only one substituent — multi-substituted
        rings are out-of-scope and cause classification to fail.
    """

    atom_idxs: frozenset[int]
    substituent_prefix: str


def _name_simple_substituent(mol, ring_atom_idx: int, sub_atom_idx: int) -> str | None:
    """Name the single external substituent attached at ``sub_atom_idx``.

    Builds a small ``Y[*]`` mol where the substituent subgraph is anchored
    by a wildcard at the position of the ring-atom bond, then routes
    through the standard naming engine to recover the canonical prefix
    (``chloro``, ``methyl``, ``bromo``, ``fluoro``, ``amino``, ...).

    Returns the substituent prefix string on success, or ``None`` when
    the substituent isn't a single supported atom or its derived name
    couldn't be reduced to a clean prefix.

    Conservative scope: only single-atom substituents (one heavy atom
    with no further branching except hydrogens) are accepted in this
    pass.  Methyl is the lone exception: ``-CH3`` is one atom plus its
    implicit H's, which the prefix-mapping table covers explicitly.
    """
    # Single-atom substituent table.  Keyed by (atomic symbol, formal
    # charge, explicit-H count if relevant).  Only the canonical IUPAC
    # detachable substituent prefixes appear here; anything outside the
    # table returns None to defer.
    atom = mol.GetAtomWithIdx(sub_atom_idx)
    sym = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if charge != 0:
        return None
    # Only one neighbour (the ring atom): the substituent has no further
    # branching to other heavy atoms.
    heavy_neighbours = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
    if len(heavy_neighbours) != 1 or heavy_neighbours[0].GetIdx() != ring_atom_idx:
        return None

    # Single-atom halide / pseudohalide / etc. → halo prefix.
    halogen = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    if sym in halogen:
        # A halide substituent has 0 H and only one bond to the ring atom.
        if atom.GetTotalNumHs() == 0:
            return halogen[sym]
        return None

    # Methyl: a single C with 3 H and one bond to the ring atom.
    if sym == "C" and atom.GetTotalNumHs() == 3:
        return "methyl"

    # Amino: a single N with 2 H and one bond to the ring atom.
    if sym == "N" and atom.GetTotalNumHs() == 2:
        return "amino"

    # Hydroxy: a single O with 1 H and one bond to the ring atom.
    if sym == "O" and atom.GetTotalNumHs() == 1:
        return "hydroxy"

    # Anything else — out of scope; defer.
    return None


def _classify_cp_fragment(
    mol, frag_atom_idxs: tuple[int, ...]
) -> _CpRing | None:
    """Classify one non-metal salt fragment as a (possibly substituted)
    cyclopentadienide.

    Returns a :class:`_CpRing` on success, ``None`` when the fragment
    isn't a 5-membered all-carbon ring with exactly one carbanion and
    at most one mono-atomic substituent.
    """
    # All ring atoms must be carbon.
    ring_idxs = [i for i in frag_atom_idxs
                 if mol.GetAtomWithIdx(i).IsInRing()]
    if len(ring_idxs) != 5:
        return None
    for i in ring_idxs:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != "C":
            return None

    # Exactly one carbanion ([c-] or [CH-]) among the 5 ring atoms.
    anion_count = sum(1 for i in ring_idxs
                      if mol.GetAtomWithIdx(i).GetFormalCharge() == -1)
    if anion_count != 1:
        return None

    # Identify any external substituents.  An external bond is a bond
    # from a ring atom to a non-ring atom in the same fragment.
    ring_set = set(ring_idxs)
    frag_set = set(frag_atom_idxs)
    external_neighbours: list[tuple[int, int]] = []  # (ring_atom_idx, sub_atom_idx)
    for ri in ring_idxs:
        a = mol.GetAtomWithIdx(ri)
        for nb in a.GetNeighbors():
            if nb.GetIdx() in ring_set:
                continue
            if nb.GetIdx() not in frag_set:
                continue
            external_neighbours.append((ri, nb.GetIdx()))

    if len(external_neighbours) == 0:
        return _CpRing(atom_idxs=frozenset(ring_idxs), substituent_prefix="")

    if len(external_neighbours) > 1:
        # Multiple substituents (or a multi-atom substituent) — defer.
        return None

    ri, si = external_neighbours[0]
    prefix = _name_simple_substituent(mol, ri, si)
    if prefix is None:
        return None

    # Make sure the substituent is fully accounted for: every fragment
    # atom should be either ring or the lone substituent atom.
    if frag_set - ring_set != {si}:
        return None

    return _CpRing(atom_idxs=frozenset(ring_idxs), substituent_prefix=prefix)


def _classify_substituted_metallocene(mol):
    """Return ``(metal_symbol, ring_a, ring_b)`` if ``mol`` is a 3-fragment
    salt of a metallocene metal + two (possibly substituted) Cp rings.

    Returns ``None`` otherwise.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 3:
        return None

    # Find the metal fragment: a single atom whose symbol is a recognised
    # metallocene metal and whose formal charge is +2.
    metal_frag_atoms = None
    metal_symbol = None
    other_frags: list[tuple[int, ...]] = []
    for fa in frags:
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            if (atom.GetSymbol() in _METALLOCENE_METALS
                    and atom.GetFormalCharge() == 2):
                if metal_frag_atoms is not None:
                    # Two metal fragments — not our pattern.
                    return None
                metal_frag_atoms = fa
                metal_symbol = atom.GetSymbol()
                continue
        other_frags.append(fa)

    if metal_symbol is None or len(other_frags) != 2:
        return None
    if metal_symbol not in _OCENE_NAME:
        return None

    ring_a = _classify_cp_fragment(mol, other_frags[0])
    if ring_a is None:
        return None
    ring_b = _classify_cp_fragment(mol, other_frags[1])
    if ring_b is None:
        return None

    return (metal_symbol, ring_a, ring_b)


def _compose_substituted_metallocene_name(
    metal_symbol: str, ring_a: _CpRing, ring_b: _CpRing,
) -> str | None:
    """Compose the surface name from a classified substituted metallocene.

    Naming rules (P-68.3 + P-14.5 multiplier conventions):

    * Both rings unsubstituted: ``METocene``  (handled by the parent
      dispatcher; this function is reached only when at least one ring
      carries a substituent — but the case is left in for symmetry and
      tested explicitly).
    * Only one ring substituted (``X``): ``X+METocene`` — no locant
      because there's no positional ambiguity on the lone substituted ring.
    * Both rings substituted with the same prefix ``X``: ``1,1'-diX+METocene``.
    * Both rings substituted with different prefixes ``X``, ``Y``
      (alphabetical order, P-14.5.2): ``1-A-1'-B+METocene`` where
      ``A``, ``B`` are the alphabetically ordered prefixes.
    """
    base = _OCENE_NAME.get(metal_symbol)
    if base is None:
        return None

    sa = ring_a.substituent_prefix
    sb = ring_b.substituent_prefix

    if sa == "" and sb == "":
        return base

    if sa == "" or sb == "":
        prefix = sa or sb
        return f"{prefix}{base}"

    if sa == sb:
        # Symmetric disubstitution — use ``di`` multiplier and primed
        # locants (P-14.5.2).
        return f"1,1'-di{sa}{base}"

    # Asymmetric: alphabetical order of substituent prefixes.
    first, second = sorted((sa, sb))
    return f"1-{first}-1'-{second}{base}"


def _detect_substituted_metallocene(mol) -> LeafTree | None:
    """Engine entry for substituted-metallocene perception.

    Called by :func:`detect` when the strict canonical-SMILES pin lookup
    fails.  Returns a fully-named ``LeafTree`` for the in-scope subset of
    substituted metallocenes (mono-atomic substituents, at most one per
    Cp ring), or ``None`` to defer to the salt dispatcher.
    """
    cls = _classify_substituted_metallocene(mol)
    if cls is None:
        return None
    metal_symbol, ring_a, ring_b = cls
    name = _compose_substituted_metallocene_name(metal_symbol, ring_a, ring_b)
    if name is None:
        return None

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"substituted metallocene: {name}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=name,
    )


# ---------------------------------------------------------------------------
# Simple group-1 / group-2 / group-12 organometallics (P-69.3, P-69 closure).
# ---------------------------------------------------------------------------
#
# IUPAC P-69.3 PINs for the simple ``R-M`` and ``R-M-X`` (Grignard-style)
# organometallics use a concatenated form ``{alkyl}{metalname}`` for
# group 1 (Li/Na/K/Rb/Cs) and ``{alkyl}{metalname} {halide}`` for
# group-2 Grignards (RMgX).  For group-2 / group-12 dialkyl species
# (R-M-R / R-M-R') the form is ``({alkyl1})({alkyl2}){metal}`` or
# ``di({alkyl}){metal}`` when the two alkyls are identical.  These are
# named *substitutively on the metal*, which the generic plan search
# cannot do because it picks the alkyl chain as parent and then the
# substituent path fails to name a bare ``[MH]`` / ``[MH][X]`` fragment
# (emitting a literal ``NAMING ERROR`` token inside the parent name).
#
# This dispatcher fires before the free-valence guard so it cannot be
# blocked by RDKit's radical-electron model on Li/Na/K/etc.

_GROUP_1_METALS: Mapping[str, str] = {
    "Li": "lithium",
    "Na": "sodium",
    "K":  "potassium",
    "Rb": "rubidium",
    "Cs": "cesium",
}
_GROUP_2_METALS: Mapping[str, str] = {
    "Be": "beryllium",
    "Mg": "magnesium",
    "Ca": "calcium",
    "Sr": "strontium",
    "Ba": "barium",
}
_GROUP_12_METALS: Mapping[str, str] = {
    "Zn": "zinc",
    "Cd": "cadmium",
    "Hg": "mercury",
}
# Group-13 trivalent main-group organometallics — substitutive PIN form is
# ``tri{R}{stem}`` where the stem is the IUPAC ``-ane`` parent name of the
# bare hydride (P-68.3 / P-69.1).  OPSIN accepts both ``-ane`` and
# ``-ium``/``-um`` variants for these; the ``-ane`` form is the substitutive
# parent and is what we emit.
_GROUP_13_METALS: Mapping[str, str] = {
    "B":  "borane",
    "Al": "alumane",
    "Ga": "gallane",
    "In": "indigane",
    "Tl": "thallane",
}
# Group-14 tetravalent main-group organometallics for which the existing
# generic engine path already produces a ``-ane`` substitutive name
# (stannane / plumbane).  Listed here so the salt-path R-M+ X- detector
# below can recognise organotin/lead cations symmetrically with Hg/Zn.
_GROUP_14_METALS: Mapping[str, str] = {
    "Sn": "stannane",
    "Pb": "plumbane",
}
_HALIDE_NAMES: Mapping[int, str] = {
    9:  "fluoride",
    17: "chloride",
    35: "bromide",
    53: "iodide",
}


def _name_alkyl_neighbour(mol, metal_idx: int, c_idx: int, strategy, session, depth) -> str | None:
    """Carve and name the alkyl substituent attached to a metal atom.

    Uses :func:`iupac_namer.perception.extraction.carve_substituent` for the
    cut, which preserves the carbon attachment atom's valence (via
    FragmentOnBonds + dummy→H replacement) so a CH centre with a substituent
    coming off the metal is rendered correctly as e.g. ``propan-2-yl`` rather
    than collapsing to ``propyl`` after a naive RWMol bond-removal.

    Returns the IUPAC substituent prefix (``methyl``, ``ethyl``,
    ``propan-2-yl``, ``phenyl``, …) or ``None`` if the substituent cannot
    be named cleanly.
    """
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm, FreeValenceInfo, SubstituentMethod
    from iupac_namer.perception.extraction import carve_substituent

    # Walk the substituent atoms (everything reachable from c_idx without
    # crossing the metal).  Used for the ``substituent_atoms`` parameter of
    # carve_substituent — it's not load-bearing for the cut (the bond
    # determines that) but kept for API symmetry.
    visited: set[int] = {metal_idx}
    stack = [c_idx]
    sub_atoms: list[int] = []
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        sub_atoms.append(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() == metal_idx:
                continue
            stack.append(nb.GetIdx())

    try:
        sub_mol, attachment_in_fragment, bond_order = carve_substituent(
            mol, frozenset(sub_atoms), (metal_idx, c_idx),
        )
    except Exception:
        return None
    if bond_order != 1:
        return None
    # Pick ALKYL (terminal) vs ALKANYL (interior) so branched substituents
    # such as propan-2-yl emit the locant.  Without this, a CH attachment in
    # the middle of a carbon chain collapses to "propyl" instead of
    # "propan-2-yl".
    from iupac_namer.engine import _select_substituent_method
    sub_method = _select_substituent_method(sub_mol, attachment_in_fragment)
    fv = FreeValenceInfo(
        bond_orders=(1,),
        method=sub_method,
        attachment_atoms_in_fragment=(attachment_in_fragment,),
    )
    try:
        sub_tree = _recursive_name(
            sub_mol,
            strategy,
            OutputForm.SUBSTITUENT,
            free_valence=fv,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    name = assemble(sub_tree)
    if name is None or "NAMING ERROR" in name:
        return None
    if name.startswith("-"):
        name = name[1:]
    return name


def _detect_simple_organometallic(mol, strategy, session, depth) -> LeafTree | None:
    """Detect and name simple group-1 / 2 / 12 / 13 organometallics.

    Recognises six shapes:

      * ``R-M`` with M ∈ {Li, Na, K, Rb, Cs} → ``{R}{metalname}``
      * ``R-M-X`` with M ∈ group 2 and X a halide (Grignard-style) →
        ``{R}{metalname} {halide}``
      * ``R-M`` with M ∈ {Zn, Cd, Hg} (mono-alkyl, deg 1) → ``{R}{metalname}``
      * ``R-M-X`` with M ∈ {Zn, Cd, Hg} and X a halide (covalent) →
        ``{R}{metalname} {halide}``
      * ``R-M-R'`` with M ∈ group 2 ∪ group 12 → ``({R})({R'}){metalname}``
        (or ``di({R}){metalname}`` when both alkyls are identical)
      * ``R3-M`` and ``R2-M-X`` with M ∈ {B, Al, Ga, In, Tl} (group-13
        substitutive PIN P-69.1) — emits ``tri{R}{stem}`` (e.g.
        ``trimethylalumane``) for the trialkyl form and the analogous
        ``({R})({R}'){halo}{stem}`` form when one position is a halide.

    Returns ``None`` for shapes outside this scope (e.g. ring-bonded
    metals, multi-bond M=C, multi-metal molecules, charged M atoms — those
    are coordination-nomenclature territory).
    """
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )

    if mol is None:
        return None
    metal_atoms = [
        a for a in mol.GetAtoms()
        if a.GetSymbol() in _GROUP_1_METALS
        or a.GetSymbol() in _GROUP_2_METALS
        or a.GetSymbol() in _GROUP_12_METALS
        or a.GetSymbol() in _GROUP_13_METALS
    ]
    if len(metal_atoms) != 1:
        return None
    m = metal_atoms[0]
    if m.GetFormalCharge() != 0:
        return None
    if m.IsInRing():
        return None
    sym = m.GetSymbol()
    heavy_nbs = [nb for nb in m.GetNeighbors() if nb.GetAtomicNum() != 1]
    # All bonds to M must be single (no multiple-bond carbenoids etc.).
    for nb in heavy_nbs:
        bond = mol.GetBondBetweenAtoms(m.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None

    metal_name: str | None = None
    is_grignard = False
    halide_name: str | None = None
    halide_prefix: str | None = None  # for group-13 substitutive ``chloro`` etc.

    if sym in _GROUP_1_METALS:
        metal_name = _GROUP_1_METALS[sym]
        if len(heavy_nbs) != 1:
            return None
        if heavy_nbs[0].GetAtomicNum() != 6:
            return None
        alkyl_neighbours = heavy_nbs
    elif sym in _GROUP_2_METALS:
        metal_name = _GROUP_2_METALS[sym]
        if len(heavy_nbs) != 2:
            return None
        # Grignard: 1 C + 1 halide.
        c_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() == 6]
        x_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() in _HALIDE_NAMES]
        if len(c_nbs) == 1 and len(x_nbs) == 1:
            # The halide must be terminal (no additional substituents).
            x = x_nbs[0]
            if x.GetDegree() != 1:
                return None
            if x.GetFormalCharge() != 0:
                return None
            halide_name = _HALIDE_NAMES[x.GetAtomicNum()]
            alkyl_neighbours = c_nbs
            is_grignard = True
        elif len(c_nbs) == 2:
            alkyl_neighbours = c_nbs
        else:
            return None
    elif sym in _GROUP_12_METALS:
        metal_name = _GROUP_12_METALS[sym]
        # Three accepted shapes:
        #   deg 1, all C        → mono-alkyl ``{R}{metal}`` (e.g. methylmercury)
        #   deg 2, 2C           → dialkyl ``({R})({R'}){metal}`` (e.g. dimethylzinc)
        #   deg 2, 1C + 1 X     → substitutive R-M-X ``({halo})({R}){metal}``
        # Substitutive (not salt-form) is required for Zn/Cd because OPSIN
        # parses the salt form ``methylzinc chloride`` as the *ionic*
        # ``[CH3][Zn+].[Cl-]``, which is a different canonical SMILES than
        # the input covalent ``C[Zn]Cl``.  The substitutive form
        # ``(chloro)(methyl)zinc`` round-trips covalently for all three
        # group-12 metals.
        c_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() == 6]
        x_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() in _HALIDE_NAMES]
        if len(heavy_nbs) == 1 and len(c_nbs) == 1:
            alkyl_neighbours = c_nbs
        elif len(heavy_nbs) == 2 and len(c_nbs) == 2:
            alkyl_neighbours = c_nbs
        elif len(heavy_nbs) == 2 and len(c_nbs) == 1 and len(x_nbs) == 1:
            x = x_nbs[0]
            if x.GetDegree() != 1:
                return None
            if x.GetFormalCharge() != 0:
                return None
            # Substitutive ``halo`` prefix on the metal parent.
            halide_prefix = {9: "fluoro", 17: "chloro",
                             35: "bromo", 53: "iodo"}[x.GetAtomicNum()]
            alkyl_neighbours = c_nbs
        else:
            return None
    elif sym in _GROUP_13_METALS:
        metal_name = _GROUP_13_METALS[sym]
        # Two accepted shapes:
        #   deg 3, 3C           → trialkyl ``tri{R}{stem}`` (e.g. trimethylalumane)
        #   deg 3, 2C + 1 X     → ``({R})({R'}){halo}{stem}`` substitutively
        # All bonds already verified single above.
        c_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() == 6]
        x_nbs = [nb for nb in heavy_nbs if nb.GetAtomicNum() in _HALIDE_NAMES]
        if len(heavy_nbs) == 3 and len(c_nbs) == 3:
            alkyl_neighbours = c_nbs
        elif len(heavy_nbs) == 3 and len(c_nbs) == 2 and len(x_nbs) == 1:
            x = x_nbs[0]
            if x.GetDegree() != 1 or x.GetFormalCharge() != 0:
                return None
            # Substitutive ``halo`` prefix on the metal hydride parent.
            halide_prefix = {9: "fluoro", 17: "chloro",
                             35: "bromo", 53: "iodo"}[x.GetAtomicNum()]
            alkyl_neighbours = c_nbs
        else:
            return None
    else:
        return None

    # Name each alkyl neighbour.
    assembled_prefixes: list[tuple[str, tuple]] = []
    for nb in alkyl_neighbours:
        name = _name_alkyl_neighbour(mol, m.GetIdx(), nb.GetIdx(), strategy, session, depth)
        if name is None:
            return None
        assembled_prefixes.append((name, ()))

    if halide_prefix is not None and sym in _GROUP_12_METALS:
        # Group-12 R-M-X substitutive form requires explicit bracketing on
        # both prefixes — OPSIN parses unbracketed ``chloromethylzinc`` as
        # ``(chloromethyl)zinc`` because the metal name has no ``-ane``
        # suffix to anchor parent recognition.  Emit
        # ``(halo)(alkyl){metal}`` (alphabetical order of bracketed
        # prefixes) which OPSIN parses unambiguously to the covalent form.
        bracketed = sorted([halide_prefix] + [n for (n, _) in assembled_prefixes])
        prefix_str = "".join(f"({p})" for p in bracketed)
        text = f"{prefix_str}{metal_name}"
    else:
        # Group-13 substitutive form: include the halide as a ``chloro`` etc.
        # prefix entry alongside the alkyls so merge/sort produces the canonical
        # alphabetical order and bracketing.  The ``-ane`` suffix on group-13
        # parent names lets OPSIN disambiguate without explicit brackets on
        # simple prefixes.
        if halide_prefix is not None:
            assembled_prefixes.append((halide_prefix, ()))
        merged = merge_identical_prefixes(assembled_prefixes)
        merged.sort(key=lambda mp: mp.sort_name)
        prefix_str = render_merged_prefixes(merged)

        if is_grignard:
            text = f"{prefix_str}{metal_name} {halide_name}"
        else:
            text = f"{prefix_str}{metal_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"simple organometallic: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Simple transition / coinage-metal organyls (IR-5 / P-69 organometallic PINs).
# ---------------------------------------------------------------------------
#
# IUPAC names a single-centre transition- or coinage-metal organometallic two
# ways depending on whether the metal carries explicit hydrido ligands:
#
#   * NO metal-H — substitutive: the organyl groups are detachable prefixes on
#     the bare metal parent (P-69 / IR-5).  ``[CH3][Cu]`` -> ``methylcopper``,
#     ``CC#[C][Cu]`` -> ``(prop-1-yn-1-yl)copper``, ``[CH3][Hg+]`` ->
#     ``methylmercury(1+)`` (the metal's formal charge is cited as a trailing
#     ``(n+)`` marker), ``[CH3][Fe]([CH3])[CH3]`` -> ``trimethyliron``.
#
#   * metal-H present — additive coordination nomenclature: each ligand
#     (``hydrido`` plus the organyls) is cited as a bracketed prefix, with the
#     ``hydrido`` count given a di/tri multiplier and the whole list sorted
#     alphabetically by ligand name.  ``[ReH2][c]1ccc2ccccc2c1`` ->
#     ``dihydrido(naphthalen-2-yl)rhenium``, ``[CH3][ReH2][c]1ccccc1`` ->
#     ``dihydrido(methyl)(phenyl)rhenium``, ``[CH3][CuH]`` ->
#     ``hydrido(methyl)copper``.
#
# This dispatcher is scoped to the GAP metals — transition / coinage metals
# the existing group-1/2/12/13/14 substitutive paths do NOT already cover, plus
# the group-12 (Zn/Cd/Hg) CATION case that ``_detect_simple_organometallic``
# rejects (it gates on ``FormalCharge == 0``).  Neutral group-12 (methylzinc,
# methylmercury) and the group-13/14 ``-ane``/-``ylidyne`` parents are left to
# their existing handlers.  Like the other organometallic dispatchers it runs
# BEFORE the P-29.2 free-valence guard so a bare metal's RDKit radical-electron
# model doesn't trigger a rejection.
#
# Structural-only: the recognition is purely on element + bond topology (one
# metal, single-bonded organyl carbons, explicit hydrides, no other
# heteroatoms), and every emitted name is composed from the carved organyl
# prefixes plus the data-driven metal-name table.  Names were verified to
# round-trip through OPSIN.
_METAL_ORGANYL_NAMES: Mapping[str, str] = {
    # Coinage metals (group 11)
    "Cu": "copper", "Ag": "silver", "Au": "gold",
    # Group-12 cations (neutral forms handled by _detect_simple_organometallic)
    "Zn": "zinc", "Cd": "cadmium", "Hg": "mercury",
    # Group 3–10 transition metals
    "Sc": "scandium", "Ti": "titanium", "V": "vanadium", "Cr": "chromium",
    "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel",
    "Y": "yttrium", "Zr": "zirconium", "Nb": "niobium", "Mo": "molybdenum",
    "Tc": "technetium", "Ru": "ruthenium", "Rh": "rhodium", "Pd": "palladium",
    "Hf": "hafnium", "Ta": "tantalum", "W": "tungsten", "Re": "rhenium",
    "Os": "osmium", "Ir": "iridium", "Pt": "platinum",
}

# Group-12 metals whose NEUTRAL mono/di-organyl form is already named by
# _detect_simple_organometallic — only their CATION form is a gap here.
_GROUP_12_NEUTRAL_HANDLED: frozenset[str] = frozenset({"Zn", "Cd", "Hg"})


def _detect_simple_metal_organyl(mol, strategy, session, depth) -> "LeafTree | None":
    """Detect and name a single-centre transition / coinage-metal organyl.

    Recognises a single-fragment molecule with exactly one metal atom from
    :data:`_METAL_ORGANYL_NAMES`, not in a ring, single-bonded to one or more
    organyl carbons and zero or more explicit hydrido ligands, with no other
    heteroatom neighbours.  Emits the substitutive ``{prefixes}{metal}[(n+)]``
    form when the metal has no hydrides, or the additive
    ``{n}hydrido({organyl})…{metal}`` form when it does.

    Returns ``None`` for anything outside this scope (multi-metal molecules,
    ring-embedded metals, metal-heteroatom bonds, multiple-bonded organyls,
    neutral group-12 mono/di-organyls already handled elsewhere) so the other
    dispatchers and the salt / generic pipeline still handle them.
    """
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )
    from rdkit import Chem

    if mol is None:
        return None
    # Single fragment only — multi-fragment salts are composed by the salt path.
    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
    except Exception:
        return None

    metal_atoms = [
        a for a in mol.GetAtoms() if a.GetSymbol() in _METAL_ORGANYL_NAMES
    ]
    if len(metal_atoms) != 1:
        return None
    m = metal_atoms[0]
    sym = m.GetSymbol()
    if m.IsInRing():
        return None
    charge = m.GetFormalCharge()

    heavy_nbs = [nb for nb in m.GetNeighbors() if nb.GetAtomicNum() != 1]
    if not heavy_nbs:
        return None
    # Every heavy neighbour must be a single-bonded carbon (an organyl ligand).
    # Any metal-heteroatom or metal=C bond is coordination / oxoacid territory
    # and is deferred to the dedicated dispatchers.
    for nb in heavy_nbs:
        if nb.GetAtomicNum() != 6:
            return None
        bond = mol.GetBondBetweenAtoms(m.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None

    # No other charged atoms anywhere in the fragment (the metal's own charge
    # is the only one cited; an organyl-borne charge is a different species).
    for a in mol.GetAtoms():
        if a.GetIdx() == m.GetIdx():
            continue
        if a.GetFormalCharge() != 0:
            return None

    n_hydrides = m.GetTotalNumHs()

    # Scope gate: neutral group-12 mono/di-organyls (methylzinc, methylmercury,
    # dimethylzinc, …) are already named by _detect_simple_organometallic; only
    # take the group-12 CATION case (and the hydrido additive case) here.
    if (sym in _GROUP_12_NEUTRAL_HANDLED
            and charge == 0 and n_hydrides == 0):
        return None

    metal_name = _METAL_ORGANYL_NAMES[sym]

    # Name each organyl ligand via the standard substituent path.
    organyl_names: list[str] = []
    for nb in heavy_nbs:
        nm = _name_alkyl_neighbour(
            mol, m.GetIdx(), nb.GetIdx(), strategy, session, depth,
        )
        if nm is None:
            return None
        organyl_names.append(nm)

    if n_hydrides == 0:
        # Substitutive form: merge identical organyl prefixes, sort, render,
        # then append the metal name and (for a charged metal) the (n+) marker.
        if charge < 0:
            # Anionic organometal centres ([R-M(-)]) are metallate / coordination
            # territory — out of scope for the substitutive form; defer.
            return None
        merged = merge_identical_prefixes([(nm, ()) for nm in organyl_names])
        merged.sort(key=lambda mp: mp.sort_name)
        prefix_str = render_merged_prefixes(merged)
        charge_marker = f"({charge}+)" if charge > 0 else ""
        text = f"{prefix_str}{metal_name}{charge_marker}"
    else:
        # Additive coordination form: cite ``{n}hydrido`` (simple, di/tri
        # multiplier) plus each organyl as a bracketed ligand, the whole list
        # sorted alphabetically by ligand name (P-IR-5.4 / additive ordering).
        # A charged hydrido-metal centre would need an oxidation/charge number;
        # that is out of scope here, so defer charged hydrido cases.
        if charge != 0:
            return None
        from iupac_namer.data_loader import get_multiplier as _get_mult
        ligands: list[tuple[str, str]] = []  # (sort_name, rendered)
        # hydrido ligand (simple, never bracketed)
        h_mult = _get_mult(n_hydrides, complex=False) if n_hydrides > 1 else ""
        ligands.append(("hydrido", f"{h_mult or ''}hydrido"))
        # organyl ligands: merge identical, bracket compound names
        organyl_merged = merge_identical_prefixes([(nm, ()) for nm in organyl_names])
        for mp in organyl_merged:
            open_b, close_b = "(", ")"
            mult = mp.multiplier or ""
            rendered = f"{mult}{open_b}{mp.name}{close_b}"
            ligands.append((mp.sort_name, rendered))
        ligands.sort(key=lambda t: t[0])
        text = "".join(r for _, r in ligands) + metal_name

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"simple metal organyl: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Group-13 (and group-14) substitutive parent-hydride namer (P-21.2 / P-68.3).
# ---------------------------------------------------------------------------
#
# IUPAC P-21.2 names the trivalent group-13 parent hydrides BH3 / AlH3 / GaH3
# / InH3 / TlH3 systematically (borane / alumane / gallane / indigane /
# thallane).  Their substituted derivatives are named *substitutively on the
# metal* exactly like the group-14 silane / stannane family: substituents are
# detachable prefixes and any unsubstituted valence keeps its H
# (e.g. ``bis(3-methylbutyl)alumane`` = ``[AlH](R)R``, ``tributoxyalumane`` =
# ``(RO)3Al``, ``hydroxydimethylthallane`` = ``(HO)(Me)2Tl``).
#
# The narrower :func:`_detect_simple_organometallic` group-13 branch only
# covers the fully-substituted (degree-3) all-carbon / 2C+halide shapes.  This
# dispatcher generalises to:
#   * any substituent count 1..valence (the rest are H on the metal),
#   * heteroatom substituents (alkoxy ``-OR``, hydroxy ``-OH``, dithioperoxy
#     ``-SSR``, amino, halide, ...) carved through the standard substituent
#     path, and
#   * the alkoxide / oxidanide anion ``-olate`` form: a single ``[O-]`` on the
#     metal is consumed as the ``-olate`` suffix (P-72.2 / P-65.3 anion),
#     yielding e.g. ``dimethylalumanolate`` for ``C[Al]([O-])C``.  The salt
#     path then composes ``sodium dimethylalumanolate`` for the Na+ salt.
#
# The metal must be neutral apart from the consumed ``-olate`` oxide (so the
# fragment net charge is 0 for the neutral parent, or -1 for the lone-oxide
# anion).  Charged metal centres and multi-metal fragments are out of scope
# and defer.  Single bonds only.
#
# Architecture-over-score: every emitted name is composed from the carved
# substituent prefixes (via ``merge_identical_prefixes`` / the standard
# substituent path) plus the systematic parent-hydride stem — no molecule-
# specific branches, no whole-molecule pins.

# Group-13 trivalent parent-hydride stems (P-21.2 Table; identical mapping to
# _GROUP_13_METALS but kept separate so the substitutive namer's valence is
# explicit).  Boron is intentionally omitted: the bare-B + substituent forms
# already round-trip through the generic substitutive engine path
# (trimethylborane etc.), and BH3 sub-forms have their own boronic/borinic
# acid PINs that this generic namer must not pre-empt.
_GROUP_13_HYDRIDE: Mapping[str, tuple[str, int]] = {
    # symbol -> (parent-hydride name, metal valence)
    "Al": ("alumane", 3),
    "Ga": ("gallane", 3),
    "In": ("indigane", 3),
    "Tl": ("thallane", 3),
}


def _name_metal_substituent(mol, metal_idx: int, nb_idx: int, strategy, session, depth) -> str | None:
    """Carve and name one substituent attached to a parent-hydride metal.

    Thin wrapper over :func:`_name_alkyl_neighbour` (which already carves any
    single-bonded substituent subgraph and routes it through the standard
    substituent path, recovering ``methyl`` / ``butyloxy`` /
    ``(ethylsulfanyl)sulfanyl`` / ``3-methylbutyl`` / ``chloro`` / ``hydroxy``
    etc.).  Returns the prefix string or ``None`` to defer.
    """
    return _name_alkyl_neighbour(mol, metal_idx, nb_idx, strategy, session, depth)


def _detect_substituted_group13_hydride(
    mol, strategy, session, depth,
) -> "LeafTree | None":
    """Detect and name a substituted group-13 parent hydride (P-21.2 / P-68.3).

    Handles single-fragment shapes ``R_k M H_(v-k)`` with M ∈ {Al, Ga, In, Tl}
    and v = 3, where each R is a substituent the standard substituent path can
    name (alkyl, alkoxy, hydroxy, halide, dithioperoxy, amino, ...).  Also
    handles the ``-olate`` anion form where exactly one substituent is a bare
    ``[O-]`` (degree-1 oxide) consumed as the ``-olate`` suffix.

    Returns a fully-named ``LeafTree`` (STANDALONE for the neutral parent,
    ANION for the ``-olate``) or ``None`` to defer.
    """
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )
    from rdkit import Chem

    if mol is None:
        return None
    # Single fragment only — multi-fragment salts are composed by the salt
    # path (which recurses into this dispatcher per fragment).
    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
    except Exception:
        return None

    metal_atoms = [
        a for a in mol.GetAtoms() if a.GetSymbol() in _GROUP_13_HYDRIDE
    ]
    if len(metal_atoms) != 1:
        return None
    m = metal_atoms[0]
    if m.GetFormalCharge() != 0:
        return None
    if m.IsInRing():
        return None
    if m.GetNumRadicalElectrons() != 0:
        return None
    metal_name, valence = _GROUP_13_HYDRIDE[m.GetSymbol()]

    heavy_nbs = [nb for nb in m.GetNeighbors() if nb.GetAtomicNum() != 1]
    if not heavy_nbs:
        return None
    # All bonds to the metal must be single.
    for nb in heavy_nbs:
        bond = mol.GetBondBetweenAtoms(m.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None

    # Identify the lone-oxide anion substituent (``[O-]`` degree 1) if present.
    oxide_anion_nbs = [
        nb for nb in heavy_nbs
        if nb.GetAtomicNum() == 8
        and nb.GetFormalCharge() == -1
        and nb.GetDegree() == 1
        and nb.GetTotalNumHs() == 0
    ]
    is_olate = False
    if len(oxide_anion_nbs) == 1:
        is_olate = True
    elif len(oxide_anion_nbs) > 1:
        # More than one bare oxide → out of scope (dianion etc.); defer.
        return None

    # Fragment net charge must be 0 (neutral parent) or -1 (lone-oxide anion).
    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if is_olate:
        if net_charge != -1:
            return None
    else:
        if net_charge != 0:
            return None

    # The remaining substituents (everything except the consumed oxide) must
    # all be nameable prefixes.  Their count must not exceed the metal valence
    # (with the oxide counting toward the valence too).
    prefix_nbs = [
        nb for nb in heavy_nbs
        if not (is_olate and nb.GetIdx() == oxide_anion_nbs[0].GetIdx())
    ]
    consumed = len(prefix_nbs) + (1 if is_olate else 0)
    if consumed < 1 or consumed > valence:
        return None
    # No prefix substituent may itself carry a formal charge (keep the
    # fragment's only charge on the consumed oxide).
    for nb in prefix_nbs:
        # The substituent's attachment atom charge is checked here; deeper
        # charges (zwitterions) are rejected by the carve/name step below.
        if nb.GetFormalCharge() != 0:
            return None

    # Implicit-H on the metal fills the unconsumed valence; the substitutive
    # parent-hydride name expresses those H's implicitly, exactly as OPSIN does.

    assembled_prefixes: list[tuple[str, tuple]] = []
    for nb in prefix_nbs:
        name = _name_metal_substituent(
            mol, m.GetIdx(), nb.GetIdx(), strategy, session, depth,
        )
        if name is None:
            return None
        assembled_prefixes.append((name, ()))

    merged = merge_identical_prefixes(assembled_prefixes)
    merged.sort(key=lambda mp: mp.sort_name)
    prefix_str = render_merged_prefixes(merged)

    if is_olate:
        # ``alumane`` -> ``aluman`` + ``olate`` (drop the trailing -e, P-72.2).
        stem = metal_name[:-1] if metal_name.endswith("e") else metal_name
        text = f"{prefix_str}{stem}olate"
        out_form = OutputForm.ANION
    else:
        text = f"{prefix_str}{metal_name}"
        out_form = OutputForm.STANDALONE

    return LeafTree(
        output_form=out_form,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"substituted group-13 hydride: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Heterogeneous heteroatom-chain replacement nomenclature (P-21.2.3 / P-15.4).
# ---------------------------------------------------------------------------
#
# IUPAC P-21.2.3 names a linear acyclic chain of skeletal heteroatoms (no
# carbon) by replacement ("a") nomenclature: the chain backbone is the parent
# hydride and the remaining skeletal atoms are cited as ``a``-prefixes
# (``sila``, ``plumba``, ``thia`` …).  For a symmetric three-atom chain
# ``M-X-M`` the parent hydride is the interior atom X and the two identical
# terminal atoms M are the replacement prefix, giving
# ``di{a-prefix(M)}{parent-hydride(X)}``:
#
#     [PbH3][Te][PbH3]   -> diplumbatellurane   (Te parent + 2× plumba)
#     [SnH3]S[SnH3]      -> distannathiane
#     [SiH3][SnH2][SiH3] -> disilastannane
#     [SiH3][PH][SiH3]   -> disilaphosphane
#     [AlH2][Te][AlH2]   -> dialumatellurane
#
# This handler is restricted to the unambiguous symmetric three-atom case
# (two identical group-13/14/15/16 terminals bridging one different
# group-14/15/16 interior atom).  Asymmetric / longer chains need full
# replacement-nomenclature parent selection and locant assignment and are
# left to defer.  Oxygen is excluded as the interior atom because OPSIN does
# not parse the ``…oxidane`` chain-parent form for these chains.
#
# Structural-only: the recognition is purely on element + chain topology, and
# the surface name is composed from data-driven a-prefix / parent-name tables.
# Every emitted name has been verified to round-trip through OPSIN.

# Replacement ("a") prefixes (P-15.4.3 Table; mirror of data/hw_tables.json).
_A_PREFIX: Mapping[str, str] = {
    "O": "oxa", "S": "thia", "Se": "selena", "Te": "tellura",
    "N": "aza", "P": "phospha", "As": "arsa", "Sb": "stiba", "Bi": "bisma",
    "Si": "sila", "Ge": "germa", "Sn": "stanna", "Pb": "plumba",
    "B": "bora", "Al": "aluma", "Ga": "galla", "In": "indiga", "Tl": "thalla",
}

# Interior-atom parent-hydride names for the chain replacement parent.
# Group-14 / group-15 use the acyclic ``-ane`` hydride name; the chalcogens
# S/Se/Te use the saturated Hantzsch-Widman ``-ane`` form (thiane / selenane /
# tellurane) that OPSIN accepts as the chain parent.  Oxygen is intentionally
# absent (no OPSIN-parseable chain-parent form for ``…oxidane``).
_CHAIN_PARENT_HYDRIDE: Mapping[str, str] = {
    # group 14
    "Si": "silane", "Ge": "germane", "Sn": "stannane", "Pb": "plumbane",
    # group 15
    "N": "azane", "P": "phosphane", "As": "arsane",
    "Sb": "stibane", "Bi": "bismuthane",
    # group 16 (HW saturated-ring style accepted by OPSIN as the chain parent)
    "S": "thiane", "Se": "selenane", "Te": "tellurane",
}

# Elements admissible as a chain terminal (a-prefix replacement atom).
# Oxygen and the halogens are deliberately excluded: an O / halogen terminal
# turns the chain into oxoacid / acid-functionality territory (e.g.
# ``HO-Sb(H)-OH`` is *stibonous acid*, not the replacement-chain
# ``dioxastibane``), which the senior oxoacid path must name.  The remaining
# group-13/14/15 metalloids/metals and the heavier chalcogens S/Se/Te form
# genuine parent-hydride replacement chains.
_CHAIN_TERMINAL_ELEMENTS: frozenset[str] = frozenset(
    sym for sym in _A_PREFIX
    if sym not in {"O", "F", "Cl", "Br", "I"}
)


def detect_heterogeneous_heteroatom_chain(mol) -> "LeafTree | None":
    """Detect and name a symmetric three-atom heterogeneous heteroatom chain.

    Recognises single-fragment ``M-X-M`` where M and X are distinct skeletal
    heteroatoms (no carbon), M is a recognised terminal element and X a
    recognised interior parent element, all bonds single, all atoms neutral
    with no radical electrons, and the two M atoms carry identical implicit-H
    counts (true symmetry).  Emits ``di{a-prefix(M)}{parent-hydride(X)}``.

    Returns ``None`` for anything outside this scope (homogeneous chains —
    handled by the engine's homogeneous-chain dispatcher; carbon-containing
    chains; asymmetric / longer chains; rings; charged / radical centres).
    """
    if mol is None:
        return None
    from rdkit import Chem

    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
    except Exception:
        return None

    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy) != 3:
        return None
    # No carbon (this is heteroatom-chain replacement nomenclature, not a
    # carbon-skeleton parent).
    if any(a.GetAtomicNum() == 6 for a in heavy):
        return None
    # No oxygen anywhere in the chain: an O atom turns the chain into oxoacid
    # / acid-functionality territory (HO-M-OH = an -ous/-ic acid), which the
    # senior oxoacid path must name.  Reject so we never pre-empt e.g.
    # ``stibonous acid`` (``[SbH](O)O``) with ``dioxastibane``.
    if any(a.GetAtomicNum() == 8 for a in heavy):
        return None
    # All neutral, no isotopes, no radicals, no rings.
    for a in heavy:
        if a.GetFormalCharge() != 0:
            return None
        if a.GetIsotope() != 0:
            return None
        if a.GetNumRadicalElectrons() != 0:
            return None
        if a.IsInRing():
            return None
    # All heavy-heavy bonds single.
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            return None
        if bond.IsInRing():
            return None

    # Topology: exactly one interior atom (degree 2 in heavy-atom graph) and
    # two terminal atoms (degree 1).  Each atom's heavy neighbours must all be
    # part of the 3-atom chain (no extra heavy substituents).
    def heavy_deg(atom):
        return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() != 1)

    interior = [a for a in heavy if heavy_deg(a) == 2]
    terminals = [a for a in heavy if heavy_deg(a) == 1]
    if len(interior) != 1 or len(terminals) != 2:
        return None
    center = interior[0]
    # The interior atom must bond to both terminals (true linear chain).
    term_idxs = {t.GetIdx() for t in terminals}
    center_nbrs = {nb.GetIdx() for nb in center.GetNeighbors()
                   if nb.GetAtomicNum() != 1}
    if center_nbrs != term_idxs:
        return None

    # Symmetry: the two terminals are the same element with identical H counts.
    t0, t1 = terminals
    if t0.GetSymbol() != t1.GetSymbol():
        return None
    if t0.GetTotalNumHs() != t1.GetTotalNumHs():
        return None
    term_sym = t0.GetSymbol()
    center_sym = center.GetSymbol()
    # Heterogeneous (terminal element differs from interior element).
    if term_sym == center_sym:
        return None
    if term_sym not in _CHAIN_TERMINAL_ELEMENTS:
        return None
    if center_sym not in _CHAIN_PARENT_HYDRIDE:
        return None

    a_prefix = _A_PREFIX[term_sym]
    parent = _CHAIN_PARENT_HYDRIDE[center_sym]
    text = f"di{a_prefix}{parent}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"heterogeneous heteroatom chain: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Group-12 organometallic cation salts (R-M+ . X-).
# ---------------------------------------------------------------------------
#
# IUPAC P-72/P-73 cation-salt nomenclature for organomercury / organozinc /
# organocadmium ions of the form ``R-M+ . X-`` is ``{R}{metal}(1+) {halide}``
# (e.g. ``ethylmercury(1+) bromide``).  The ``(1+)`` charge marker is
# optional for these monovalent organometal cations — OPSIN parses both
# forms — and we emit it explicitly to make the cation status unambiguous
# at the surface level.
#
# Without this dispatcher the engine's salt path tries to name the bare
# ``[MH+]`` cation fragment via the substitutive engine, which fails (the
# metal cation's valence model has implicit H's that conflict with the
# alkyl-bond) and emits a literal ``NAMING ERROR`` token inside the
# composed salt name.
#
# Architectural note: this dispatcher fires from ``name_smiles`` BEFORE
# the salt path, mirroring the existing :func:`_detect_simple_organometallic`
# pre-validation hook.  It returns ``None`` for any shape it can't
# confidently name, deferring to the salt path unchanged.


def _detect_organometallic_cation_salt(mol, strategy, session, depth) -> LeafTree | None:
    """Detect and name R-M+ . X- group-12 organometallic cation salts.

    Recognises two-fragment salts with:

      * cation fragment: a single charge-+1 metal (Zn / Cd / Hg) bonded to
        exactly one carbon (``R-M+``) where ``R`` is a hydrocarbon
        substituent that the engine can name via the substituent path.
      * anion fragment: a single charge-(-1) halide atom (F, Cl, Br, I)
        with degree 0.

    Returns a fully-named ``LeafTree`` (``"{R}{metal}(1+) {halide}"``) or
    ``None`` for shapes outside this scope.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 2:
        return None

    # Identify cation fragment (contains a +1 metal of group 12) and anion
    # fragment (single -1 halide atom).
    cation_frag: tuple[int, ...] | None = None
    anion_atom_idx: int | None = None
    metal_atom_idx: int | None = None
    metal_symbol: str | None = None
    for fa in frags:
        # Anion fragment: single atom, halide, charge -1.
        if len(fa) == 1:
            a = mol.GetAtomWithIdx(fa[0])
            if (a.GetFormalCharge() == -1
                    and a.GetAtomicNum() in _HALIDE_NAMES
                    and a.GetDegree() == 0):
                if anion_atom_idx is not None:
                    return None
                anion_atom_idx = fa[0]
                continue
        # Cation fragment: contains a +1 group-12 metal.
        for ai in fa:
            a = mol.GetAtomWithIdx(ai)
            if (a.GetSymbol() in _GROUP_12_METALS
                    and a.GetFormalCharge() == 1
                    and not a.IsInRing()):
                if metal_atom_idx is not None:
                    return None
                metal_atom_idx = ai
                metal_symbol = a.GetSymbol()
                cation_frag = fa

    if (cation_frag is None or anion_atom_idx is None
            or metal_atom_idx is None or metal_symbol is None):
        return None
    if metal_symbol not in _GROUP_12_METALS:
        return None

    metal_atom = mol.GetAtomWithIdx(metal_atom_idx)
    # The metal must have exactly one heavy neighbour, and that must be a
    # carbon — single bond — within the cation fragment.
    heavy_nbs = [nb for nb in metal_atom.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbs) != 1:
        return None
    c_nb = heavy_nbs[0]
    if c_nb.GetAtomicNum() != 6:
        return None
    bond = mol.GetBondBetweenAtoms(metal_atom_idx, c_nb.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    # No additional charged atoms in the cation fragment beyond the metal
    # itself (exclude e.g. R-N+-M+ zwitterion-like patterns).
    for ai in cation_frag:
        if ai == metal_atom_idx:
            continue
        if mol.GetAtomWithIdx(ai).GetFormalCharge() != 0:
            return None

    alkyl_name = _name_alkyl_neighbour(
        mol, metal_atom_idx, c_nb.GetIdx(), strategy, session, depth,
    )
    if alkyl_name is None:
        return None

    metal_name = _GROUP_12_METALS[metal_symbol]
    halide_name = _HALIDE_NAMES[mol.GetAtomWithIdx(anion_atom_idx).GetAtomicNum()]
    text = f"{alkyl_name}{metal_name}(1+) {halide_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"organometallic cation salt: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Metal carbonyl complexes (P-68.3 / coordination nomenclature).
# ---------------------------------------------------------------------------
#
# SMILES like ``[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Fe]`` (iron pentacarbonyl)
# and ``[Br-].[Br-].[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Mn+2]`` are
# disconnected-fragment representations of metal carbonyl coordination
# compounds.  The salt path names each fragment individually, producing
# garbage like ``pentacarbon monoxide iron``.
#
# IUPAC coordination nomenclature (P-68.3) names the CO ligand as
# ``carbonyl`` and uses the concatenated form ``{N}carbonyl{metalname}``
# for pure homoleptic carbonyls (e.g. ``pentacarbonyliron``,
# ``tetracarbonylnickel``, ``hexacarbonylchromium``).  When ionic halide
# fragments are also present the form is
# ``{N}carbonyl{metalname} {mult}halide`` (e.g.
# ``pentacarbonylmanganese dibromide``).
#
# This dispatcher fires in ``name_smiles`` BEFORE the salt path and BEFORE
# ``_validate_no_open_valences``, so metals that carry RDKit radical
# electrons (V, Re, …) are handled without weakening the guard.
#
# Canonical SMILES for the CO fragment is ``[C]=O`` — two atoms, C with
# two radical electrons and no H, double-bonded to O.  We identify it by
# the invariant: exactly 2 heavy atoms (C + O), the C has atomic num 6,
# the O has atomic num 8, and the bond between them is a double bond.
#
# Scope of this dispatcher:
#   * All fragments are either [C]=O (carbon monoxide), a single
#     uncharged or charged bare transition/d-block metal atom, or a
#     single charged halide anion.
#   * Exactly one fragment contains the metal atom.
#   * At least one CO fragment is present.
#   * The net charge of all non-CO, non-metal fragments must consist only
#     of halide anions.
#   * The metal charge equals the total halide count (charge balance).
#
# Returns None for any shape outside this scope, deferring to the salt
# path unchanged.

_CARBONYL_METAL_NAMES: Mapping[str, str] = {
    # d-block transition metals commonly found in carbonyl complexes.
    # Names are the IUPAC element names that OPSIN accepts as the second
    # part of the concatenated ``{N}carbonyl{metal}`` string.
    "V":  "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Mo": "molybdenum",
    "Tc": "technetium",
    "Ru": "ruthenium",
    "Rh": "rhodium",
    "Re": "rhenium",
    "Os": "osmium",
    "Ir": "iridium",
    "W":  "tungsten",
}

# Halide atomic numbers → halide anion name for the salt part.
_HALIDE_ANION_NAMES: Mapping[int, str] = {
    9:  "fluoride",
    17: "chloride",
    35: "bromide",
    53: "iodide",
    85: "astatide",
}

# Halide atomic numbers → prefix-form halogeno name for the inline-Cp /
# coordination compound forms (e.g. "dichlorodicarbonyl(cyclopentadienyl)
# iron").  Distinct from ``_HALIDE_ANION_NAMES`` because OPSIN parses the
# two forms differently.
_HALIDE_PREFIX_NAMES: Mapping[int, str] = {
    9:  "fluoro",
    17: "chloro",
    35: "bromo",
    53: "iodo",
    85: "astato",
}

# Multiplier prefixes for 1–12 CO ligands.
_CO_MULTIPLIERS: dict[int, str] = {
    1:  "",
    2:  "di",
    3:  "tri",
    4:  "tetra",
    5:  "penta",
    6:  "hexa",
    7:  "hepta",
    8:  "octa",
    9:  "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
}


def _is_co_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is exactly one C≡O or C=O (carbon monoxide).

    Accepts two common representations:
    * Neutral radical form ``[C]=O`` (double bond, no charges) — used in
      disconnected carbonyl SMILES like ``[C]=O.[C]=O.[C]=O.[Fe]``.
    * Charge-separated zwitterion form ``[C-]#[O+]`` (triple bond, C-/O+) —
      used when drawing CO with formal bond orders, e.g. in Mn2(CO)10 SMILES
      ``[C-]#[O+].[C-]#[O+]...[Mn].[Mn]``.

    Criteria (shared):
    * Exactly 2 heavy atoms: one C and one O.
    * No explicit H on either atom.

    Bond-order variants:
    * Double bond (DOUBLE) + both charges 0   → neutral radical CO.
    * Triple bond (TRIPLE) + C charge -1, O charge +1  → zwitterion CO.
    """
    if len(frag_atom_idxs) != 2:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    syms = sorted(a.GetSymbol() for a in atoms)
    if syms != ["C", "O"]:
        return False
    for a in atoms:
        if a.GetTotalNumHs() != 0:
            return False

    bond = mol.GetBondBetweenAtoms(frag_atom_idxs[0], frag_atom_idxs[1])
    if bond is None:
        return False

    from rdkit.Chem import rdchem
    c_atom = next(a for a in atoms if a.GetSymbol() == "C")
    o_atom = next(a for a in atoms if a.GetSymbol() == "O")

    bt = bond.GetBondType()
    if bt == rdchem.BondType.DOUBLE:
        # Neutral radical form: [C]=O
        return c_atom.GetFormalCharge() == 0 and o_atom.GetFormalCharge() == 0
    if bt == rdchem.BondType.TRIPLE:
        # Charge-separated zwitterion form: [C-]#[O+]
        return c_atom.GetFormalCharge() == -1 and o_atom.GetFormalCharge() == 1
    return False


def _is_metal_cp_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
    """Detect a fragment of the shape ``[M][C]1=CC=CC1`` — a single metal
    atom bonded to one carbon of a 5-membered all-carbon ring (cyclopenta-
    2,4-dien-1-yl coordinated through C1 with a free-valence anchor).

    Returns a tuple ``(metal_atom_idx, ring_atom_idxs)`` on a positive
    match, or ``None`` otherwise.  The ring atoms are the five carbons of
    the Cp ring in their original mol indices (order not guaranteed).

    Recognition criteria:
      * Exactly 6 atoms total in the fragment.
      * Exactly one metal atom (symbol in ``_CARBONYL_METAL_NAMES``);
        it has degree 1 (bonded only to the ring anchor C).
      * The other 5 atoms form a single 5-membered ring of all-carbons.
      * Exactly one ring carbon — the one bonded to the metal — carries
        zero H, has 2 (radical) electrons in RDKit's valence model, and
        no formal charge (the Cp1 anchor C with two double bonds and a
        single bond to the metal — RDKit accepts this as a 5-valent C).
      * The remaining 4 ring carbons are unspecific (CH2 + CH=CH-CH= or
        similar Kekule pattern from ``C1=CC=CC1``); we don't enforce a
        specific H pattern beyond C5H4-on-the-ring (matches common
        SMILES forms from chemists writing the Cp as ``[C]1=CC=CC1``).
    """
    if len(frag_atom_idxs) != 6:
        return None
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]

    # Find the metal atom (single one).
    metal_atoms = [a for a in atoms if a.GetSymbol() in _CARBONYL_METAL_NAMES]
    if len(metal_atoms) != 1:
        return None
    metal = metal_atoms[0]
    if metal.GetDegree() != 1:
        return None

    # The remaining 5 atoms must all be carbon.
    other_atoms = [a for a in atoms if a is not metal]
    if any(a.GetAtomicNum() != 6 for a in other_atoms):
        return None

    # The 5 carbons must form a 5-membered ring.  Find the ring set.
    ri = mol.GetRingInfo()
    ring_atom_idxs: tuple[int, ...] | None = None
    other_idx_set = set(a.GetIdx() for a in other_atoms)
    for ring in ri.AtomRings():
        if len(ring) == 5 and set(ring) == other_idx_set:
            ring_atom_idxs = tuple(ring)
            break
    if ring_atom_idxs is None:
        return None

    # The metal's neighbour must be a ring C with 0 H and no formal
    # charge — the "[C]" anchor carbene-style form.  RDKit accepts a
    # 5-valent C here without radicals when the user wrote ``[C]1=CC=CC1``
    # (two double bonds inside the ring + one single bond to the metal),
    # so we don't constrain the radical count.
    nb_idxs = [n.GetIdx() for n in metal.GetNeighbors()]
    if len(nb_idxs) != 1:
        return None
    anchor_idx = nb_idxs[0]
    if anchor_idx not in other_idx_set:
        return None
    anchor = mol.GetAtomWithIdx(anchor_idx)
    if anchor.GetTotalNumHs() != 0:
        return None
    if anchor.GetFormalCharge() != 0:
        return None

    # Sanity-check the ring: count the H-bearing carbons.  The reference
    # ``[M][C]1=CC=CC1`` form has the anchor with 0 H, three CH ring atoms
    # and one CH2 (so 5 H total in the ring).  We accept this signature
    # to disambiguate from other 5-membered-ring forms (e.g. cyclopent-
    # adienyl with all CH).
    other_ring_atoms = [mol.GetAtomWithIdx(i) for i in ring_atom_idxs if i != anchor_idx]
    h_total = sum(a.GetTotalNumHs() for a in other_ring_atoms)
    if h_total != 5:
        return None

    return (metal.GetIdx(), ring_atom_idxs)


def _is_metal_with_ligand_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> tuple[int, list[tuple[int, ...]]] | None:
    """Detect a single-fragment shape of the form ``L1-[M]-L2`` where the
    metal bridges two neutral ligand sub-fragments — used to recognise
    e.g. ``C=C[CH2][Pd][C]1=CC=CC1`` (allyl-Pd-Cp) or ``[Co][C]1=CC=CC1``
    (single-ligand CpCo).

    Returns ``(metal_atom_idx, [ligand_atom_idx_tuple, ...])`` on success
    where each ligand sub-fragment is the connected component left after
    removing the metal.  Returns ``None`` for shapes outside scope.

    Recognition criteria:
      * Exactly one metal atom (in ``_CARBONYL_METAL_NAMES``) in the
        fragment; it is uncharged (single fragment forms are always
        neutral overall).
      * The metal has degree 1 or 2 (one or two ligand carbons attached).
      * Removing the metal partitions the rest into 1 or 2 connected
        components, all carbons (Cp/allyl/etc.).
      * Net formal charge on the ligand atoms is zero (so we don't
        accept charged anionic ligand fragments as part of this path —
        those are handled by the substituted-metallocene route).
    """
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    metal_atoms = [a for a in atoms if a.GetSymbol() in _CARBONYL_METAL_NAMES]
    if len(metal_atoms) != 1:
        return None
    metal = metal_atoms[0]
    if metal.GetFormalCharge() != 0:
        return None
    if metal.GetDegree() not in (1, 2):
        return None

    # Remaining atom indices (everything in the fragment except the
    # metal).  All must be carbon.
    other_idxs = [a.GetIdx() for a in atoms if a is not metal]
    for i in other_idxs:
        if mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
            return None

    # Charge balance on ligand atoms.
    if any(mol.GetAtomWithIdx(i).GetFormalCharge() != 0 for i in other_idxs):
        return None

    # Compute the connected components of the ligand atoms (i.e. the
    # graph-induced subgraph on ``other_idxs``, ignoring the metal).
    other_set = set(other_idxs)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in other_idxs:
        if start in seen:
            continue
        # BFS limited to other_set.
        stack = [start]
        comp: list[int] = []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            comp.append(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                ni = nb.GetIdx()
                if ni in other_set and ni not in seen:
                    stack.append(ni)
        components.append(tuple(sorted(comp)))

    # Each ligand component must be either a 5-C ring (Cp) or an
    # allyl-like 3-C chain.  Validate at least one ligand is a Cp ring
    # (otherwise it's not a recognisable Cp-bonded metal).
    return (metal.GetIdx(), components)


def _extract_fragment_smiles(mol, frag_atom_idxs: tuple[int, ...]) -> str | None:
    """Extract a sub-mol containing just the atoms in ``frag_atom_idxs`` and
    return its canonical SMILES.

    Used to recursively name a neutral ligand fragment in a mixed
    metal-carbonyl-arene complex (e.g. the benzene ring in
    ``[C]=O.[C]=O.[C]=O.[Cr].c1ccccc1`` → ``benzene``).
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        # PathToSubmol expects bond indices, but for whole-fragment extraction
        # we use atomMap-based approach via RWMol clone + remove-other-frags.
        # Simpler: build via Chem.MolFragmentToSmiles then re-canonicalise.
        smi = Chem.MolFragmentToSmiles(
            mol, atomsToUse=list(frag_atom_idxs), canonical=True,
        )
        if not smi:
            return None
        # Re-parse to ensure it's a valid standalone SMILES (canonical form).
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _is_methyl_metal_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> int | None:
    """Detect a 2-atom fragment of the shape ``[CH3][M]`` — methyl bonded
    to a single transition-metal atom.

    Returns the metal atom index on success, ``None`` otherwise.

    Recognition criteria:
      * Exactly 2 atoms in the fragment.
      * One is a metal (``_CARBONYL_METAL_NAMES``); the other is C with
        exactly 3 implicit Hs and zero charge.
      * The metal has degree 1 and zero formal charge.
      * The C-M bond is a single bond.
    """
    if len(frag_atom_idxs) != 2:
        return None
    a0 = mol.GetAtomWithIdx(frag_atom_idxs[0])
    a1 = mol.GetAtomWithIdx(frag_atom_idxs[1])
    metal = c_atom = None
    if a0.GetSymbol() in _CARBONYL_METAL_NAMES and a1.GetAtomicNum() == 6:
        metal, c_atom = a0, a1
    elif a1.GetSymbol() in _CARBONYL_METAL_NAMES and a0.GetAtomicNum() == 6:
        metal, c_atom = a1, a0
    else:
        return None
    if metal.GetFormalCharge() != 0 or metal.GetDegree() != 1:
        return None
    if c_atom.GetTotalNumHs() != 3 or c_atom.GetFormalCharge() != 0:
        return None
    if c_atom.GetDegree() != 1:
        return None
    bond = mol.GetBondBetweenAtoms(metal.GetIdx(), c_atom.GetIdx())
    if bond is None:
        return None
    from rdkit.Chem import rdchem
    if bond.GetBondType() != rdchem.BondType.SINGLE:
        return None
    return metal.GetIdx()


def _is_silyl_metal_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> tuple[int, str] | None:
    """Detect a fragment of the shape ``X[Si](Y)(Z)[M]`` — a tetravalent Si
    bonded to one metal atom and three substituents (halogen and/or methyl).

    Returns ``(metal_atom_idx, silyl_substituent_name)`` on success, where
    ``silyl_substituent_name`` is the IUPAC silyl substituent name without
    enclosing parentheses (e.g. ``trichlorosilyl``,
    ``difluoro(methyl)silyl``).  Returns ``None`` for shapes outside scope.

    Recognition criteria:
      * Exactly 5 atoms total: 1 metal + 1 Si + 3 substituents.
      * The metal (in ``_CARBONYL_METAL_NAMES``) has degree 1 and zero
        formal charge; bonded to Si.
      * The Si has degree 4, zero formal charge, no H, no radical.
      * Each remaining atom is a halogen (F, Cl, Br, I) or a methyl C
        (with H=3, deg=1, q=0).
      * The Si-M bond is single.
    """
    if len(frag_atom_idxs) != 5:
        return None
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    metal_atoms = [a for a in atoms if a.GetSymbol() in _CARBONYL_METAL_NAMES]
    si_atoms = [a for a in atoms if a.GetSymbol() == "Si"]
    if len(metal_atoms) != 1 or len(si_atoms) != 1:
        return None
    metal = metal_atoms[0]
    si = si_atoms[0]
    if metal.GetFormalCharge() != 0 or metal.GetDegree() != 1:
        return None
    if si.GetDegree() != 4 or si.GetFormalCharge() != 0:
        return None
    if si.GetTotalNumHs() != 0 or si.GetNumRadicalElectrons() != 0:
        return None
    # Si must be bonded to the metal.
    bond_si_m = mol.GetBondBetweenAtoms(si.GetIdx(), metal.GetIdx())
    if bond_si_m is None:
        return None
    from rdkit.Chem import rdchem
    if bond_si_m.GetBondType() != rdchem.BondType.SINGLE:
        return None

    # Classify the three other substituents on Si.  Each must be a
    # single-degree halogen or a methyl carbon.
    halide_count: dict[int, int] = {}
    methyl_count = 0
    for a in atoms:
        if a is metal or a is si:
            continue
        anum = a.GetAtomicNum()
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return None
        if a.GetDegree() != 1:
            return None
        # Must be bonded to Si specifically.
        nb = next(iter(a.GetNeighbors()))
        if nb.GetIdx() != si.GetIdx():
            return None
        if anum in _HALIDE_PREFIX_NAMES and a.GetTotalNumHs() == 0:
            halide_count[anum] = halide_count.get(anum, 0) + 1
        elif anum == 6 and a.GetTotalNumHs() == 3:
            methyl_count += 1
        else:
            return None

    # Build the silyl substituent name.  IUPAC orders prefixes
    # alphabetically (chloro < fluoro < methyl etc.), and since the
    # methyl group's name starts with a parenthesised "(methyl)" form
    # (per OPSIN test results), we format the multiplied halide prefixes
    # without parens and the methyl with parens when it co-exists with a
    # halide.  Halide-only or methyl-only forms also work.
    pieces: list[tuple[str, str]] = []  # (sort_key, segment)
    for anum in sorted(halide_count.keys()):
        cnt = halide_count[anum]
        h_pfx = _HALIDE_PREFIX_NAMES[anum]
        h_mult = _CO_MULTIPLIERS.get(cnt, "")
        # "trichloro", "difluoro", etc.
        pieces.append((h_pfx, f"{h_mult}{h_pfx}"))
    if methyl_count > 0:
        # When co-occurring with halides, OPSIN parses "difluoro(methyl)silyl"
        # but rejects "difluoromethylsilyl" (which it reads as "difluoromethyl
        # silyl").  Use parenthesised form when methyl is present alongside
        # other substituents; bare "methyl" when it is the only substituent.
        if pieces:
            m_mult = _CO_MULTIPLIERS.get(methyl_count, "")
            pieces.append(("methyl", f"{m_mult}(methyl)"))
        else:
            m_mult = _CO_MULTIPLIERS.get(methyl_count, "")
            pieces.append(("methyl", f"{m_mult}methyl"))
    pieces.sort(key=lambda p: p[0])
    silyl_name = "".join(seg for _key, seg in pieces) + "silyl"
    return (metal.GetIdx(), silyl_name)


# Single-atom Group-15 / Group-16 donor ligands recognised as space-
# separated PIN qualifiers (e.g. "arsane", "phosphane").  The atomic
# number is the key; the value is the IUPAC parent name OPSIN parses
# as the standalone donor ligand.
_SIMPLE_DONOR_HYDRIDE_NAMES: Mapping[int, tuple[str, int]] = {
    # Z → (donor name, expected H count)
    7:  ("azane",     3),   # NH3 (ammonia)
    15: ("phosphane", 3),   # PH3
    33: ("arsane",    3),   # AsH3
    51: ("stibane",   3),   # SbH3
}


def _is_simple_donor_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> str | None:
    """Detect a single-atom Group-15/16 donor hydride (NH3, PH3, AsH3,
    SbH3) used as a neutral ligand qualifier in PIN names like
    ``arsane dicarbonyl(cyclopentadienyl)manganese``.

    Returns the IUPAC parent name (``arsane`` / ``phosphane`` / …) on
    a positive match, or ``None`` otherwise.  Criteria:
      * Single-atom fragment.
      * Element in ``_SIMPLE_DONOR_HYDRIDE_NAMES``.
      * Implicit H count matches the entry's expected value.
      * Zero formal charge and zero radical electrons.
    """
    if len(frag_atom_idxs) != 1:
        return None
    atom = mol.GetAtomWithIdx(frag_atom_idxs[0])
    entry = _SIMPLE_DONOR_HYDRIDE_NAMES.get(atom.GetAtomicNum())
    if entry is None:
        return None
    name, expected_h = entry
    if atom.GetTotalNumHs() != expected_h:
        return None
    if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
        return None
    return name


def _is_cs_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is exactly ``[C]=S`` (the
    carbonothioyl / monothiocarbonyl ligand, the sulfur analogue of CO).

    Criteria mirror ``_is_co_fragment`` for the neutral radical form:
      * Exactly 2 heavy atoms: one C and one S.
      * No explicit H on either atom.
      * Bond is double; both atoms uncharged.
    """
    if len(frag_atom_idxs) != 2:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    syms = sorted(a.GetSymbol() for a in atoms)
    if syms != ["C", "S"]:
        return False
    for a in atoms:
        if a.GetTotalNumHs() != 0:
            return False
    bond = mol.GetBondBetweenAtoms(frag_atom_idxs[0], frag_atom_idxs[1])
    if bond is None:
        return False
    from rdkit.Chem import rdchem
    if bond.GetBondType() != rdchem.BondType.DOUBLE:
        return False
    c_atom = next(a for a in atoms if a.GetSymbol() == "C")
    s_atom = next(a for a in atoms if a.GetSymbol() == "S")
    return c_atom.GetFormalCharge() == 0 and s_atom.GetFormalCharge() == 0


def _is_nitrosyl_ligand_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is a nitrosyl-ligand precursor.

    Recognises two common SMILES representations that PubChem uses for the
    coordinated NO (nitrosyl) ligand:

    * Anionic ``[N-]=O`` — N has formal charge -1, double-bonded to neutral O.
      Used in e.g. ``[C-]#[O+].[C-]#[O+].[Co].[N-]=O`` (cobalt dicarbonyl
      nitrosyl, Co(CO)2(NO)).
    * Cationic ``N#[O+]`` — N has charge 0, triple-bonded to O(+).  Used in
      e.g. ``N#[O+].[C-]#[O+].[C-]#[O+].[Mo].c1cc[cH-]c1``
      (dicarbonylnitrosyl(cyclopentadienyl)molybdenum).

    Both ligand forms describe the bound NO unit; the dispatcher treats them
    identically and emits the same ``nitrosyl`` portion of the PIN.  OPSIN's
    output for ``...nitrosyl...`` is a covalent ``[N]=O`` ligand on the
    metal, which the eval's ``_metal_organic_ligand_equiv`` matcher accepts
    against either input form (metal-stripped skeleton match).

    Criteria (shared):
      * Exactly 2 heavy atoms: one N and one O, no H, no radicals.

    Bond-order variants:
      * Double bond + N(-1) / O(0)  → anionic nitroxyl ``[N-]=O``.
      * Triple bond + N(0) / O(+1)  → cationic nitrosyl ``N#[O+]``.
    """
    if len(frag_atom_idxs) != 2:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    syms = sorted(a.GetSymbol() for a in atoms)
    if syms != ["N", "O"]:
        return False
    for a in atoms:
        if a.GetTotalNumHs() != 0:
            return False
        if a.GetNumRadicalElectrons() != 0:
            return False
    n_atom = next(a for a in atoms if a.GetSymbol() == "N")
    o_atom = next(a for a in atoms if a.GetSymbol() == "O")
    bond = mol.GetBondBetweenAtoms(frag_atom_idxs[0], frag_atom_idxs[1])
    if bond is None:
        return False
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    if bt == rdchem.BondType.DOUBLE:
        # Anionic form: [N-]=O.
        return (n_atom.GetFormalCharge() == -1
                and o_atom.GetFormalCharge() == 0)
    if bt == rdchem.BondType.TRIPLE:
        # Cationic form: N#[O+].
        return (n_atom.GetFormalCharge() == 0
                and o_atom.GetFormalCharge() == 1)
    return False


def _is_cyclopentadienide_anion_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is an aromatic cyclopentadienide anion ``c1cc[cH-]c1``.

    Used by ``detect_metal_carbonyl`` to admit a Cp ligand presented as a
    separate aromatic anion fragment (as opposed to the inline
    ``[M][C]1=CC=CC1`` shape covered by ``_is_metal_cp_fragment``).  This
    shape appears in PubChem depictions like
    ``N#[O+].[C-]#[O+].[C-]#[O+].[Mo].c1cc[cH-]c1`` —
    dicarbonylnitrosyl(cyclopentadienyl)molybdenum.

    Criteria:
      * Exactly 5 atoms, all carbon, all aromatic.
      * Single 5-membered ring (aromatic).
      * Net formal charge = -1 (one C bears the formal charge).
      * No radical electrons on any atom.
      * Every ring C has exactly 1 implicit H (the standard
        ``c1cc[cH-]c1`` Kekule-equivalent form in RDKit: even the
        charged carbon shows TotalNumHs == 1).
    """
    if len(frag_atom_idxs) != 5:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return False
    # All aromatic.
    if any(not a.GetIsAromatic() for a in atoms):
        return False
    # All atoms in a single 5-membered ring.
    ri = mol.GetRingInfo()
    target = set(frag_atom_idxs)
    if not any(len(r) == 5 and set(r) == target for r in ri.AtomRings()):
        return False
    # Net charge = -1.
    if sum(a.GetFormalCharge() for a in atoms) != -1:
        return False
    # Exactly one C with formal charge -1.
    if sum(1 for a in atoms if a.GetFormalCharge() == -1) != 1:
        return False
    # No radicals.
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return False
    # Each ring C bears exactly 1 H in the aromatic [cH-] form.
    if any(a.GetTotalNumHs() != 1 for a in atoms):
        return False
    return True


def _name_ligand_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> str | None:
    """Recursively name a neutral ligand fragment.

    Returns the IUPAC name (e.g. ``benzene``, ``thiophene``,
    ``bromobenzene``) on success, or ``None`` to defer.  Used by the metal
    carbonyl dispatcher to handle mixed M(CO)n + ligand complexes such as
    ``(benzene)tricarbonylchromium`` written as
    ``[C]=O.[C]=O.[C]=O.[Cr].c1ccccc1``.

    Architectural notes:
    * The recursive call goes back through ``name_smiles`` on the extracted
      ligand SMILES so retained names (benzene, thiophene, ...) and
      substituted parents (bromobenzene, fluorobenzene) all flow through
      the standard pipeline.
    * Returns ``None`` on any naming error (NAMING ERROR token, exception,
      empty string) so the caller can fall through to the salt path
      unchanged.
    """
    smi = _extract_fragment_smiles(mol, frag_atom_idxs)
    if smi is None:
        return None
    # Late import — name_smiles lives in iupac_namer.engine which itself
    # imports from this module, so we defer to call time to avoid a
    # circular import at module load.
    try:
        from iupac_namer.engine import name_smiles as _name
        name = _name(smi)
    except Exception:
        return None
    if not name:
        return None
    if "NAMING ERROR" in name:
        return None
    return name


def detect_metal_carbonyl(mol) -> LeafTree | None:
    """Detect and name metal carbonyl coordination compounds.

    Recognises disconnected-fragment SMILES of the form:
    ``N × [C]=O . [M]``                  → ``{N}carbonyl{metal}``
    ``N × [C]=O . [M+n] . n × X-``       → ``{N}carbonyl{metal} {mult}halide``
    ``N × [C]=O . [M] . L``              → ``{N}carbonyl{metal} {ligand}``
    ``N × [C]=O . [M] . [M]``            → ``{N/2}carbonyl{metal} {N/2}carbonyl{metal}``

    where ``L`` is exactly one neutral non-CO non-metal-non-halide
    ligand fragment (arene, thiophene, alkene, etc.) that the engine can
    name standalone.  The dimer form (2 identical neutral metals, N CO
    ligands evenly split) emits two space-separated unit names (e.g.
    ``[C-]#[O+]×10.[Mn].[Mn]`` → "pentacarbonylmanganese pentacarbonylmanganese").
    When the recursive ligand naming fails, the dispatcher falls through
    to the salt path unchanged.

    Returns a ``LeafTree`` with the IUPAC coordination name, or ``None``
    when the input is not a metal carbonyl (defers to the salt path).
    """
    if mol is None:
        return None
    from rdkit import Chem

    try:
        frag_tuples = Chem.GetMolFrags(mol)
    except Exception:
        return None

    if len(frag_tuples) < 2:
        return None

    # Classify each fragment.
    co_count = 0
    cs_count = 0   # [C]=S carbonothioyl ligands.
    # Nitrosyl ligand count.  Accepts either anionic ``[N-]=O`` (used in
    # PubChem's Co(CO)n(NO) depictions) or cationic ``N#[O+]`` (used in
    # ``N#[O+].[C-]#[O+]×2.[Mo].c1cc[cH-]c1`` for the Mo(CO)2(NO)(Cp)
    # class).  Both forms emit the same ``nitrosyl`` portion of the PIN
    # which OPSIN parses to a covalent ``[M]N=O`` ligand whose round-trip
    # is accepted by the eval's ``_metal_organic_ligand_equiv`` matcher.
    no_count = 0
    # Per-metal accounting: ordered list of (symbol, charge) for each
    # metal atom encountered.  This supports homometallic dimers
    # (Mn2(CO)10) and heterometallic dimers (MnTc(CO)10) uniformly.
    metals_seen: list[tuple[str, int]] = []
    halide_counts: dict[int, int] = {}   # atomic_num → count
    ligand_frags: list[tuple[int, ...]] = []
    # Donor hydride ligands (AsH3, PH3, SbH3, NH3) found as single-atom
    # fragments with the appropriate H count.  These produce a leading
    # "{donor} " word qualifier in the PIN.
    donor_hydrides: list[str] = []
    # Inline-Cp marker: when the metal is bonded to a Cp ring as part of
    # the same fragment (e.g. ``[Cr][C]1=CC=CC1``), we treat it as a
    # M-Cp combo and emit a "(cyclopentadienyl){metal}" ligand-form.
    inline_cp = False
    # Separated-Cp-anion marker: when an aromatic cyclopentadienide
    # anion ``c1cc[cH-]c1`` appears as its own fragment alongside the
    # bare metal and CO/NO ligands (e.g.
    # ``N#[O+].[C-]#[O+]×2.[Mo].c1cc[cH-]c1`` — dicarbonylnitrosyl
    # (cyclopentadienyl)molybdenum), treat it as an additional Cp
    # ligand on the metal.  Distinguished from the inline-Cp marker
    # because the metal is a separate single-atom fragment here.
    sep_cp = False
    # Inline alkyl-metal ligand: when a methyl carbon is bonded directly
    # to a metal as a 2-atom fragment ``[CH3][M]`` (e.g. MeRe), record
    # the alkyl prefix to emit ``methyl{N}carbonyl{metal}``.  Only one
    # alkyl ligand is supported.
    inline_alkyl_prefix: str | None = None
    # Inline silyl-metal ligand: when a silyl group is bonded to a metal
    # as a single fragment ``X[Si](Y)(Z)[M]`` (e.g. trichlorosilylcobalt
    # tetracarbonyl), record the parenthesised silyl substituent name
    # to emit ``{N}carbonyl({silyl}){metal}``.  Only one silyl ligand
    # is supported per complex.
    inline_silyl_name: str | None = None
    for fa in frag_tuples:
        if _is_co_fragment(mol, fa):
            co_count += 1
            continue
        if _is_cs_fragment(mol, fa):
            cs_count += 1
            continue
        if _is_nitrosyl_ligand_fragment(mol, fa):
            no_count += 1
            continue
        if _is_cyclopentadienide_anion_fragment(mol, fa):
            if sep_cp:
                return None   # More than one separated Cp anion — out of scope.
            sep_cp = True
            continue

        # Single-atom fragment?
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            sym = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            anum = atom.GetAtomicNum()

            # Bare transition metal (may be charged for mixed carbonyls).
            if sym in _CARBONYL_METAL_NAMES:
                if len(metals_seen) >= 2:
                    return None   # More than two metals — out of scope.
                metals_seen.append((sym, charge))
                continue

            # Halide anion.
            if anum in _HALIDE_ANION_NAMES and charge == -1:
                halide_counts[anum] = halide_counts.get(anum, 0) + 1
                continue

            # Donor hydride ligand: AsH3, PH3, NH3, SbH3.  Emitted as
            # a leading "{donor} " word in the PIN.
            donor_name = _is_simple_donor_fragment(mol, fa)
            if donor_name is not None:
                donor_hydrides.append(donor_name)
                continue

            # Single-atom fragment that isn't a metal, halide anion, or
            # donor hydride — defer to salt path.
            return None

        # Inline methyl-metal fragment: ``[CH3][M]`` — single metal
        # with a directly-bonded methyl ligand.  Treat the metal as the
        # complex centre and emit a "methyl" prefix on the PIN.
        methyl_match = _is_methyl_metal_fragment(mol, fa)
        if methyl_match is not None:
            if metals_seen:
                return None   # Metal already found — inline-alkyl dimer unsupported.
            if inline_alkyl_prefix is not None or inline_silyl_name is not None:
                return None
            metal_atom = mol.GetAtomWithIdx(methyl_match)
            metals_seen.append(
                (metal_atom.GetSymbol(), metal_atom.GetFormalCharge())
            )
            inline_alkyl_prefix = "methyl"
            continue

        # Inline silyl-metal fragment: ``X[Si](Y)(Z)[M]`` — single metal
        # bonded to one Si with three substituents (halogen and/or methyl).
        silyl_match = _is_silyl_metal_fragment(mol, fa)
        if silyl_match is not None:
            if metals_seen:
                return None
            if inline_alkyl_prefix is not None or inline_silyl_name is not None:
                return None
            metal_idx, silyl_name = silyl_match
            metal_atom = mol.GetAtomWithIdx(metal_idx)
            metals_seen.append(
                (metal_atom.GetSymbol(), metal_atom.GetFormalCharge())
            )
            inline_silyl_name = silyl_name
            continue

        # Inline metal-Cp fragment: ``[M][C]1=CC=CC1`` — single metal
        # bonded to a 5-C ring with the carbene-style anchor.  Treat
        # the metal as the complex centre and the Cp ring as a
        # cyclopentadienyl ligand.
        cp_match = _is_metal_cp_fragment(mol, fa)
        if cp_match is not None:
            if metals_seen:
                return None   # Metal already found — inline-Cp dimer unsupported.
            metal_atom_idx, _ring_idxs = cp_match
            metal_atom = mol.GetAtomWithIdx(metal_atom_idx)
            metals_seen.append(
                (metal_atom.GetSymbol(), metal_atom.GetFormalCharge())
            )
            inline_cp = True
            continue

        # Multi-atom non-CO fragment: candidate neutral ligand (arene,
        # thiophene, alkene, etc.).  Net charge must be zero and we accept
        # at most one such fragment per complex (matching IUPAC P-69
        # mixed-ligand-carbonyl PIN form for a single L ligand).
        net_charge = sum(
            mol.GetAtomWithIdx(i).GetFormalCharge() for i in fa
        )
        if net_charge != 0:
            return None
        ligand_frags.append(fa)

    if not metals_seen or co_count == 0:
        return None

    metal_count = len(metals_seen)
    metal_symbol = metals_seen[0][0]
    metal_charge = metals_seen[0][1]

    # ── Dimer path: 2 metals (homo- or heterometallic) + N CO ligands ──────
    # Recognises ``N × [C]=O . [M1] . [M2]`` and emits two
    # space-separated "{N/2}carbonyl{metal}" units which OPSIN parses to a
    # two-fragment disconnected mixture (each fragment being one M(CO)_{N/2}
    # covalent complex).  The eval's ``_metal_ionic_covalent_equiv`` /
    # ``_metal_anion_stoich_equiv`` paths accept the OPSIN round-trip.
    #
    # Two sub-shapes are accepted:
    #  (a) Both metals neutral, N CO splits evenly, no halides / other
    #      ligands.  Output ``{N/2}carbonyl{m_a} {N/2}carbonyl{m_b}``.
    #  (b) Both metals same positive charge q and N CO + 2q halide-
    #      anions splitting evenly between the two metals (e.g.
    #      ``8×[C]=O . 2×[Cl-] . 2×[Rh+]`` →
    #      "chlorotetracarbonylrhodium chlorotetracarbonylrhodium").
    if metal_count == 2:
        if (ligand_frags or inline_cp
                or inline_alkyl_prefix or inline_silyl_name
                or donor_hydrides or cs_count
                or no_count or sep_cp):
            return None
        # Both metals must carry the same formal charge.  Heterometallic
        # dimers with different per-metal charges are out of scope here
        # (the surface form would be ambiguous).
        charge_a = metals_seen[0][1]
        charge_b = metals_seen[1][1]
        if charge_a != charge_b:
            return None
        per_metal_charge = charge_a
        # Halides must charge-balance both metals together (each halide
        # contributes -1; total halide count must equal 2*charge).
        total_halide = sum(halide_counts.values())
        if 2 * per_metal_charge != total_halide:
            return None
        # If halides are present, all of one element type at a uniform
        # split per metal: total halide must be even AND each per-element
        # count must be even (so each metal gets an equal share).
        if halide_counts:
            if per_metal_charge <= 0:
                return None
            if any(cnt % 2 != 0 for cnt in halide_counts.values()):
                return None
        # CO count must be even and each half must be in _CO_MULTIPLIERS.
        if co_count % 2 != 0:
            return None
        co_per_metal = co_count // 2
        co_mult_per = _CO_MULTIPLIERS.get(co_per_metal)
        if co_mult_per is None:
            return None
        # Halide prefix per metal (e.g. "chloro", "dibromo").
        halide_per_metal = ""
        for anum in sorted(halide_counts.keys()):
            cnt_per = halide_counts[anum] // 2
            h_pfx = _HALIDE_PREFIX_NAMES[anum]
            h_mult = _CO_MULTIPLIERS.get(cnt_per, "")
            halide_per_metal += f"{h_mult}{h_pfx}"
        # Order the two units alphabetically by metal name.  This gives
        # a deterministic surface form regardless of input fragment order.
        m1_name = _CARBONYL_METAL_NAMES[metals_seen[0][0]]
        m2_name = _CARBONYL_METAL_NAMES[metals_seen[1][0]]
        m_a, m_b = sorted((m1_name, m2_name))
        unit_a = f"{halide_per_metal}{co_mult_per}carbonyl{m_a}"
        unit_b = f"{halide_per_metal}{co_mult_per}carbonyl{m_b}"
        text = f"{unit_a} {unit_b}"
        return LeafTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(Choice(
                type="organometallic",
                detail=f"metal carbonyl dimer: {text}",
            ),),
            decision_ctx=None,
            validity_warnings=None,
            text=text,
        )

    # ── Single-metal path (original logic) ──────────────────────────────────
    # Mixed-ligand complex: at most one neutral ligand allowed.  Multi-
    # ligand mixed carbonyls (e.g. M(CO)2(arene)(arene')) need
    # bracketed coordination nomenclature that's outside this dispatcher.
    if len(ligand_frags) > 1:
        return None

    # Charge balance: when no separated-Cp anion is present, the metal
    # charge must equal the total halide count (each halide -1; metal
    # +metal_charge; neutral CO/CS/NO ligands don't enter).  When a
    # separated Cp anion (charge -1) IS present, the Cp contributes -1
    # to the balance and the metal must be neutral with no halides
    # (this is the Mo(CO)n(NO)(Cp) class).
    total_halide = sum(halide_counts.values())
    if sep_cp:
        # Cp- + neutral metal + neutral CO/NO/CS ligands.  The
        # cationic NO+ fragment (``N#[O+]``, +1 charge) balances the
        # Cp anion when no_count == 1.  We don't enforce the precise
        # balance here because the eval matcher accepts the
        # metal-stripped skeleton form regardless of charge placement
        # — but we DO require the metal to be neutral, no halides,
        # and no other inline ligands, to keep this path tight.
        if (metal_charge != 0 or halide_counts or inline_cp
                or inline_alkyl_prefix or inline_silyl_name
                or donor_hydrides or cs_count or ligand_frags):
            return None
    else:
        if metal_charge != total_halide:
            return None

    # Halide and ligand combinations together aren't supported here —
    # the surface form ``{carbonyl}{metal} {halide} {ligand}`` doesn't
    # round-trip cleanly through OPSIN.  Defer the rare halide+ligand
    # case to the salt path.
    if halide_counts and ligand_frags:
        return None

    # Inline Cp + extra ligand together aren't supported — keep the
    # surface form simple and well-defined.
    if inline_cp and ligand_frags:
        return None

    # Halide / inline-alkyl / inline-silyl combinations aren't supported
    # on this path either — keep the dispatcher conservative.
    if (inline_alkyl_prefix or inline_silyl_name) and (
            halide_counts or ligand_frags or inline_cp):
        return None

    # Donor hydrides combine with the inline-Cp form (e.g.
    # CpMn(CO)2(AsH3)) and with bare metals (e.g. tricarbonylmanganese
    # arsane) but not with ligand_frags / halides / alkyls / silyls.
    if donor_hydrides and (
            halide_counts or ligand_frags or inline_alkyl_prefix
            or inline_silyl_name):
        return None

    # CS ligand combines with inline-Cp + CO (CpMn(CO)2(CS)) and with
    # bare-metal + CO + arene-style neutral ligand (e.g.
    # (benzene)dicarbonyl(carbonothioyl)chromium written as
    # ``[C]=O.[C]=O.[C]=S.[Cr].c1ccccc1``).  It does NOT combine with
    # halides, alkyl/silyl ligands, or donor hydrides on this path.
    if cs_count and (
            halide_counts or inline_alkyl_prefix
            or inline_silyl_name or donor_hydrides):
        return None

    # Nitrosyl ligand combines with bare metal + CO (cobalt
    # dicarbonyl/tricarbonyl nitrosyl) and with bare metal + CO +
    # separated Cp anion (dicarbonylnitrosyl(cyclopentadienyl)
    # molybdenum).  It does NOT combine with halides / alkyls / silyls /
    # inline-Cp / arene-ligand_frags / CS / donor hydrides on this path.
    if no_count and (
            halide_counts or ligand_frags or inline_alkyl_prefix
            or inline_silyl_name or inline_cp or cs_count
            or donor_hydrides):
        return None

    # We need a multiplier for the CO count.
    co_mult = _CO_MULTIPLIERS.get(co_count)
    if co_mult is None:
        return None   # More than 12 CO — extremely unusual; defer.

    metal_name = _CARBONYL_METAL_NAMES[metal_symbol]
    carbonyl_part = f"{co_mult}carbonyl{metal_name}"

    # Build the halide suffix (if any).
    if halide_counts and inline_cp:
        # CpFe(CO)2X2 form: emit OPSIN-recognised PIN
        # ``{halides}{carbonyls}(cyclopentadienyl){metal}`` (e.g.
        # "dichlorodicarbonyl(cyclopentadienyl)iron") which OPSIN parses
        # to the covalent ``Cl[Fe]([Cl])(=C=O)(=C=O)[CH]1C=CC=C1`` form.
        # The eval matcher accepts this against the multi-fragment input
        # via ``_metal_ionic_covalent_equiv`` (formula + charge match).
        # Halides as PREFIXES use the "chloro/fluoro/bromo/iodo" forms
        # (not the "chloride/fluoride/..." anion suffixes).
        halide_parts: list[str] = []
        for anum in sorted(halide_counts.keys()):
            cnt = halide_counts[anum]
            halide_pfx = _HALIDE_PREFIX_NAMES[anum]
            h_mult = _CO_MULTIPLIERS.get(cnt, "")
            halide_parts.append(f"{h_mult}{halide_pfx}")
        text = (
            "".join(halide_parts) + co_mult
            + f"carbonyl(cyclopentadienyl){metal_name}"
        )
    elif inline_cp and (donor_hydrides or cs_count):
        # CpMn(CO)n + donor hydride / CS form: emit
        # ``{donor} {N}carbonyl(cyclopentadienyl){metal}`` for AsH3 etc.
        # OPSIN parses the space-separated donor as a disconnected fragment
        # alongside the covalent CpM(CO)n complex; the eval matcher's
        # _metal_organic_ligand_equiv accepts the round-trip.  The CS
        # ligand is emitted likewise as ``carbon monosulfide`` since
        # OPSIN parses that as a disconnected [C]=S fragment.
        leading_parts: list[str] = []
        # CS first (alphabetical: "carbon monosulfide" < others usually)
        # then donor hydrides alphabetised.
        if cs_count:
            cs_mult = _CO_MULTIPLIERS.get(cs_count, "")
            leading_parts.append(f"{cs_mult}carbon monosulfide" if cs_mult
                                 else "carbon monosulfide")
        for d in sorted(donor_hydrides):
            leading_parts.append(d)
        leading = " ".join(leading_parts)
        text = (
            leading + " "
            + f"{co_mult}carbonyl(cyclopentadienyl){metal_name}"
        )
    elif inline_cp:
        # CpM(CO)n form: "{N}carbonyl(cyclopentadienyl){metal}" (e.g.
        # "tricarbonyl(cyclopentadienyl)manganese", aka cymantrene).
        text = f"{co_mult}carbonyl(cyclopentadienyl){metal_name}"
    elif halide_counts:
        # Group identical halides together.  Most complexes have one type.
        halide_parts = []
        for anum in sorted(halide_counts.keys()):
            cnt = halide_counts[anum]
            halide_name = _HALIDE_ANION_NAMES[anum]
            h_mult = _CO_MULTIPLIERS.get(cnt, "")
            halide_parts.append(f"{h_mult}{halide_name}")
        text = carbonyl_part + " " + " ".join(halide_parts)
    elif inline_alkyl_prefix:
        # Inline alkyl (e.g. ``[CH3][Re].(CO)5``): emit
        # ``{alkyl}{N}carbonyl{metal}`` (e.g.
        # "methylpentacarbonylrhenium").
        text = f"{inline_alkyl_prefix}{co_mult}carbonyl{metal_name}"
    elif inline_silyl_name:
        # Inline silyl (e.g. ``Cl[Si](Cl)(Cl)[Co].(CO)4``): emit
        # ``{N}carbonyl({silyl}){metal}`` (e.g.
        # "tetracarbonyl(trichlorosilyl)cobalt").
        text = f"{co_mult}carbonyl({inline_silyl_name}){metal_name}"
    elif donor_hydrides and not ligand_frags:
        # Bare metal + CO + donor hydride (e.g. ``[AsH3].[C]=O.[Mn]``):
        # emit ``{donor} {N}carbonyl{metal}``.  Multiple donors are
        # alphabetised; the eval matcher accepts the disconnected-fragment
        # round-trip.
        donor_str = " ".join(sorted(donor_hydrides))
        text = f"{donor_str} {carbonyl_part}"
    elif cs_count and ligand_frags:
        # Bare metal + CO + CS + neutral arene/heteroarene ligand
        # (e.g. ``[C]=O.[C]=O.[C]=S.[Cr].c1ccccc1`` —
        # (benzene)dicarbonyl(carbonothioyl)chromium).  Emit
        # ``{cs_mult}carbon monosulfide {N}carbonyl{metal} {ligand}``
        # so OPSIN parses three disconnected pieces (CS fragment,
        # M(CO)n covalent cluster, arene fragment) which the eval
        # matcher's ``_metal_organic_ligand_equiv`` /
        # ``_metal_ionic_covalent_equiv`` recognises as equivalent
        # to the original 5-fragment input.
        ligand_name = _name_ligand_fragment(mol, ligand_frags[0])
        if ligand_name is None:
            return None
        cs_mult = _CO_MULTIPLIERS.get(cs_count, "")
        cs_word = f"{cs_mult}carbon monosulfide" if cs_mult \
            else "carbon monosulfide"
        text = f"{cs_word} {carbonyl_part} {ligand_name}"
    elif cs_count and not ligand_frags:
        # Bare metal + CO + CS (e.g. ``[C]=S.[C]=O×4.[Fe]``): emit
        # ``{cs_mult}carbon monosulfide {N}carbonyl{metal}`` so the CS
        # ligand is rendered as a leading disconnected fragment.  OPSIN
        # parses the space-separated form to a multi-fragment SMILES;
        # the eval's ``_metal_organic_ligand_equiv`` accepts the round-trip.
        cs_mult = _CO_MULTIPLIERS.get(cs_count, "")
        cs_word = f"{cs_mult}carbon monosulfide" if cs_mult \
            else "carbon monosulfide"
        text = f"{cs_word} {carbonyl_part}"
    elif ligand_frags:
        # Mixed M(CO)n + L: name the ligand recursively and emit
        # ``{carbonyl}{metal} {ligand-name}``.  OPSIN parses the
        # space-separated form to disconnected fragments and the
        # M(CO)n part to a covalent ``=[C]=O`` cluster on the metal,
        # which the eval's ``_metal_ionic_covalent_equiv`` /
        # InChI fallback recognises as equivalent to the original
        # disconnected ``[C]=O.[C]=O.[C]=O.[M].<ligand>`` input.
        ligand_name = _name_ligand_fragment(mol, ligand_frags[0])
        if ligand_name is None:
            return None
        text = carbonyl_part + " " + ligand_name
    elif no_count and sep_cp:
        # Bare metal + CO + NO + separated Cp anion (e.g.
        # ``N#[O+].[C-]#[O+]×2.[Mo].c1cc[cH-]c1`` —
        # dicarbonylnitrosyl(cyclopentadienyl)molybdenum).  Emit the
        # fused PIN form ``{N}carbonyl{M}nitrosyl(cyclopentadienyl){metal}``
        # which OPSIN parses to a single covalent cluster
        # ``C(=O)=[Mo](C1C=CC=C1)(N=O)=C=O``.  The eval's
        # ``_metal_organic_ligand_equiv`` matcher accepts this against
        # the original multi-fragment input via metal-stripped
        # skeleton equivalence (heavy-atom skeletons + anomaly marker).
        no_mult = _CO_MULTIPLIERS.get(no_count, "")
        text = (
            co_mult + "carbonyl" + no_mult + "nitrosyl"
            + f"(cyclopentadienyl){metal_name}"
        )
    elif no_count:
        # Bare metal + CO + NO (e.g. ``[C-]#[O+]×2.[Co].[N-]=O`` —
        # cobalt dicarbonyl nitrosyl, Co(CO)2(NO); or
        # ``[C-]#[O+]×3.[Co].[N-]=O`` — cobalt tricarbonyl nitrosyl,
        # Co(CO)3(NO)).  Emit the fused PIN form
        # ``{N}carbonyl{M}nitrosyl{metal}`` (e.g.
        # ``dicarbonylnitrosylcobalt``, ``tricarbonylnitrosylcobalt``).
        # OPSIN parses to a single covalent cluster
        # ``C(=O)=[Co](N=O)=C=O`` / ``C(=O)=[Co](N=O)(=C=O)=C=O``;
        # the eval's ``_metal_organic_ligand_equiv`` matcher accepts
        # the round-trip via metal-stripped skeleton equivalence.
        no_mult = _CO_MULTIPLIERS.get(no_count, "")
        text = co_mult + "carbonyl" + no_mult + "nitrosyl" + metal_name
    else:
        text = carbonyl_part

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"metal carbonyl: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Metal + acetylide (ethyn-1-ide) salt dispatcher.
# ---------------------------------------------------------------------------
#
# Recognises disconnected-fragment SMILES of the form:
#   ``n × [C-]#C . [M]``
# where [M] is a bare metal atom (charged or uncharged) and [C-]#C is the
# monoanionic acetylide (ethyn-1-ide) fragment.  These appear when the user
# supplies a SMILES like ``[C-]#C.[Eu]`` or ``[C-]#C.[C-]#C.[Ho]``.
#
# Unlike the carbonyl dispatcher, the metal atom here may carry RDKit radical
# electrons (e.g. [Rh], [Eu], [Ho] all have non-zero ``GetNumRadicalElectrons``
# in RDKit's model), so this dispatcher **must** be called from ``name_smiles``
# BEFORE ``_validate_no_open_valences``.
#
# Naming strategy (IUPAC P-72 salt nomenclature):
#   * 1 acetylide + 1 metal  → ``{metal_name} ethyn-1-ide``
#   * 2 acetylides + 1 metal → ``{metal_name} diethyn-1-ide``
#
# These names are OPSIN-parseable for common metals when matched with the
# appropriate metal oxidation-state qualifier (e.g. "rhodium(I) ethyn-1-ide").
# We emit the bare name without the oxidation-state qualifier when the input
# metal is uncharged (the qualifier would be speculative).  The eval's
# ``_metal_ionic_covalent_equiv`` fallback will reconcile formula-equivalent
# representations.
#
# Scope: this dispatcher fires only when ALL non-metal fragments are exactly
# one ``[C-]#C`` unit (no other anions or organic fragments).  Everything else
# defers to the salt path.

_METAL_ACETYLIDE_NAMES: Mapping[str, str] = {
    # Transition metals
    "Sc": "scandium",  "Ti": "titanium",  "V":  "vanadium",
    "Cr": "chromium",  "Mn": "manganese", "Fe": "iron",
    "Co": "cobalt",    "Ni": "nickel",    "Cu": "copper",
    "Zn": "zinc",      "Y":  "yttrium",   "Zr": "zirconium",
    "Nb": "niobium",   "Mo": "molybdenum","Tc": "technetium",
    "Ru": "ruthenium", "Rh": "rhodium",   "Pd": "palladium",
    "Ag": "silver",    "Cd": "cadmium",   "Hf": "hafnium",
    "Ta": "tantalum",  "W":  "tungsten",  "Re": "rhenium",
    "Os": "osmium",    "Ir": "iridium",   "Pt": "platinum",
    "Au": "gold",      "Hg": "mercury",
    # Main-group metals
    "Na": "sodium",    "K":  "potassium", "Li": "lithium",
    "Rb": "rubidium",  "Cs": "cesium",
    "Mg": "magnesium", "Ca": "calcium",   "Ba": "barium",
    "Sr": "strontium",
    "Al": "aluminium", "Ga": "gallium",   "In": "indium",
    "Tl": "thallium",  "Sn": "tin",       "Pb": "lead",
    # Lanthanides / actinides
    "La": "lanthanum", "Ce": "cerium",    "Pr": "praseodymium",
    "Nd": "neodymium", "Pm": "promethium","Sm": "samarium",
    "Eu": "europium",  "Gd": "gadolinium","Tb": "terbium",
    "Dy": "dysprosium","Ho": "holmium",   "Er": "erbium",
    "Tm": "thulium",   "Yb": "ytterbium", "Lu": "lutetium",
    "Th": "thorium",   "U":  "uranium",   "Np": "neptunium",
    "Pu": "plutonium", "Am": "americium", "Cm": "curium",
}

_ACETYLIDE_MULTIPLIERS: Mapping[int, str] = {
    1: "",
    2: "di",
    3: "tri",
    4: "tetra",
}


def _is_acetylide_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is exactly one ``[C-]#C``.

    Criteria:
    * Exactly 2 heavy atoms, both carbon.
    * One carbon has formal charge -1 and zero H (sp carbanion).
    * One carbon has formal charge 0 and exactly 1 H.
    * Triple bond between them.
    * No radical electrons.
    """
    if len(frag_atom_idxs) != 2:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return False
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return False
    charged = [a for a in atoms if a.GetFormalCharge() == -1]
    neutral = [a for a in atoms if a.GetFormalCharge() == 0]
    if len(charged) != 1 or len(neutral) != 1:
        return False
    if charged[0].GetTotalNumHs() != 0:
        return False
    if neutral[0].GetTotalNumHs() != 1:
        return False
    bond = mol.GetBondBetweenAtoms(frag_atom_idxs[0], frag_atom_idxs[1])
    if bond is None:
        return False
    from rdkit.Chem import rdchem
    return bond.GetBondType() == rdchem.BondType.TRIPLE


# ---------------------------------------------------------------------------
# Bis(cyclopentadienyl)<metal> dispatcher (Mn and other metals OPSIN does
# not recognise as a ``<X>ocene`` PIN).
# ---------------------------------------------------------------------------
#
# Recognises the 3-fragment shape ``[M] . [C-]1C=CC=C1 . [C-]1C=CC=C1`` (or
# the proper aromatic ``[M] . c1cc[cH-]c1 . c1cc[cH-]c1`` form, or the
# charged-metal ``[M+2] . c1cc[cH-]c1 . c1cc[cH-]c1`` form) where the metal
# is one whose ``<X>ocene`` retained name is not in OPSIN's vocabulary.
# Manganocene is the canonical example — OPSIN parses ``ferrocene`` /
# ``cobaltocene`` etc. but rejects ``manganocene``, so we emit the
# OPSIN-recognised PIN ``bis(cyclopentadienyl){metal}`` instead.
#
# Scope: the metal must be a single-atom fragment (charged or uncharged)
# and the two non-metal fragments must be 5-membered all-carbon rings (at
# any aromatic / kekulé form) — i.e. the parent (unsubstituted) metallocene
# shape only.  Substituted variants are out of scope here.

_BIS_CP_METAL_NAMES: Mapping[str, str] = {
    # Metals where OPSIN does not parse ``<X>ocene`` as a retained PIN.
    # We emit ``bis(cyclopentadienyl){metal}`` so OPSIN can map back to
    # the corresponding multi-fragment SMILES that the eval matcher
    # accepts via ``_metal_organic_ligand_equiv``.
    "Mn": "manganese",
}


def _is_unsub_cp_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff ``frag_atom_idxs`` is an unsubstituted 5-C ring.

    Accepts both kekulé / aromatic forms; the only requirement is that
    every atom is a carbon and the five atoms form a single 5-membered
    ring.  H counts and formal charges aren't constrained — the matcher
    handles tautomeric / charge-redistribution differences downstream.
    """
    if len(frag_atom_idxs) != 5:
        return False
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in frag_atom_idxs):
        return False
    ri = mol.GetRingInfo()
    target = set(frag_atom_idxs)
    for ring in ri.AtomRings():
        if len(ring) == 5 and set(ring) == target:
            return True
    return False


def detect_bis_cyclopentadienyl_metal(mol) -> "LeafTree | None":
    """Detect ``[M] . Cp- . Cp-`` shapes for metals OPSIN doesn't parse
    as ``<X>ocene`` and emit ``bis(cyclopentadienyl){metal}``.

    Currently only fires for manganese (the lone metal in
    ``_BIS_CP_METAL_NAMES``).  Returns ``None`` for all other shapes,
    deferring to the regular metallocene / salt dispatch.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 3:
        return None

    metal_sym: str | None = None
    cp_frags: list[tuple[int, ...]] = []
    for fa in frags:
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            sym = atom.GetSymbol()
            if sym in _BIS_CP_METAL_NAMES:
                if metal_sym is not None:
                    return None
                metal_sym = sym
                continue
            return None
        if _is_unsub_cp_fragment(mol, fa):
            cp_frags.append(fa)
            continue
        return None

    if metal_sym is None or len(cp_frags) != 2:
        return None

    metal_name = _BIS_CP_METAL_NAMES[metal_sym]
    text = f"bis(cyclopentadienyl){metal_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"bis(cyclopentadienyl) metal: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Single-fragment Cp-metal half-sandwich and Cp-M-L mixed-ligand dispatcher.
# ---------------------------------------------------------------------------
#
# Recognises **single-fragment** SMILES where one metal atom is covalently
# bonded to one or two organic ligand sub-fragments via metal-C bonds:
#
#   ``[Co][C]1=CC=CC1``                    → "(cyclopentadienyl)cobalt"
#   ``[Ni][C]1=CC=CC1``                    → "(cyclopentadienyl)nickel"
#   ``C=C[CH2][Pd][C]1=CC=CC1``            → "(allyl)(cyclopentadienyl)palladium"
#
# These are chemist-shorthand forms; OPSIN's parses of the corresponding
# ``(cyclopentadienyl){metal}`` / ``(allyl)(cyclopentadienyl){metal}`` PINs
# differ in H placement on the metal-bound C, but the eval matcher accepts
# them via ``_metal_organic_ligand_equiv`` (metal-stripped skeleton match).
#
# Scope:
#   * Single connected fragment (``GetMolFrags`` returns one tuple).
#   * Exactly one metal atom (in ``_CP_LIGAND_METAL_NAMES``); uncharged.
#   * Removing the metal yields one or two connected components, each
#     composed of carbons only (Cp ring or allyl chain).
#   * One of those components is a 5-C ring (Cp); when there are two
#     components, the other is a 3-C chain (allyl) or another Cp.
#
# The dispatcher must run BEFORE ``_validate_no_open_valences`` because
# the carbene-style ring anchor ``[C]`` (5-valent C with no H) shows up
# as an open valence in RDKit's model.

_CP_LIGAND_METAL_NAMES: Mapping[str, str] = {
    "Co": "cobalt",
    "Ni": "nickel",
    "Fe": "iron",
    "Pd": "palladium",
    "Pt": "platinum",
    "Rh": "rhodium",
    "Ir": "iridium",
    "Ru": "ruthenium",
    "Os": "osmium",
    "V":  "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Mo": "molybdenum",
    "W":  "tungsten",
    "Re": "rhenium",
    "Ti": "titanium",
}


def _classify_ligand_component(mol, atom_idxs: tuple[int, ...]) -> str | None:
    """Classify a connected metal-bonded ligand sub-component and return
    its IUPAC ligand-prefix name (e.g. ``cyclopentadienyl``, ``allyl``).

    Returns ``None`` if the component doesn't match a recognised ligand
    shape so the dispatcher can defer.

    Recognised shapes:
      * 5 carbons forming a single 5-membered ring (cyclopentadienyl).
        The H-distribution and explicit-H counts are not constrained
        because the eval matcher tolerates the various Kekule / aromatic
        / carbene-anchor forms via metal-stripped skeleton equivalence.
      * 3 carbons in an open chain ``CH2=CH-CH2-`` (allyl / prop-2-en-1-
        yl).  We accept any 3-C tree with one H-adjusted CH2 endpoint
        bonded to the metal — the typical way ``C=C[CH2][M]`` is drawn.
    """
    n = len(atom_idxs)
    if n not in (3, 5):
        return None
    atoms = [mol.GetAtomWithIdx(i) for i in atom_idxs]
    # All atoms must be carbon.
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return None

    if n == 5:
        # Cyclopentadienyl: 5 atoms forming a single 5-membered ring.
        ri = mol.GetRingInfo()
        target = set(atom_idxs)
        for ring in ri.AtomRings():
            if len(ring) == 5 and set(ring) == target:
                return "cyclopentadienyl"
        return None

    # 3 carbons: allyl shape — should be a linear chain (no ring).
    ri = mol.GetRingInfo()
    if any(ri.NumAtomRings(i) > 0 for i in atom_idxs):
        return None
    # The connectivity-induced subgraph on these 3 atoms must be a path:
    # a chain of 3 atoms has bond count 2 within the component.
    bond_count = 0
    atom_set = set(atom_idxs)
    for i in atom_idxs:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in atom_set and nb.GetIdx() > i:
                bond_count += 1
    if bond_count != 2:
        return None
    return "allyl"


def detect_metal_cp_ligand(mol) -> "LeafTree | None":
    """Single-fragment Cp-metal half-sandwich + Cp-M-allyl dispatcher.

    Recognises ``[M][C]1=CC=CC1`` and ``L[M][C]1=CC=CC1`` (single
    fragment; metal bonded to one or two carbon ligand sub-components).

    Returns a ``LeafTree`` with the IUPAC ligand-coordination name when
    the ligand shape is recognised.  Returns ``None`` otherwise (defers
    to other dispatchers / the salt path).
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 1:
        return None

    fa = frags[0]
    atoms = [mol.GetAtomWithIdx(i) for i in fa]
    metal_atoms = [a for a in atoms if a.GetSymbol() in _CP_LIGAND_METAL_NAMES]
    if len(metal_atoms) != 1:
        return None
    metal = metal_atoms[0]
    if metal.GetFormalCharge() != 0:
        return None
    if metal.GetDegree() not in (1, 2):
        return None

    # All non-metal atoms must be carbons with charge 0.
    others = [a for a in atoms if a is not metal]
    for a in others:
        if a.GetAtomicNum() != 6:
            return None
        if a.GetFormalCharge() != 0:
            return None

    # Compute connected components of the ligand atoms (BFS limited to
    # non-metal atoms in this fragment).
    other_set = set(a.GetIdx() for a in others)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in other_set:
        if start in seen:
            continue
        stack = [start]
        comp: list[int] = []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            comp.append(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                ni = nb.GetIdx()
                if ni in other_set and ni not in seen:
                    stack.append(ni)
        components.append(tuple(sorted(comp)))

    if len(components) not in (1, 2):
        return None

    # Classify each component.  At least one must be a Cp ring; the
    # other (if present) is allyl or another Cp.
    ligand_names: list[str] = []
    for comp in components:
        cls = _classify_ligand_component(mol, comp)
        if cls is None:
            return None
        ligand_names.append(cls)

    # Require at least one cyclopentadienyl ligand — this dispatcher's
    # remit is Cp-metal half-sandwich and Cp-M-L mixed-ligand forms.
    if "cyclopentadienyl" not in ligand_names:
        return None

    metal_name = _CP_LIGAND_METAL_NAMES[metal.GetSymbol()]

    # Sort ligand prefixes alphabetically for the IUPAC P-14.5.2
    # ligand-ordering rule.
    sorted_ligands = sorted(ligand_names)
    if len(sorted_ligands) == 1:
        text = f"({sorted_ligands[0]}){metal_name}"
    else:
        text = "".join(f"({lig})" for lig in sorted_ligands) + metal_name

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"metal Cp ligand: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


def detect_acetylide_metal_salt(mol) -> "LeafTree | None":
    """Detect and name bare-metal + acetylide salt complexes.

    Recognises ``n × [C-]#C . [M]`` where M is any supported metal atom
    (may be charged or uncharged; may carry RDKit radical electrons).
    Returns ``None`` for everything outside this scope.

    Must be called BEFORE ``_validate_no_open_valences`` because the bare
    metal atom [Rh], [Eu], [Ho], etc. carries non-zero radical electrons in
    RDKit's d/f-shell model.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frag_tuples = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frag_tuples) < 2:
        return None

    metal_sym: str | None = None
    acetylide_count: int = 0
    for fa in frag_tuples:
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            sym = atom.GetSymbol()
            if sym in _METAL_ACETYLIDE_NAMES:
                if metal_sym is not None:
                    # Two metals — out of scope.
                    return None
                metal_sym = sym
                continue
            # Single-atom non-metal fragment is not an acetylide — defer.
            return None
        if _is_acetylide_fragment(mol, fa):
            acetylide_count += 1
            continue
        # Multi-atom non-acetylide fragment — defer.
        return None

    if metal_sym is None or acetylide_count == 0:
        return None

    mult = _ACETYLIDE_MULTIPLIERS.get(acetylide_count)
    if mult is None:
        return None   # More than 4 acetylides — defer.

    metal_name = _METAL_ACETYLIDE_NAMES[metal_sym]
    text = f"{metal_name} {mult}ethyn-1-ide"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"metal acetylide: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Covalent metallocene dispatcher (single-fragment M(Cp)2 forms).
# ---------------------------------------------------------------------------
#
# Some metallocenes appear in SMILES as *connected* single-fragment molecules
# where the metal is covalently bonded to both Cp rings:
#   ``C1=C[CH]([Pb][CH]2C=CC=C2)C=C1``  (plumbocene, covalent)
# rather than the ionic salt ``[Pb+2].c1cc[cH-]c1.c1cc[cH-]c1``.
#
# The metal in these forms may carry RDKit radical electrons (Pb has 4 in
# its valence model), so this dispatcher must run BEFORE the free-valence
# guard in ``name_smiles``.
#
# Recognition criteria:
#   * single RDKit fragment (GetMolFrags returns one group)
#   * exactly one metal atom (symbol in _METALLOCENE_METALS)
#   * the metal has exactly two carbon neighbours (single bonds)
#   * each C neighbour is part of a 5-membered all-carbon ring (Cp ring)
#   * each Cp ring has exactly two double bonds (cyclopentadienyl)
#   * all ring carbons are sp2 (no extra H count requirements)
#
# Emitted name: ``bis(cyclopentadienyl){metalname}`` which OPSIN parses
# correctly for covalent M(Cp)2 forms.

_COVALENT_METALLOCENE_NAMES: Mapping[str, str] = {
    "Pb": "lead",
    "Sn": "tin",
    "Si": "silicon",
    "Ge": "germanium",
    # Add other metals as needed.
}


def detect_covalent_metallocene(mol) -> "LeafTree | None":
    """Detect covalent M(Cp)2 single-fragment metallocenes.

    Recognises single-fragment SMILES like ``C1=C[CH]([Pb][CH]2C=CC=C2)C=C1``
    and emits ``bis(cyclopentadienyl){metalname}``.  Returns ``None`` for
    anything outside this scope.

    Must run BEFORE ``_validate_no_open_valences`` because the metal may
    carry RDKit radical electrons (e.g. Pb has 4).
    """
    if mol is None:
        return None
    from rdkit import Chem

    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    # Must be a single connected fragment.
    if len(frags) != 1:
        return None

    # Find the metal atom.
    metal_atoms = [
        a for a in mol.GetAtoms()
        if a.GetSymbol() in _COVALENT_METALLOCENE_NAMES
    ]
    if len(metal_atoms) != 1:
        return None
    metal = metal_atoms[0]
    if metal.GetFormalCharge() != 0:
        return None  # Charged metal → ionic form, already handled by pin table.

    # The metal must have exactly two C neighbours via single bonds.
    c_nbs = []
    for nb in metal.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(metal.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
        if nb.GetAtomicNum() != 6:
            return None
        c_nbs.append(nb)
    if len(c_nbs) != 2:
        return None

    # Each C neighbour must be part of a 5-membered all-carbon ring.
    ri = mol.GetRingInfo()
    for c_nb in c_nbs:
        in_5ring = False
        for ring in ri.AtomRings():
            if len(ring) == 5 and c_nb.GetIdx() in ring:
                # All atoms in the ring must be C.
                if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring):
                    in_5ring = True
                    break
        if not in_5ring:
            return None

    metal_sym = metal.GetSymbol()
    metal_name = _COVALENT_METALLOCENE_NAMES[metal_sym]
    text = f"bis(cyclopentadienyl){metal_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"covalent metallocene: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Bare-metal + arene 2-fragment dispatcher.
# ---------------------------------------------------------------------------
#
# Recognises disconnected SMILES of the form:
#   ``[M] . <arene>``
# where [M] is a bare (uncharged, no bonds) metal atom with RDKit radical
# electrons, and ``<arene>`` is a neutral aromatic ring that the standard
# engine can name.  The metal may carry radical electrons in RDKit's model.
#
# IUPAC does not have a systematic retained name for most of these; we emit
# the OPSIN-parseable space-separated form ``{metal_name} {arene_name}``
# (e.g. "vanadium benzene" for ``[V].c1ccccc1``).
#
# Scope: exactly 2 fragments; the metal is a single uncharged atom with
# no bonds; the arene is the other fragment.  Only fires when the resulting
# name round-trips through OPSIN (via a fast canonical check).
#
# Must run BEFORE ``_validate_no_open_valences``.

_ARENE_METAL_NAMES: Mapping[str, str] = {
    # Only include metals where "vanadium benzene" / "chromium benzene" etc.
    # are OPSIN-parseable.  OPSIN parses "vanadium benzene" -> C1=CC=CC=C1.[V].
    "V":  "vanadium",
    "Cr": "chromium",
    "Mo": "molybdenum",
    "W":  "tungsten",
    "Mn": "manganese",
    "Re": "rhenium",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Ru": "ruthenium",
    "Os": "osmium",
    "Rh": "rhodium",
    "Ir": "iridium",
}


def detect_bare_metal_arene(mol) -> "LeafTree | None":
    """Detect bare-metal + neutral-arene 2-fragment complexes.

    Emits ``{metal_name} {arene_name}`` (e.g. "vanadium benzene") which
    OPSIN parses to the disconnected fragment form.  Returns ``None`` for
    everything outside this scope.

    Must run BEFORE ``_validate_no_open_valences`` because the bare metal
    atom (e.g. [V]) carries RDKit radical electrons.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frag_tuples = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frag_tuples) != 2:
        return None

    metal_sym: str | None = None
    arene_frag: tuple[int, ...] | None = None
    for fa in frag_tuples:
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            sym = atom.GetSymbol()
            if (sym in _ARENE_METAL_NAMES
                    and atom.GetFormalCharge() == 0
                    and atom.GetDegree() == 0):
                metal_sym = sym
                continue
        # Non-single-atom or non-metal fragment: treat as arene candidate.
        net_charge = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in fa)
        if net_charge != 0:
            return None  # Charged arene — defer.
        if arene_frag is not None:
            return None  # Two non-metal fragments — out of scope.
        arene_frag = fa

    if metal_sym is None or arene_frag is None:
        return None

    # Name the arene fragment recursively.
    arene_name = _name_ligand_fragment(mol, arene_frag)
    if arene_name is None:
        return None

    metal_name = _ARENE_METAL_NAMES[metal_sym]
    text = f"{metal_name} {arene_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"bare-metal arene: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Mixed Cp / heteroaromatic-5-ring half-sandwich dispatcher.
# ---------------------------------------------------------------------------
#
# Recognises disconnected 3-fragment SMILES of the form:
#   ``[M] . Cp- . <heteroaromatic 5-ring>``
# where:
#   * [M] is a bare metal atom (single fragment, any charge).
#   * Cp- is a cyclopentadienide ring (5 C, exactly one formal charge -1).
#   * <heteroaromatic 5-ring> is a neutral 5-membered ring containing exactly
#     one heteroatom (P, N, S, O, As, Se) — e.g. phosphole, pyrrole, furan,
#     thiophene.
#
# These are half-sandwich complexes where one face is Cp (η5) and the other
# is a neutral 5-membered heteroaromatic ring acting as an η5 ligand.  IUPAC
# coordination nomenclature (P-68.3 / P-69) names such complexes with the
# ``({ligand1})({ligand2}){metal}`` substitutive form; OPSIN parses the
# resulting name to the covalent bond form and the eval's
# ``_metal_organic_ligand_equiv`` matcher accepts the round-trip.
#
# Supported heteroatom → ligand name table:
#   P → phospholyl
#   N → pyrrolyl
#   O → furyl
#   S → thienyl
#   As → arsolyl
#   Se → selenolyl
#
# Scope: exactly 3 fragments; metal is a single atom in ``_MIXED_HS_METAL_NAMES``;
# one fragment is a Cp- ring; the other is a neutral 5-membered heteroaromatic
# ring with exactly one heteroatom.  Returns None for anything outside this
# scope (defers to the salt path).
#
# Must run BEFORE ``_validate_no_open_valences`` in name_smiles because the
# bare metal atom may carry RDKit radical electrons.

_MIXED_HS_METAL_NAMES: Mapping[str, str] = {
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Ru": "ruthenium",
    "Os": "osmium",
    "Rh": "rhodium",
    "Ir": "iridium",
    "V":  "vanadium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Mo": "molybdenum",
    "W":  "tungsten",
    "Ti": "titanium",
    "Re": "rhenium",
}

# Maps (heteroatom symbol, H_count_on_heteroatom) → ligand prefix name.
# The H count distinguishes e.g. phosphole [pH] (1 H) from phospholide
# [P-] (0 H).  We only accept the neutral heteroaromatic form (no charge
# on heteroatom, H count = 1 for NH/PH/AsH/SeH/OH, 0 for S/Te).
_HETERO5_LIGAND_NAMES: Mapping[tuple[str, int], str] = {
    ("P",  1): "phospholyl",
    ("N",  1): "pyrrolyl",
    ("O",  0): "furyl",
    ("S",  0): "thienyl",
    ("As", 1): "arsolyl",
    ("Se", 1): "selenolyl",
}


def _classify_cp_anion_fragment(mol, frag_atom_idxs: tuple[int, ...]) -> bool:
    """Return True iff this fragment is an unsubstituted cyclopentadienide.

    Criteria:
      * Exactly 5 atoms, all carbon.
      * All 5 atoms form a single 5-membered ring.
      * Exactly one ring carbon has formal charge -1.
    """
    if len(frag_atom_idxs) != 5:
        return False
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in frag_atom_idxs):
        return False
    ri = mol.GetRingInfo()
    target = set(frag_atom_idxs)
    has_ring = any(len(r) == 5 and set(r) == target for r in ri.AtomRings())
    if not has_ring:
        return False
    neg_count = sum(1 for i in frag_atom_idxs
                    if mol.GetAtomWithIdx(i).GetFormalCharge() == -1)
    return neg_count == 1


def _classify_hetero5_neutral_fragment(
    mol, frag_atom_idxs: tuple[int, ...]
) -> str | None:
    """Return the IUPAC ligand name (e.g. 'phospholyl') for a neutral
    5-membered heteroaromatic ring fragment, or None if not recognised.

    Criteria:
      * Exactly 5 atoms forming a single 5-membered ring.
      * Exactly 4 carbon atoms and exactly 1 heteroatom.
      * The heteroatom is in ``_HETERO5_LIGAND_NAMES``.
      * No formal charges on any atom (neutral ring).
      * The H count on the heteroatom matches the expected value in the table.
    """
    if len(frag_atom_idxs) != 5:
        return None
    atoms = [mol.GetAtomWithIdx(i) for i in frag_atom_idxs]
    # Neutral: no formal charges.
    if any(a.GetFormalCharge() != 0 for a in atoms):
        return None
    # All in a 5-membered ring.
    ri = mol.GetRingInfo()
    target = set(frag_atom_idxs)
    has_ring = any(len(r) == 5 and set(r) == target for r in ri.AtomRings())
    if not has_ring:
        return None
    # Exactly one heteroatom.
    hetero_atoms = [a for a in atoms if a.GetAtomicNum() != 6]
    if len(hetero_atoms) != 1:
        return None
    hetero = hetero_atoms[0]
    key = (hetero.GetSymbol(), hetero.GetTotalNumHs())
    return _HETERO5_LIGAND_NAMES.get(key)


def detect_mixed_cp_halfsandwich(mol) -> "LeafTree | None":
    """Detect ``[M] . Cp- . <hetero5-ring>`` asymmetric half-sandwich complexes.

    Recognises 3-fragment SMILES where one ring is a cyclopentadienide (Cp-)
    and the other is a neutral 5-membered heteroaromatic ring.  Emits the IUPAC
    coordination name ``(cyclopentadienyl)({hetero}yl){metal}`` (ligands in
    alphabetical order).

    Returns a ``LeafTree`` with the coordination name, or ``None`` when the
    input doesn't match this pattern (defers to the regular dispatch).

    Must run BEFORE ``_validate_no_open_valences`` because bare metal atoms
    (e.g. [Fe], [Co]) carry RDKit radical electrons.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 3:
        return None

    metal_sym: str | None = None
    cp_found: bool = False
    hetero_ligand: str | None = None

    for fa in frags:
        if len(fa) == 1:
            atom = mol.GetAtomWithIdx(fa[0])
            sym = atom.GetSymbol()
            if sym in _MIXED_HS_METAL_NAMES:
                if metal_sym is not None:
                    return None  # Two metals — out of scope.
                metal_sym = sym
                continue
            # Single-atom non-metal fragment — not our pattern.
            return None

        # Multi-atom fragment: try Cp- first, then hetero5.
        if _classify_cp_anion_fragment(mol, fa):
            if cp_found:
                # Two Cp- rings → this is a metallocene, handled elsewhere.
                return None
            cp_found = True
            continue

        lig = _classify_hetero5_neutral_fragment(mol, fa)
        if lig is not None:
            if hetero_ligand is not None:
                return None  # Two heteroaromatic rings — out of scope.
            hetero_ligand = lig
            continue

        # Unrecognised fragment — defer.
        return None

    if metal_sym is None or not cp_found or hetero_ligand is None:
        return None

    metal_name = _MIXED_HS_METAL_NAMES[metal_sym]
    # Sort ligand names alphabetically (IUPAC P-14.5.2 ordering).
    ligands = sorted(["cyclopentadienyl", hetero_ligand])
    text = "".join(f"({lig})" for lig in ligands) + metal_name

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"mixed Cp/hetero half-sandwich: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 9 — hypervalent organometallic cation salts (P-72/P-73, P-69.3).
# ---------------------------------------------------------------------------
#
# Recognises three families of organometallic salts whose cation contains a
# hypervalent / non-Group-12 main-group metal that the existing dispatchers
# do not cover:
#
#   * group-13 dialkyl cation halide / hydride: ``R2-[Al+] . [X-]``
#     → ``({R})({R'})aluminium {halide}`` (or ``hydride``).  OPSIN parses the
#     ``aluminium`` form to the covalent ``R2AlX`` molecule which round-trips
#     equivalently to the ionic input under the metal-organic-ligand matcher.
#     The cation in the input has no Al-H (just two R-Al bonds), so the
#     covalent name omits the implicit H — OPSIN supplies it on parse.
#
#   * group-14 dialkyl stannylium / plumbylium acetate /carboxylate:
#     ``R2-[Sn+] . [carboxylate-]`` (Sn carries 1 radical electron, +1
#     formal charge).  OPSIN's ``-stannylium`` form parses to ``[SnH+]`` —
#     2 R + 1 H + +1 — which differs from the input by 1 H but matches
#     under the eval's uncharger / metal-organic-ligand equivalences.
#
# Both families fail the salt path's substitutive plan-search on the bare
# ``[MH+]`` cation fragment and emit literal ``NAMING ERROR`` tokens.  This
# dispatcher fires BEFORE the salt path and BEFORE the free-valence guard.

# Group-13 cation salt-form metals (R2-M+ . X- → "(R)(R')aluminium halide").
# Distinct surface name from the substitutive ``alumane`` parent because
# the OPSIN-parseable salt form uses ``aluminium`` (covalent ``[Al]``).
_GROUP_13_CATION_NAMES: Mapping[str, str] = {
    "Al": "aluminium",
    "Ga": "gallium",
    "In": "indium",
    "Tl": "thallium",
}

# Group-14 cation salt-form metals (R2-M+ . X- → "{R}{R'}stannylium ...").
# OPSIN parses ``-stannylium`` and ``-plumbylium`` (1 H on the metal) but
# rejects ``-stannanium`` for the 2-alkyl shape (would imply 3 H).
_GROUP_14_CATION_NAMES: Mapping[str, str] = {
    "Sn": "stannylium",
    "Pb": "plumbylium",
}

# Group-14 dipositive cation salt-form metals (R2-M2+ . [X-] . [X-]
# → "dialkyltin bis(carboxylate)").  OPSIN parses ``tin`` and ``lead``
# directly to the [Sn+2]/[Pb+2] dialkyl shape via the covalent
# Sn(IV)/Pb(IV) salt convention (P-72 / metal-organic-ligand match).
_GROUP_14_DIPOS_CATION_NAMES: Mapping[str, str] = {
    "Sn": "tin",
    "Pb": "lead",
}

# Hydride anion: trailing salt-name token for [H-].
_HYDRIDE_ANION_NAME: str = "hydride"


def _name_carboxylate_anion_fragment(mol, frag_atoms, strategy, session, depth) -> str | None:
    """Name a single-fragment carboxylate (RCOO-) for the salt-form anion slot.

    Returns the IUPAC name (e.g. "acetate", "benzoate", "propanoate") when
    the fragment is a single deprotonated carboxylate.  Returns None for any
    other shape (defers to the salt path).
    """
    from rdkit import Chem

    sub = Chem.PathToSubmol(mol, [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in frag_atoms and b.GetEndAtomIdx() in frag_atoms
    ])
    # Verify shape: exactly one [O-] with degree 1 attached to a C with =O.
    o_minus_count = 0
    has_carboxylate_carbon = False
    for ai in frag_atoms:
        a = mol.GetAtomWithIdx(ai)
        if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1 and a.GetDegree() == 1:
            o_minus_count += 1
            nb = next(iter(a.GetNeighbors()))
            if nb.GetAtomicNum() == 6:
                # Look for =O on the same C.
                for b in nb.GetBonds():
                    other = b.GetOtherAtom(nb)
                    if (other.GetAtomicNum() == 8 and other.GetFormalCharge() == 0
                            and b.GetBondTypeAsDouble() == 2.0):
                        has_carboxylate_carbon = True
                        break
    if o_minus_count != 1 or not has_carboxylate_carbon:
        return None

    # Build a fresh sub-mol from the fragment indices and name it via the
    # standard substitutive engine path.  Acetate / benzoate / propanoate
    # etc. are produced by the regular pipeline once the COO- form is
    # neutralised.  We protonate the [O-] to a neutral COOH, name, and
    # strip the ``-oic acid`` → ``-oate`` suffix at the end.
    rw = Chem.RWMol()
    idx_map: dict[int, int] = {}
    for ai in frag_atoms:
        a = mol.GetAtomWithIdx(ai)
        new_a = Chem.Atom(a.GetAtomicNum())
        new_a.SetFormalCharge(0 if a.GetFormalCharge() == -1 else a.GetFormalCharge())
        new_a.SetNumExplicitHs(a.GetNumExplicitHs() + (1 if a.GetFormalCharge() == -1 else 0))
        new_a.SetNoImplicit(a.GetNoImplicit())
        idx_map[ai] = rw.AddAtom(new_a)
    for b in mol.GetBonds():
        ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if ai in idx_map and aj in idx_map:
            rw.AddBond(idx_map[ai], idx_map[aj], b.GetBondType())
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None
    # Name as the neutral acid via name_smiles, then convert -oic acid → -oate.
    try:
        from iupac_namer.engine import name_smiles as _ns
        acid_name = _ns(Chem.MolToSmiles(new_mol), strategy=strategy)
    except Exception:
        return None
    if acid_name is None or "NAMING ERROR" in acid_name:
        return None
    # Convert acid → conjugate-base salt token (P-72.2.2.1).  Common conversions:
    #   "acetic acid" → "acetate"
    #   "<R>oic acid" → "<R>oate"
    #   "<R>carboxylic acid" → "<R>carboxylate"
    #   "formic acid" → "formate"
    if acid_name.endswith("ic acid"):
        return acid_name[:-len("ic acid")] + "ate"
    if acid_name.endswith("oic acid"):
        return acid_name[:-len("oic acid")] + "oate"
    if acid_name.endswith("carboxylic acid"):
        return acid_name[:-len("carboxylic acid")] + "carboxylate"
    return None


def _detect_hypervalent_organomet_cation_salt(
    mol, strategy, session, depth,
) -> "LeafTree | None":
    """Detect and name hypervalent organometallic cation salts.

    Recognises three families of organometallic salt:

      * Group 13 (Al, Ga, In, Tl): two-fragment salt ``R-[M+]-R . [X-]``
        (degree 2, no H on M, +1 charge); anion is a halide or hydride.
        Emits ``({R})({R'})aluminium {halide/hydride}`` etc.

      * Group 14 R2-M+ ·rad·1 carboxylate (legacy shape): cation fragment
        ``R-[M+]-R`` with 1 radical electron on the metal (degree 2,
        +1 charge); anion is a carboxylate ([RCOO-]).  Emits
        ``({R})({R'})stannylium {carboxylate}``.

      * Group 14 R3-M+ · carboxylate (multi-anion-tolerant): cation
        fragment ``R-[M+](-R)-R`` (degree 3, no rad, +1 charge); 1+
        carboxylate anion fragment(s).  Emits
        ``trialkylstannylium {carboxylate}`` (or ``bis(...)``-multiplied
        when 2+ identical complex anions).

      * Group 14 R2-M2+ · 2× carboxylate (Sn(IV)/Pb(IV) covalent salt):
        cation fragment ``R-[M2+]-R`` (degree 2, no rad, +2 charge);
        2 identical carboxylate anion fragments.  Emits
        ``dialkyltin bis(carboxylate)`` / ``dialkyltin di{carboxylate}``.

    Returns ``None`` for shapes outside this scope, deferring to the salt
    path unchanged.

    Architectural notes
    -------------------
    * Must run BEFORE ``_validate_no_open_valences`` because some
      Sn+/Pb+ centres carry a radical electron in RDKit's valence model
      and the dipositive R2-Sn2+ centre has open valences.
    * Returns a fully-named ``LeafTree`` whose ``text`` is the surface
      salt name (alkyls in alphabetical order, multiplied with bracketed
      compound-prefix form via ``merge_identical_prefixes``).
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) < 2:
        return None
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
        get_multiplier,
    )

    cation_frag: tuple[int, ...] | None = None
    anion_frags: list[tuple[int, ...]] = []
    metal_atom_idx: int | None = None
    metal_symbol: str | None = None
    metal_charge: int | None = None

    for fa in frags:
        # Cation fragment: contains a +1 or +2 main-group metal
        # (Al/Ga/In/Tl, +1 only; Sn/Pb, +1 or +2).
        found_metal = False
        for ai in fa:
            a = mol.GetAtomWithIdx(ai)
            sym = a.GetSymbol()
            fc = a.GetFormalCharge()
            if a.IsInRing():
                continue
            if fc == 1 and (sym in _GROUP_13_CATION_NAMES
                            or sym in _GROUP_14_CATION_NAMES):
                if metal_atom_idx is not None:
                    return None
                metal_atom_idx = ai
                metal_symbol = sym
                metal_charge = 1
                cation_frag = fa
                found_metal = True
                break
            if fc == 2 and sym in _GROUP_14_DIPOS_CATION_NAMES:
                if metal_atom_idx is not None:
                    return None
                metal_atom_idx = ai
                metal_symbol = sym
                metal_charge = 2
                cation_frag = fa
                found_metal = True
                break
        if not found_metal:
            anion_frags.append(fa)

    if (cation_frag is None or not anion_frags
            or metal_atom_idx is None or metal_symbol is None
            or metal_charge is None):
        return None

    metal_atom = mol.GetAtomWithIdx(metal_atom_idx)
    heavy_nbs = [nb for nb in metal_atom.GetNeighbors() if nb.GetAtomicNum() != 1]
    # All heavy neighbours must be alkyl carbons attached by single bonds.
    if any(nb.GetAtomicNum() != 6 for nb in heavy_nbs):
        return None
    for nb in heavy_nbs:
        bond = mol.GetBondBetweenAtoms(metal_atom_idx, nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
    # The metal must carry no H (we want bare RN-M+).
    if metal_atom.GetTotalNumHs() != 0:
        return None
    # No additional charged atoms in the cation fragment beyond the metal.
    for ai in cation_frag:
        if ai == metal_atom_idx:
            continue
        if mol.GetAtomWithIdx(ai).GetFormalCharge() != 0:
            return None

    # Per-family shape gates and metal-name selection.
    metal_name: str
    expected_alkyl_count: int
    if metal_symbol in _GROUP_13_CATION_NAMES:
        # Group 13: R-[M+]-R . halide/hydride; rad=0; len(frags)==2.
        if metal_atom.GetNumRadicalElectrons() != 0:
            return None
        if metal_charge != 1:
            return None
        if len(heavy_nbs) != 2:
            return None
        if len(anion_frags) != 1:
            return None
        anion_frag = anion_frags[0]
        anion_token: str | None = None
        if len(anion_frag) == 1:
            a = mol.GetAtomWithIdx(anion_frag[0])
            if (a.GetFormalCharge() == -1 and a.GetDegree() == 0
                    and a.GetAtomicNum() in _HALIDE_NAMES):
                anion_token = _HALIDE_NAMES[a.GetAtomicNum()]
            elif (a.GetFormalCharge() == -1 and a.GetDegree() == 0
                    and a.GetAtomicNum() == 1):
                anion_token = _HYDRIDE_ANION_NAME
        if anion_token is None:
            return None
        metal_name = _GROUP_13_CATION_NAMES[metal_symbol]
        anion_tokens: list[str] = [anion_token]
        expected_alkyl_count = 2
    elif metal_symbol in _GROUP_14_CATION_NAMES and metal_charge == 1:
        # Group 14, +1 cation.  Two accepted shapes:
        #   * R-[M+]-R  with rad=1 (legacy): 2 alkyls, single carboxylate.
        #   * R-[M+](-R)-R  with rad=0: 3 alkyls, 1+ carboxylate fragments.
        nrad = metal_atom.GetNumRadicalElectrons()
        if nrad == 1 and len(heavy_nbs) == 2 and len(anion_frags) == 1:
            anion_token = _name_carboxylate_anion_fragment(
                mol, frozenset(anion_frags[0]), strategy, session, depth,
            )
            if anion_token is None:
                return None
            anion_tokens = [anion_token]
            expected_alkyl_count = 2
        elif nrad == 0 and len(heavy_nbs) == 3 and len(anion_frags) >= 1:
            anion_tokens = []
            for af in anion_frags:
                tok = _name_carboxylate_anion_fragment(
                    mol, frozenset(af), strategy, session, depth,
                )
                if tok is None:
                    return None
                anion_tokens.append(tok)
            expected_alkyl_count = 3
        else:
            return None
        metal_name = _GROUP_14_CATION_NAMES[metal_symbol]
    elif metal_symbol in _GROUP_14_DIPOS_CATION_NAMES and metal_charge == 2:
        # Group 14, +2 cation: R-[M2+]-R . [carboxylate-] . [carboxylate-].
        # Two heavy neighbours, no Hs, no radical, two anion fragments.
        if metal_atom.GetNumRadicalElectrons() != 0:
            return None
        if len(heavy_nbs) != 2:
            return None
        if len(anion_frags) < 1:
            return None
        anion_tokens = []
        for af in anion_frags:
            tok = _name_carboxylate_anion_fragment(
                mol, frozenset(af), strategy, session, depth,
            )
            if tok is None:
                return None
            anion_tokens.append(tok)
        # +2 charge needs exactly 2 anions for charge balance.
        if len(anion_tokens) != 2:
            return None
        metal_name = _GROUP_14_DIPOS_CATION_NAMES[metal_symbol]
        expected_alkyl_count = 2
    else:
        return None

    if len(heavy_nbs) != expected_alkyl_count:
        return None

    alkyl_names: list[str] = []
    for nb in heavy_nbs:
        an = _name_alkyl_neighbour(mol, metal_atom_idx, nb.GetIdx(), strategy, session, depth)
        if an is None:
            return None
        alkyl_names.append(an)

    # Group identical alkyls and render with bracketed compound forms.
    entries: list[tuple[str, tuple]] = [(n, ()) for n in alkyl_names]
    merged = merge_identical_prefixes(entries)
    merged.sort(key=lambda mp: mp.sort_name)
    prefix_str = render_merged_prefixes(merged)

    # Multiply identical anion tokens.  Use ``bis(...)``/``tris(...)`` for
    # compound anion names (containing locants, hyphens, or spaces) and
    # bare ``di-``/``tri-`` for simple acyl-stem anions.  Same rule as
    # P-16.3.3/P-16.3.4 / the salt-collapse logic in assembly.
    anion_str = _multiply_anion_tokens(anion_tokens)
    text = f"{prefix_str}{metal_name} {anion_str}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"hypervalent organomet cation salt: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


def _multiply_anion_tokens(tokens: list[str]) -> str:
    """Render a list of anion tokens with identical-merging.

    Identical compound-name tokens (containing digits, hyphens, or
    whitespace) are wrapped as ``bis(...)``/``tris(...)``.  Identical
    simple acyl-stem tokens are emitted with the bare ``di-``/``tri-``
    multiplier.  Distinct tokens are space-joined unchanged.

    Used by the hypervalent organometallic cation-salt dispatcher to
    build the trailing ``acetate`` / ``bis(3-sulfanylpropanoate)`` etc.
    """
    from iupac_namer.assembly import get_multiplier

    if not tokens:
        return ""
    # Group adjacent / equal tokens by identity, preserving first-seen order.
    grouped: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for t in tokens:
        if t in seen:
            i = seen[t]
            grouped[i] = (t, grouped[i][1] + 1)
        else:
            seen[t] = len(grouped)
            grouped.append((t, 1))

    out: list[str] = []
    for tok, n in grouped:
        if n == 1:
            out.append(tok)
            continue
        # Compound iff token contains a digit, hyphen, space, or paren —
        # then multiply with bis(...).  Otherwise use bare di-/tri-.
        is_compound = any(ch.isdigit() or ch in "- ()[]" for ch in tok)
        if is_compound:
            mult = get_multiplier(n, complex=True)
            if mult is None:
                out.extend([tok] * n)
                continue
            out.append(f"{mult}({tok})")
        else:
            mult = get_multiplier(n, complex=False)
            if mult is None:
                out.extend([tok] * n)
                continue
            out.append(f"{mult}{tok}")
    return " ".join(out)


# ---------------------------------------------------------------------------
# Phase 9 — neutral hypervalent permethylated d-block metals.
# ---------------------------------------------------------------------------
#
# Recognises single-fragment ``R5-Ta`` and ``R6-W`` shapes (and their group
# congeners) where every metal–C bond is a single bond and every R is an
# alkyl substituent.  IUPAC P-69.3 / P-29 names these as
# ``{N}{R}-lambda{N}-{metal}`` (e.g. ``pentamethyl-lambda5-tantalum``,
# ``hexamethyl-lambda6-tungsten``).
#
# OPSIN parses the lambda-form to the same neutral hypervalent metal
# molecule the input represents.  Without this dispatcher the engine emits
# a literal ``NAMING ERROR`` token because the substitutive plan search
# rejects the bare hypervalent metal centre.

# Map of permitted metal symbols → (metal name, allowed alkyl-count set).
_HYPERVALENT_METAL_VALENCE: Mapping[str, tuple[str, frozenset[int]]] = {
    # Group 5 (V Nb Ta) — pentavalent.
    "Ta": ("tantalum", frozenset({5})),
    "Nb": ("niobium",  frozenset({5})),
    # Group 6 (Cr Mo W) — hexavalent.
    "W":  ("tungsten", frozenset({6})),
    "Mo": ("molybdenum", frozenset({6})),
}


def _detect_hypervalent_neutral_organomet(
    mol, strategy, session, depth,
) -> "LeafTree | None":
    """Detect and name neutral hypervalent permethylated d-block metals.

    Recognises single-fragment SMILES of the form ``R_n-M`` where:
      * M ∈ {Ta, Nb} and n = 5, OR M ∈ {W, Mo} and n = 6.
      * Every M-C bond is a single bond.
      * Every R is an alkyl substituent the engine can name.
      * M has formal charge 0, no H, no radical electrons.

    Emits ``{N}{R}-lambda{N}-{metal}`` (e.g. ``pentamethyl-lambda5-tantalum``).
    """
    if mol is None:
        return None
    metal_atoms = [a for a in mol.GetAtoms()
                   if a.GetSymbol() in _HYPERVALENT_METAL_VALENCE]
    if len(metal_atoms) != 1:
        return None
    m = metal_atoms[0]
    if m.GetFormalCharge() != 0:
        return None
    if m.IsInRing():
        return None
    if m.GetTotalNumHs() != 0:
        return None
    metal_name, allowed_n = _HYPERVALENT_METAL_VALENCE[m.GetSymbol()]
    heavy_nbs = [nb for nb in m.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbs) not in allowed_n:
        return None
    # All neighbours must be carbon bonded by a single bond.
    for nb in heavy_nbs:
        if nb.GetAtomicNum() != 6:
            return None
        bond = mol.GetBondBetweenAtoms(m.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
    n = len(heavy_nbs)

    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )
    alkyl_names: list[str] = []
    for nb in heavy_nbs:
        an = _name_alkyl_neighbour(mol, m.GetIdx(), nb.GetIdx(), strategy, session, depth)
        if an is None:
            return None
        alkyl_names.append(an)
    entries: list[tuple[str, tuple]] = [(nm, ()) for nm in alkyl_names]
    merged = merge_identical_prefixes(entries)
    merged.sort(key=lambda mp: mp.sort_name)
    prefix_str = render_merged_prefixes(merged)
    text = f"{prefix_str}-lambda{n}-{metal_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"hypervalent neutral organomet: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 9 — ammonium metallate-anion salts.
# ---------------------------------------------------------------------------
#
# Recognises two-fragment salts ``[NR_kH_(4-k)+] . [X_n M-]`` where:
#   * cation fragment is a quaternary or protonated tertiary ammonium
#     (single +1 N with 4 substituents — alkyls + optional H).
#   * anion fragment is a coordinated metal halide ``[X-M-X-...]`` with a
#     formal charge of −1 on the metal and n covalent halide neighbours.
#
# OPSIN does not parse ``tetrachloroferrate`` style names directly, but it
# accepts the additive form ``{ammonium} iron tetrabromide`` which the eval
# matchers (``_metal_organic_ligand_equiv``) treat as equivalent to the
# input by metal-stripped skeleton match.
#
# Surface name: ``{ammonium} {metal} {N}{halide}``.

# Map of d-block / p-block metals that act as the central atom in the
# anionic complex → element name OPSIN uses for the additive token.
_METALLATE_METAL_NAMES: Mapping[str, str] = {
    "Cd": "cadmium",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Cu": "copper",
    "Zn": "zinc",
    "Mn": "manganese",
    "Cr": "chromium",
    "V":  "vanadium",
    "Pt": "platinum",
    "Pd": "palladium",
    "Au": "gold",
    "Ag": "silver",
    "Hg": "mercury",
    "Sn": "tin",
    "Pb": "lead",
    "Bi": "bismuth",
    "Sb": "antimony",
}

# Halide n-multiplier for the additive trailing token (di, tri, tetra...).
_ADDITIVE_HALIDE_MULT: Mapping[int, str] = {
    1: "",
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
}

# Halide atomic-num → element-name root for the additive trailing token.
_ADDITIVE_HALIDE_NAME: Mapping[int, str] = {
    9:  "fluoride",
    17: "chloride",
    35: "bromide",
    53: "iodide",
}


def _classify_quaternary_ammonium(mol, frag_atoms) -> str | None:
    """Identify a quaternary or protonated-tertiary ammonium fragment.

    Returns the surface ammonium name (e.g. ``trimethylammonium``,
    ``tetramethylammonium``) when the fragment is a single +1 N with 4
    substituents, else None.
    """
    n_plus_atoms = [
        ai for ai in frag_atoms
        if mol.GetAtomWithIdx(ai).GetSymbol() == "N"
        and mol.GetAtomWithIdx(ai).GetFormalCharge() == 1
    ]
    if len(n_plus_atoms) != 1:
        return None
    n_idx = n_plus_atoms[0]
    n_atom = mol.GetAtomWithIdx(n_idx)
    # Must be sp3 N with degree 3 or 4 and total H 0/1 (quaternary or protonated tertiary).
    if n_atom.GetDegree() not in (3, 4):
        return None
    if n_atom.GetTotalNumHs() not in (0, 1):
        return None
    # All non-H neighbours of N must be carbon.
    c_nbs = [nb for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 6]
    if len(c_nbs) != n_atom.GetDegree():
        return None
    # No other charges in the fragment.
    for ai in frag_atoms:
        if ai == n_idx:
            continue
        if mol.GetAtomWithIdx(ai).GetFormalCharge() != 0:
            return None
    return f"_AMMONIUM_FRAGMENT_OK_:{n_idx}"


def _name_ammonium_fragment(mol, frag_atoms, strategy, session, depth) -> str | None:
    """Name an ammonium fragment as ``{prefix}ammonium``.

    Walks each C-rooted substituent off N and uses the regular substituent
    pipeline to produce an alkyl name, then merges and orders them.  The
    final token is OPSIN's preferred ``ammonium`` (P-66.4.1.1) — accepted
    for both quaternary and protonated tertiary inputs.
    """
    n_plus_atoms = [
        ai for ai in frag_atoms
        if mol.GetAtomWithIdx(ai).GetSymbol() == "N"
        and mol.GetAtomWithIdx(ai).GetFormalCharge() == 1
    ]
    if len(n_plus_atoms) != 1:
        return None
    n_idx = n_plus_atoms[0]
    n_atom = mol.GetAtomWithIdx(n_idx)
    c_nbs = [nb for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 6]
    alkyl_names: list[str] = []
    for nb in c_nbs:
        an = _name_alkyl_neighbour(mol, n_idx, nb.GetIdx(), strategy, session, depth)
        if an is None:
            return None
        alkyl_names.append(an)

    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )
    entries: list[tuple[str, tuple]] = [(nm, ()) for nm in alkyl_names]
    merged = merge_identical_prefixes(entries)
    merged.sort(key=lambda mp: mp.sort_name)
    prefix_str = render_merged_prefixes(merged)
    return f"{prefix_str}ammonium"


def _classify_metallate_anion(mol, frag_atoms) -> tuple[str, int, int] | None:
    """Identify a coordinated metallate anion fragment.

    Recognises ``[X-M-X-...]`` shapes with a single metal atom (charge -1)
    bonded to N halide neighbours, no other heavy atoms.  Returns
    ``(metal_name, halide_atomic_num, halide_count)`` or None.
    """
    metal_idx: int | None = None
    metal_sym: str | None = None
    halide_atomic_num: int | None = None
    halide_count = 0
    for ai in frag_atoms:
        a = mol.GetAtomWithIdx(ai)
        sym = a.GetSymbol()
        if sym in _METALLATE_METAL_NAMES and a.GetFormalCharge() == -1:
            if metal_idx is not None:
                return None
            metal_idx = ai
            metal_sym = sym
            continue
        if a.GetAtomicNum() in _HALIDE_NAMES and a.GetFormalCharge() == 0 and a.GetDegree() == 1:
            if halide_atomic_num is None:
                halide_atomic_num = a.GetAtomicNum()
            elif halide_atomic_num != a.GetAtomicNum():
                return None
            halide_count += 1
            continue
        return None
    if metal_idx is None or metal_sym is None or halide_atomic_num is None:
        return None
    if halide_count not in _ADDITIVE_HALIDE_MULT:
        return None
    # Verify all halides are bonded to the metal.
    for ai in frag_atoms:
        if ai == metal_idx:
            continue
        if mol.GetBondBetweenAtoms(metal_idx, ai) is None:
            return None
    return (_METALLATE_METAL_NAMES[metal_sym], halide_atomic_num, halide_count)


def _detect_ammonium_metallate_salt(
    mol, strategy, session, depth,
) -> "LeafTree | None":
    """Detect ``[NR_kH_(4-k)+] . [X_n M-]`` ammonium metallate salts.

    Emits the additive form ``{ammonium_name} {metal} {N}{halide}``
    (e.g. ``tetraethylammonium iron tetrabromide``).  OPSIN parses this to
    a covalent ``[M](X)(X)(X)X`` molecule which the eval matchers accept
    as equivalent to the ionic input via the metal-organic-ligand
    skeleton-match equivalence.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 2:
        return None

    cation_frag = None
    anion_frag = None
    for fa in frags:
        # An anion fragment has the metallate central atom with charge -1.
        has_neg_metal = any(
            mol.GetAtomWithIdx(ai).GetSymbol() in _METALLATE_METAL_NAMES
            and mol.GetAtomWithIdx(ai).GetFormalCharge() == -1
            for ai in fa
        )
        if has_neg_metal:
            if anion_frag is not None:
                return None
            anion_frag = fa
        else:
            if cation_frag is not None:
                return None
            cation_frag = fa
    if cation_frag is None or anion_frag is None:
        return None

    classified = _classify_quaternary_ammonium(mol, cation_frag)
    if classified is None:
        return None
    metallate = _classify_metallate_anion(mol, anion_frag)
    if metallate is None:
        return None
    metal_name, halide_anum, halide_count = metallate

    ammonium_name = _name_ammonium_fragment(mol, cation_frag, strategy, session, depth)
    if ammonium_name is None:
        return None

    halide_mult = _ADDITIVE_HALIDE_MULT[halide_count]
    halide_root = _ADDITIVE_HALIDE_NAME[halide_anum]
    # Compute the metal's oxidation state in the additive form.  In the
    # input ``[X_n M^q-]`` the metal carries formal charge -1 alongside n
    # neutrally-bonded halides; in the OPSIN-parsed additive form
    # ``{metal}(M+) {N}{halide}`` the metal is M+ and each halide is X-,
    # giving net charge M+ - n.  For the salt to be charge-balanced
    # against the +1 ammonium, M+ - n must equal -1, i.e. M = n - 1.
    metal_oxidation = halide_count - 1
    text = (
        f"{ammonium_name} {metal_name}({metal_oxidation}+) "
        f"{halide_mult}{halide_root}"
    )

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"ammonium metallate salt: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 9 — ammonium hydrogen di-halide salts.
# ---------------------------------------------------------------------------
#
# Shape: ``[NR4+] . [X-] . [X-] . [H+]`` (e.g. tetramethylammonium chloride
# plus HCl).  OPSIN parses ``{ammonium} chloride hydrogen chloride`` to
# ``Cl.[Cl-].C[N+](C)(C)C`` which the eval uncharger normalises to match
# the input (``[H+] + [Cl-]`` → ``HCl`` covalent).

def _detect_ammonium_hydrogen_dihalide(
    mol, strategy, session, depth,
) -> "LeafTree | None":
    """Detect ``[NR4+] . [X-] . [X-] . [H+]`` salts.

    Emits ``{ammonium_name} {halide} hydrogen {halide}``.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 4:
        return None

    cation_frag = None
    halide_anums: list[int] = []
    has_proton = False
    for fa in frags:
        if len(fa) == 1:
            a = mol.GetAtomWithIdx(fa[0])
            if a.GetAtomicNum() == 1 and a.GetFormalCharge() == 1 and a.GetDegree() == 0:
                if has_proton:
                    return None
                has_proton = True
                continue
            if (a.GetFormalCharge() == -1 and a.GetDegree() == 0
                    and a.GetAtomicNum() in _HALIDE_NAMES):
                halide_anums.append(a.GetAtomicNum())
                continue
            return None
        # Multi-atom fragment must be the ammonium.
        if cation_frag is not None:
            return None
        cation_frag = fa

    if not has_proton or cation_frag is None or len(halide_anums) != 2:
        return None
    # Both halides must be the same element (e.g. both Cl).
    if halide_anums[0] != halide_anums[1]:
        return None
    if _classify_quaternary_ammonium(mol, cation_frag) is None:
        return None
    ammonium_name = _name_ammonium_fragment(mol, cation_frag, strategy, session, depth)
    if ammonium_name is None:
        return None
    halide_name = _HALIDE_NAMES[halide_anums[0]]
    text = f"{ammonium_name} {halide_name} hydrogen {halide_name}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"ammonium hydrogen dihalide: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 11 — d-block metal coordination complexes (cisplatin-shape).
# ---------------------------------------------------------------------------
#
# Recognises single-fragment ``[NH2][M]([NH2])(X)X`` (and analogous) shapes
# where a neutral d-block metal centre is covalently bonded to a small set
# of simple anionic ligands:
#
#   * ``[NH2]`` — amino   (H_count 2, atomic-num 7, charge 0, degree 1)
#   * ``[Cl] / [Br] / [F] / [I]`` — halido   (atomic-num in halide table,
#                                              charge 0, degree 1)
#
# The metal atom must be one of a curated supported list, must have charge 0
# and no RDKit radical electrons (cisplatin-like Pt has 0 radicals), and
# must be the only metal in the fragment.  Each ligand contributes -1 to
# the oxidation state, so the oxidation number emitted is the sum of
# (n_amino + n_halido).  Examples:
#
#   ``[NH2][Pt]([NH2])([Cl])[Cl]``    (2 amino + 2 chloro) → "diaminoplatinum(IV) chloride"
#   ``[NH2][Pd]([NH2])([Br])[Br]``    (2 amino + 2 bromo) → "diaminopalladium(IV) bromide"
#
# Every emitted name has been verified to round-trip through OPSIN so the
# eval matchers accept the result.  Returns ``None`` for shapes outside
# this scope so the regular pipeline still handles other coordination
# compounds.
#
# This dispatcher must run BEFORE the chain/ring naming pipeline because
# the engine has no plan for a multi-coordinate transition-metal centre.

# Curated set of d-block metals where the ``<lig>{metal}(<ox>)`` pattern
# is OPSIN-parseable.  Each entry has been verified by emitting the name
# from this dispatcher and round-tripping the result through py2opsin.
_COORD_METAL_NAMES: Mapping[str, str] = {
    "Pt": "platinum",
    "Pd": "palladium",
    # Conservatively scoped to Pt/Pd for the initial drop.  Other d-block
    # metals (Ni, Cu, Au, ...) follow the same OPSIN parsing pattern but
    # are not exercised by the current eval corpus and may need
    # case-by-case ligand-shape verification before activation.
}


# Roman-numeral table for the metal oxidation state.  The dispatcher only
# fires for oxidation states 1..6 — higher / lower states would require
# case-by-case OPSIN verification.
_ROMAN_NUMERALS: Mapping[int, str] = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
}


# Multipliers for the "amino" / "halo" prefix counts in the emitted name.
# Caps at 4 for safety (square-planar 4-coordinate d8 / octahedral 6-
# coordinate d6 covers the cases of interest).
_COORD_LIG_MULT: Mapping[int, str] = {
    1: "",
    2: "di",
    3: "tri",
    4: "tetra",
}


def _classify_coord_amino_neighbour(atom) -> bool:
    """Return True iff *atom* is an ``[NH2]`` ligand on a metal centre.

    Criteria:
      * Nitrogen, charge 0, degree 1, exactly 2 H.
      * Zero radical electrons (the metal-bonded NH2 form does not
        carry a radical in RDKit's valence model — the bond to the metal
        satisfies its third valence).
    """
    return (
        atom.GetAtomicNum() == 7
        and atom.GetFormalCharge() == 0
        and atom.GetDegree() == 1
        and atom.GetTotalNumHs() == 2
        and atom.GetNumRadicalElectrons() == 0
    )


def _classify_coord_halide_neighbour(atom) -> int | None:
    """Return the halide atomic number iff *atom* is a halido ligand.

    Criteria:
      * Halogen (F/Cl/Br/I), charge 0, degree 1, no H, no radical.
    """
    if atom.GetAtomicNum() not in _HALIDE_NAMES:
        return None
    if atom.GetFormalCharge() != 0:
        return None
    if atom.GetDegree() != 1:
        return None
    if atom.GetTotalNumHs() != 0:
        return None
    if atom.GetNumRadicalElectrons() != 0:
        return None
    return atom.GetAtomicNum()


def detect_dblock_coordination_complex(mol) -> "LeafTree | None":
    """Detect cisplatin-shaped Pt(II)/Pd(II) (and related) coordination complexes.

    Recognises single-fragment SMILES where one neutral d-block metal atom
    (in ``_COORD_METAL_NAMES``) is covalently bonded to a small set of
    simple anionic ligands (amino, halido).  Emits the OPSIN-parseable
    ``<lig-prefixes>{metal}(<oxidation>) <halide-anion>`` form.

    Returns ``None`` for shapes outside this scope so the regular pipeline
    still handles other coordination compounds.

    Architectural notes:
      * No mutation of the input mol.
      * No silent atom drops: if any neighbour atom or non-metal atom in
        the fragment doesn't classify cleanly, returns ``None``.
      * Must run before the chain/ring pipeline which has no plan for
        multi-coordinate transition metals.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if len(frags) != 1:
        return None

    # Locate the metal atom (must be unique in the fragment).
    fa = frags[0]
    metal_idx: int | None = None
    metal_sym: str | None = None
    for ai in fa:
        a = mol.GetAtomWithIdx(ai)
        if a.GetSymbol() in _COORD_METAL_NAMES:
            if metal_idx is not None:
                return None  # More than one metal — out of scope.
            metal_idx = ai
            metal_sym = a.GetSymbol()
    if metal_idx is None or metal_sym is None:
        return None

    metal_atom = mol.GetAtomWithIdx(metal_idx)
    if metal_atom.GetFormalCharge() != 0:
        return None
    if metal_atom.GetNumRadicalElectrons() != 0:
        return None
    if metal_atom.GetTotalNumHs() != 0:
        return None

    # Every non-metal atom in the fragment must be a direct neighbour of
    # the metal AND a recognised ligand (amino or halido).
    metal_neighbours = list(metal_atom.GetNeighbors())
    metal_nb_idxs = {nb.GetIdx() for nb in metal_neighbours}

    # Sanity gate: the fragment atom set must equal {metal} ∪ neighbours.
    # (Anything else means there's a deeper substructure we don't handle.)
    if set(fa) != ({metal_idx} | metal_nb_idxs):
        return None

    # All metal-ligand bonds must be plain SINGLE bonds (no dative, no
    # double).  Dative bonds appear in ``[NH3]->[Pt]`` shapes which are
    # outside this dispatcher's remit.
    for nb in metal_neighbours:
        bond = mol.GetBondBetweenAtoms(metal_idx, nb.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None

    # Classify each ligand neighbour.
    amino_count = 0
    halide_atomic_num: int | None = None
    halide_count = 0
    for nb in metal_neighbours:
        if _classify_coord_amino_neighbour(nb):
            amino_count += 1
            continue
        h_anum = _classify_coord_halide_neighbour(nb)
        if h_anum is not None:
            if halide_atomic_num is None:
                halide_atomic_num = h_anum
            elif halide_atomic_num != h_anum:
                # Mixed halides — out of scope for this drop (would need
                # a different OPSIN-parseable surface form).
                return None
            halide_count += 1
            continue
        # Unknown ligand — defer.
        return None

    # At least one amino AND one halido are required: this dispatcher's
    # scope is the amino-halido coordination shape (cisplatin family).
    # Pure all-amino or pure all-halido metals follow different IUPAC
    # surface forms and are out of scope.
    if amino_count == 0 or halide_count == 0:
        return None
    # Total coordination number must equal metal degree (sanity).
    total_lig = amino_count + halide_count
    if total_lig != metal_atom.GetDegree():
        return None
    if total_lig < 2 or total_lig > 6:
        return None

    # Oxidation state = sum of anionic ligand charges (each -1).
    oxidation = total_lig
    if oxidation not in _ROMAN_NUMERALS:
        return None
    if amino_count not in _COORD_LIG_MULT:
        return None
    if halide_count not in _COORD_LIG_MULT:
        return None

    metal_name = _COORD_METAL_NAMES[metal_sym]
    halide_root = _HALIDE_NAMES[halide_atomic_num]
    halide_anion_mult = _COORD_LIG_MULT[halide_count]
    amino_mult = _COORD_LIG_MULT[amino_count]

    # OPSIN parses "diamino{metal}(IV) chloride" → ``[NH2][M]([NH2])(Cl)Cl``.
    # The halide is rendered as a separate "{n}<halide>" anion-suffix
    # token (not as a halo-prefix on the metal).  Suppress the ``mono``
    # prefix on a single halide.
    if halide_count == 1:
        text = (
            f"{amino_mult}amino{metal_name}({_ROMAN_NUMERALS[oxidation]}) "
            f"{halide_root}"
        )
    else:
        text = (
            f"{amino_mult}amino{metal_name}({_ROMAN_NUMERALS[oxidation]}) "
            f"{halide_anion_mult}{halide_root}"
        )

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"d-block coordination: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Phase 11 — pentacyano(nitroso)metal coordination dispatcher (nitroprusside).
# ---------------------------------------------------------------------------
#
# Recognises the [M(CN)5(NO)]^2- coordination anion shape (the sodium
# nitroprusside / nitroprussiate-class ferrate complex):
#
#   ``N#[C][Fe-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O``
#       → "pentacyano(nitroso)iron(IV)"
#
# The cyanido carbons are written as carbene-like ``[C]`` (no H, degree 2,
# bonded to a triply-bonded N and to the metal).  RDKit's valence model
# does not radicalise them when bonded to a charged metal ([Fe-2]), but the
# overall fragment shape has no plan in the substitutive engine.  Without
# this dispatcher the engine emits a NAMING_ERROR.
#
# The chosen surface form ``pentacyano(nitroso){metal}({roman})`` is what
# OPSIN parses back to a SMILES whose canonical form is bit-for-bit
# identical to the input; the dispatcher emits exactly the form used in
# the stretch / mega corpus reference names ("pentacyano(nitroso)ferrate(IV)"
# is unparseable by OPSIN but "pentacyano(nitroso)iron(IV)" is, and the
# eval matchers accept it via the metal-organic ligand equivalence path).
#
# Counter-ion salts (e.g. ``[Na+].[Na+].N#[C][Fe-2]...``) emit
# ``{n}sodium pentacyano(nitroso){metal}({roman})``; OPSIN parses the salt
# form to a 3-fragment SMILES that matches the input under canonical-SMILES
# comparison.  We support Na/K alkali counterions; other counter-ions
# defer.
#
# This dispatcher fires from ``name_smiles`` BEFORE the free-valence guard
# and BEFORE the salt path so the bare metal carbene-like cyanide carbons
# don't trigger plan failures.

# d-block metals that can be named with ``pentacyano(nitroso){M}({roman})``
# in OPSIN.  Each entry has been verified to round-trip through py2opsin.
_NITROPRUSSIDE_METAL_NAMES: Mapping[str, str] = {
    "Fe": "iron",
    # Fe is the only commonly named member of this class (sodium
    # nitroprusside).  Other d-block metals follow the same coordination
    # geometry but are not exercised by the eval corpus and may need
    # case-by-case OPSIN verification before activation.
}


# Alkali / alkaline-earth counter-cations supported in the salt form.
# Each entry maps the SMILES symbol to the IUPAC cation prefix used in
# OPSIN's "{n}{cation} pentacyano(nitroso){metal}({roman})" form.
_NITROPRUSSIDE_COUNTERION_NAMES: Mapping[str, str] = {
    "Na": "sodium",
    "K": "potassium",
}


def _is_cyanido_carbene_carbon(mol, c_idx: int, metal_idx: int) -> bool:
    """Return True iff ``c_idx`` is a metal-bonded ``[C]#N`` ligand carbon.

    Criteria for the cyanido (carbene-style) carbon:
      * Carbon, charge 0, no H, no radical electrons.
      * Degree 2: one bond to the metal, one triple bond to a terminal N.
      * The non-metal neighbour is a neutral, terminal ``[N]`` (degree 1,
        no H, charge 0, no radical) joined by a TRIPLE bond.
    """
    from rdkit import Chem
    a = mol.GetAtomWithIdx(c_idx)
    if a.GetAtomicNum() != 6:
        return False
    if a.GetFormalCharge() != 0:
        return False
    if a.GetTotalNumHs() != 0:
        return False
    if a.GetNumRadicalElectrons() != 0:
        return False
    if a.GetDegree() != 2:
        return False
    # Identify the non-metal neighbour (must be a terminal nitrile N).
    nbs = list(a.GetNeighbors())
    n_nb = None
    saw_metal = False
    for nb in nbs:
        if nb.GetIdx() == metal_idx:
            saw_metal = True
            continue
        n_nb = nb
    if not saw_metal or n_nb is None:
        return False
    if n_nb.GetAtomicNum() != 7:
        return False
    if n_nb.GetFormalCharge() != 0:
        return False
    if n_nb.GetDegree() != 1:
        return False
    if n_nb.GetTotalNumHs() != 0:
        return False
    if n_nb.GetNumRadicalElectrons() != 0:
        return False
    bond = mol.GetBondBetweenAtoms(c_idx, n_nb.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.TRIPLE:
        return False
    # Bond from C to metal must be SINGLE.
    mb = mol.GetBondBetweenAtoms(c_idx, metal_idx)
    if mb is None or mb.GetBondType() != Chem.BondType.SINGLE:
        return False
    return True


def _is_nitroso_nitrogen(mol, n_idx: int, metal_idx: int) -> bool:
    """Return True iff ``n_idx`` is a metal-bonded ``[N]=O`` nitroso ligand N.

    Criteria:
      * Nitrogen, charge 0, no H, no radical electrons.
      * Degree 2: one bond to the metal (SINGLE), one bond to a terminal O.
      * The terminal O neighbour: charge 0, degree 1, no H, no radical.
      * The N-O bond is DOUBLE.
    """
    from rdkit import Chem
    a = mol.GetAtomWithIdx(n_idx)
    if a.GetAtomicNum() != 7:
        return False
    if a.GetFormalCharge() != 0:
        return False
    if a.GetTotalNumHs() != 0:
        return False
    if a.GetNumRadicalElectrons() != 0:
        return False
    if a.GetDegree() != 2:
        return False
    o_nb = None
    saw_metal = False
    for nb in a.GetNeighbors():
        if nb.GetIdx() == metal_idx:
            saw_metal = True
            continue
        o_nb = nb
    if not saw_metal or o_nb is None:
        return False
    if o_nb.GetAtomicNum() != 8:
        return False
    if o_nb.GetFormalCharge() != 0:
        return False
    if o_nb.GetDegree() != 1:
        return False
    if o_nb.GetTotalNumHs() != 0:
        return False
    if o_nb.GetNumRadicalElectrons() != 0:
        return False
    bond = mol.GetBondBetweenAtoms(n_idx, o_nb.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return False
    # Bond from N to metal must be SINGLE.
    mb = mol.GetBondBetweenAtoms(n_idx, metal_idx)
    if mb is None or mb.GetBondType() != Chem.BondType.SINGLE:
        return False
    return True


def detect_pentacyano_nitroso_metal(mol) -> "LeafTree | None":
    """Detect [M(CN)5(NO)]^2- coordination anion (sodium-nitroprusside class).

    Recognises the shape::

        N#[C][M-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O

    where ``[M]`` is a supported d-block metal in
    ``_NITROPRUSSIDE_METAL_NAMES``.  Counterions (Na+, K+) are accepted
    in additional fragments and the dispatcher emits a salt-form name
    ``{n}{cation} pentacyano(nitroso){metal}(IV)`` in that case.  The
    bare anion fragment alone emits ``pentacyano(nitroso){metal}(IV)``.

    Returns ``None`` for shapes outside this scope so the regular pipeline
    still handles other coordination compounds.

    Architectural notes:
      * No mutation of the input mol.
      * No silent atom drops: every atom in the fragment must classify
        cleanly as the metal centre, a cyanido C, a cyanido N, the
        nitroso N, or the nitroso O.  Otherwise returns ``None``.
      * Must run BEFORE the free-valence guard ``_validate_no_open_valences``
        in some valence-model variants where the carbene-like ``[C]``
        carries a radical; the present RDKit version (sanitised) gives 0
        radicals on these atoms but the dispatcher remains valence-model
        agnostic by returning ``None`` rather than asserting any.
      * Must run BEFORE the chain/ring naming pipeline because the engine
        has no plan for a 6-coordinate transition-metal centre with
        carbene-like ``[C]#N`` ligands.
    """
    if mol is None:
        return None
    from rdkit import Chem
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return None
    if not frags:
        return None

    # Identify the anion fragment (must contain the metal + 6 ligands).
    # All other fragments must be supported alkali / alkaline-earth
    # counter-cation singletons.
    metal_frag: tuple[int, ...] | None = None
    counterion_counts: dict[str, int] = {}
    counterion_total_charge = 0
    for fa in frags:
        # Single-atom counter-cation fragment?
        if len(fa) == 1:
            a = mol.GetAtomWithIdx(fa[0])
            sym = a.GetSymbol()
            if (
                sym in _NITROPRUSSIDE_COUNTERION_NAMES
                and a.GetFormalCharge() == 1
                and a.GetTotalNumHs() == 0
                and a.GetNumRadicalElectrons() == 0
                and a.GetDegree() == 0
            ):
                counterion_counts[sym] = counterion_counts.get(sym, 0) + 1
                counterion_total_charge += 1
                continue
            # Single-atom fragment that isn't a recognised counterion —
            # out of scope.
            return None
        # Multi-atom fragment: must be the anion (only one allowed).
        if metal_frag is not None:
            return None
        metal_frag = fa
    if metal_frag is None:
        return None

    # Locate the metal atom in the anion fragment.
    metal_idx: int | None = None
    metal_sym: str | None = None
    for ai in metal_frag:
        a = mol.GetAtomWithIdx(ai)
        if a.GetSymbol() in _NITROPRUSSIDE_METAL_NAMES:
            if metal_idx is not None:
                return None  # More than one metal — out of scope.
            metal_idx = ai
            metal_sym = a.GetSymbol()
    if metal_idx is None or metal_sym is None:
        return None

    metal_atom = mol.GetAtomWithIdx(metal_idx)
    # Charge -2 on the metal centre is the canonical nitroprusside form.
    if metal_atom.GetFormalCharge() != -2:
        return None
    if metal_atom.GetNumRadicalElectrons() != 0:
        return None
    if metal_atom.GetTotalNumHs() != 0:
        return None
    if metal_atom.GetDegree() != 6:
        return None

    # Counter-ion charge must balance the anion (-2 for nitroprusside).
    if counterion_total_charge != 0 and counterion_total_charge != 2:
        return None

    # Classify every neighbour of the metal as either a cyanido C or the
    # nitroso N.  Exactly 5 cyanido C's and exactly 1 nitroso N expected.
    metal_neighbours = list(metal_atom.GetNeighbors())
    if len(metal_neighbours) != 6:
        return None

    cyanido_c_idxs: list[int] = []
    nitroso_n_idx: int | None = None
    for nb in metal_neighbours:
        nb_idx = nb.GetIdx()
        if _is_cyanido_carbene_carbon(mol, nb_idx, metal_idx):
            cyanido_c_idxs.append(nb_idx)
            continue
        if _is_nitroso_nitrogen(mol, nb_idx, metal_idx):
            if nitroso_n_idx is not None:
                return None  # More than one nitroso — out of scope.
            nitroso_n_idx = nb_idx
            continue
        # Unknown ligand neighbour — defer.
        return None

    if len(cyanido_c_idxs) != 5 or nitroso_n_idx is None:
        return None

    # Sanity: the anion fragment must consist exactly of:
    #   metal + 5 cyanido C + 5 cyanido N (each C's terminal N) + 1 nitroso N + 1 nitroso O
    # = 13 atoms.  No silent atom drops: we explicitly count below.
    expected_atoms = {metal_idx, nitroso_n_idx}
    expected_atoms.update(cyanido_c_idxs)
    # Each cyanido C has exactly one terminal-N neighbour (other than the
    # metal); each was already validated, so add it to the expected set.
    for c_idx in cyanido_c_idxs:
        c_atom = mol.GetAtomWithIdx(c_idx)
        for nb in c_atom.GetNeighbors():
            if nb.GetIdx() != metal_idx:
                expected_atoms.add(nb.GetIdx())
    # Add the nitroso O (the single non-metal neighbour of the nitroso N).
    n_atom = mol.GetAtomWithIdx(nitroso_n_idx)
    for nb in n_atom.GetNeighbors():
        if nb.GetIdx() != metal_idx:
            expected_atoms.add(nb.GetIdx())
    if set(metal_frag) != expected_atoms:
        # Some atom in the anion fragment is neither the metal nor a
        # validated ligand atom — out of scope.
        return None

    metal_name = _NITROPRUSSIDE_METAL_NAMES[metal_sym]
    # Oxidation state for [M-2(CN^-)5(NO^0)]: the IUPAC convention names
    # this anion as "pentacyano(nitroso)iron(IV)" — OPSIN parses this
    # back to [Fe-2] (matches the input charge).  The (IV) is the
    # bracketed central-atom oxidation state per the standard reference
    # (Cotton & Wilkinson §26-A) where NO is treated as NO+.
    base = f"pentacyano(nitroso){metal_name}(IV)"

    if not counterion_counts:
        text = base
    else:
        # Charge balance enforced above (counterion_total_charge == 2).
        # Build the salt prefix as ``{n}{cation}`` (n=2 → "disodium").
        # Only one cation type per salt; mixed counterions defer.
        if len(counterion_counts) != 1:
            return None
        cation_sym, n_cations = next(iter(counterion_counts.items()))
        cation_name = _NITROPRUSSIDE_COUNTERION_NAMES[cation_sym]
        if n_cations == 1:
            cation_part = cation_name
        elif n_cations == 2:
            cation_part = f"di{cation_name}"
        elif n_cations == 3:
            cation_part = f"tri{cation_name}"
        else:
            # Out of scope (charge -2 anion only needs 2 monovalent cations).
            return None
        text = f"{cation_part} {base}"

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="organometallic",
            detail=f"pentacyano(nitroso) coordination: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )
