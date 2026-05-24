"""
iupac_namer/perception/skeletal_chain.py

Skeletal replacement ("a") nomenclature for ACYCLIC chains (P-15.4.3 / P-51.4.1).

When an unbranched chain contains four or more skeletal *heterounits* together
with at least one carbon atom, and none of the heteroatoms constitute all or
part of a principal characteristic group, IUPAC requires the chain to be named
by skeletal replacement nomenclature rather than substitutively or
multiplicatively (P-51.4.1.1).  Examples:

    CCO[Te][Se]SCC  -> 3-oxa-6-thia-5-selena-4-telluraoctane   (PIN)
    COCCOCCOCCOC    -> 2,5,8,11-tetraoxadodecane               (PIN)
    COCOCOCOC       -> 2,4,6,8-tetraoxanonane                  (PIN)

The replacement parent hydride is the alkane whose chain length equals the
total number of skeletal atoms (carbons + heteroatoms).  Skeletal heteroatoms
are designated by nondetachable "a" prefixes (oxa, thia, selena, tellura, aza,
phospha, sila, ...) placed in front of the alkane name in seniority order
(P-15.4.1.2 / Appendix 1: O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge >
Sn > Pb > B > Al > Ga > In > Tl), each with its locant(s).

Numbering is fixed (P-15.4.3.2.1): the chain is numbered from the end that
gives the lower locant set to the heteroatoms considered together as a set
without regard to kind; ties are broken by giving low locants to the
heteroatom(s) cited earliest in the seniority order.

CARBON-FREE chains take a different rule.  Replacement ("a") nomenclature above
requires at least one carbon (P-21.2.3.2); an all-heteroatom chain is instead a
*parent hydride*: a homogeneous one (P-21.2.2, e.g. trisulfane — produced by the
engine's homogeneous-heteroatom-chain dispatcher) or, when two heteroatoms
strictly alternate and the chain is terminated at both ends by the same
(less-senior) element, a heterogeneous ``a(ba)n`` parent hydride (P-21.2.3.1).
This module names the alternating case (:func:`_compute_alternating_heterochain`):

    SOS                   -> dithioxane           (HS-O-SH)
    [SiH3]S[SiH3]         -> disilathiane
    SOSOS                 -> trithioxane
    [SiH3]O[SiH2]O[SiH3]  -> trisiloxane

Nitrogen-containing heterogeneous chains use amine names (P-21.2.3.1 note) and
are NOT named by the alternating rule (declined here).

Scope of THIS module (the cleanly round-tripping common cases):
  - A single acyclic molecule (no rings, one fragment).
  - A saturated, unbranched main chain (no double/triple bonds, no branches).
  - Skeletal heteroatoms drawn from the standard-bonding-number replacement
    set with two-coordinate (O/S/Se/Te chalcogen), three-coordinate
    (N/P/As/Sb/Bi, B/Al/Ga/In/Tl) or four-coordinate (Si/Ge/Sn/Pb) standard
    valences only.
  - No charges, isotopes, radicals on skeletal atoms (terminal SiH3 etc. have
    no radical electrons in RDKit's model, so they pass).
  - No principal characteristic group anywhere on the molecule (an oxo/ol/etc.
    would change the numbering and unit count; deferred — see DOCUMENTED LIMITS
    in the build notes).
  - Heterounit count >= 4 (the P-51.4.1.1 threshold for the PIN).

Anything outside this scope returns ``None`` so the regular pipeline (which
already handles e.g. simple ethers substitutively) is used unchanged.
"""

from __future__ import annotations

import logging

from rdkit import Chem

from iupac_namer.data_loader import get_chain_stem, get_multiplier

logger = logging.getLogger(__name__)


# Element -> skeletal "a" prefix (P-15.4.1.1, Table 1.5).  Only the elements
# that can be skeletal replacement atoms in a chain with standard bonding
# numbers are listed here; loaded once at import from the curated data table.
def _load_a_prefixes() -> dict[str, str]:
    from iupac_namer.data_loader import _DATA_DIR, _load

    rows = _load(_DATA_DIR / "bluebook" / "skeletal_a_prefixes.json")
    return {r["element"]: r["a_prefix"] for r in rows}


_A_PREFIX: dict[str, str] = _load_a_prefixes()


# Seniority order for chain heteroatoms (P-15.4.1.2 / Appendix 1).  Lower index
# = senior (cited first, gets low locant on a tie).  Loaded from the curated
# element_seniority.json "chain_heteroatom" list.
def _load_seniority() -> dict[str, int]:
    from iupac_namer.data_loader import _DATA_DIR, _load

    data = _load(_DATA_DIR / "element_seniority.json")
    order = data["chain_heteroatom"]
    return {sym: i for i, sym in enumerate(order)}


_SENIORITY: dict[str, int] = _load_seniority()


# Standard bonding numbers for the skeletal replacement elements.  A skeletal
# atom must have exactly this many sigma bonds (heavy + H) to be a *standard*
# valence replacement atom.  Non-standard valences (lambda-convention) are out
# of scope for this module.
_STD_BONDING: dict[str, int] = {
    "O": 2, "S": 2, "Se": 2, "Te": 2,
    "N": 3, "P": 3, "As": 3, "Sb": 3, "Bi": 3,
    "B": 3, "Al": 3, "Ga": 3, "In": 3, "Tl": 3,
    "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4,
}

# Heteroatoms that may TERMINATE a heterochain (P-15.4.3.1 / P-51.4.1.4).
# Chalcogens (O/S/Se/Te) and N may NOT be chain terminals.
_TERMINAL_OK: frozenset[str] = frozenset(
    {"P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl"}
)


def compute_name(mol) -> str | None:
    """Return the skeletal-replacement PIN for an acyclic heterochain, or None.

    See module docstring for the exact (deliberately conservative) scope.
    """
    if mol is None:
        return None
    # Single fragment only.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    # No rings.
    if mol.GetRingInfo().NumRings() != 0:
        return None

    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy) < 3:
        # The smallest in-scope chain is a 3-atom carbon-free a(ba) parent
        # hydride (P-21.2.3.1, e.g. SOS -> dithioxane).  The carbon-bearing
        # skeletal-replacement path additionally requires >=4 heterounits +
        # >=1 carbon (>=5 skeletal atoms), enforced by its own n_units guard
        # below; this outer gate only rejects 1- and 2-atom skeletons that
        # neither rule can name.
        return None

    # Reject any charge / isotope / radical on heavy atoms; those are handled
    # by other dispatchers and would change the name (lambda / ide / ium).
    for a in heavy:
        if a.GetFormalCharge() != 0:
            return None
        if a.GetIsotope() != 0:
            return None
        if a.GetNumRadicalElectrons() != 0:
            return None

    # All heavy atoms must be C or a recognised skeletal-replacement element.
    has_carbon = False
    for a in heavy:
        sym = a.GetSymbol()
        if sym == "C":
            has_carbon = True
            continue
        if sym not in _STD_BONDING:
            return None

    # All bonds between heavy atoms must be single (saturated chain) and acyclic.
    # A double/triple bond would need ene/yne handling (deferred).
    for bond in mol.GetBonds():
        ba, ea = bond.GetBeginAtom(), bond.GetEndAtom()
        if ba.GetAtomicNum() == 1 or ea.GetAtomicNum() == 1:
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            return None

    # The heavy-atom skeleton must be a single unbranched path: exactly two
    # atoms of heavy-degree 1 (terminals) and the rest of heavy-degree 2.
    heavy_idx = {a.GetIdx() for a in heavy}

    def heavy_deg(a) -> int:
        return sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in heavy_idx)

    deg = {a.GetIdx(): heavy_deg(a) for a in heavy}
    if any(d > 2 for d in deg.values()):
        return None  # branched -> not an unbranched chain (deferred)
    terminals = [i for i, d in deg.items() if d == 1]
    if len(terminals) != 2:
        return None  # not a simple path
    if any(d == 0 for d in deg.values()):
        return None

    # Each skeletal atom must carry exactly its standard bonding number of
    # bonds (heavy + H), with no extra heavy substituents (unbranched).  This
    # also rejects e.g. a chain carbon bearing a halogen/OH substituent, which
    # would introduce a characteristic group / prefix substituent -> deferred.
    for a in heavy:
        sym = a.GetSymbol()
        total_bonds = a.GetTotalDegree()  # heavy + implicit/explicit H
        if sym == "C":
            if total_bonds != 4:
                return None
        else:
            if total_bonds != _STD_BONDING[sym]:
                return None
        # No heavy substituents beyond the chain (degree already <= 2 ensures
        # this for the path, but guard explicitly).
        if heavy_deg(a) > 2:
            return None

    # Order the skeleton as a linear path from one terminal to the other.
    path = _order_path(mol, terminals[0], heavy_idx)
    if path is None or len(path) != len(heavy):
        return None

    # Build the per-position element list along the path.
    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in path]

    # Carbon-free chains use a DIFFERENT rule.  Replacement ("a") nomenclature
    # (P-21.2.3.2) requires at least one carbon; a chain made entirely of
    # heteroatoms is named either as a homogeneous parent hydride (P-21.2.2,
    # e.g. trisulfane — handled by the engine's homogeneous-chain dispatcher)
    # or, when two different heteroatoms strictly alternate, as a P-21.2.3.1
    # ``a(ba)n`` heterogeneous parent hydride (e.g. dithioxane).  Delegate the
    # alternating case here; everything else carbon-free returns None so the
    # homogeneous dispatcher / substitutive pipeline handles it.
    if not has_carbon:
        return _compute_alternating_heterochain(elems)

    # P-51.4.1.4 / P-15.4.3.1: the chain must be terminated by C or one of
    # P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl.  A terminal chalcogen
    # (O/S/Se/Te) or N is NOT a skeletal terminus — it is a hydroxyl / thiol /
    # amine etc. characteristic group, which is out of scope for this module
    # (e.g. HO-CH2-O-CH2-...-OH must be named as a diol, not a terminal-oxa
    # chain).  Decline if either terminus is such an atom.
    for end_elem in (elems[0], elems[-1]):
        if end_elem == "C":
            continue
        if end_elem not in _TERMINAL_OK:
            return None

    # There must be at least one heteroatom.
    hetero_positions = [k for k, e in enumerate(elems) if e != "C"]
    if not hetero_positions:
        return None

    # Count heterounits: a maximal run of consecutive *same-element* heteroatoms
    # counts as one unit (it has a parent-hydride group name, e.g. -SS-
    # disulfanediyl); a heterogeneous adjacent run contributes one unit per
    # atom (P-51.4.1.1 worked examples).  Carbons separate units.
    n_units = _count_heterounits(elems)
    if n_units < 4:
        return None

    # Choose the numbering direction (P-15.4.3.2.1).
    direction = _choose_direction(elems)
    if direction == -1:
        elems = elems[::-1]
        path = path[::-1]

    # Build (locant, element) pairs (1-based).
    locant_elem = [(k + 1, e) for k, e in enumerate(elems) if e != "C"]

    # Compose the replacement-prefix segment list in seniority order.
    prefix = _build_prefix(locant_elem)
    if prefix is None:
        return None

    # The parent alkane: chain length = total skeletal atoms.
    n = len(elems)
    stem = get_chain_stem(n)
    if stem is None:
        return None
    parent = stem + "ane"

    return prefix + parent


def _order_path(mol, start: int, heavy_idx: set[int]) -> list[int] | None:
    """Walk the unbranched heavy-atom path from *start*; return ordered idx list."""
    order = [start]
    prev = -1
    cur = start
    while True:
        nbrs = [
            nb.GetIdx()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
            if nb.GetIdx() in heavy_idx and nb.GetIdx() != prev
        ]
        if len(nbrs) == 0:
            break
        if len(nbrs) > 1:
            return None  # branch encountered (should not happen after guards)
        prev, cur = cur, nbrs[0]
        order.append(cur)
    return order


def _count_heterounits(elems: list[str]) -> int:
    """Count skeletal heterounits along the element sequence (P-51.4.1.1).

    A maximal run of consecutive heteroatoms of the SAME element is one unit
    (parent-hydride group name, e.g. -SS- disulfanediyl).  In a heterogeneous
    adjacent run each atom is its own unit.  Carbons are not heteroatoms.
    """
    units = 0
    i = 0
    n = len(elems)
    while i < n:
        if elems[i] == "C":
            i += 1
            continue
        # Start of a hetero run.
        j = i
        while j < n and elems[j] != "C":
            j += 1
        run = elems[i:j]
        # Split the run into maximal same-element sub-runs; each is one unit.
        k = 0
        while k < len(run):
            m = k
            while m < len(run) and run[m] == run[k]:
                m += 1
            units += 1
            k = m
        i = j
    return units


def _locant_set(elems: list[str]) -> list[int]:
    """1-based locants of heteroatoms for the chosen orientation."""
    return [k + 1 for k, e in enumerate(elems) if e != "C"]


def _seniority_locant_key(elems: list[str]) -> list[int]:
    """Per-seniority-rank locant list for the seniority tie-break.

    Returns the concatenation, over seniority ranks (most senior first), of the
    sorted locants of heteroatoms at that rank.  Comparing two orientations'
    keys lexicographically realises "low locants to the heteroatom cited first
    in the seniority order" (P-15.4.3.2.1).
    """
    by_rank: dict[int, list[int]] = {}
    for k, e in enumerate(elems):
        if e == "C":
            continue
        rank = _SENIORITY.get(e, 999)
        by_rank.setdefault(rank, []).append(k + 1)
    key: list[int] = []
    for rank in sorted(by_rank):
        key.extend(sorted(by_rank[rank]))
    return key


def _choose_direction(elems: list[str]) -> int:
    """Return +1 to keep, -1 to reverse, per P-15.4.3.2.1.

    Rule order:
      1. Lower locant SET for heteroatoms as a whole (without regard to kind).
      2. If tied, low locants to heteroatoms cited first in seniority order.
    """
    fwd = elems
    rev = elems[::-1]

    fwd_set = sorted(_locant_set(fwd))
    rev_set = sorted(_locant_set(rev))
    if fwd_set < rev_set:
        return 1
    if rev_set < fwd_set:
        return -1

    # Heteroatom locant sets equal -> seniority tie-break.
    fwd_key = _seniority_locant_key(fwd)
    rev_key = _seniority_locant_key(rev)
    if rev_key < fwd_key:
        return -1
    return 1


def _build_prefix(locant_elem: list[tuple[int, str]]) -> str | None:
    """Build the ``2,5,8,11-tetraoxa`` style prefix string (with trailing '-').

    Segments are ordered by seniority (most senior element first); within an
    element the locants are sorted ascending and a multiplicative prefix
    (di/tri/...) is used for repeats.  Returns None if any element lacks an
    "a" prefix.
    """
    by_elem: dict[str, list[int]] = {}
    for loc, elem in locant_elem:
        by_elem.setdefault(elem, []).append(loc)

    ordered_elems = sorted(by_elem, key=lambda e: _SENIORITY.get(e, 999))

    segments: list[str] = []
    for elem in ordered_elems:
        a_pref = _A_PREFIX.get(elem)
        if a_pref is None:
            return None
        locs = sorted(by_elem[elem])
        if len(locs) > 1:
            mult = get_multiplier(len(locs))
            if mult is None:
                return None
            a_pref = mult + a_pref
        loc_str = ",".join(str(x) for x in locs)
        segments.append(f"{loc_str}-{a_pref}")

    # Concatenate segments with '-' (e.g. "3-oxa-6-thia-5-selena-4-tellura").
    # The caller appends the alkane stem directly with NO separator, so the
    # final "a"-prefix abuts the stem ("...4-tellura" + "octane" =
    # "4-telluraoctane").  No elision occurs (each "a"-prefix concatenates
    # cleanly, and every alkane stem begins with a consonant).
    return "-".join(segments)


# Carbon-free alternating heterochains (P-21.2.3.1).
#
# Nitrogen is deliberately excluded from the alternating rule: P-21.2.3.1 states
# that "When nitrogen atoms are present, amine names (see P-62) should be used
# because of the higher functionality of amines."  So an N-containing
# heterogeneous chain (e.g. SiH3-NH-SiH3) is named as an amine derivative, not
# as an ``a(ba)n`` parent hydride — declined here and left to other dispatchers.
#
# Boron is excluded because OPSIN does not support polyborane parent hydrides,
# so such a name could not be round-trip verified.
_ALTERNATING_EXCLUDED: frozenset[str] = frozenset({"N", "B"})


def _compute_alternating_heterochain(elems: list[str]) -> str | None:
    """Return the P-21.2.3.1 ``a(ba)n`` parent-hydride PIN, or None.

    A carbon-free unbranched chain of two strictly alternating heteroatom
    kinds, terminated at *both* ends by the same element, is a heterogeneous
    parent hydride composed of alternating atoms (P-21.2.3.1).  Its name is::

        {multiplier-for-terminal-count}{a-term of terminal element}
        {a-term of middle element}ane

    where the terminal element is the one coming *later* (less senior) in the
    chain-heteroatom seniority order and the middle element comes *earlier*.
    Examples (all OPSIN-verified):

        HS-O-SH               -> dithioxane
        SiH3-S-SiH3           -> disilathiane
        PH2-Se-PH2            -> diphosphaselenane
        S-O-S-O-S             -> trithioxane
        SiH3-O-SiH2-O-SiH3    -> trisiloxane

    Homogeneous carbon-free chains (one element only) are handled by the
    engine's homogeneous-heteroatom-chain dispatcher (P-21.2.2); this function
    returns None for them and for anything that is not a clean two-element
    alternating pattern.
    """
    n = len(elems)
    if n < 3 or n % 2 == 0:
        # An a(ba)n chain has an odd number of atoms (k terminals of element a,
        # k-1 of element b): 2k-1, k >= 2 -> n >= 3 and odd.
        return None

    distinct = set(elems)
    if len(distinct) != 2:
        return None  # homogeneous (handled elsewhere) or >2 kinds (not a(ba)n)

    term = elems[0]
    if elems[-1] != term:
        return None  # not terminated by two identical atoms

    # The two interleaved positions must each be a single element: even indices
    # (0-based) are the terminal element, odd indices the middle element.
    mid_candidates = {elems[i] for i in range(1, n, 2)}
    term_candidates = {elems[i] for i in range(0, n, 2)}
    if len(mid_candidates) != 1 or len(term_candidates) != 1:
        return None
    mid = next(iter(mid_candidates))
    if term_candidates != {term} or mid == term:
        return None

    # P-21.2.3.1: excludes carbon and the halogens (already filtered upstream),
    # and nitrogen / boron (see note above).
    if term in _ALTERNATING_EXCLUDED or mid in _ALTERNATING_EXCLUDED:
        return None

    # Both elements must be standard-valence chain heteroatoms with a-terms and
    # a defined seniority.
    for e in (term, mid):
        if e not in _STD_BONDING or e not in _A_PREFIX or e not in _SENIORITY:
            return None

    # The terminal element must be the one coming LATER (less senior) in the
    # seniority order; the middle element the one coming EARLIER (more senior).
    # If the structure has the senior element at the termini it is not a valid
    # a(ba)n parent hydride in this orientation (and, being symmetric, cannot be
    # reoriented to fix it) -> decline.
    if _SENIORITY[term] <= _SENIORITY[mid]:
        return None

    k = (n + 1) // 2  # number of terminal-element atoms
    mult = get_multiplier(k)
    if mult is None:
        return None

    term_a = _A_PREFIX[term]
    mid_a = _A_PREFIX[mid]

    # Assemble with IUPAC elision (P-21.2.3.1): the terminal letter 'a' of an
    # 'a' term is elided when followed by a vowel; the multiplying-prefix vowel
    # is never elided.  Order: mult + term_a + mid_a + "ane".
    body = _elide_join(term_a, mid_a)
    body = _elide_join(body, "ane")
    return f"{mult}{body}"


def _elide_join(left: str, right: str) -> str:
    """Join two morphemes, eliding a terminal 'a' on *left* before a vowel.

    Implements the P-21.2.3.1 elision: "the terminal letter 'a' of an 'a' term
    is elided when followed by a vowel".  Only a terminal 'a' is elided (the
    multiplying-prefix vowel rule is handled by not passing multipliers through
    this helper).
    """
    if left.endswith("a") and right[:1] in "aeiou":
        return left[:-1] + right
    return left + right
