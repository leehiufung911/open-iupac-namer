"""
iupac_namer/ring_naming/fused.py

Fused ring naming.

Stages 1 + 2 + 3 implement IUPAC P-25.3 fused-bicyclic and fused-tricyclic
naming for ortho-fused systems where a SMALLER component (5- or 6-membered
ring with two heteroatoms in the 1,3-pattern) is fused onto a BASE component
that has a known IUPAC name (retained or systematic).

Stage 1 (5-ring smaller, mono-ring base, 0-1 hetero in base):
    [1,3]dioxolo[4,5-b]benzene, [1,3]dithiolo[4,5-b]pyridine, ...

Stage 2 extensions:
    A) Multi-hetero aromatic base: pyrazine / pyrimidine / pyridazine
        e.g. [1,3]dioxolo[4,5-b]pyrazine, [1,3]dioxolo[4,5-d]pyrimidine
    B) Multi-ring fused base: naphthalene / quinoline (a 2-ring fused
       system whose retained name is looked up via the existing retained
       lookup on a carved 2-ring sub-mol)
        e.g. [1,3]dioxolo[4,5-b]naphthalene, [1,3]dithiolo[4,5-b]quinoline
    C) 6-ring smaller component (1,3-dioxin / dithiin / oxathiin)
        e.g. [1,3]dioxino[4,5-b]benzene, [1,3]dioxino[4,5-b]pyridine

Stage 3: saturated / partly-saturated base parents.  When the BASE ring(s)
of the fused system are not fully aromatic in the input molecule, the name
receives a hydro-prefix describing which atoms have been saturated relative
to the fully-aromatic canonical parent:

    * Fully-saturated mono-ring base: ``hexahydro-[1,3]dioxolo[4,5-b]benzene``
      (saturated cyclohexane base), ``hexahydro-[1,3]dioxolo[4,5-b]pyridine``
      (saturated piperidine base), ``hexahydro-[1,3]dioxino[4,5-b]benzene``.
    * Fully-saturated multi-ring base: ``decahydro-[1,3]dioxolo[4,5-b]naphthalene``.
    * Partly-saturated mono-ring base: ``4,5,6,7-tetrahydro-[1,3]dioxolo[4,5-b]benzene``,
      ``4,5-dihydro-[1,3]dioxolo[4,5-b]benzene`` — emitted with explicit hydro
      locants on the base's sp3 positions.

Existing retained-name lookups (1,3-benzodioxole etc.) take priority via
the dispatcher; this module's output is scored under a dedicated
``fused_hetero_hydro`` method rank (below retained/HW, above VB) so a
retained name always wins where it exists but the fused-hydro systematic
form beats VB fallbacks for saturated dioxolo-type heterocycles.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from iupac_namer.types import Locant, NamedParent, Numbering

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HW-prefix → fusion-prefix mapping for the smaller (5- or 6-ring) partner
# ---------------------------------------------------------------------------
#
# For a 5-ring partner with two heteroatoms in 1,3 positions (fusion edge =
# locants 4,5), the partner stem is "ole" → fusion-form "olo" (e.g. dioxolo).
#
# For a 6-ring partner with two heteroatoms in 1,3 positions (fusion edge =
# locants 4,5; non-fusion locants = 1,2,3,6), the partner stem is "ine" →
# fusion-form "ino" (e.g. dioxino).
#
# Standard hetero seniority for HW (P-22.1.2.1.1): O > S > Se > Te > N > P
# > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl

# Fusion-prefix base (HW prefix with trailing 'a').  Maps element symbol to
# the bare 'a'-prefix used in fusion naming.
_FUSION_PREFIX: dict[str, str] = {
    "O":  "oxa",
    "S":  "thia",
    "Se": "selena",
    "Te": "tellura",
    "N":  "aza",
    "P":  "phospha",
    "As": "arsa",
    "Si": "sila",
    "Ge": "germa",
    "B":  "bora",
}

# HW seniority (higher = more senior) — only used to order mixed-element
# pairs in the fusion prefix.
_HW_SENIORITY: dict[str, int] = {
    "O":  16, "S": 15, "Se": 14, "Te": 13,
    "N":  12, "P":  11, "As": 10, "Sb":  9, "Bi": 8,
    "Si":  7, "Ge":  6, "Sn":  5, "Pb":  4,
    "B":   3, "Al":  2, "Ga":  1,
}

# Per-size unsaturated HW stem.  The fusion-prefix form replaces the trailing
# 'e' with 'o'.
_RING_STEM_BY_SIZE: dict[int, str] = {
    5: "ole",   # → "olo"
    6: "ine",   # → "ino"
}


# ---------------------------------------------------------------------------
# Mono-heteroatom smaller-component fusion prefixes (P-25.3.1.3)
# ---------------------------------------------------------------------------
#
# A 5-membered smaller component with a SINGLE ring heteroatom has a retained
# fusion prefix derived from the parent monocycle (furan → furo, thiophene →
# thieno, pyrrole → pyrrolo, ...).  Unlike the [1,3]-dihetero forms these are
# not built from HW 'a'-prefix + stem; they are the IUPAC retained
# fusion-prefix forms (Table 28.1 / P-25.3.1.3).  The single heteroatom takes
# locant 1 in the smaller component's standalone numbering and the attachment
# (fusion-edge) locants are derived from that numbering — they are NOT fixed
# at [4,5] the way the symmetric [1,3]-dihetero edge is.
_MONO_FUSION_PREFIX_5: dict[str, str] = {
    "O":  "furo",
    "S":  "thieno",
    "Se": "selenolo",
    "Te": "tellurolo",
    "N":  "pyrrolo",
    "P":  "phospholo",
    "As": "arsolo",
    "Si": "silolo",
    "Ge": "germolo",
    "B":  "borolo",
}


def _build_smaller_prefix(
    elements_in_order: list[str], smaller_size: int
) -> str | None:
    """Build the smaller-ring fusion prefix for a 5- or 6-ring smaller
    partner with the given heteroatom elements (in IUPAC seniority order,
    most senior first).

    Returns e.g. "dioxolo", "dithiolo", "oxathiolo" (5-ring) or "dioxino",
    "dithiino", "oxathiino" (6-ring), or None if any element lacks a fusion
    prefix or the size is unsupported.
    """
    if len(elements_in_order) != 2:
        return None
    stem = _RING_STEM_BY_SIZE.get(smaller_size)
    if stem is None:
        return None

    e1, e2 = elements_in_order
    p1 = _FUSION_PREFIX.get(e1)
    p2 = _FUSION_PREFIX.get(e2)
    if p1 is None or p2 is None:
        return None

    # Same element: use multiplier "di"
    if e1 == e2:
        raw_prefix = "di" + p1  # e.g. "dioxa", "dithia"
    else:
        raw_prefix = p1 + p2    # e.g. "oxathia", "oxaaza"

    # Append fusion-form ring stem (terminal e → o)
    fusion_stem = stem[:-1] + "o"  # "ole"→"olo", "ine"→"ino"

    # HW elision: trailing 'a' of the prefix elides before a vowel-initial
    # stem.  Both "olo" and "ino" begin with vowels.
    name = raw_prefix
    if name.endswith("a") and fusion_stem and fusion_stem[0] in "aeiou":
        name = name[:-1]
    name = name + fusion_stem

    # Internal elision between mixed-element prefixes: e.g. "oxaaza" → "oxaza".
    name = name.replace("aaz", "az")
    name = name.replace("aar", "ar")
    return name


# ---------------------------------------------------------------------------
# Smaller / base ring identification
# ---------------------------------------------------------------------------

def _identify_smaller_and_base(
    rings: tuple[frozenset[int], ...],
    fusion_atoms_list: tuple[tuple[int, int], ...],
    mol,
) -> tuple[
    frozenset[int],            # smaller ring
    tuple[frozenset[int], ...], # base rings (1 or more)
    tuple[int, int],           # the smaller↔base fusion edge
] | None:
    """Choose which ring is the smaller (attached) component.  The remaining
    rings together form the BASE.

    Returns (smaller_ring, base_rings_tuple, smaller_base_fusion_edge), or
    None if no unambiguous smaller component fits the [1,3]-dihetero pattern.

    Stage 1: 2-ring system, smaller is 5-ring with 2 hetero, fusion size = 1.
    Stage 2: 3-ring system, smaller is 5-ring with 2 hetero, base is the
             other 2 rings (themselves ortho-fused), fusion size = 2 edges.
             Or smaller is a 6-ring with 1,3-dihetero (dioxino case).
    """
    if not rings:
        return None

    # Build a per-ring count of hetero atoms (only O / S / Se / Te / N / P …
    # supported by _FUSION_PREFIX) AMONG NON-FUSION ATOMS.
    edge_set = {frozenset(e) for e in fusion_atoms_list}

    candidates: list[tuple[frozenset[int], tuple[int, int]]] = []
    for r in rings:
        if len(r) not in _RING_STEM_BY_SIZE:
            continue
        # Find this ring's fusion edge to the rest of the system.
        # A ring belongs to AT MOST ONE shared edge with the base for a true
        # smaller-component candidate (Stage 1/2).
        my_edges = [e for e in edge_set if e.issubset(r)]
        if len(my_edges) != 1:
            continue
        edge = tuple(sorted(my_edges[0]))
        # Heteroatom count on the non-fusion atoms only.
        non_fusion = r - set(edge)
        het_count = sum(
            1 for a in non_fusion
            if mol.GetAtomWithIdx(a).GetSymbol() in _FUSION_PREFIX
            and mol.GetAtomWithIdx(a).GetSymbol() != "C"
        )
        if het_count != 2:
            continue
        candidates.append((r, edge))

    if not candidates:
        return None

    # If multiple rings qualify by the [1,3]-dihetero non-fusion pattern (e.g.
    # in dioxolo[4,5-b]pyrazine the small dioxole ring AND the pyrazine 6-ring
    # both have 2 heteroatoms among their non-fusion atoms), prefer the one
    # that is structurally the SMALLER component (smaller ring size; tie-broken
    # by total heteroatom count INCLUDING fusion-edge atoms — the smaller
    # IUPAC partner is the one whose 1,3-pattern uses non-N heteroatoms like
    # O/S so the fusion-prefix list is well-defined).
    def _candidate_priority(cand: tuple[frozenset[int], tuple[int, int]]):
        ring, edge = cand
        non_fusion = ring - set(edge)
        # Most senior heteroatom present in the non-fusion arc.
        senior_score = max(
            (_HW_SENIORITY.get(mol.GetAtomWithIdx(a).GetSymbol(), 0)
             for a in non_fusion
             if mol.GetAtomWithIdx(a).GetSymbol() in _FUSION_PREFIX
             and mol.GetAtomWithIdx(a).GetSymbol() != "C"),
            default=0,
        )
        # Prefer SMALLER ring; then prefer ring with MORE-SENIOR hetero
        # (O > S > Se > Te > N > ...).  This lands the dioxole/dithiole on
        # the "smaller" side and the diazine on the "base" side.
        return (len(ring), -senior_score)

    candidates.sort(key=_candidate_priority)
    smaller_ring, smaller_edge = candidates[0]
    base_rings = tuple(r for r in rings if r != smaller_ring)
    if not base_rings:
        return None

    # All non-smaller rings must form the base.  For Stage 2 multi-ring base,
    # we additionally require the base sub-system to be a single connected
    # set of ortho-fused rings (a 2-ring fused base is the only Stage 2 form
    # we attempt).
    return smaller_ring, base_rings, smaller_edge


# ---------------------------------------------------------------------------
# Smaller ring [1,3]-dihetero pattern check (5- and 6-ring)
# ---------------------------------------------------------------------------

def _is_dihetero_pattern(
    smaller_ring: frozenset[int],
    fusion_atoms: tuple[int, int],
    mol,
) -> tuple[list[str], int, list[int]] | None:
    """Verify that ``smaller_ring`` is a 5- or 6-ring whose two heteroatoms
    sit in the 1,3-pattern: hetero — C — hetero on the side opposite to the
    fusion edge.  Carbons fill the rest.

    Returns ``(elements_in_seniority_order, middle_carbon_idx,
                extra_carbons_idx_list)`` on success, where extra_carbons is
    the list of additional sp3 ring carbons (locant 6 for 6-ring; empty for
    5-ring).  Returns None if the ring does not match.
    """
    size = len(smaller_ring)
    if size not in (5, 6):
        return None
    fa = set(fusion_atoms)
    non_fusion = sorted(smaller_ring - fa)
    if len(non_fusion) != size - 2:
        return None

    # Build adjacency within the smaller ring
    adj: dict[int, set[int]] = {a: set() for a in smaller_ring}
    for a in smaller_ring:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            nb_i = nb.GetIdx()
            if nb_i in smaller_ring:
                adj[a].add(nb_i)

    if size == 5:
        # 5-ring: locants 1, 2, 3 on the non-fusion arc.  Middle (locant 2)
        # is the C between the two heteroatoms; it has NO fusion neighbor.
        middle_candidates = [
            a for a in non_fusion
            if (set(non_fusion) - {a}).issubset(adj[a]) and not (adj[a] & fa)
        ]
        if len(middle_candidates) != 1:
            return None
        middle = middle_candidates[0]
        others = [a for a in non_fusion if a != middle]
        extra_carbons: list[int] = []
    else:  # size == 6
        # 6-ring: non-fusion arc is locants 1, 2, 3, 6.  Locants 1 and 3 are
        # the heteroatoms; locant 2 is C between them; locant 6 is C adjacent
        # to one of the fusion atoms (the 4a/8a-side carbon).
        # Identify atoms in the non-fusion arc by walking from one fusion
        # atom around the non-fusion side back to the other fusion atom.
        cycle = _ring_cycle_order(smaller_ring, mol, start=fusion_atoms[0])
        if cycle is None or len(cycle) != 6:
            return None
        # cycle[0] = fa0, cycle[5] = fa1 (or reversed).  Ensure cycle[-1]
        # is the other fusion atom; otherwise rotate-direction.
        if cycle[-1] != fusion_atoms[1]:
            # Try the reverse direction
            cycle = [cycle[0]] + list(reversed(cycle[1:]))
            if cycle[-1] != fusion_atoms[1]:
                return None
        # Now arc-positions 1..4 of the cycle are the non-fusion atoms.
        # Going AWAY from cycle[0] (fusion atom) around the non-fusion side:
        #   arc[0] is adjacent to fusion_atom_0 → this is locant 6
        #     (or locant 4 — see below; the smaller is symmetric in either
        #     sense for 1,3-dihetero, so we pick the orientation where
        #     locant 1 = senior heteroatom).
        # Identify the heteroatom positions in the arc
        arc = cycle[1:5]  # 4 atoms
        # Identify the two heteroatoms in the arc
        het_positions = [i for i, a in enumerate(arc)
                         if mol.GetAtomWithIdx(a).GetSymbol() != "C"]
        if len(het_positions) != 2:
            return None
        # Heteroatoms must be in 1,3-pattern within the arc:
        # arc indices 0..3.  For 1,3 with fusion at 4,5, heteroatoms are at
        # arc positions {1, 3} (locants 3, 1) — separated by exactly one C.
        # Check that the two heteroatom positions differ by 2.
        if abs(het_positions[0] - het_positions[1]) != 2:
            return None
        # Identify middle carbon: the arc atom BETWEEN the two heteroatoms.
        mid_pos = (het_positions[0] + het_positions[1]) // 2
        middle = arc[mid_pos]
        others = [arc[i] for i in het_positions]
        # The "extra" non-fusion carbon is the arc atom NOT in {middle} and
        # NOT in others.
        extra_carbons = [arc[i] for i in range(4)
                         if i != mid_pos and i not in het_positions]

    elem_others = [mol.GetAtomWithIdx(a).GetSymbol() for a in others]
    middle_elem = mol.GetAtomWithIdx(middle).GetSymbol()

    if middle_elem != "C":
        return None
    for elem in elem_others:
        if elem == "C":
            return None
        if elem not in _FUSION_PREFIX:
            return None
    for c in extra_carbons:
        if mol.GetAtomWithIdx(c).GetSymbol() != "C":
            return None

    elements_in_order = sorted(
        elem_others, key=lambda e: -_HW_SENIORITY.get(e, 0)
    )
    # Architectural guard: do NOT emit a generic HW fusion prefix
    # ("[1,3]diazolo", "[1,3]diphospholo", ...) for an all-pnictogen 1,3-
    # dihetero smaller ring.  IUPAC P-25.3.1.3 reserves the retained
    # fusion forms (imidazo, pyrazolo, etc.) for those skeletons, and the
    # corresponding fused parents have well-established retained names
    # (purine, imidazo[4,5-d]pyrimidine, ...).  Emitting a [1,3]diazolo
    # form here would compete (and on numbering grounds sometimes win)
    # against the legitimate retained name; that's a regression hazard.
    # Stage 1+2 therefore require at least one chalcogen (O/S/Se/Te) in
    # the smaller ring's 1,3-dihetero pair.
    chalcogens = {"O", "S", "Se", "Te"}
    if not any(e in chalcogens for e in elements_in_order):
        return None
    return elements_in_order, middle, extra_carbons


# ---------------------------------------------------------------------------
# Mono-heteroatom smaller-component identification + descriptor (P-25.3.1.3)
# ---------------------------------------------------------------------------

def _identify_mono_smaller_and_base(
    rings: tuple[frozenset[int], ...],
    fusion_atoms_list: tuple[tuple[int, int], ...],
    mol,
) -> tuple[frozenset[int], tuple[frozenset[int], ...], tuple[int, int]] | None:
    """Choose a SINGLE-heteroatom 5-ring smaller component and the base.

    Mirror of ``_identify_smaller_and_base`` but for a smaller ring carrying
    exactly ONE ring heteroatom (furo / thieno / pyrrolo / selenolo / ...).
    The heteroatom may sit anywhere in the ring (including adjacent to a
    fusion atom), so — unlike the symmetric [1,3]-dihetero case — we count
    heteroatoms over the WHOLE smaller ring, not only its non-fusion arc.

    Returns ``(smaller_ring, base_rings, smaller_base_edge)`` or None.
    """
    if not rings:
        return None
    edge_set = {frozenset(e) for e in fusion_atoms_list}

    candidates: list[tuple[frozenset[int], tuple[int, int]]] = []
    for r in rings:
        # Only a 5-ring smaller component is supported for the mono-hetero
        # path (the 6-ring mono-hetero prefixes — pyrano / thiopyrano — name
        # a non-fully-aromatic partner and need indicated-hydrogen handling;
        # see the gap note in ``name_fused``).
        if len(r) != 5:
            continue
        my_edges = [e for e in edge_set if e.issubset(r)]
        if len(my_edges) != 1:
            continue
        edge = tuple(sorted(my_edges[0]))
        het_count = sum(
            1 for a in r
            if mol.GetAtomWithIdx(a).GetSymbol() in _MONO_FUSION_PREFIX_5
            and mol.GetAtomWithIdx(a).GetSymbol() != "C"
        )
        if het_count != 1:
            continue
        candidates.append((r, edge))

    if not candidates:
        return None

    # Prefer the structurally smaller / more-senior-hetero ring as the
    # attached component (same rationale as the dihetero identifier).
    def _priority(cand):
        ring, edge = cand
        senior = max(
            (_HW_SENIORITY.get(mol.GetAtomWithIdx(a).GetSymbol(), 0)
             for a in ring
             if mol.GetAtomWithIdx(a).GetSymbol() in _MONO_FUSION_PREFIX_5
             and mol.GetAtomWithIdx(a).GetSymbol() != "C"),
            default=0,
        )
        return (len(ring), -senior)

    candidates.sort(key=_priority)
    smaller_ring, smaller_edge = candidates[0]
    base_rings = tuple(r for r in rings if r != smaller_ring)
    if not base_rings:
        return None
    return smaller_ring, base_rings, smaller_edge


def _mono_smaller_prefix(smaller_ring: frozenset[int], mol) -> tuple[str, str] | None:
    """Return ``(fusion_prefix, hetero_element)`` for a 5-ring smaller
    component with exactly one ring heteroatom, or None.

    e.g. ``("furo", "O")``, ``("thieno", "S")``, ``("pyrrolo", "N")``.
    """
    hets = [
        a for a in smaller_ring
        if mol.GetAtomWithIdx(a).GetSymbol() != "C"
    ]
    if len(hets) != 1:
        return None
    elem = mol.GetAtomWithIdx(hets[0]).GetSymbol()
    prefix = _MONO_FUSION_PREFIX_5.get(elem)
    if prefix is None:
        return None
    return prefix, elem


def _mono_fusion_descriptor(
    smaller_ring: frozenset[int],
    fusion_atoms: tuple[int, int],
    base_numbering: dict[int, int] | dict[int, str],
    mol,
) -> tuple[int, int] | None:
    """Compute the smaller-component attachment-locant pair ``[X,Y]`` for a
    mono-hetero 5-ring fused to a base using ``base_numbering`` (the chosen
    base peripheral numbering, atom→locant).

    Algorithm (P-25.3.1.3):
      1. Number the smaller ring standalone with its heteroatom = locant 1,
         in whichever of the two cyclic directions gives the fusion edge the
         LOWEST locant pair.
      2. Cite the attachment locants in the order that corresponds to the
         base's lettering/numbering direction: the first attachment locant is
         the smaller-component locant of the fusion atom that carries the
         LOWER base locant; the second is that of the higher-base-locant atom.

    Returns ``(d1, d2)`` (the smaller-component locants in citation order) or
    None.
    """
    hets = [a for a in smaller_ring if mol.GetAtomWithIdx(a).GetSymbol() != "C"]
    if len(hets) != 1:
        return None
    het = hets[0]
    fa0, fa1 = fusion_atoms

    # Both standalone smaller-ring directions (het = locant 1).
    dir_numberings = _all_directional_numberings(smaller_ring, mol, het)
    if not dir_numberings:
        return None

    # Pick directions giving the lowest fusion-edge locant pair.
    best_pair: tuple[int, int] | None = None
    for sn in dir_numberings:
        pair = tuple(sorted((sn[fa0], sn[fa1])))
        if best_pair is None or pair < best_pair:
            best_pair = pair
    best_dirs = [
        sn for sn in dir_numberings
        if tuple(sorted((sn[fa0], sn[fa1]))) == best_pair
    ]

    # Resolve base locants as integers for the low/high comparison.
    def _base_int(atom: int) -> int | None:
        loc = base_numbering.get(atom)
        if loc is None:
            return None
        if isinstance(loc, int):
            return loc
        # string label like "4a" — use its numeric anchor + suffix ordering
        n, suffix = _locant_sort_key(loc)
        return n * 100 + (ord(suffix) if suffix else 0)

    b0 = _base_int(fa0)
    b1 = _base_int(fa1)
    if b0 is None or b1 is None:
        return None
    if b0 < b1:
        low_atom, high_atom = fa0, fa1
    else:
        low_atom, high_atom = fa1, fa0

    # Among the smaller-ring directions achieving the lowest edge pair, the
    # citation order is fixed by the base direction; choose the lexically
    # lowest (d1, d2) to break any residual symmetry (symmetric smaller ring).
    cand: list[tuple[int, int]] = []
    for sn in best_dirs:
        cand.append((sn[low_atom], sn[high_atom]))
    if not cand:
        return None
    return min(cand)


def _mono_hetero_aromatic_base_name(
    base_ring: frozenset[int], mol
) -> str | None:
    """Derive the canonical AROMATIC mono-ring base name from heteroatom
    topology ALONE, for the mono-hetero fusion path.

    Why not reuse ``_try_monocyclic_base_name`` / ``_canonical_mono_base_name``?
    Those go through the retained lookup, which keys on the carved fragment's
    indicated-hydrogen.  In a mono-hetero fused system the whole-molecule
    indicated H can be parked by RDKit's aromaticity perception on a BASE ring
    nitrogen (e.g. ``pyrrolo[3,4-b]pyridine`` canonicalises with the [nH] on
    the 6-ring), which makes the retained lookup return ``1H-pyridine`` instead
    of ``pyridine``.  The base parent identity for fusion naming is determined
    by the ring's heteroatom skeleton, not by where a tautomeric H currently
    sits, so we infer the name from topology and let OPSIN place the indicated
    H when it re-parses the fusion name.

    Supports the common 6-ring aromatic bases:
      0 N  → benzene
      1 N  → pyridine
      2 N  → pyridazine (1,2) / pyrimidine (1,3) / pyrazine (1,4)
    Other heteroatom skeletons fall back to the retained lookup.
    """
    if len(base_ring) != 6:
        return _try_monocyclic_base_name(base_ring, mol)

    symbols = {a: mol.GetAtomWithIdx(a).GetSymbol() for a in base_ring}
    het = sorted(a for a, s in symbols.items() if s != "C")

    if len(het) == 0:
        if all(s == "C" for s in symbols.values()):
            return "benzene"
        return _try_monocyclic_base_name(base_ring, mol)

    if len(het) == 1 and symbols[het[0]] == "N":
        return "pyridine"

    if len(het) == 2 and all(symbols[a] == "N" for a in het):
        cycle = _ring_cycle_order(base_ring, mol)
        if cycle is None or len(cycle) != 6:
            return _try_monocyclic_base_name(base_ring, mol)
        pos = {atom: i for i, atom in enumerate(cycle)}
        d = abs(pos[het[0]] - pos[het[1]])
        d = min(d, 6 - d)
        if d == 1:
            return "pyridazine"
        if d == 2:
            return "pyrimidine"
        if d == 3:
            return "pyrazine"

    # Fall back to retained lookup for other (O/S-in-base, triazine, ...) forms.
    return _try_monocyclic_base_name(base_ring, mol)


# ---------------------------------------------------------------------------
# Ring cycle utilities
# ---------------------------------------------------------------------------

def _ring_cycle_order(
    ring_atoms: frozenset[int], mol, start: int | None = None
) -> list[int] | None:
    """Walk a single ring as an ordered cycle.  Optional ``start`` atom; if
    not given, the lowest-index atom is used.  Returns None if the atoms do
    not form a single closed cycle in the molecular graph.
    """
    if not ring_atoms:
        return None
    if start is None:
        start = min(ring_atoms)
    if start not in ring_atoms:
        return None
    adj: dict[int, list[int]] = {a: [] for a in ring_atoms}
    for a in ring_atoms:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            nb_i = nb.GetIdx()
            if nb_i in ring_atoms:
                adj[a].append(nb_i)
    if any(len(adj[a]) < 2 for a in ring_atoms):
        return None
    ordered = [start]
    prev = -1
    curr = start
    n = len(ring_atoms)
    while len(ordered) < n:
        nexts = [nb for nb in adj[curr] if nb != prev]
        if not nexts:
            return None
        nxt = nexts[0]
        if nxt == start:
            return None
        ordered.append(nxt)
        prev = curr
        curr = nxt
    if start not in adj[curr]:
        return None
    return ordered


def _all_directional_numberings(
    base_ring: frozenset[int], mol, anchor_locant_1: int
) -> list[dict[int, int]]:
    """Return both walk directions starting from ``anchor_locant_1`` as
    locant 1.  Each result maps atom_idx → locant.
    """
    cycle = _ring_cycle_order(base_ring, mol, start=anchor_locant_1)
    if cycle is None:
        return []
    forward = {a: i + 1 for i, a in enumerate(cycle)}
    reversed_cycle = [cycle[0]] + list(reversed(cycle[1:]))
    backward = {a: i + 1 for i, a in enumerate(reversed_cycle)}
    return [forward, backward]


# ---------------------------------------------------------------------------
# Mono-ring base: name + numbering
# ---------------------------------------------------------------------------

# Multi-hetero monocyclic 6-ring base patterns we support (Stage 2A).  Each
# entry maps a sorted tuple of (locant, element) to the retained base name.
# Locants reflect IUPAC canonical numbering for the diazine/triazine.
_MULTI_HETERO_BASE_PATTERNS: dict[
    tuple[tuple[int, str], ...], str
] = {
    # Diazines: positions of the N atoms in the canonical numbering.
    ((1, "N"), (2, "N")):                "pyridazine",
    ((1, "N"), (3, "N")):                "pyrimidine",
    ((1, "N"), (4, "N")):                "pyrazine",
    # (Triazines and beyond deferred — they don't typically host a
    # dioxolo fusion in routine chemistry without complicating tautomers.)
}


def _try_monocyclic_base_name(
    base_ring: frozenset[int], mol
) -> str | None:
    """Return the base ring's retained or systematic name (or None).

    Uses the existing retained_lookup on a synthetic monocyclic RingSystem
    pointing at the FULL mol so the lookup respects the actual heteroatom
    positions (and any [nH] tautomer).  Falls back to "benzene" for an
    all-carbon aromatic 6-ring that retained_lookup happens to miss.
    """
    from iupac_namer.ring_naming.retained_lookup import try_retained_name
    from iupac_namer.types import RingSystem as RS

    aromatic = all(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_ring
    )
    base_rs = RS(
        atom_indices=base_ring,
        rings=(base_ring,),
        type="monocyclic",
        aromatic=aromatic,
        bridge_sizes=None,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=tuple(),
        ring_size=len(base_ring),
    )
    try:
        retained = try_retained_name(base_rs, mol)
        if retained:
            return retained[0].name
    except Exception as e:
        logger.debug("Stage 2 mono-base retained lookup failed: %s", e)

    # Fallback: aromatic carbocyclic 6-ring → "benzene"
    if aromatic and all(
        mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in base_ring
    ) and len(base_ring) == 6:
        return "benzene"
    return None


def _canonical_mono_base_name(
    base_ring: frozenset[int], mol
) -> str | None:
    """Return the canonical AROMATIC name of a mono-ring base, regardless
    of whether the input base is actually aromatic in ``mol``.

    This is the Stage 3 analogue of ``_try_monocyclic_base_name``: for a
    saturated pyridine-like base in the input, we still want the canonical
    parent name to be "pyridine" (so the hydro-prefix + fusion descriptor
    can be built on top).  We determine the aromatic form by heteroatom
    arrangement alone:

      * 0 hetero, 6 atoms → ``benzene``
      * 1 hetero, 6 atoms → one of pyridine / pyran-like / ... — we
        restrict to the cases our numbering module already supports, i.e.
        a single N → ``pyridine``.  Other 1-hetero saturated 6-rings are
        rejected (deferred).
      * 2 hetero, 6 atoms → pyrazine / pyrimidine / pyridazine based on
        the 1,2 / 1,3 / 1,4 pattern of the two heteroatom positions.
    """
    aromatic = all(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_ring
    )
    if aromatic:
        return _try_monocyclic_base_name(base_ring, mol)

    # Saturated / partly saturated input.  Infer canonical aromatic name
    # from heteroatom topology.
    if len(base_ring) != 6:
        return None  # 5-ring saturated base deferred (cyclopentane etc.)

    symbols = {
        a: mol.GetAtomWithIdx(a).GetSymbol() for a in base_ring
    }
    het_atoms = sorted(a for a, s in symbols.items() if s != "C")
    if len(het_atoms) == 0:
        return "benzene"

    if len(het_atoms) == 1:
        if symbols[het_atoms[0]] == "N":
            return "pyridine"
        # Other 1-hetero saturated 6-rings (pyran, thiopyran, ...): deferred.
        # Those would need indicated-hydrogen + hydro prefix logic the Stage
        # 3 mono-ring-base numbering module doesn't yet produce.
        return None

    if len(het_atoms) == 2:
        # Both heteros must be N for the supported diazine names.
        if not all(symbols[a] == "N" for a in het_atoms):
            return None
        # Determine their topological separation inside the ring.
        cycle = _ring_cycle_order(base_ring, mol)
        if cycle is None or len(cycle) != 6:
            return None
        pos = {atom: i for i, atom in enumerate(cycle)}
        d = abs(pos[het_atoms[0]] - pos[het_atoms[1]])
        d = min(d, 6 - d)
        if d == 1:
            return "pyridazine"
        if d == 2:
            return "pyrimidine"
        if d == 3:
            return "pyrazine"
        return None

    return None


def _enumerate_monocyclic_base_numberings(
    base_ring: frozenset[int],
    mol,
    base_name: str,
) -> list[dict[int, int]]:
    """Enumerate all IUPAC-valid (atom_idx → locant) numberings for the
    monocyclic base ring, given its known retained name.

    Returns numberings consistent with:
      * 0-hetero (benzene/cyclohexane/etc.): every atom can be locant 1.
      * 1-hetero (pyridine etc.): the heteroatom is locant 1.
      * 2-hetero diazines (pyrazine/pyrimidine/pyridazine): the heteroatom
        positions match the canonical numbering for that name.
    """
    n = len(base_ring)
    hetero_idxs = sorted(
        a for a in base_ring
        if mol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
    )

    # 0 hetero: anchor anywhere
    if len(hetero_idxs) == 0:
        anchors = sorted(base_ring)
        out: list[dict[int, int]] = []
        for anchor in anchors:
            out.extend(_all_directional_numberings(base_ring, mol, anchor))
        return out

    # 1 hetero: heteroatom = locant 1
    if len(hetero_idxs) == 1:
        return _all_directional_numberings(base_ring, mol, hetero_idxs[0])

    # 2 hetero in a 6-ring: pyrazine / pyrimidine / pyridazine
    if len(hetero_idxs) == 2 and n == 6:
        # Both heteroatoms must be N for the supported diazine names.
        if not all(
            mol.GetAtomWithIdx(a).GetSymbol() == "N" for a in hetero_idxs
        ):
            return []
        # Identify required N-locant pattern from base_name.
        required: tuple[int, int] | None = None
        if base_name == "pyridazine":
            required = (1, 2)
        elif base_name == "pyrimidine":
            required = (1, 3)
        elif base_name == "pyrazine":
            required = (1, 4)
        if required is None:
            return []
        # Try every (anchor, direction) pair and keep those where the N's
        # land at exactly the required locants.
        out = []
        for anchor in sorted(base_ring):
            for numbering in _all_directional_numberings(
                base_ring, mol, anchor
            ):
                n_locants = tuple(sorted(
                    numbering[h] for h in hetero_idxs
                ))
                if n_locants == required:
                    out.append(numbering)
        return out

    return []


# ---------------------------------------------------------------------------
# Multi-ring base (Stage 2B): naphthalene / quinoline
# ---------------------------------------------------------------------------

def _build_aromatic_surrogate_mol(
    base_rings: tuple[frozenset[int], ...], mol
) -> tuple[object, dict[int, int]] | None:
    """Build a synthetic aromatic-only RDKit mol for the base ring atoms.

    Used in Stage 3 when the input's base is non-aromatic but we want to
    look up the canonical AROMATIC retained name (``naphthalene`` /
    ``quinoline`` etc.) so we can prepend a hydro- prefix instead of
    getting back the saturated retained form (``decalin`` etc.).

    Returns ``(surrogate_mol, orig_to_surrogate_idx)`` or None on failure.
    The surrogate contains only the base atoms, connected by the same
    ring bonds, but all set to their aromatic forms (C -> c, N -> n, ...)
    with every ring bond aromatic.  Only C/N-heterocycles are supported.
    """
    base_atoms = frozenset().union(*base_rings)
    # Enforce: base must be all C/N (naphthalene/quinoline class) for a
    # safe aromatic surrogate.  Saturated O/S-containing multi-ring bases
    # (decahydro-isobenzofuran etc.) are deferred.
    for a in base_atoms:
        if mol.GetAtomWithIdx(a).GetSymbol() not in ("C", "N"):
            return None

    from rdkit.Chem import RWMol, Atom, BondType

    rw = RWMol()
    orig_to_new: dict[int, int] = {}
    for a in sorted(base_atoms):
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        new_atom = Atom(sym)
        new_atom.SetIsAromatic(True)
        new_idx = rw.AddAtom(new_atom)
        orig_to_new[a] = new_idx

    # Add one bond per pair of base atoms that are bonded in the source mol.
    # All are aromatic in the surrogate.
    seen_pairs: set[tuple[int, int]] = set()
    for a in base_atoms:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            nb_i = nb.GetIdx()
            if nb_i not in base_atoms:
                continue
            pair = tuple(sorted((a, nb_i)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rw.AddBond(orig_to_new[a], orig_to_new[nb_i], BondType.AROMATIC)

    try:
        surrogate = rw.GetMol()
        Chem.SanitizeMol(surrogate)
    except Exception as e:
        logger.debug("Stage 3 aromatic surrogate mol build failed: %s", e)
        return None
    return surrogate, orig_to_new


def _try_multiring_base_name_and_numberings(
    base_rings: tuple[frozenset[int], ...],
    mol,
    force_aromatic: bool = False,
) -> tuple[str, list[dict[int, str]]] | None:
    """For a base composed of MULTIPLE fused rings (Stage 2B), look up the
    retained name and return the per-atom numberings as ``{atom_idx: locant_label}``.

    Locant labels are strings (e.g. ``"1"``, ``"4a"``, ``"8a"``) so the
    fusion-edge letter computation can use them directly.

    When ``force_aromatic`` is set (Stage 3 saturated/partly-saturated
    case), the retained lookup is forced to see the base as aromatic so
    the canonical aromatic parent name (``naphthalene`` / ``quinoline``)
    is returned, letting the caller prepend a hydro- prefix instead of
    getting back the saturated retained form (``decalin`` etc.).

    Returns None if the multi-ring base has no retained name.
    """
    from iupac_namer.ring_naming.retained_lookup import try_retained_name
    from iupac_namer.types import RingSystem as RS, FusionInfo

    if len(base_rings) < 2:
        return None
    # Compute shared edges between base rings to populate FusionInfo.
    shared_edges: list[tuple[int, int]] = []
    for i, r1 in enumerate(base_rings):
        for r2 in base_rings[i + 1:]:
            common = sorted(r1 & r2)
            if len(common) == 2:
                shared_edges.append(tuple(common))
    if not shared_edges:
        return None
    base_atoms = frozenset().union(*base_rings)

    # Stage 3 path: build an aromatic-surrogate mol so the retained lookup
    # returns the canonical aromatic name (naphthalene / quinoline) that
    # we can prepend a hydro- prefix to, rather than the saturated retained
    # form (decalin) that doesn't compose with the fusion descriptor.
    if force_aromatic and not all(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_atoms
    ):
        sur = _build_aromatic_surrogate_mol(base_rings, mol)
        if sur is None:
            return None
        surrogate_mol, orig_to_new = sur
        new_base_atoms = frozenset(orig_to_new[a] for a in base_atoms)
        new_rings = tuple(
            frozenset(orig_to_new[a] for a in r) for r in base_rings
        )
        new_shared_edges = tuple(
            tuple(orig_to_new[a] for a in e) for e in shared_edges
        )
        new_fi = FusionInfo(
            shared_edges=new_shared_edges,
            fusion_atoms=new_shared_edges,
        )
        base_rs = RS(
            atom_indices=new_base_atoms,
            rings=new_rings,
            type="fused",
            aromatic=True,
            bridge_sizes=None,
            spiro_sizes=None,
            fusion_info=new_fi,
            heteroatoms=tuple(),
            ring_size=len(new_base_atoms),
        )
        lookup_mol = surrogate_mol
        new_to_orig = {v: k for k, v in orig_to_new.items()}
    else:
        aromatic = all(
            mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_atoms
        )
        base_rs = RS(
            atom_indices=base_atoms,
            rings=tuple(base_rings),
            type="fused",
            aromatic=aromatic,
            bridge_sizes=None,
            spiro_sizes=None,
            fusion_info=FusionInfo(
                shared_edges=tuple(shared_edges),
                fusion_atoms=tuple(shared_edges),
            ),
            heteroatoms=tuple(),
            ring_size=len(base_atoms),
        )
        lookup_mol = mol
        new_to_orig = None

    try:
        retained = try_retained_name(base_rs, lookup_mol)
    except Exception as e:
        logger.debug("Stage 2/3 multi-ring base retained lookup failed: %s", e)
        return None
    if not retained:
        return None
    np = retained[0]
    if not np.numbering_options:
        return None

    # Stage2 fusion-base opt-out gate.  Curated retained-name entries carrying
    # ``stage2_fusion_base: False`` (data_loader._RING_CURATED_SMILES) provide
    # atom_locants for the substituent-locant rendering path but MUST NOT be
    # selectable as a multi-ring fusion base.  This decouples atom_locants
    # (needed for naming "2-chloroanthracene" correctly) from Stage 2B
    # eligibility (where anthracene as a 3-ring base would let a 4-ring
    # anthracene-dioxole emit ``[1,3]dioxolo[4,5-b]anthracene``, violating the
    # ≤3-ring Stage 2 invariant — see
    # tests/test_fused_ring_hetero.py::test_stage2_excludes_four_plus_ring_systems).
    from iupac_namer.ring_naming.common import extract_ring_mol
    from iupac_namer.ring_naming.retained_lookup import (
        is_stage2_fusion_base_eligible,
    )
    try:
        gate_ring_mol = extract_ring_mol(base_rs, lookup_mol)
        gate_smiles = (
            Chem.MolToSmiles(gate_ring_mol)
            if gate_ring_mol is not None
            else None
        )
    except Exception:
        gate_smiles = None
    if not is_stage2_fusion_base_eligible(gate_smiles):
        return None
    # Convert each Numbering's _assignments to a {atom: label_string} dict.
    # When we used a surrogate mol, translate atom indices back to the
    # original mol's indexing so downstream fusion-letter / substituent
    # placement logic keeps working.
    out_numberings: list[dict[int, str]] = []
    for numbering in np.numbering_options:
        atom_to_label: dict[int, str] = {}
        for atom_idx, locant in numbering._assignments:
            orig_idx = new_to_orig[atom_idx] if new_to_orig else atom_idx
            atom_to_label[orig_idx] = locant.label
        out_numberings.append(atom_to_label)
    return np.name, out_numberings


# ---------------------------------------------------------------------------
# Fusion-letter computation
# ---------------------------------------------------------------------------

def _fusion_letter_from_string_locants(
    base_numbering: dict[int, str],
    fusion_atoms: tuple[int, int],
) -> str | None:
    """Compute the IUPAC fusion letter for an edge whose endpoints carry the
    given string-form locant labels (e.g. ``"1"``, ``"4a"``, ``"8a"``).

    Letter assignment in fused-ring nomenclature: edge between locants 1-2 is
    'a', 2-3 is 'b', 3-4 is 'c', etc.  Locants like ``"4a"`` (peri/bridgehead)
    follow the numeric one: edge 4-4a is between numeric 4 and the next
    numeric position; we treat it as edge after locant 4 in the alphabetical
    sequence.

    Returns the lower-case letter, or None if the locants are unrecognised
    or the edge doesn't have a single-letter assignment.
    """
    a, b = fusion_atoms
    la = base_numbering.get(a)
    lb = base_numbering.get(b)
    if la is None or lb is None:
        return None

    # Convert locant labels to a comparable "index in the alphabetical
    # sequence".  In a fused multi-ring system numbered 1..N with
    # bridgehead locants like "4a", the alphabetical sequence walks
    # 1,2,3,4,4a,5,6,7,8,8a,1 and edges between consecutive entries are
    # labelled 'a','b','c','d','e','f','g','h','i','j','k' starting from
    # the 1-2 edge as 'a'.  We map each locant label to its index in that
    # walk (1-based) using a parse + secondary-key sort.
    #
    # For a generic mono- or fused ring with numeric locants 1..N and
    # optional 'a' suffixes after their numeric anchor, the canonical
    # alphabetical order is:
    #   1, 2, 3, ..., k, ka, k+1, ka+1, ...
    # We rebuild a sorted list of all locant labels in the numbering and
    # use position to compute the letter.
    all_labels = sorted(
        set(base_numbering.values()),
        key=_locant_sort_key,
    )
    try:
        ia = all_labels.index(la)
        ib = all_labels.index(lb)
    except ValueError:
        return None
    lo, hi = sorted((ia, ib))
    n = len(all_labels)
    # Consecutive in the cyclic order: edge index = lo (0-based), letter = chr('a' + lo).
    if hi - lo == 1:
        idx = lo + 1  # 1-based: edge between positions 0,1 → 'a'
    elif lo == 0 and hi == n - 1:
        # Wrap-around edge between last and first.
        idx = n
    else:
        return None
    if idx < 1 or idx > 26:
        return None
    return chr(ord("a") + idx - 1)


def _locant_sort_key(label: str) -> tuple[int, str]:
    """Sort key for IUPAC locant labels.  Numeric locants sort by integer;
    suffixed locants ('4a', '8a', '8b') sort immediately after their numeric
    anchor in alphabetical suffix order.

    "4" → (4, '')
    "4a" → (4, 'a')
    "4b" → (4, 'b')
    """
    digits = ""
    suffix = ""
    for c in label:
        if c.isdigit():
            digits += c
        else:
            suffix += c
    try:
        n = int(digits) if digits else 0
    except ValueError:
        n = 0
    return n, suffix


def _fusion_letter_from_int_locants(
    base_numbering: dict[int, int],
    fusion_atoms: tuple[int, int],
    base_ring_size: int,
) -> str | None:
    """Stage 1 monocyclic helper: edge between consecutive locants k, k+1
    → letter ``chr('a' + k - 1)``.  Wrap-around edge n–1 → last letter.
    """
    str_numbering = {a: str(loc) for a, loc in base_numbering.items()}
    return _fusion_letter_from_string_locants(str_numbering, fusion_atoms)


# ---------------------------------------------------------------------------
# Fusion-edge selection (mono-ring base)
# ---------------------------------------------------------------------------

def _select_mono_base_numbering(
    base_ring: frozenset[int],
    mol,
    fusion_atoms: tuple[int, int],
    base_name: str,
) -> tuple[dict[int, int], str] | None:
    """Choose the base ring numbering (anchor + direction) that gives the
    lowest fusion letter for the fusion edge.  Stage 1+2A monocyclic.

    Returns ``(numbering, letter)`` or None.
    """
    n = len(base_ring)
    candidates: list[tuple[str, dict[int, int]]] = []
    for numbering in _enumerate_monocyclic_base_numberings(
        base_ring, mol, base_name
    ):
        letter = _fusion_letter_from_int_locants(numbering, fusion_atoms, n)
        if letter is not None:
            candidates.append((letter, numbering))

    if not candidates:
        return None

    # P-25.3.1.3: prefer lowest fusion letter.  Stage 1 preserves the
    # historical preference for 'b' over 'a' in benzo/pyridino fusions
    # (the dioxolo-side numbering forces the heteroatoms onto locants 1,3
    # of the small ring, which in turn forces fusion atoms onto positions
    # 3a,Na — the local edge label in the BASE numbering becomes 'b').
    candidates.sort(key=lambda t: (
        0 if t[0] == "b" else 1 if t[0] == "a" else 2,
        t[0],
    ))
    return candidates[0][1], candidates[0][0]


def _select_multi_base_numbering(
    base_rings: tuple[frozenset[int], ...],
    mol,
    fusion_atoms: tuple[int, int],
    force_aromatic: bool = False,
) -> tuple[str, dict[int, str], str] | None:
    """Stage 2B: select the BASE name + numbering + fusion letter for a
    multi-ring fused base whose retained name was looked up via
    ``_try_multiring_base_name_and_numberings``.

    Returns ``(base_name, atom_to_label_numbering, fusion_letter)``.
    """
    base_lookup = _try_multiring_base_name_and_numberings(
        base_rings, mol, force_aromatic=force_aromatic
    )
    if base_lookup is None:
        return None
    base_name, numberings = base_lookup
    base_atoms = frozenset().union(*base_rings)
    candidates: list[tuple[str, dict[int, str]]] = []
    for numbering in numberings:
        # Completeness guard: the retained-name lookup for some 4-ring bases
        # (e.g. cyclopenta[a]phenanthrene, where atom_locants intentionally
        # omit the 4a/4b/8a/10a/11a junction positions) returns numberings
        # missing peri locants.  ``_fusion_letter_from_string_locants`` would
        # then produce a deep alphabetical letter (`o`, `p`, ...) that OPSIN
        # rejects when round-tripping the composed fused name.  Reject such
        # numberings here so VB fallback wins instead of an invalid fused
        # form being emitted.  Stage 4 will infer the missing junction
        # locants from ring topology so 4-ring retained bases can flow
        # through this pipeline cleanly.
        if len(numbering) != len(base_atoms):
            continue
        letter = _fusion_letter_from_string_locants(numbering, fusion_atoms)
        if letter is not None:
            candidates.append((letter, numbering))
    if not candidates:
        return None
    # P-25.3.1.3: lowest fusion letter wins.  Same 'b'-preference heuristic
    # as the monocyclic case (verified against OPSIN for naphthalene[d]/[g]).
    candidates.sort(key=lambda t: (
        0 if t[0] == "b" else 1 if t[0] == "a" else 2,
        t[0],
    ))
    return base_name, candidates[0][1], candidates[0][0]


def _select_mono_base_numbering_lowest_letter(
    base_ring: frozenset[int],
    mol,
    fusion_atoms: tuple[int, int],
    base_name: str,
) -> tuple[dict[int, int], str] | None:
    """Mono-hetero analogue of ``_select_mono_base_numbering`` for a single
    monocyclic base ring.

    Unlike the symmetric [1,3]-dihetero fusion (whose smaller-ring numbering
    forces fusion atoms onto the base 'b' edge), a mono-hetero smaller
    component can fuse at the genuinely lowest-lettered base edge, so we apply
    the plain P-25.3.3 rule: choose the base numbering giving the LOWEST
    fusion letter.  Ties (symmetric bases) are broken by the descriptor
    selection downstream.

    Returns ``(base_numbering atom->int, fusion_letter)`` or None.
    """
    n = len(base_ring)
    candidates: list[tuple[str, dict[int, int]]] = []
    for numbering in _enumerate_monocyclic_base_numberings(
        base_ring, mol, base_name
    ):
        letter = _fusion_letter_from_int_locants(numbering, fusion_atoms, n)
        if letter is not None:
            candidates.append((letter, numbering))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1], candidates[0][0]


def _select_multi_base_numbering_lowest_letter(
    base_rings: tuple[frozenset[int], ...],
    mol,
    fusion_atoms: tuple[int, int],
) -> tuple[str, dict[int, str], str] | None:
    """Mono-hetero analogue of ``_select_multi_base_numbering``: lowest fusion
    letter (plain P-25.3.3), no 'b'-preference heuristic.

    Returns ``(base_name, atom_to_label_numbering, fusion_letter)`` or None.
    """
    base_lookup = _try_multiring_base_name_and_numberings(base_rings, mol)
    if base_lookup is None:
        return None
    base_name, numberings = base_lookup
    base_atoms = frozenset().union(*base_rings)
    candidates: list[tuple[str, dict[int, str]]] = []
    for numbering in numberings:
        if len(numbering) != len(base_atoms):
            continue
        letter = _fusion_letter_from_string_locants(numbering, fusion_atoms)
        if letter is not None:
            candidates.append((letter, numbering))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return base_name, candidates[0][1], candidates[0][0]


# ---------------------------------------------------------------------------
# Numbering construction for the full fused system
# ---------------------------------------------------------------------------

def _build_full_numbering_mono_base(
    smaller_ring: frozenset[int],
    base_ring: frozenset[int],
    fusion_atoms: tuple[int, int],
    middle_carbon: int,
    extra_carbons: list[int],
    smaller_elements_in_order: list[str],
    mol,
) -> Numbering | None:
    """Build per-atom IUPAC locants for the full fused system when the BASE
    is a single monocyclic ring.

    Smaller-ring locants:
      5-ring case: heteroatoms at 1,3; middle C at 2; fusion atoms at 3a, Na.
      6-ring case: heteroatoms at 1,3; middle C at 2; fusion atoms at 4a, Na;
                   extra C at 4 (between N-side hetero and fusion).

    Wait — re-deriving from OPSIN [1,3]dioxino[4,5-b]benzene SMILES
    'c1ccc2c(c1)COCO2': for the smaller 6-ring, locants 1=O, 2=CH2, 3=O,
    then 4 is the next non-fusion atom, then fusion atoms get 4a, 8a.
    Actually the smaller ring canonical numbering is 1,2,3,4,4a,8a where
    4 is a non-fusion CH2 and 4a is the first fusion atom adjacent to it.

    So for the 6-ring smaller, walk the smaller ring as:
      1 (senior het) → 2 (CH2) → 3 (junior het) → 4 (CH2 = extra carbon
      adjacent to fusion atom Na) ... wait, then 4a would be on the OTHER
      side.  Let me re-derive:

    For [1,3]dioxino[4,5-b]benzene the smaller-ring fusion edge is
    locants 4,5 — but those locants don't exist as such in the canonical
    numbering of the FULL bicyclic, where they become 4a and 8a.  The
    descriptor "[4,5-b]" refers to the SMALLER ring's standalone numbering
    (1,2,3,4,5,6 cyclical) before fusion.

    In the FULL bicyclic numbering of [1,3]dioxino[4,5-b]benzene, the small
    ring contributes positions 1,2,3,4,4a,8a; the benzene ring contributes
    4a,5,6,7,8,8a.

    So for the 6-ring smaller component, atoms in walk order from h_senior
    around the non-fusion arc are:
      1: h_senior (= O)
      2: middle C
      3: h_junior (= O)
      4: extra non-fusion C (= CH2 between h_junior and fusion atom)
      4a: fusion atom adjacent to extra C
      8a: fusion atom adjacent to h_senior
    """
    fa1, fa2 = fusion_atoms
    smaller_non_fusion = smaller_ring - set(fusion_atoms)
    hetero_atoms = sorted(smaller_non_fusion - {middle_carbon} - set(extra_carbons))
    if len(hetero_atoms) != 2:
        return None

    senior_elem = smaller_elements_in_order[0]
    junior_elem = smaller_elements_in_order[-1]

    def elem(idx: int) -> str:
        return mol.GetAtomWithIdx(idx).GetSymbol()

    if senior_elem == junior_elem:
        h_senior = hetero_atoms[0]
        h_junior = hetero_atoms[1]
    else:
        h_senior = next((h for h in hetero_atoms if elem(h) == senior_elem), None)
        h_junior = next((h for h in hetero_atoms if elem(h) == junior_elem), None)
        if h_senior is None or h_junior is None:
            return None

    # For 5-ring smaller: h_junior is adjacent to a fusion atom (3a-side).
    # For 6-ring smaller: h_junior is adjacent to the extra CH2 (locant 4),
    #   which in turn is adjacent to fusion atom 4a; h_senior is adjacent
    #   directly to fusion atom 8a.
    h_junior_neighbors_in_ring = {
        nb.GetIdx() for nb in mol.GetAtomWithIdx(h_junior).GetNeighbors()
        if nb.GetIdx() in smaller_ring
    }
    h_senior_neighbors_in_ring = {
        nb.GetIdx() for nb in mol.GetAtomWithIdx(h_senior).GetNeighbors()
        if nb.GetIdx() in smaller_ring
    }

    base_size = len(base_ring)

    if not extra_carbons:
        # 5-ring smaller (Stage 1 path).
        fa_adj_junior = h_junior_neighbors_in_ring & set(fusion_atoms)
        fa_adj_senior = h_senior_neighbors_in_ring & set(fusion_atoms)
        if len(fa_adj_junior) != 1 or len(fa_adj_senior) != 1:
            return None
        atom_3a = next(iter(fa_adj_junior))   # locant 3a
        atom_n_a = next(iter(fa_adj_senior))  # locant Na (e.g. 7a for benzene base)
        if atom_3a == atom_n_a:
            return None

        # Walk base ring from atom_3a in the direction AWAY from atom_n_a.
        cycle = _ring_cycle_order(base_ring, mol, start=atom_3a)
        if cycle is None:
            return None
        if cycle[1] == atom_n_a:
            cycle = [cycle[0]] + list(reversed(cycle[1:]))
        if cycle[1] == atom_n_a or cycle[-1] != atom_n_a:
            return None
        last_numeric = 4 + (base_size - 2) - 1

        locant_map: dict[int, Locant] = {}
        locant_map[h_senior] = Locant.numeric(1)
        locant_map[middle_carbon] = Locant.numeric(2)
        locant_map[h_junior] = Locant.numeric(3)
        locant_map[atom_3a] = Locant.numeric(3, "a")
        for pos, atom in enumerate(cycle[1:-1]):
            locant_map[atom] = Locant.numeric(4 + pos)
        locant_map[atom_n_a] = Locant.numeric(last_numeric, "a")
    else:
        # 6-ring smaller (Stage 2C: dioxino).  Smaller-ring locants:
        # 1 = h_senior, 2 = middle, 3 = h_junior, 4 = extra C, 4a, ..., Na.
        if len(extra_carbons) != 1:
            return None
        extra_c = extra_carbons[0]
        # extra_c's neighbors in the smaller ring: one fusion atom (= 4a)
        # and h_junior.
        extra_neighbors = {
            nb.GetIdx() for nb in mol.GetAtomWithIdx(extra_c).GetNeighbors()
            if nb.GetIdx() in smaller_ring
        }
        fa_adj_extra = extra_neighbors & set(fusion_atoms)
        fa_adj_senior = h_senior_neighbors_in_ring & set(fusion_atoms)
        if len(fa_adj_extra) != 1 or len(fa_adj_senior) != 1:
            return None
        atom_4a = next(iter(fa_adj_extra))   # locant 4a
        atom_n_a = next(iter(fa_adj_senior))  # locant Na (8a for benzene base)
        if atom_4a == atom_n_a:
            return None

        # Walk base ring from atom_4a away from atom_n_a (assigns 5,6,...).
        cycle = _ring_cycle_order(base_ring, mol, start=atom_4a)
        if cycle is None:
            return None
        if cycle[1] == atom_n_a:
            cycle = [cycle[0]] + list(reversed(cycle[1:]))
        if cycle[1] == atom_n_a or cycle[-1] != atom_n_a:
            return None
        # Numeric base locants are 5..last_numeric; total = 4 + (base_size - 2)
        last_numeric = 4 + (base_size - 2)  # base_size=6 → last_numeric = 8

        locant_map = {}
        locant_map[h_senior] = Locant.numeric(1)
        locant_map[middle_carbon] = Locant.numeric(2)
        locant_map[h_junior] = Locant.numeric(3)
        locant_map[extra_c] = Locant.numeric(4)
        locant_map[atom_4a] = Locant.numeric(4, "a")
        for pos, atom in enumerate(cycle[1:-1]):
            locant_map[atom] = Locant.numeric(5 + pos)
        locant_map[atom_n_a] = Locant.numeric(last_numeric, "a")

    assignments = tuple(sorted(locant_map.items(), key=lambda kv: kv[0]))
    locant_set = tuple(sorted(
        locant_map.values(),
        key=lambda l: (l._numeric_value or 0, getattr(l, "suffix", "") or ""),
    ))
    return Numbering(_assignments=assignments, locant_set=locant_set)


# ---------------------------------------------------------------------------
# Stage 3: hydro-prefix computation
# ---------------------------------------------------------------------------

def _base_saturation(
    base_atoms: frozenset[int], mol
) -> tuple[bool, bool, set[int]]:
    """Classify the base ring system's saturation state.

    Returns ``(is_fully_aromatic, is_fully_saturated, sp3_like_atoms)`` where:
      * ``is_fully_aromatic``: every base atom has ``GetIsAromatic() == True``.
      * ``is_fully_saturated``: every base atom is non-aromatic AND has no
        double/triple/aromatic bond to another base atom.
      * ``sp3_like_atoms``: the subset of base atoms that have NO
        double/triple/aromatic bond to another base atom (i.e. positions
        that would carry a hydro-locant relative to the fully-aromatic
        canonical parent).  Empty when the base is fully aromatic; equals
        ``base_atoms`` when fully saturated.
    """
    fully_aromatic = all(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_atoms
    )
    sp3_like: set[int] = set()
    any_unsat = False
    for a in base_atoms:
        atom = mol.GetAtomWithIdx(a)
        has_unsat_in_base = False
        if atom.GetIsAromatic():
            has_unsat_in_base = True
        else:
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom).GetIdx()
                if other not in base_atoms:
                    continue
                bt = bond.GetBondTypeAsDouble()
                if bt >= 1.5:  # aromatic (1.5) or double (2.0) or triple (3.0)
                    has_unsat_in_base = True
                    break
        if has_unsat_in_base:
            any_unsat = True
        else:
            sp3_like.add(a)
    fully_saturated = (not fully_aromatic) and (len(sp3_like) == len(base_atoms))
    return fully_aromatic, fully_saturated, sp3_like


def _hydro_prefix_mono_base(
    sp3_like_atoms: set[int],
    base_ring: frozenset[int],
    numbering: Numbering | None,
    fully_saturated: bool,
) -> str:
    """Build the hydro-prefix for a mono-ring base.

    ``numbering`` is the full-system Numbering from
    ``_build_full_numbering_mono_base``; the base atoms are mapped to
    integer locants (e.g. 3a, 4, 5, 6, 7, 7a for a benzene base fused with
    a dioxolo smaller ring).

    Returns either ``""`` (fully aromatic, no hydros), a bare
    ``"<multiplier>hydro-"`` (fully saturated, no explicit locants), or an
    explicit ``"<locant-list>-<multiplier>hydro-"`` (partly saturated).
    """
    from iupac_namer.data_loader import get_multiplier

    if not sp3_like_atoms:
        return ""

    count = len(sp3_like_atoms)
    mult = get_multiplier(count)
    # "hydro" alone with count 1 would be "monohydro" which IUPAC doesn't
    # support in this context; a single sp3 atom in an otherwise aromatic
    # base is unusual and we defer it.
    if mult is None:
        return ""

    if fully_saturated:
        # Omit explicit locants per OPSIN's canonical form (hexahydro- rather
        # than 3a,4,5,6,7,7a-hexahydro-).  Both are accepted; the shorter
        # form is preferred in P-31.1.4.2.4 examples.
        return f"{mult}hydro-"

    # Partly saturated: need explicit locants.  Look them up in the numbering.
    if numbering is None:
        return ""
    atom_to_loc: dict[int, Locant] = dict(numbering._assignments)
    locants_for_hydros: list[Locant] = []
    for a in sorted(sp3_like_atoms):
        loc = atom_to_loc.get(a)
        if loc is None:
            return ""
        locants_for_hydros.append(loc)

    locants_for_hydros.sort(
        key=lambda l: (l._numeric_value or 0, getattr(l, "suffix", "") or "")
    )
    loc_str = ",".join(l.label for l in locants_for_hydros)
    return f"{loc_str}-{mult}hydro-"


def _hydro_prefix_multi_base_fully_saturated(
    base_atoms: frozenset[int],
) -> str:
    """Hydro-prefix for a multi-ring fused base (Stage 2B) when it is fully
    saturated.  Partial saturation on a multi-ring base is not attempted
    (numbering integration with the carved retained lookup is deferred)."""
    from iupac_namer.data_loader import get_multiplier
    count = len(base_atoms)
    mult = get_multiplier(count)
    if mult is None:
        return ""
    return f"{mult}hydro-"


# ---------------------------------------------------------------------------
# Mono-heteroatom smaller-component naming path (P-25.3.1.3)
# ---------------------------------------------------------------------------

def _try_mono_hetero_fused(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Name an ortho-fused system whose SMALLER component is a 5-ring with a
    single ring heteroatom (furo / thieno / pyrrolo / selenolo / ...) fused
    onto a nameable aromatic base (benzene / pyridine / diazine / naphthalene
    / quinoline / ...).

    Generative, structural — no molecule pins.  Emits

        ``{prefix}[{d1},{d2}-{letter}]{base}``  (P-25.3.1.3)

    where ``{d1},{d2}`` are the smaller component's attachment locants (its
    heteroatom = locant 1, fusion edge as low as possible, cited in the base's
    lettering direction) and ``{letter}`` is the base fusion-edge letter
    (lowest possible per P-25.3.3).

    Scope notes (documented gaps):
      * Only a 5-ring mono-hetero smaller component is handled.  6-ring
        mono-hetero prefixes (pyrano / thiopyrano / pyrido) name a NON-fully-
        aromatic partner and require indicated-hydrogen handling that this
        path does not yet produce.
      * The base must be FULLY AROMATIC.  Saturated / partly-saturated bases
        for the mono-hetero path are deferred (the hydro-prefix numbering
        machinery is wired only for the dihetero smaller component, whose
        full-system numbering is fixed; a mono-hetero smaller ring's
        peripheral numbering varies with the heteroatom position).
      * Where a retained fused name exists (indole, isoindole, benzofuran,
        benzothiophene, the purine family, ...) the retained lookup wins
        upstream; this path only fires for systematically-named scaffolds
        like furo[3,2-b]pyridine / thieno[2,3-d]pyrimidine.
    """
    fi = ring_system.fusion_info
    if fi is None or not fi.shared_edges:
        return []

    # Retained fused names (indole, isoindole, benzofuran, benzothiophene,
    # purine, ...) take absolute priority (P-25.3.1.3 / P-31.1.3): the
    # systematic mono-hetero fusion form is a FALLBACK for scaffolds without a
    # retained name.  Emitting a competing numbering-less systematic candidate
    # for a retained fused system (e.g. a substituted indole) can disrupt
    # substituent-locant resolution downstream, so we skip the mono-hetero
    # path whenever the whole ring system already has a retained name.
    try:
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        if try_retained_name(ring_system, mol):
            return []
    except Exception as e:
        logger.debug("mono-hetero retained pre-check failed: %s", e)

    triple = _identify_mono_smaller_and_base(
        ring_system.rings, fi.fusion_atoms, mol
    )
    if triple is None:
        return []
    smaller_ring, base_rings, smaller_base_edge = triple

    # Smaller component must be fully aromatic (furo/thieno/pyrrolo name the
    # aromatic mancude partner; a saturated 5-ring mono-hetero smaller would
    # need its own indicated-H / hydro handling).
    if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in smaller_ring):
        return []

    pref = _mono_smaller_prefix(smaller_ring, mol)
    if pref is None:
        return []
    smaller_prefix, _hetero_elem = pref

    # Base must be fully aromatic for the mono-hetero path (see scope notes).
    base_atoms = frozenset().union(*base_rings)
    if not all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in base_atoms):
        return []

    base_name: str
    fusion_letter: str
    base_numbering: dict[int, int] | dict[int, str]

    if len(base_rings) == 1:
        base_ring = base_rings[0]
        base_name_opt = _mono_hetero_aromatic_base_name(base_ring, mol)
        if base_name_opt is None:
            return []
        base_name = base_name_opt
        sel = _select_mono_base_numbering_lowest_letter(
            base_ring, mol, smaller_base_edge, base_name
        )
        if sel is None:
            return []
        base_numbering, fusion_letter = sel
    else:
        sel_multi = _select_multi_base_numbering_lowest_letter(
            base_rings, mol, smaller_base_edge
        )
        if sel_multi is None:
            return []
        base_name, base_numbering, fusion_letter = sel_multi

    descriptor = _mono_fusion_descriptor(
        smaller_ring, smaller_base_edge, base_numbering, mol,
    )
    if descriptor is None:
        return []
    d1, d2 = descriptor

    name_str = f"{smaller_prefix}[{d1},{d2}-{fusion_letter}]{base_name}"
    stem_str = name_str[:-1] if name_str.endswith("e") else name_str

    # FR-5.3 / P-25.3.3 canonical peripheral numbering.  Without an explicit
    # numbering the downstream generic fused-numbering fallback numbers every
    # atom 1..N sequentially (no ring-fusion ``Na`` suffixes, different
    # heteroatom-priority tie-break), producing substituent locants that are
    # inconsistent with — and sometimes outside the range of — the emitted
    # fusion name (e.g. ``9-chlorofuro[2,3-b]pyridine``).  Compute the genuine
    # peripheral numbering so substituent locants match the fusion name.
    #
    # ``compute_peripheral_numberings`` returns exactly ONE numbering: the
    # FR-5.3 winner for unambiguous systems, and — for the near-symmetric mirror
    # families (pyrrolo/furo/thieno[3,4-b]pyrazine and [3,4-d]pyridazine) where
    # the bare-skeleton criteria tie — the one selected by the FR-5.2
    # 2D-orientation tie-break (P-25.3.3.1.2 (f), low locants to indicated
    # hydrogen).  For the asymmetric (pyrrolo NH) members this picks OPSIN's
    # preferred mirror; for the symmetric (furo/thieno) members the two mirrors
    # are automorphism-equivalent so a deterministic representative is returned,
    # round-tripping correctly either way.  We attach that single numbering.
    numbering_options: tuple[Numbering, ...] = ()
    try:
        from iupac_namer.ring_naming.fr_orientation import (
            compute_peripheral_numberings,
        )
        fr_numberings = compute_peripheral_numberings(
            ring_system.atom_indices, mol
        )
        if fr_numberings:
            numbering_options = (fr_numberings[0],)
    except Exception as e:  # defensive — never crash the naming pass
        logger.debug("FR-5.3 peripheral numbering failed: %s", e)

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem_str,
        alkyl_stem=None,
        # Aromatic fused parent: catch-all "systematic" rank so any retained
        # fused name (indole / benzofuran / purine / ...) wins upstream.
        naming_method="systematic",
        indicated_hydrogen=None,
        numbering_options=numbering_options,
    )]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _name_fused_hetero(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Generate fused-ring nomenclature names for ortho-fused systems with a
    [1,3]-dihetero 5- or 6-ring smaller component and a known-name base.

    Stage 1: 2-ring system, 5-ring smaller, mono-ring base (0-1 hetero).
    Stage 2A: 2-ring system, 5-ring smaller, multi-hetero mono-ring base
              (pyrazine / pyrimidine / pyridazine).
    Stage 2B: 3-ring system, 5-ring smaller, 2-ring multi-ring base
              (naphthalene / quinoline).
    Stage 2C: 2-ring system, 6-ring smaller (dioxino), mono-ring base.
    Stage 3:  saturated / partly-saturated base parents.  Emits a hydro-
              prefix ("hexahydro-", "decahydro-", "<locants>-tetrahydro-",
              ...) where the base ring atoms are not all aromatic in the
              input molecule.  Stage 3 partial saturation is supported on
              the mono-ring base path only; Stage 3 multi-ring bases must
              be fully saturated for a valid name.

    Stage 4 investigation (4-ring base, cyclopenta[a]phenanthrene):
      The 5-ring total cap (``len(rings) > 5`` gate) admits 4-ring retained
      bases like cyclopenta[a]phenanthrene fused to a dioxolo smaller.  The
      downstream completeness guard in ``_select_multi_base_numbering``
      currently rejects these because the retained-table atom_locants for
      the various cyclopenta[a]phenanthrene saturation-state entries omit
      junction positions (typically locant 5, sometimes 9, 10).

      A full Stage 4 implementation would need TWO pieces of machinery:
        (i) peri-locant / junction-locant inference to fill in the missing
            atom_locants (addressable by chloro-probing OPSIN and skeletal-
            topology substructure matching — verified feasible in April
            2026 investigation);
       (ii) a peripheral-walk-aware fusion-letter computation.  The current
            ``_fusion_letter_from_string_locants`` assumes the base's
            locants form a simple 1..N cycle; for cyclopenta[a]phenanthrene
            the alphabetical walk skips internal junctions and OPSIN only
            accepts letters a..n (14 letters), whereas the naive sort over
            locants 1..17 would assign letter 'p' to the 16-17 (D-ring) edge.

      More critically, OPSIN does NOT accept ANY
      ``[1,3]dioxolo[4,5-X]cyclopenta[a]phenanthrene`` form that round-trips
      to the amcinonide-partner skeleton (the 16,17-dioxolane fused to the
      steroid D-ring).  OPSIN instead parses this compound as
      ``16,17-methylenedioxyhexadecahydro-1H-cyclopenta[a]phenanthrene`` — a
      methylenedioxy BRIDGE naming, not a ring-fusion naming.

      Probed April 2026 (py2opsin with OpenJDK21): all 14 valid fusion
      letters ``a..n`` for ``[1,3]dioxolo[4,5-<L>]cyclopenta[a]phenanthrene``
      tested fully saturated (hexadecahydro); none matches the skeletal
      topology of amcinonide's polycyclic partner
      (``C1CCC2C(C1)CCC1C2CCC2C1CC1OCOC12``).  This confirms Stage 4 via a
      fused-hetero path is architecturally the wrong approach for
      amcinonide — the methylenedioxy-bridge path (handled elsewhere) is
      the correct canonical IUPAC form.  The completeness guard therefore
      stays in place as a Stage 4 safety net, and amcinonide continues to
      emit the polyspiro articulation / VB form that round-trips through
      OPSIN cleanly.

    Returns a list with one NamedParent (or empty).  Ranked at a dedicated
    ``fused_hetero_hydro`` method priority — below retained/HW (so retained
    names always win where they exist) but above VB fallbacks (so saturated
    dioxolo-type heterocycles prefer the fused systematic name over von-
    Baeyer decomposition).
    """
    fi = ring_system.fusion_info
    if fi is None:
        return []
    if not fi.shared_edges:
        return []
    if len(ring_system.rings) < 2:
        return []
    # Architectural gate: the smaller-component approach handles one 5- or
    # 6-ring smaller fused to a 1-, 2-, 3-, or 4-ring base.  Larger systems
    # are deferred (peri-fused / >4-ring bases need a renumbering walk that
    # the current retained-lookup-pinned numbering doesn't supply).
    #
    # Note: a downstream completeness guard in
    # ``_select_multi_base_numbering`` ensures that retained-name numberings
    # which OMIT junction locants (e.g. cyclopenta[a]phenanthrene's atom_locants
    # skips position 5, sometimes 9 and 10) are rejected here rather than
    # emitted with a bogus fusion letter.  See Stage 4 investigation notes in
    # the ``name_fused`` docstring above: OPSIN does not accept any
    # ``[1,3]dioxolo[4,5-X]cyclopenta[a]phenanthrene`` form for amcinonide's
    # D-ring 16,17 edge — the canonical IUPAC form is methylenedioxy-bridge
    # on hexadecahydrocyclopenta[a]phenanthrene, so the completeness guard
    # stays in place as a safety net.
    if len(ring_system.rings) > 5:
        return []
    # Number of shared edges must equal n_rings - 1 (true ortho-fusion of a
    # tree of rings — no peri-fusion or shared atoms).
    if len(fi.shared_edges) != len(ring_system.rings) - 1:
        return []

    # Identify the smaller (5- or 6-ring) and base components.
    triple = _identify_smaller_and_base(
        ring_system.rings, fi.fusion_atoms, mol
    )
    if triple is None:
        # No [1,3]-dihetero smaller component — try the mono-heteroatom
        # smaller-component path (furo / thieno / pyrrolo / ...).
        return _try_mono_hetero_fused(ring_system, candidate, mol)
    smaller_ring, base_rings, smaller_base_edge = triple

    # Verify the smaller ring's [1,3]-dihetero pattern.
    pattern = _is_dihetero_pattern(smaller_ring, smaller_base_edge, mol)
    if pattern is None:
        return _try_mono_hetero_fused(ring_system, candidate, mol)
    elements_in_order, middle_carbon, extra_carbons = pattern

    # Build the smaller-ring fusion prefix (dioxolo, dithiolo, dioxino, ...)
    smaller_prefix = _build_smaller_prefix(elements_in_order, len(smaller_ring))
    if smaller_prefix is None:
        return []

    # Classify base saturation state.  Stage 1/2 handle fully aromatic
    # bases; Stage 3 extends to fully saturated and partly saturated.
    base_atoms = frozenset().union(*base_rings)
    base_fully_aromatic, base_fully_saturated, base_sp3_atoms = _base_saturation(
        base_atoms, mol
    )

    # Resolve base name + numbering + fusion letter.  The base NAME is
    # always the fully-aromatic canonical form; we add a hydro-prefix to
    # express any saturation state.
    base_name: str
    numbering: Numbering | None = None
    fusion_letter: str
    hydro_prefix: str = ""

    if len(base_rings) == 1:
        # Mono-ring base path (Stage 1 + 2A + 2C, plus Stage 3 hydro).
        base_ring = base_rings[0]
        # Retained-lookup is keyed on the ACTUAL base-ring aromaticity, so
        # a fully-saturated base won't resolve to "benzene" / "pyridine" via
        # try_retained_name.  Pick the canonical aromatic name by calling
        # the helper with a synthesised aromatic "view" of the base.
        base_name_opt = _canonical_mono_base_name(base_ring, mol)
        if base_name_opt is None:
            return []
        base_name = base_name_opt
        sel = _select_mono_base_numbering(
            base_ring, mol, smaller_base_edge, base_name
        )
        if sel is None:
            return []
        _, fusion_letter = sel
        numbering = _build_full_numbering_mono_base(
            smaller_ring=smaller_ring,
            base_ring=base_ring,
            fusion_atoms=smaller_base_edge,
            middle_carbon=middle_carbon,
            extra_carbons=extra_carbons,
            smaller_elements_in_order=elements_in_order,
            mol=mol,
        )
        if not base_fully_aromatic:
            # Stage 3: any sp3 positions → add hydro prefix.
            hydro_prefix = _hydro_prefix_mono_base(
                sp3_like_atoms=base_sp3_atoms,
                base_ring=base_ring,
                numbering=numbering,
                fully_saturated=base_fully_saturated,
            )
            # If the base has any unsaturation but we couldn't synthesise
            # a valid hydro prefix (partly-sat without numbering, 1-hydro
            # case, etc.), bail rather than emit an incorrect name.
            if hydro_prefix == "":
                return []
    else:
        # Multi-ring base path (Stage 2B + Stage 3 fully-sat only).
        sel_multi = _select_multi_base_numbering(
            base_rings, mol, smaller_base_edge,
            force_aromatic=(not base_fully_aromatic),
        )
        if sel_multi is None:
            return []
        base_name, base_numbering_labels, fusion_letter = sel_multi
        # Numbering for the full fused system is harder to compute reliably
        # for the multi-ring base case (it requires renumbering to give the
        # smaller-ring atoms lowest locants).  Stage 2B emits the name
        # without a per-atom Numbering — substituent placement in the
        # smaller ring still works (the parent locants 1, 2, 3 are pinned)
        # but downstream sub-substitution in the multi-ring base portion
        # would need numbering refinement.  This matches OPSIN's behaviour
        # of accepting the base-only form.
        numbering = None
        if not base_fully_aromatic:
            # Stage 3: only emit for fully-saturated multi-ring bases; partial
            # saturation requires per-atom label numbering that the multi-ring
            # path doesn't currently compute.
            if not base_fully_saturated:
                return []
            hydro_prefix = _hydro_prefix_multi_base_fully_saturated(base_atoms)
            if hydro_prefix == "":
                return []

    name_str = f"{hydro_prefix}[1,3]{smaller_prefix}[4,5-{fusion_letter}]{base_name}"
    stem_str = name_str[:-1] if name_str.endswith("e") else name_str

    numbering_options: tuple[Numbering, ...] = (numbering,) if numbering else ()

    # Stage 1/2 fully-aromatic forms keep the catch-all "systematic" method
    # rank so retained names win where present.  Stage 3 hydro-forms use a
    # dedicated "fused_hetero_hydro" rank so they beat the VB fallback on
    # saturated dioxolo-type heterocycles while still losing to retained/HW.
    naming_method = (
        "systematic" if base_fully_aromatic else "fused_hetero_hydro"
    )

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem_str,
        alkyl_stem=None,
        naming_method=naming_method,
        indicated_hydrogen=None,
        numbering_options=numbering_options,
    )]


def _has_endocyclic_unsaturation(ring_atoms: frozenset[int], mol) -> bool:
    """Return True if any ring↔ring bond within ``ring_atoms`` is a double,
    triple, or aromatic bond (i.e. the fused system carries endocyclic
    unsaturation relative to the fully-saturated skeleton).

    Used by the von Baeyer fallback to restrict the tricyclic+ (rank >= 3)
    branch to systems WITH endocyclic unsaturation — the gap this fallback
    targets — so fully-saturated polycyclic fused bases (perhydro steroids,
    methylenedioxy/polyspiro acetonide partners) stay on their preferred
    retained / polyspiro / methylenedioxy-bridge paths.
    """
    seen: set[tuple[int, int]] = set()
    for a in ring_atoms:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            other = bond.GetOtherAtom(mol.GetAtomWithIdx(a)).GetIdx()
            if other not in ring_atoms:
                continue
            key = (min(a, other), max(a, other))
            if key in seen:
                continue
            seen.add(key)
            if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() >= 2.0:
                return True
    return False


def _name_fused_von_baeyer_fallback(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Von Baeyer fallback for ortho-fused systems with no fused-hetero name.

    Perception classifies a two-or-more-ring ortho-fused system as
    ``type="fused"`` even when no retained name and no ``[1,3]``-dihetero
    fused-prefix name applies — e.g. an all-carbon benzo-fused four-ring with
    endocyclic unsaturation (benzocyclobutadiene,
    ``bicyclo[4.2.0]octa-1,3,5,7-tetraene``) or a partially unsaturated
    fused β-lactam (2,3-didehydropenam,
    ``4-thia-1-azabicyclo[3.2.0]hept-2-ene``).  Because the ring is typed
    "fused", the dispatcher never routes it to :func:`name_bridged`, so the
    engine produces NO plan and errors out.

    IUPAC P-23.2.5 von Baeyer nomenclature is the general systematic name for
    ANY polycyclic ring system that lacks a retained/fusion name; it handles
    endocyclic unsaturation (P-23.3 ``-ene``/``-yne`` locants) and skeletal
    heteroatom replacement (P-31.1.3 ``-thia-``/``-aza-`` prefixes) directly.
    This fallback decomposes the fused skeleton into its von Baeyer main
    bridges (:func:`vb_decompose.decompose_ring_system`) and re-routes through
    :func:`bridged.name_bridged`, which already builds the unsaturation suffix
    by Kekulising aromatic ring bonds.

    The fallback is STRUCTURAL, not molecule-specific: it fires for every
    fused system the fused-hetero / retained paths cannot name, and produces
    the same von Baeyer descriptor those systems would receive if perception
    had typed them "bridged".  It is offered as an additional candidate at
    the ``von_baeyer`` method rank, so any retained or fused-hetero name still
    wins in strategy where one exists.
    """
    from iupac_namer.ring_naming.vb_decompose import (
        _circuit_rank,
        decompose_ring_system,
    )
    from iupac_namer.ring_naming.bridged import name_bridged
    from iupac_namer.types import CandidateParent as CP
    from iupac_namer.types import RingSystem as RS

    # Scope the fallback to POLYCYCLIC fused systems (circuit rank >= 2):
    # bicyclic (rank 2) and tricyclic+ (rank >= 3).  These are the cases the
    # gap targets — an ortho-fused two-or-more-ring system with endocyclic
    # unsaturation and no retained / fusion name (benzocyclobutadiene,
    # 2,3-didehydropenam, and their tricyclic+ congeners).
    #
    # Wave-21 originally scoped this fallback to rank == 2 because, at that
    # time, every NON-aromatic tricyclic+ fused system was already routed to
    # ``name_bridged`` by perception's alternate-classification path
    # (perception/rings.py marks a non-aromatic ``len(rings) >= 3`` fused
    # system ``classification_ambiguous`` with ``alternate_type="bridged"``).
    # The fallback was therefore redundant for rank >= 3 and left disabled to
    # avoid shadowing the polyspiro / methylenedioxy-bridge paths.
    #
    # Widening to rank >= 3 makes ``name_fused`` self-sufficient: it now
    # produces the SAME von Baeyer descriptor for a tricyclic+ fused system
    # whether or not perception happened to inject the alternate path (the two
    # routes call the identical ``decompose_ring_system`` + ``name_bridged``
    # machinery, and the result is deduplicated by name in
    # ``ring_naming.__init__``).  The aromatic and charged-ring-atom exclusions
    # below, plus the rank >= 3 endocyclic-unsaturation requirement and the
    # methylenedioxy-bridge exclusion, keep this from displacing the preferred
    # name for the higher-rank saturated polycyclic bases (amcinonide's
    # cyclopenta[a]phenanthrene partner, methylenedioxy/polyspiro steroids).
    rank = _circuit_rank(ring_system.atom_indices, mol)
    if rank < 2:
        return []

    # Do NOT offer von Baeyer for a FULLY mancude (aromatic) ortho-fused
    # system.  Such systems are the domain of fusion nomenclature (P-25) —
    # acridine, thioxanthone, etc. — which the engine builds through the
    # retained / fusion-name paths (handled outside ``name_fused``).  A von
    # Baeyer fallback here would emit a fully *saturated* ``...decane`` /
    # ``...decan`` skeleton (``name_bridged`` only Kekulises the ring when it
    # can recover the localised double bonds, which it does not for these
    # peri/ortho aromatics), silently dropping every ring double bond and
    # outranking the correct fusion name.  The von Baeyer fallback is reserved
    # for partially-unsaturated / non-mancude fused systems
    # (benzocyclobutadiene, 2,3-didehydropenam, ...) where no fusion or
    # retained name applies.
    if ring_system.aromatic:
        return []

    # Do NOT offer von Baeyer when any ring atom carries a formal charge.  A
    # "fused" system with a charged ring atom is almost always a charged arene
    # — e.g. naphthalen-1-ylium ([C+]1=CC=Cc2ccccc21), where the cation centre
    # defeats RDKit's aromaticity perception on one ring so ``ring_system.
    # aromatic`` is False, but the system is still a mancude naphthalene cation
    # whose name is the fusion form ``naphthalen-1-ylium``.  A von Baeyer
    # fallback emits a saturated/neutral ``bicyclo[4.4.0]deca...`` skeleton
    # that round-trips to a DIFFERENT structure (charge + Kekulé pattern lost)
    # — worse than the prior NAMING_ERROR.  True charged von Baeyer cages
    # (e.g. a quaternary-N bridgehead salt) are perceived as ``type="bridged"``
    # and reach ``name_bridged`` directly, not through this fused fallback.
    if any(
        mol.GetAtomWithIdx(a).GetFormalCharge() != 0
        for a in ring_system.atom_indices
    ):
        return []

    # ----- rank >= 3 (tricyclic+) safeguards -------------------------------
    #
    # The bicyclic (rank == 2) path is unchanged; the additional guards below
    # apply only to tricyclic+ systems newly admitted by the widened gate.
    if rank >= 3:
        # (a) Require endocyclic unsaturation.  The gap this widening closes is
        # specifically tricyclic+ ortho-fused systems WITH endocyclic C=C/C#C
        # that lack a retained/fusion name.  A FULLY SATURATED tricyclic+ fused
        # base (perhydro steroid skeletons, methylenedioxy/polyspiro acetonide
        # partners after the bridge is carved, decahydronaphthalene-class cages)
        # is the domain of the retained / polyspiro / methylenedioxy-bridge
        # paths — emitting a von Baeyer ``...icosane`` alternative there is both
        # unnecessary (those paths rank higher) and a regression hazard.  A von
        # Baeyer name only needs to *fall out* of the fused fallback for the
        # unsaturated case the alternate-classification path also targets, so
        # gating on endocyclic unsaturation keeps the saturated bases on their
        # preferred paths.
        if not _has_endocyclic_unsaturation(ring_system.atom_indices, mol):
            return []

        # (b) Do NOT offer von Baeyer when the system carries an O-CH2-O
        # methylenedioxy bridge ring.  Such a system (e.g. amcinonide's
        # polycyclic partner: the steroid scaffold with a 16,17-methylenedioxy
        # ring and a ring-A enone) HAS endocyclic unsaturation but is the
        # canonical domain of the methylenedioxy-bridge naming path
        # (``16,17-methylenedioxy-...-cyclopenta[a]phenanthrene``, per the
        # Stage-4 OPSIN-probe finding).  The methylenedioxy-bridge path runs as
        # a separate dispatch in ``ring_naming.__init__`` and ranks above von
        # Baeyer, so a VB alternative here would not win — but per the Stage-4
        # guard contract we keep it from being emitted at all, so the von
        # Baeyer fallback never shadows that path's intermediate machinery.
        try:
            from iupac_namer.ring_naming.methylenedioxy_bridge import (
                _find_methylenedioxy_ring,
            )
            if _find_methylenedioxy_ring(ring_system, mol) is not None:
                return []
        except Exception as e:  # defensive — never crash the naming pass
            logger.debug("VB fallback methylenedioxy pre-check failed: %s", e)

    try:
        decomps = decompose_ring_system(ring_system.atom_indices, mol)
    except Exception as e:  # defensive — never crash the naming pass
        logger.debug("VB fallback decomposition failed: %s", e)
        return []
    if not decomps:
        return []

    decomp = decomps[0]
    bridge_sizes = decomp.main_bridge_sizes
    secondary_bridges = decomp.secondary_bridges or None

    vb_rs = RS(
        atom_indices=ring_system.atom_indices,
        rings=ring_system.rings,
        type="bridged",
        aromatic=ring_system.aromatic,
        bridge_sizes=bridge_sizes,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=ring_system.heteroatoms,
        ring_size=ring_system.ring_size,
        secondary_bridges=secondary_bridges,
    )
    vb_candidate = CP(
        atom_indices=candidate.atom_indices,
        type="bridged",
        length=candidate.length,
        ring_system=vb_rs,
        unsaturation=candidate.unsaturation,
        element=candidate.element,
        lambda_value=candidate.lambda_value,
    )
    try:
        return name_bridged(vb_rs, vb_candidate, mol)
    except Exception as e:  # defensive
        logger.debug("VB fallback name_bridged failed: %s", e)
        return []


def name_fused(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Public fused-ring naming entry point.

    Tries the fused-hetero / fusion-prefix path first
    (:func:`_name_fused_hetero`).  When that path produces no name — the
    system has no retained name and no ``[1,3]``-dihetero fusion descriptor
    applies — fall back to von Baeyer nomenclature
    (:func:`_name_fused_von_baeyer_fallback`) so that ortho-fused carbocyclic
    and heterocyclic systems with endocyclic unsaturation still receive a
    systematic name (P-23.2.5) instead of erroring with no plan.
    """
    results = _name_fused_hetero(ring_system, candidate, mol)
    if results:
        return results
    return _name_fused_von_baeyer_fallback(ring_system, candidate, mol)
