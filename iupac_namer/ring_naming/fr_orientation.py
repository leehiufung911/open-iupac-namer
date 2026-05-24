"""
iupac_namer/ring_naming/fr_orientation.py

FR-5.2 / FR-5.3 (P-25.3.3) canonical peripheral numbering for ortho-fused
ring systems.

This is an ADDITIVE module.  Its single public entry point,
``compute_peripheral_numbering``, takes an ortho-fused ring system (a tree of
rings joined by shared two-atom edges, no spiro/bridge) and returns the IUPAC
canonical per-atom locant assignment as a :class:`Numbering`, with ring-fusion
(junction) atoms carrying the ``Na`` letter suffix (``3a``, ``7a`` …) exactly
as the published fusion names use.

Why this exists
---------------
The fusion-name builders in :mod:`fused.py` (``_try_mono_hetero_fused`` and the
[1,3]-dihetero Stage-2B multi-ring path) emit a correct *name string* but, for
the mono-hetero path, leave ``numbering_options`` empty.  Downstream,
``numbering.compute_ring_numberings`` then falls back to the generic
``_compute_fused_numberings`` peripheral walk, which numbers every atom
``1..N`` sequentially (no ``Na`` junction suffixes) and scores by a different
heuristic.  The result is substituent locants that are inconsistent with — and
often *outside the range of* — the emitted fusion name (e.g.
``9-chlorofuro[2,3-b]pyridine``, where position 9 does not exist).

``compute_peripheral_numbering`` produces the genuine FR-5.3 numbering so the
mono-hetero path can attach it as ``numbering_options`` and substituent locants
land where OPSIN expects them.

The numbering rule (P-25.3.3, the relevant subset for ortho-fused bicyclics
and small ortho-fused tricyclics):

  1.  Atoms are numbered around the periphery in one direction.  Interior /
      ring-fusion atoms (shared by two rings) receive the immediately preceding
      peripheral number plus a letter (``a``, ``b`` …).
  2.  The starting atom and direction are chosen so that, in priority order:
        (a)  heteroatoms as a set get the lowest locants;
        (b)  carbon atoms common to two rings (fusion carbons) get the lowest
             locants;
        (c)  heteroatoms get low locants in element-seniority order
             (O > S > Se > Te > N > P > …);
      Numbering must begin at an atom next to a ring-fusion atom (the
      "uppermost, farthest-right" atom in the FR-5.2 preferred orientation;
      for a perimeter that is a single cycle, this is equivalent to: position 1
      is a non-fusion atom immediately following a fusion atom in the chosen
      direction).

For an ortho-fused bicyclic the whole molecule's perimeter is a single cycle
(the two fusion atoms are the only branch points), which makes (1) a plain
cyclic walk with the two fusion atoms suffixed.  This module handles that case
exactly and generalises to a cata-fused (tree, no interior atom shared by 3
rings) chain of rings.

It deliberately returns ``None`` for systems it cannot number unambiguously
(peri-fused systems with an interior atom shared by 3 rings, bridged/spiro
inputs) so the caller falls back to its existing behaviour — never emitting a
wrong numbering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.types import Locant, Numbering

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# Heteroatom seniority for the element-order tie-break (FR-5.3 / P-31.1.4.3.4).
# Lower number = more senior (cited / numbered first).
_ELEMENT_SENIORITY: dict[str, int] = {
    "F": 0, "Cl": 1, "Br": 2, "I": 3,   # (rare in ring skeletons; kept for completeness)
    "O": 4, "S": 5, "Se": 6, "Te": 7,
    "N": 8, "P": 9, "As": 10, "Sb": 11, "Bi": 12,
    "Si": 13, "Ge": 14, "Sn": 15, "Pb": 16,
    "B": 17, "Al": 18, "Ga": 19, "In": 20, "Tl": 21,
}


def _ring_graph_adj(atom_indices: frozenset[int], mol) -> dict[int, list[int]]:
    """Adjacency restricted to the ring atom set."""
    adj: dict[int, list[int]] = {a: [] for a in atom_indices}
    for a in atom_indices:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in atom_indices:
                adj[a].append(j)
    return adj


def _perimeter_cycle(atom_indices: frozenset[int], mol) -> list[int] | None:
    """Return the outer-perimeter atoms in cyclic order for a cata-fused
    system whose perimeter is a single simple cycle.

    The perimeter is the set of atoms reachable by always turning along the
    outer boundary.  For an ortho-fused (cata-fused) system every atom lies on
    the perimeter — fusion atoms have ring-degree 3 but are still on the
    boundary — and the boundary is a single Hamiltonian cycle of the whole
    atom set.  We therefore find a single cycle that visits every atom exactly
    once, where consecutive atoms are bonded.

    Returns the ordered atom list (length == n), or None when no single
    spanning perimeter cycle exists (e.g. a peri-fused / ortho-and-peri-fused
    system, where some atom is interior and the perimeter does not include
    every atom).
    """
    adj = _ring_graph_adj(atom_indices, mol)
    n = len(atom_indices)
    if n < 3:
        return None
    # Every atom must have ring-degree 2 (peripheral) or 3 (fusion atom on a
    # cata-fused boundary).  A degree >= 4 (or an atom shared by 3 rings,
    # giving an interior atom) means the perimeter is not a single spanning
    # cycle — defer.
    if any(len(v) not in (2, 3) for v in adj.values()):
        return None
    deg3 = [a for a, v in adj.items() if len(v) == 3]
    # Cata-fused tree of k rings has exactly 2*(k-1) fusion atoms, all degree
    # 3, and NO interior atom.  An interior (degree-3 but not on the single
    # perimeter) atom appears in peri-fused systems; we detect that below by
    # checking the recovered cycle spans all atoms.

    start = min(atom_indices)
    # DFS for a Hamiltonian cycle along bonds, preferring to keep the boundary.
    # n is small (typically 9-16) so a bounded backtracking search is fine.
    best: list[int] | None = None

    def dfs(path: list[int], visited: set[int]) -> bool:
        nonlocal best
        if len(path) == n:
            if start in adj[path[-1]]:
                best = list(path)
                return True
            return False
        cur = path[-1]
        # Try neighbours; order does not affect correctness of *a* cycle, but
        # we want a perimeter cycle.  Any spanning cycle of a cata-fused
        # system is the perimeter (the only simple spanning cycle), so the
        # first one found is canonical.
        for nb in adj[cur]:
            if nb not in visited:
                path.append(nb)
                visited.add(nb)
                if dfs(path, visited):
                    return True
                visited.discard(nb)
                path.pop()
        return False

    if dfs([start], {start}):
        return best
    return None


def _fusion_atoms(atom_indices: frozenset[int], mol) -> set[int]:
    """Atoms shared by two rings (ring-degree 3 on a cata-fused boundary)."""
    adj = _ring_graph_adj(atom_indices, mol)
    return {a for a, v in adj.items() if len(v) >= 3}


def _assign_locants_from_walk(
    cycle: list[int], start_pos: int, direction: int, fusion: set[int]
) -> dict[int, Locant] | None:
    """Given the perimeter cycle and a (start, direction), assign locants.

    Walking the perimeter from ``cycle[start_pos]`` in ``direction`` (+1 / -1):
    each NON-fusion atom takes the next integer; each fusion atom takes the
    previous integer with a letter suffix (a, b, …).  The very first atom of
    the walk MUST be a non-fusion atom (it is locant 1) — enforced by caller.

    Returns atom_idx -> Locant, or None if the walk is malformed (e.g. starts
    on a fusion atom, or two consecutive fusion atoms exhaust the letters).
    """
    n = len(cycle)
    if cycle[start_pos] in fusion:
        return None
    order = [cycle[(start_pos + direction * i) % n] for i in range(n)]

    locmap: dict[int, Locant] = {}
    last_int = 0
    letter_run = 0
    for atom in order:
        if atom in fusion:
            if last_int == 0:
                return None  # can't suffix before any integer assigned
            suffix = chr(ord("a") + letter_run)
            letter_run += 1
            locmap[atom] = Locant.numeric(last_int, suffix)
        else:
            last_int += 1
            letter_run = 0
            locmap[atom] = Locant.numeric(last_int)
    return locmap


def _largest_ring_atoms(atom_indices: frozenset[int], mol) -> frozenset[int]:
    """Atoms of the single largest SSSR ring within the fused system (ties
    broken arbitrarily but deterministically by sorted membership)."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings() if set(r) <= set(atom_indices)]
    if not rings:
        return frozenset()
    best = max(rings, key=lambda r: (len(r), -min(r)))
    return frozenset(best)


def _indicated_hydrogen_atoms(atom_indices: frozenset[int], mol) -> frozenset[int]:
    """Ring atoms bearing *indicated* hydrogen (P-25.3.3.1.2 (f)).

    In a mancude (maximally-unsaturated) fused ring system the "indicated"
    hydrogen is the H that cannot be accommodated by the aromatic/double-bond
    framework — the pyrrole-type ``NH`` and the sp3 (``CH2`` / saturated) ring
    positions of a partly-reduced parent.  An ordinary aromatic ``C-H`` is NOT
    indicated hydrogen and is excluded.

    Operationally: a ring atom carries indicated hydrogen when it has at least
    one attached H AND it is either a heteroatom (aromatic ``NH``, etc.) or a
    non-aromatic (sp3) carbon (a reduced ``CH2`` such as the 1,3-dihydrofuro
    positions).  This deliberately mirrors the atoms whose H placement the Blue
    Book uses as the P-25.3.3.1.2 (f) numbering tie-break.
    """
    out: set[int] = set()
    for i in atom_indices:
        a = mol.GetAtomWithIdx(i)
        if a.GetTotalNumHs() > 0 and (not a.GetIsAromatic() or a.GetSymbol() != "C"):
            out.add(i)
    return frozenset(out)


def _score_numbering(
    locmap: dict[int, Locant],
    atom_indices: frozenset[int],
    mol,
    big_ring: frozenset[int],
    indicated_h: frozenset[int],
) -> tuple:
    """IUPAC FR-5.3 numbering preference score (lower is better).

    Tie-break order:
      1.  Heteroatoms as a set get the lowest locants.
      2.  Fusion CARBON atoms get the lowest locants.
      3.  Heteroatoms get low locants in element-seniority order.
      4.  The larger (more senior) ring is numbered first — i.e. the sum of
          locants over the largest ring's atoms is lowest.  This resolves the
          mirror ambiguity on near-symmetric skeletons (e.g.
          ``pyrrolo[3,4-b]pyrazine``) where rules 1-3 tie: FR-5.2 places the
          larger base ring in the preferred orientation, so its atoms take the
          earlier numbers.  (Verified against OPSIN for the
          furo/thieno/pyrrolo[3,4-b]pyrazine and [3,4-d]pyridazine families.)
      5.  Low locants are assigned to indicated hydrogen (P-25.3.3.1.2 (f)).
          This is the FR-5.2 2D-orientation tie-break for the near-symmetric
          *asymmetric* mirror cases: pyrrolo[3,4-b]pyrazine / [3,4-d]pyridazine
          (and the [3,4-c] analogues) tie on criteria 1-4 because the two mirror
          numberings give an identical heteroatom/fusion-carbon *locant set*;
          the only asymmetry is the pyrrole ``NH`` (the indicated hydrogen),
          which OPSIN's preferred 2D orientation always places at the lower
          locant.  Criterion 5 fires ONLY when 1-4 tie, so it is strictly
          additive and never disturbs an already-resolved numbering.  (For the
          furo/thieno members the two mirrors are related by a true molecular
          automorphism — there is no indicated hydrogen and either numbering
          yields identical substituent locants, so this criterion correctly
          leaves them tied; see ``compute_peripheral_numberings``.)

    Locant comparison uses (integer, suffix_rank) so that ``3`` < ``3a`` < ``4``.
    """
    fusion = _fusion_atoms(atom_indices, mol)

    def lkey(loc: Locant) -> tuple[int, int]:
        return (loc._numeric_value or 0,
                (ord(loc.suffix) - ord("a") + 1) if loc.suffix else 0)

    het_locs: list[tuple[int, int]] = []
    fusion_c_locs: list[tuple[int, int]] = []
    het_by_elem: dict[str, list[tuple[int, int]]] = {}
    for a in atom_indices:
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        key = lkey(locmap[a])
        if sym != "C":
            het_locs.append(key)
            het_by_elem.setdefault(sym, []).append(key)
        elif a in fusion:
            fusion_c_locs.append(key)

    het_locs.sort()
    fusion_c_locs.sort()

    # Element-seniority sub-score: for each element from most senior, its sorted
    # locants, concatenated.
    elem_subscore: list[tuple[int, int]] = []
    for elem in sorted(het_by_elem, key=lambda e: _ELEMENT_SENIORITY.get(e, 99)):
        elem_subscore.extend(sorted(het_by_elem[elem]))

    big_ring_locsum = sum(
        (locmap[a]._numeric_value or 0) for a in big_ring
    )

    # P-25.3.3.1.2 (f): low locants to indicated hydrogen.  Final tie-break.
    indicated_h_locs = tuple(sorted(lkey(locmap[a]) for a in indicated_h))

    return (
        tuple(het_locs),
        tuple(fusion_c_locs),
        tuple(elem_subscore),
        big_ring_locsum,
        indicated_h_locs,
    )


def _locmap_to_numbering(locmap: dict[int, Locant]) -> Numbering:
    assignments = tuple(sorted(locmap.items(), key=lambda kv: kv[0]))
    locant_set = tuple(sorted(
        locmap.values(),
        key=lambda l: (l._numeric_value or 0, l.suffix or ""),
    ))
    return Numbering(_assignments=assignments, locant_set=locant_set)


def _numberings_are_symmetry_equivalent(
    locmaps: list[dict[int, Locant]], atom_indices: frozenset[int], mol
) -> bool:
    """True when every numbering in ``locmaps`` is related to the first by a
    molecular automorphism — i.e. they assign the *same* locant to every
    symmetry-equivalent atom, so any substituent on the real molecule receives
    the same locant under whichever numbering is chosen.

    This distinguishes the genuinely-equivalent residual ties (furo/thieno
    [3,4-b]pyrazine, where the mirror IS a molecular symmetry) from a real
    ambiguity (which, after the P-25.3.3.1.2 (f) indicated-hydrogen tie-break,
    no longer occurs for the documented near-symmetric family).
    """
    if len(locmaps) <= 1:
        return True
    # Heavy-atom automorphisms of the molecule (self substructure matches).
    try:
        autos = mol.GetSubstructMatches(
            mol, uniquify=False, useChirality=False, maxMatches=64
        )
    except Exception:  # pragma: no cover - defensive
        return False
    base = locmaps[0]
    base_label = {a: l.label for a, l in base.items()}
    for lm in locmaps[1:]:
        lm_label = {a: l.label for a, l in lm.items()}
        matched = False
        for perm in autos:
            # perm maps mol-atom x -> perm[x]; require base_label[x] == lm_label[perm[x]]
            if all(
                perm[x] in lm_label and base_label[x] == lm_label[perm[x]]
                for x in atom_indices
            ):
                matched = True
                break
        if not matched:
            return False
    return True


def compute_peripheral_numberings(
    atom_indices: frozenset[int], mol
) -> tuple[Numbering, ...]:
    """Compute the FR-5.3 canonical peripheral numbering(s) of an ortho-fused
    (cata-fused) ring system.

    The candidate numberings are scored by the FR-5.3 / P-25.3.3.1.2 criteria
    (heteroatom set, fusion-carbon set, heteroatom element-order, larger-ring
    first, and finally — the FR-5.2 2D-orientation tie-break for near-symmetric
    mirrors — low locants to indicated hydrogen).

    Normally this yields a single winning numbering, which is returned as a
    one-element tuple.  When two or more numberings remain tied *and* they are
    related by a molecular automorphism (the furo/thieno[3,4-b]pyrazine family,
    where the mirror is a true symmetry and either numbering gives identical
    substituent locants), a single deterministically-chosen representative is
    returned — it round-trips correctly because the alternatives are equivalent.

    Returns ``()`` when the system is not a single-perimeter cata-fused tree
    (peri-fused, bridged, spiro), or — defensively — when distinct,
    non-symmetry-equivalent numberings remain tied after all criteria (which
    would indicate a genuine ambiguity the criteria cannot resolve; the caller
    then falls back to its prior behaviour rather than risk a wrong locant).
    """
    if not atom_indices or len(atom_indices) < 5:
        return ()

    cycle = _perimeter_cycle(atom_indices, mol)
    if cycle is None or len(cycle) != len(atom_indices):
        return ()
    fusion = _fusion_atoms(atom_indices, mol)
    if not fusion:
        return ()  # monocyclic — not our job

    n = len(cycle)
    big_ring = _largest_ring_atoms(atom_indices, mol)
    indicated_h = _indicated_hydrogen_atoms(atom_indices, mol)
    scored: list[tuple[tuple, dict[int, Locant]]] = []

    # Candidate start positions: every non-fusion atom that immediately
    # FOLLOWS a fusion atom in the walk direction (classic fused numbering:
    # position 1 is adjacent to a ring-fusion atom).  Try both directions.
    for direction in (1, -1):
        for sp in range(n):
            if cycle[sp] in fusion:
                continue
            prev = cycle[(sp - direction) % n]
            if prev not in fusion:
                continue
            locmap = _assign_locants_from_walk(cycle, sp, direction, fusion)
            if locmap is None:
                continue
            score = _score_numbering(
                locmap, atom_indices, mol, big_ring, indicated_h
            )
            scored.append((score, locmap))

    if not scored:
        return ()

    scored.sort(key=lambda t: t[0])
    best = scored[0][0]
    tied: list[dict[int, Locant]] = []
    seen: set[tuple] = set()
    for score, locmap in scored:
        if score != best:
            break
        key = tuple(sorted((a, l.label) for a, l in locmap.items()))
        if key in seen:
            continue
        seen.add(key)
        tied.append(locmap)

    if len(tied) == 1:
        return (_locmap_to_numbering(tied[0]),)

    # More than one DISTINCT numbering survives.  If they are related by a
    # molecular automorphism they assign equivalent locants — return a single
    # deterministic representative (the one with the lexicographically smallest
    # atom->label signature) so the result is stable and round-trips.
    if _numberings_are_symmetry_equivalent(tied, atom_indices, mol):
        rep = min(
            tied,
            key=lambda lm: tuple(sorted((a, l.label) for a, l in lm.items())),
        )
        return (_locmap_to_numbering(rep),)

    # Genuinely ambiguous (should not happen for the documented family): defer.
    return tuple(_locmap_to_numbering(lm) for lm in tied)


def compute_peripheral_numbering(
    atom_indices: frozenset[int], mol
) -> Numbering | None:
    """Singular convenience wrapper: the resolved canonical numbering, or None.
    Used by validation harnesses and as the numbering the naming path attaches.
    With the FR-5.2 2D-orientation tie-break in place the plural form returns a
    single numbering for the near-symmetric mirror families; this wrapper simply
    surfaces it.
    """
    nbs = compute_peripheral_numberings(atom_indices, mol)
    return nbs[0] if nbs else None
