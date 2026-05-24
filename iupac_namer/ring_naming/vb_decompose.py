"""
iupac_namer/ring_naming/vb_decompose.py

Von Baeyer polycycle decomposition (IUPAC P-23.2.5, extensions for tricyclic+).

Decomposes a bridged ring system into:

  1. Two main bridgeheads (bh1, bh2).
  2. Three (or two) "main bridges" — disjoint simple paths between bh1/bh2.
     These contribute the first three numbers of the descriptor:
       bicyclo[a.b.c]...                         (2 bridges + trivial 0 third)
       bicyclo[a.b]...                           (degenerate fused-edge case)
       tricyclo[a.b.c.d^{p,q}]...                (main + 1 secondary)
       tetracyclo[a.b.c.d^{p,q}.e^{r,s}]...      (main + 2 secondaries)
       ...
  3. Zero or more "secondary bridges" — each is a simple path between two
     atoms already on the skeleton so far, going through zero or more
     uncovered atoms.  Listed in descending bridge-size order.

The decomposition is chosen by IUPAC selection rules (greatest coverage,
then main ring size, then main bridge size, then descending secondary
bridges, then lowest superscript locants).  This module returns the atomic
topology; locant assignment and name emission live in :mod:`bridged`.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VBDecomposition:
    """Structural decomposition of a bridged ring system for VB naming.

    Attributes
    ----------
    bh1, bh2
        Atom indices of the two principal bridgeheads.
    main_bridges
        Tuple of simple paths (each a tuple of atom indices from bh1 to bh2).
        Length 2 (degenerate fused bicyclic) or 3 (normal).  Sorted by length
        descending — longest first.
    secondary_bridges
        Tuple of secondary bridges, each ``(endpoints, interior_atoms)`` where
        ``endpoints = (a, b)`` are atom indices already on the skeleton and
        ``interior_atoms`` is the tuple of atoms *between* a and b on this
        bridge (possibly empty for a direct bond).  Listed largest first.
    """
    bh1: int
    bh2: int
    main_bridges: tuple[tuple[int, ...], ...]
    secondary_bridges: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]

    @property
    def main_bridge_sizes(self) -> tuple[int, ...]:
        """Sizes (= len(path) - 2) of main bridges, largest first."""
        return tuple(len(p) - 2 for p in self.main_bridges)

    @property
    def secondary_bridge_sizes(self) -> tuple[int, ...]:
        return tuple(len(interior) for _, interior in self.secondary_bridges)


# ---------------------------------------------------------------------------
# Graph / path helpers
# ---------------------------------------------------------------------------


def _ring_neighbors(atom_idx: int, ring_atom_set: frozenset[int], mol) -> list[int]:
    """Neighbours of ``atom_idx`` that lie within the ring system."""
    atom = mol.GetAtomWithIdx(atom_idx)
    return [
        nb.GetIdx()
        for nb in atom.GetNeighbors()
        if nb.GetIdx() in ring_atom_set
    ]


def _all_simple_paths(
    start: int,
    end: int,
    allowed: frozenset[int],
    forbidden_interior: frozenset[int],
    mol,
    max_count: int = 1000,
) -> list[list[int]]:
    """DFS enumerate simple paths from ``start`` to ``end`` within ``allowed``.

    ``forbidden_interior`` atoms may not appear as interior nodes of a path
    (but may be endpoints).  Stops after ``max_count`` paths to avoid blow-up
    on very dense graphs.
    """
    results: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(start, [start])]
    while stack:
        curr, path = stack.pop()
        if curr == end:
            results.append(path)
            if len(results) >= max_count:
                break
            continue
        for nb_idx in _ring_neighbors(curr, allowed, mol):
            if nb_idx in path:
                continue
            if nb_idx != end and nb_idx in forbidden_interior:
                continue
            stack.append((nb_idx, path + [nb_idx]))
    return results


def _circuit_rank(ring_atom_set: frozenset[int], mol) -> int:
    """Return the cyclomatic number (# independent rings) of the ring subgraph."""
    n_atoms = len(ring_atom_set)
    n_edges = 0
    seen: set[tuple[int, int]] = set()
    for a in ring_atom_set:
        for nb in _ring_neighbors(a, ring_atom_set, mol):
            bkey = (min(a, nb), max(a, nb))
            if bkey in seen:
                continue
            seen.add(bkey)
            n_edges += 1
    # Connected ring subgraph → rank = E - V + 1
    return n_edges - n_atoms + 1


def _bridgehead_candidates(ring_atom_set: frozenset[int], mol) -> list[int]:
    """Atoms in the ring system with 3+ ring-neighbours — bridgehead candidates."""
    return sorted(
        a for a in ring_atom_set
        if len(_ring_neighbors(a, ring_atom_set, mol)) >= 3
    )


# ---------------------------------------------------------------------------
# Enumeration of disjoint-path triples between bh1, bh2
# ---------------------------------------------------------------------------


def _enumerate_disjoint_triples(
    bh1: int,
    bh2: int,
    ring_atom_set: frozenset[int],
    mol,
    k_target: int,
    max_path_count: int = 500,
) -> list[tuple[list[int], ...]]:
    """Enumerate all sets of ``k_target`` pairwise-disjoint simple paths bh1→bh2.

    Paths are disjoint in their *interior* atoms only (bh1 and bh2 are
    shared).

    Also enumerates smaller sets when k_target paths don't exist: returns
    best-effort tuples of 2 disjoint paths (when only 2 exist).
    """
    all_paths = _all_simple_paths(
        bh1, bh2, ring_atom_set, frozenset(), mol, max_count=max_path_count
    )
    if not all_paths:
        return []

    # Convert to tuples; sort by length descending so larger paths come first
    all_paths.sort(key=len, reverse=True)

    results: list[tuple[list[int], ...]] = []

    # Backtracking: pick k_target disjoint paths
    def _backtrack(chosen: list[list[int]], used: set[int], start_idx: int) -> None:
        if len(chosen) == k_target:
            results.append(tuple(chosen))
            return
        for i in range(start_idx, len(all_paths)):
            p = all_paths[i]
            interior = set(p[1:-1])
            if interior & used:
                continue
            chosen.append(p)
            _backtrack(chosen, used | interior, i + 1)
            chosen.pop()
            # Limit result set to avoid blow-up
            if len(results) >= 200:
                return

    _backtrack([], set(), 0)

    return results


# ---------------------------------------------------------------------------
# Secondary bridge enumeration (given a partially-covered skeleton)
# ---------------------------------------------------------------------------


def _enumerate_secondary_bridges(
    covered_atoms: frozenset[int],
    remaining_atoms: frozenset[int],
    all_atoms: frozenset[int],
    mol,
) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    """Enumerate all possible single secondary bridges.

    A secondary bridge is a simple path  a -- x1 -- x2 -- ... -- xk -- b
    where a, b are in ``covered_atoms`` (distinct) and x_i are in
    ``remaining_atoms``.  k = 0 is allowed — a direct bond between two
    covered atoms.

    Returns a list of ``((a, b), (x1, ..., xk))`` tuples, with a <= b for
    the (a, b) tuple to deduplicate.
    """
    results: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()

    # Case 1: k == 0 (direct bond between two covered atoms, not on main paths)
    covered_list = sorted(covered_atoms)
    for i in range(len(covered_list)):
        for j in range(i + 1, len(covered_list)):
            a, b = covered_list[i], covered_list[j]
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None:
                continue
            # This edge must not already be accounted for by a main-bridge
            # edge — but we don't have that info here; caller will filter.
            key = ((a, b), ())
            if key not in seen:
                seen.add(key)
                results.append(key)

    # Case 2: k >= 1 (path through remaining atoms)
    if remaining_atoms:
        allowed = covered_atoms | remaining_atoms
        for i in range(len(covered_list)):
            for j in range(i + 1, len(covered_list)):
                a, b = covered_list[i], covered_list[j]
                # Find simple paths a->b with interior entirely in remaining
                paths = _all_simple_paths(
                    a, b, allowed,
                    # forbid other covered atoms as interior
                    forbidden_interior=covered_atoms - {a, b},
                    mol=mol,
                    max_count=200,
                )
                for p in paths:
                    if len(p) < 3:
                        continue  # len 2 = direct bond (handled above)
                    interior = tuple(p[1:-1])
                    # Must use only remaining atoms in interior
                    if not all(x in remaining_atoms for x in interior):
                        continue
                    key = ((a, b), interior)
                    if key not in seen:
                        seen.add(key)
                        results.append(key)

    return results


# ---------------------------------------------------------------------------
# Main decomposition
# ---------------------------------------------------------------------------


def decompose_ring_system(
    ring_atom_set: frozenset[int],
    mol,
    max_decompositions: int = 200,
) -> list[VBDecomposition]:
    """Enumerate candidate VB decompositions of the ring system.

    Returns a list sorted from MOST to LEAST preferred under IUPAC P-23.2.5:

    1. More atoms covered (must be all, but the scoring prefers completeness).
    2. Main ring (= largest_main_bridge + second_largest_main_bridge) as
       large as possible.
    3. Main bridge (= third largest main bridge) as large as possible.
    4. Secondary bridges sorted descending; compare lexicographically,
       larger better.
    5. (Tie-breakers on superscript locants handled in bridged.py during
       numbering selection.)

    If no valid complete decomposition exists, returns ``[]``.
    """
    rank = _circuit_rank(ring_atom_set, mol)
    if rank < 2:
        return []

    bh_candidates = _bridgehead_candidates(ring_atom_set, mol)
    if len(bh_candidates) < 2:
        return []

    n_main_paths_needed = 3 if rank >= 2 else 2

    # For each bh pair, enumerate disjoint path triples.  Then for each
    # triple, add secondary bridges to account for remaining rank - 2 rings.
    candidates: list[tuple[tuple, VBDecomposition]] = []

    for bh1, bh2 in itertools.combinations(bh_candidates, 2):
        # Try k=3 first, fall back to k=2 if no 3-disjoint exists
        for k_target in (3, 2):
            if k_target > rank + 1:
                continue
            triples = _enumerate_disjoint_triples(
                bh1, bh2, ring_atom_set, mol, k_target
            )
            if not triples:
                continue
            for path_set in triples:
                # Sort paths by length descending
                main_paths = tuple(
                    tuple(p) for p in sorted(path_set, key=len, reverse=True)
                )
                covered: set[int] = set()
                for p in main_paths:
                    covered.update(p)
                # Edges used by main bridges
                main_edges: set[tuple[int, int]] = set()
                for p in main_paths:
                    for i in range(len(p) - 1):
                        e = (min(p[i], p[i+1]), max(p[i], p[i+1]))
                        main_edges.add(e)

                remaining_atoms = frozenset(ring_atom_set) - covered
                n_secondaries_needed = rank - (len(main_paths) - 1)
                # If all atoms covered and rank == len(main)-1, no secondaries
                if n_secondaries_needed <= 0 and not remaining_atoms:
                    decomp = VBDecomposition(
                        bh1=bh1, bh2=bh2,
                        main_bridges=main_paths,
                        secondary_bridges=(),
                    )
                    score = _score_decomposition(decomp, ring_atom_set)
                    candidates.append((score, decomp))
                    if len(candidates) > max_decompositions * 3:
                        break
                    continue
                if n_secondaries_needed <= 0:
                    continue  # atoms remain but no secondaries expected

                # Enumerate secondary bridges
                sec_choices = _enumerate_secondary_bridges(
                    frozenset(covered), remaining_atoms, ring_atom_set, mol
                )
                # Filter: for size-0 secondary (direct bond), the bond must
                # not already be a main-bridge edge.
                filtered: list[tuple[tuple[int, int], tuple[int, ...]]] = []
                for (a, b), interior in sec_choices:
                    if not interior:
                        e = (min(a, b), max(a, b))
                        if e in main_edges:
                            continue
                    filtered.append(((a, b), interior))
                if not filtered:
                    continue

                # Try all combinations of n_secondaries_needed secondaries
                # that collectively (a) cover all remaining atoms and
                # (b) close rank - 2 independent cycles.
                for sec_combo in itertools.combinations(filtered, n_secondaries_needed):
                    # Validate atom coverage
                    sec_interior_atoms: set[int] = set()
                    sec_edges: set[tuple[int, int]] = set()
                    ok = True
                    for (a, b), interior in sec_combo:
                        if set(interior) & sec_interior_atoms:
                            ok = False; break
                        sec_interior_atoms.update(interior)
                        # Track edges
                        path_full = (a,) + tuple(interior) + (b,)
                        for i in range(len(path_full) - 1):
                            e = (min(path_full[i], path_full[i+1]),
                                 max(path_full[i], path_full[i+1]))
                            if e in sec_edges or e in main_edges:
                                # Another bridge already uses this edge
                                ok = False
                                break
                            sec_edges.add(e)
                        if not ok:
                            break
                    if not ok:
                        continue
                    # All remaining atoms must be covered
                    if sec_interior_atoms != set(remaining_atoms):
                        continue

                    # Sort secondaries by size (= len(interior)) descending;
                    # tiebreak irrelevant here — locant selection picks.
                    sec_sorted = tuple(sorted(
                        sec_combo,
                        key=lambda sb: (-len(sb[1]),),
                    ))
                    decomp = VBDecomposition(
                        bh1=bh1, bh2=bh2,
                        main_bridges=main_paths,
                        secondary_bridges=sec_sorted,
                    )
                    score = _score_decomposition(decomp, ring_atom_set)
                    candidates.append((score, decomp))
                    if len(candidates) > max_decompositions * 3:
                        break
                if len(candidates) > max_decompositions * 3:
                    break
            if candidates:
                # If we got complete coverage at k=3, don't fall back to k=2
                break

    if not candidates:
        return []

    # Sort by score descending (higher = better)
    candidates.sort(key=lambda sd: sd[0], reverse=True)
    return [d for _, d in candidates[:max_decompositions]]


def _score_decomposition(
    decomp: VBDecomposition, ring_atom_set: frozenset[int]
) -> tuple:
    """IUPAC selection score (higher tuple = more preferred)."""
    main_sizes = decomp.main_bridge_sizes  # sorted desc
    sec_sizes = decomp.secondary_bridge_sizes  # sorted desc

    # 1. Coverage (guaranteed complete by filter but included for safety)
    atoms_covered = len(set(
        a for p in decomp.main_bridges for a in p
    ) | set(
        a
        for (ea, eb), interior in decomp.secondary_bridges
        for a in (ea, eb, *interior)
    ))
    coverage = 1 if atoms_covered == len(ring_atom_set) else 0

    # 2. Main ring size = largest + second-largest main bridge
    if len(main_sizes) >= 2:
        main_ring = main_sizes[0] + main_sizes[1] + 2  # +2 for bridgeheads
    else:
        main_ring = main_sizes[0] if main_sizes else 0

    # 3. Main bridge = third main bridge
    main_bridge = main_sizes[2] if len(main_sizes) >= 3 else -1

    # 4. Secondary bridge sizes (descending; as tuple)
    sec_tup = sec_sizes

    return (coverage, main_ring, main_bridge, sec_tup)
