"""
iupac_namer/ring_naming/numbering.py

Ring numbering computation (P-14.4).

For monocyclic rings: try all starting atoms and both directions (CW/CCW).
For fused rings: try all starting atoms and both traversal directions,
  select the numbering that gives heteroatoms the lowest locant set (P-14.5).
For bridged/spiro: simplified single-pass numbering.

Returns tuple[Numbering, ...] with valid orderings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.types import Locant, Numbering

if TYPE_CHECKING:
    from iupac_namer.types import NamedParent, RingSystem

logger = logging.getLogger(__name__)

# Heteroatom priority for ring numbering (P-14.5 / P-31.1.2.2):
# O > S > Se > Te > N > P > B > Si > Ge > Sn
_HETEROATOM_PRIORITY: dict[str, int] = {
    "O":  1, "S":  2, "Se": 3, "Te": 4,
    "N":  5, "P":  6, "B":  7, "Si": 8, "Ge": 9, "Sn": 10,
}


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def compute_ring_numberings(
    ring_system: "RingSystem",
    mol,
    named_parent: "NamedParent",
) -> tuple[Numbering, ...]:
    """Compute valid numberings for a ring parent.

    Returns a tuple of Numbering objects (may be empty on failure).
    The BEST numbering (lowest heteroatom locants per P-14.5) is first.

    If named_parent has pre-computed numbering_options (set from curated
    ring data), those are returned directly.
    """
    # Use pre-computed locant map if available (from curated ring data)
    if named_parent is not None and named_parent.numbering_options:
        return named_parent.numbering_options

    rs_type = ring_system.type
    if rs_type == "monocyclic":
        return _compute_monocyclic_numberings(ring_system, mol)
    elif rs_type == "fused":
        return _compute_fused_numberings(ring_system, mol)
    elif rs_type == "bridged":
        # Full IUPAC P-23 von Baeyer numbering: enumerate all valid orderings
        # so the strategy can pick the one with lowest substituent locants.
        if ring_system.bridge_sizes:
            from iupac_namer.ring_naming.bridged import compute_vb_numberings
            vb_nbs = compute_vb_numberings(ring_system, ring_system.bridge_sizes, mol)
            if vb_nbs:
                return vb_nbs
        # Fallback: sorted atom index order
        return _compute_simple_numbering(ring_system)
    elif rs_type == "spiro":
        return _compute_simple_numbering(ring_system)
    return ()


# ---------------------------------------------------------------------------
# Monocyclic numbering
# ---------------------------------------------------------------------------

def _compute_monocyclic_numberings(
    ring_system: "RingSystem", mol
) -> tuple[Numbering, ...]:
    """Try all starting atoms and both directions for monocyclic rings.

    Returns up to 2*n orderings where n = ring size.
    For efficiency, we return only distinct orderings (deduplication by
    assignment tuple).
    """
    ring_atoms = sorted(ring_system.atom_indices)
    if not ring_atoms:
        return ()

    ordered = _order_ring_atoms_mol(ring_atoms, mol)
    if not ordered:
        # Fallback: use sorted order
        ordered = ring_atoms

    n = len(ordered)
    seen: set[tuple] = set()
    numberings: list[Numbering] = []

    for start_idx in range(n):
        # Clockwise
        atoms_cw = [ordered[(start_idx + i) % n] for i in range(n)]
        nb_cw = _make_numbering(atoms_cw)
        key = nb_cw._assignments
        if key not in seen:
            seen.add(key)
            numberings.append(nb_cw)

        # Counter-clockwise
        atoms_ccw = [ordered[(start_idx - i) % n] for i in range(n)]
        nb_ccw = _make_numbering(atoms_ccw)
        key = nb_ccw._assignments
        if key not in seen:
            seen.add(key)
            numberings.append(nb_ccw)

    return tuple(numberings)


# ---------------------------------------------------------------------------
# Fused ring numbering (P-14.4 / P-31.1.2.2)
# ---------------------------------------------------------------------------

def _build_ring_adj(atom_indices: frozenset, mol) -> dict[int, list[int]]:
    """Build adjacency list restricted to the ring atom set."""
    adj: dict[int, list[int]] = {idx: [] for idx in atom_indices}
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in atom_indices:
                adj[idx].append(nb_idx)
    return adj


def _peripheral_traversal(start: int, direction_neighbor: int,
                           adj: dict[int, list[int]]) -> list[int] | None:
    """Traverse the peripheral path of a fused ring system.

    For a fused ring, peripheral atoms have degree 2 in the ring graph;
    junction atoms have degree 3 (or more). IUPAC numbering follows the
    outer perimeter: when at a peripheral atom, move to the next peripheral
    atom; when at a junction, move along the perimeter (not the interior).

    Implements a DFS-like perimeter walk:
    - Start at `start`, first step is to `direction_neighbor`.
    - At each step, prefer to continue in the same direction (non-backtrack neighbor).
    - If we reach a junction (degree >= 3), choose the neighbor that keeps us
      on the perimeter (degree-2 neighbor if available, otherwise the
      already-visited neighbor is skipped via backtrack avoidance).
    - Return the ordered list of atoms, or None if the traversal fails.

    This is a heuristic; for complex polycyclics, the correct traversal may
    require a full IUPAC P-14.4 implementation.  For bicyclics (most retained
    rings), this works correctly.
    """
    n = len(adj)
    ordered = [start, direction_neighbor]
    visited = {start, direction_neighbor}
    prev = start
    current = direction_neighbor

    for _ in range(n - 2):
        neighbors = adj[current]
        # Exclude the direction we came from
        candidates = [nb for nb in neighbors if nb != prev]
        if not candidates:
            return None  # dead end

        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            # At a junction: prefer atoms NOT yet visited (peripheral path).
            # If all are visited, we've completed the perimeter.
            unvisited = [nb for nb in candidates if nb not in visited]
            if unvisited:
                # Choose the unvisited neighbor with the fewest ring-adj neighbors
                # (i.e., prefer degree-2 atoms = peripheral atoms)
                unvisited.sort(key=lambda x: len(adj[x]))
                nxt = unvisited[0]
            else:
                # All candidates visited — we may have completed a cycle
                nxt = candidates[0]

        if nxt in visited and len(ordered) > 2:
            # Check if we've returned to start (completed the perimeter)
            if nxt == start:
                break
        ordered.append(nxt)
        visited.add(nxt)
        prev = current
        current = nxt

    return ordered if len(ordered) == n else None


def _numbering_score(ordered: list[int], mol, atom_indices: frozenset) -> tuple:
    """Score a numbering for IUPAC P-14.5 heteroatom priority.

    Lower score = better (heteroatoms get lower locants).
    Returns a tuple for lexicographic comparison:
    (element-weighted heteroatom locant sum, total heteroatom locant sum)
    """
    hetero_positions: list[tuple[int, int]] = []  # (priority, locant_1based)
    for i, atom_idx in enumerate(ordered):
        sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        prio = _HETEROATOM_PRIORITY.get(sym)
        if prio is not None:
            hetero_positions.append((prio, i + 1))  # 1-based locant

    if not hetero_positions:
        return (0,)

    # Primary: sum of locants weighted by element priority (lower O/S locants matter more)
    # We sort by priority (O first), then compare locant sets lexicographically
    hetero_by_prio: dict[int, list[int]] = {}
    for prio, loc in hetero_positions:
        hetero_by_prio.setdefault(prio, []).append(loc)

    # Build comparison tuple: for each element priority (ascending), sorted locants
    score = []
    for prio in sorted(hetero_by_prio.keys()):
        score.extend(sorted(hetero_by_prio[prio]))

    return tuple(score)


def _compute_fused_numberings(
    ring_system: "RingSystem", mol
) -> tuple[Numbering, ...]:
    """Compute numberings for fused ring systems using peripheral traversal.

    Tries all starting atoms and both traversal directions, selects the
    numbering that gives heteroatoms the lowest locants (P-14.5).

    Returns a tuple with the best numbering first.
    """
    atom_indices = ring_system.atom_indices
    n = len(atom_indices)
    if n == 0:
        return ()

    adj = _build_ring_adj(atom_indices, mol)

    seen: set[tuple] = set()
    candidates: list[tuple[tuple, list[int]]] = []  # (score, ordered_atoms)

    sorted_atoms = sorted(atom_indices)

    for start in sorted_atoms:
        for direction_nb in adj[start]:
            ordered = _peripheral_traversal(start, direction_nb, adj)
            if ordered is None or len(ordered) != n:
                continue
            key = tuple(ordered)
            if key in seen:
                continue
            seen.add(key)
            score = _numbering_score(ordered, mol, atom_indices)
            candidates.append((score, ordered))

    if not candidates:
        # Fallback: sorted atom index order
        return _compute_simple_numbering(ring_system)

    # Sort by score (best = lowest heteroatom locants first)
    candidates.sort(key=lambda x: x[0])

    # Return top numberings (all that tie for best score, plus a few alternatives)
    best_score = candidates[0][0]
    result = []
    for score, ordered in candidates:
        if score == best_score:
            result.append(_make_numbering(ordered))
        if len(result) >= 10:  # cap at 10 best
            break

    # If we only have the best, also include a few runner-up for strategy scoring
    if len(result) < 4:
        for score, ordered in candidates[len(result):len(result)+4]:
            result.append(_make_numbering(ordered))

    return tuple(result)


# ---------------------------------------------------------------------------
# Simple numbering (bridged/spiro — single pass)
# ---------------------------------------------------------------------------

def _compute_simple_numbering(ring_system: "RingSystem") -> tuple[Numbering, ...]:
    """Compute a single numbering from sorted atom indices.

    Used for bridged/spiro where the full IUPAC numbering algorithm
    is complex and deferred. This gives each atom a locant in sorted order.
    """
    atom_indices = sorted(ring_system.atom_indices)
    if not atom_indices:
        return ()
    nb = _make_numbering(atom_indices)
    return (nb,)


# ---------------------------------------------------------------------------
# Ring atom ordering helpers
# ---------------------------------------------------------------------------

def _order_ring_atoms_mol(atom_indices: list[int], mol) -> list[int]:
    """Order ring atoms into a cyclic sequence using BFS on the ring graph.

    Returns a list of atom indices in cyclic order, or [] on failure.
    """
    atom_set = set(atom_indices)
    if not atom_set:
        return []

    start = atom_indices[0]
    ordered = [start]
    visited = {start}

    # Build adjacency in the ring-atom subgraph
    adj: dict[int, list[int]] = {idx: [] for idx in atom_indices}
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in atom_set:
                adj[idx].append(nb_idx)

    # Walk the ring: each node should have exactly 2 ring neighbors
    current = start
    prev = -1
    for _ in range(len(atom_indices) - 1):
        neighbors_in_ring = [nb for nb in adj[current] if nb != prev]
        if not neighbors_in_ring:
            return []  # broken ring graph
        next_atom = neighbors_in_ring[0]
        if next_atom in visited:
            # Try the other neighbor
            if len(neighbors_in_ring) > 1:
                next_atom = neighbors_in_ring[1]
            else:
                break  # cycle completed early
        ordered.append(next_atom)
        visited.add(next_atom)
        prev = current
        current = next_atom

    if len(ordered) != len(atom_indices):
        # Ring traversal incomplete — return sorted order as fallback
        return atom_indices
    return ordered


def _make_numbering(ordered_atoms: list[int]) -> Numbering:
    """Build a Numbering from an ordered atom list (1-indexed)."""
    assignments = tuple(
        (atom, Locant.numeric(i + 1))
        for i, atom in enumerate(ordered_atoms)
    )
    locant_set = tuple(Locant.numeric(i + 1) for i in range(len(ordered_atoms)))
    return Numbering(_assignments=assignments, locant_set=locant_set)
