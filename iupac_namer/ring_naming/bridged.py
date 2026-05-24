"""
iupac_namer/ring_naming/bridged.py

Von Baeyer nomenclature for bridged polycyclic ring systems (P-22.2).

Produces names like:
  bicyclo[2.2.1]heptane  (norbornane)
  bicyclo[2.2.2]octane   (DABCO precursor skeleton)
  1,4-diazabicyclo[2.2.2]octane  (DABCO)
  tricyclo[2.2.1.0^{2,6}]heptane

Heteroatom replacement (P-31.1.3):
  When the bridged ring contains non-carbon atoms, aza/oxa/thia/... prefixes
  are prepended to the carbocycle name, with locants from von Baeyer numbering.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from iupac_namer.data_loader import get_chain_stem, get_hw_tables, get_multiplier
from iupac_namer.types import NamedParent

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, Numbering, RingSystem
else:
    from iupac_namer.types import Numbering  # runtime use in annotations

logger = logging.getLogger(__name__)

# Cycle-count to Von Baeyer prefix
_CYCLE_PREFIX: dict[int, str] = {
    2: "bicyclo",
    3: "tricyclo",
    4: "tetracyclo",
    5: "pentacyclo",
    6: "hexacyclo",
    7: "heptacyclo",
    8: "octacyclo",
    9: "nonacyclo",
    10: "decacyclo",
}

# Heteroatom replacement priority (P-31.1.3 uses same order as HW: O > S > Se > Te > N > P > As)
_REPLACEMENT_PRIORITY: dict[str, int] = {
    "O":  8,
    "S":  7,
    "Se": 6,
    "Te": 5,
    "N":  4,
    "P":  3,
    "As": 2,
    "Si": 1,
    "Ge": 0,
}


def _find_simple_paths(
    start: int,
    end: int,
    allowed: frozenset[int],
    exclude_intermediate: set[int],
    mol,
) -> list[list[int]]:
    """Find all simple paths from start to end within allowed atoms.

    Intermediate nodes cannot be in exclude_intermediate.
    """
    all_paths: list[list[int]] = []
    stack: list[tuple[int, list[int]]] = [(start, [start])]
    while stack:
        curr, path = stack.pop()
        if curr == end:
            all_paths.append(path)
            continue
        for nb in mol.GetAtomWithIdx(curr).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in allowed:
                continue
            if nb_idx in path:
                continue
            if nb_idx != end and nb_idx in exclude_intermediate:
                continue
            stack.append((nb_idx, path + [nb_idx]))
    return all_paths


def _find_bridgeheads_and_paths(
    ring_system: "RingSystem",
    bridge_sizes: tuple[int, ...],
    mol,
) -> list[tuple[int, int, list[list[int]]]]:
    """Find all valid (bh1, bh2, sorted_paths) triples for a bridged ring.

    Returns a list of triples where:
    - bh1, bh2 are the two principal bridgeheads
    - sorted_paths is the list of bridge paths sorted by length descending,
      with bh1 as path[0] and bh2 as path[-1]

    Tries all candidate bridgehead pairs in priority order.
    """
    all_atoms = ring_system.atom_indices
    rings = ring_system.rings

    # Count how many SSSR rings each atom appears in
    atom_ring_count: Counter[int] = Counter()
    for ring in rings:
        for atom in ring:
            atom_ring_count[atom] += 1

    # Bridgehead candidates: atoms in 2 or more SSSR rings
    all_bridgeheads = sorted(
        {a for a, c in atom_ring_count.items() if c >= 2}
    )
    if len(all_bridgeheads) < 2:
        return []

    # Target bridge sizes (sorted descending)
    target_sizes = sorted(bridge_sizes, reverse=True)
    n_bridges = len(target_sizes)

    # Try pairs of bridgehead candidates
    max_ring_count = max(atom_ring_count[a] for a in all_bridgeheads)
    priority_bhs = sorted({a for a in all_bridgeheads if atom_ring_count[a] == max_ring_count})

    candidate_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(len(priority_bhs)):
        for j in range(i + 1, len(priority_bhs)):
            pair = (priority_bhs[i], priority_bhs[j])
            candidate_pairs.append(pair)
            seen_pairs.add(pair)
    for i in range(len(all_bridgeheads)):
        for j in range(i + 1, len(all_bridgeheads)):
            pair = (all_bridgeheads[i], all_bridgeheads[j])
            if pair not in seen_pairs:
                candidate_pairs.append(pair)
                seen_pairs.add(pair)

    results: list[tuple[int, int, list[list[int]]]] = []

    for bh1, bh2 in candidate_pairs:
        paths = _find_simple_paths(bh1, bh2, all_atoms,
                                   exclude_intermediate=set(), mol=mol)
        if len(paths) != n_bridges:
            continue

        paths_sorted = sorted(paths, key=len, reverse=True)
        sizes_found = [len(p) - 2 for p in paths_sorted]
        if sizes_found != target_sizes:
            continue

        covered: set[int] = set()
        for p in paths_sorted:
            covered.update(p)
        if covered != set(all_atoms):
            continue

        results.append((bh1, bh2, paths_sorted))

    return results


def _build_locant_map(
    bh1: int,
    bh2: int,
    paths_sorted: list[list[int]],
    bridge_directions: list[bool],
) -> dict[int, int] | None:
    """Assign VB locants given a bridgehead pair, sorted paths, and direction flags.

    bridge_directions[i] == True means traverse path i forward (bh1→bh2),
    False means traverse reversed (only the intermediate atoms are reversed).

    Returns atom→locant dict or None if assignment fails.
    """
    locant: dict[int, int] = {}
    current_loc = 1
    locant[bh1] = current_loc
    current_loc += 1

    for path_idx, path in enumerate(paths_sorted):
        intermediates = path[1:-1]  # strip bridgeheads
        if not bridge_directions[path_idx]:
            intermediates = list(reversed(intermediates))
        for atom in intermediates:
            if atom not in locant:
                locant[atom] = current_loc
                current_loc += 1
        if path_idx == 0:
            locant[bh2] = current_loc
            current_loc += 1

    if len(locant) == len(set(a for p in paths_sorted for a in p)):
        return locant
    return None


def _compute_vb_numbering(
    ring_system: "RingSystem",
    bridge_sizes: tuple[int, ...],
    mol,
) -> dict[int, int] | None:
    """Compute ONE von Baeyer atom → locant mapping for the ring system.

    Returns the first valid mapping found (used as a fallback).
    Prefer _choose_best_vb_locant_map() when the ring has unsaturation or
    heteroatoms, since that function picks the numbering with the lowest
    locant set.
    """
    triples = _find_bridgeheads_and_paths(ring_system, bridge_sizes, mol)
    if not triples:
        return None
    bh1, bh2, paths_sorted = triples[0]
    n_bridges = len(paths_sorted)
    directions = [True] * n_bridges
    return _build_locant_map(bh1, bh2, paths_sorted, directions)


def _choose_best_vb_locant_map(
    ring_system: "RingSystem",
    bridge_sizes: tuple[int, ...],
    mol,
) -> dict[int, int] | None:
    """Choose the VB locant map that gives the lowest locant set for the ring.

    Priority (IUPAC P-23):
    1. Heteroatoms get lowest possible locants.
    2. Double/triple bonds get the lowest possible locants.
    3. Other criteria (handled by strategy layer, not here).

    Tries all valid numberings (both bridgehead orderings × all permutations
    of equal-length bridges) and returns the one with the lexicographically
    smallest locant tuple for [heteroatom_locants, unsaturation_locants].
    """
    import itertools as _itertools
    from rdkit.Chem import BondType

    triples = _find_bridgeheads_and_paths(ring_system, bridge_sizes, mol)
    if not triples:
        return None

    ring_atom_set = ring_system.atom_indices
    heteroatom_idxs = frozenset(
        hp.atom_idx for hp in (ring_system.heteroatoms or ())
    )

    # Find unsaturated bonds (double or triple) within the ring
    unsat_bonds: list[tuple[int, int]] = []
    seen_b: set[tuple[int, int]] = set()
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in ring_atom_set:
                continue
            bkey = (min(atom_idx, nb_idx), max(atom_idx, nb_idx))
            if bkey in seen_b:
                continue
            seen_b.add(bkey)
            bond = mol.GetBondBetweenAtoms(atom_idx, nb_idx)
            if bond is not None and bond.GetBondType() in (BondType.DOUBLE, BondType.TRIPLE):
                unsat_bonds.append(bkey)

    best_map: dict[int, int] | None = None
    best_score: tuple | None = None

    def _score(locant_map: dict[int, int]) -> tuple:
        """Lower score = better (lowest locants first)."""
        # Score 1: sorted heteroatom locants
        hetero_locs = sorted(locant_map.get(a, 9999) for a in heteroatom_idxs)
        # Score 2: sorted unsaturation locants (lower atom of each bond)
        unsat_locs = sorted(
            min(locant_map.get(a, 9999), locant_map.get(b, 9999))
            for a, b in unsat_bonds
        )
        # Score 3: full locant set as tiebreaker
        all_locs = sorted(locant_map.values())
        return tuple(hetero_locs + unsat_locs + all_locs)

    for bh1, bh2, paths_sorted in triples:
        for bh_swap in (False, True):
            if bh_swap:
                actual_bh1, actual_bh2 = bh2, bh1
                base_paths = [list(reversed(p)) for p in paths_sorted]
            else:
                actual_bh1, actual_bh2 = bh1, bh2
                base_paths = list(paths_sorted)

            # Build runs of equal-length bridges for ALL bridges
            runs_all: list[list] = []
            if base_paths:
                cr: list = [base_paths[0]]
                for p in base_paths[1:]:
                    if len(p) == len(cr[0]):
                        cr.append(p)
                    else:
                        runs_all.append(cr)
                        cr = [p]
                runs_all.append(cr)

            run_perms: list[list] = [[]]
            for run in runs_all:
                new_rp: list[list] = []
                for perm in _itertools.permutations(run):
                    for existing in run_perms:
                        new_rp.append(existing + list(perm))
                run_perms = new_rp

            for ordered_paths in run_perms:
                # IUPAC P-23.2.5 alternating-direction convention.  See
                # compute_vb_numberings for rationale.
                n_bridges = len(ordered_paths)
                dirs = [i % 2 == 0 for i in range(n_bridges)]

                lm = _build_locant_map(actual_bh1, actual_bh2, ordered_paths, dirs)
                if lm is None:
                    continue
                score = _score(lm)
                if best_score is None or score < best_score:
                    best_score = score
                    best_map = lm

    return best_map


def _extend_locant_map_over_secondaries(
    main_locant_map: dict[int, int],
    secondary_bridges: "tuple[tuple[tuple[int, int], tuple[int, ...]], ...]",
    total_atoms: int,
) -> dict[int, int] | None:
    """Extend a main-bicycle locant map over secondary bridge interior atoms.

    Locants continue from ``max(main_locant_map.values()) + 1``.  Secondary
    bridges are processed in descending size order (largest first); within
    each bridge, the direction is chosen so the **lower** endpoint locant
    comes first (matches IUPAC convention).

    Returns the extended map, or ``None`` if any interior atom has already
    been assigned (which would indicate an inconsistent decomposition).
    """
    locant_map = dict(main_locant_map)
    next_loc = max(locant_map.values()) + 1 if locant_map else 1

    # Sort secondary bridges: largest bridge first; tiebreak by lower-endpoint
    # locant ascending so numbering of interiors is deterministic.
    def _sort_key(sb: tuple[tuple[int, int], tuple[int, ...]]) -> tuple:
        (a, b), interior = sb
        la = locant_map.get(a, 9999)
        lb = locant_map.get(b, 9999)
        lower = min(la, lb)
        return (-len(interior), lower)

    ordered = sorted(secondary_bridges, key=_sort_key)

    for (a, b), interior in ordered:
        if not interior:
            continue  # 0-atom bridge: no new locants needed
        la = locant_map.get(a)
        lb = locant_map.get(b)
        if la is None or lb is None:
            return None
        # Traverse from lower-locant endpoint into the bridge
        if la <= lb:
            seq = list(interior)
        else:
            seq = list(reversed(interior))
        for atom in seq:
            if atom in locant_map:
                return None
            locant_map[atom] = next_loc
            next_loc += 1

    if len(locant_map) != total_atoms:
        return None
    return locant_map


def _choose_best_vb_locant_map_with_secondaries(
    ring_system: "RingSystem",
    bridge_sizes: tuple[int, ...],
    mol,
) -> dict[int, int] | None:
    """Choose the VB locant map for a ring with secondary bridges.

    Enumerates all main-bicycle numberings (via the existing triple / path
    permutation machinery), extends each with secondary bridge interiors,
    and picks the one minimising (in priority order):

    1. Heteroatom locant set (P-31.1.3)
    2. Unsaturation locants (P-23.3)
    3. Secondary bridge superscript locants (P-23.2.5)
    4. Full locant set

    Falls back to ``_choose_best_vb_locant_map`` if no secondary bridges are
    defined.
    """
    import itertools as _itertools
    from rdkit.Chem import BondType

    secondary_bridges = ring_system.secondary_bridges or ()
    if not secondary_bridges:
        return _choose_best_vb_locant_map(ring_system, bridge_sizes, mol)

    triples = _find_bridgeheads_and_paths(ring_system, bridge_sizes, mol)
    # The structured decomposition's bridgeheads may not appear in ``triples``
    # because that helper filters by exact bridge-size match.  If we can't get
    # main-bicycle triples directly, derive them from the decomposition
    # bridgeheads.
    if not triples:
        # Derive from ring_system.atom_indices + mol — use any valid main
        # bridgehead pair with 2+ disjoint paths of the required sizes.
        from iupac_namer.ring_naming.vb_decompose import (
            decompose_ring_system,
        )
        decomps = decompose_ring_system(ring_system.atom_indices, mol)
        if not decomps:
            return None
        d = decomps[0]
        main_paths = [list(p) for p in d.main_bridges]
        main_paths.sort(key=len, reverse=True)
        triples = [(d.bh1, d.bh2, main_paths)]

    ring_atom_set = ring_system.atom_indices
    heteroatom_idxs = frozenset(
        hp.atom_idx for hp in (ring_system.heteroatoms or ())
    )

    unsat_bonds: list[tuple[int, int]] = []
    seen_b: set[tuple[int, int]] = set()
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in ring_atom_set:
                continue
            bkey = (min(atom_idx, nb_idx), max(atom_idx, nb_idx))
            if bkey in seen_b:
                continue
            seen_b.add(bkey)
            bond = mol.GetBondBetweenAtoms(atom_idx, nb_idx)
            if bond is not None and bond.GetBondType() in (BondType.DOUBLE, BondType.TRIPLE):
                unsat_bonds.append(bkey)

    total_atoms = len(ring_atom_set)

    # Identify ring atoms that carry at least one external (non-ring) neighbour.
    # These are likely substituent-bearing positions.  IUPAC P-23.2.5 ranks
    # substituent locants (criterion d) ABOVE secondary-bridge superscripts
    # (criterion e), so when the main heteroatom/unsaturation criteria are
    # tied the numbering that lowers substituent-bearing locants must win —
    # otherwise the bridge descriptor emitted here disagrees with the
    # numbering the strategy layer picks (which scores by substituents),
    # producing chemically impossible names like
    # ``tricyclo[5.4.3.0^{1,8}]tetradecan-1-yl`` where locant 1 is a
    # quaternary bridgehead that cannot accept another substituent.
    substituent_atoms: set[int] = set()
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in ring_atom_set:
                substituent_atoms.add(atom_idx)
                break

    def _score(locant_map: dict[int, int]) -> tuple:
        hetero_locs = sorted(locant_map.get(a, 9999) for a in heteroatom_idxs)
        unsat_locs = sorted(
            min(locant_map.get(a, 9999), locant_map.get(b, 9999))
            for a, b in unsat_bonds
        )
        # Substituent-bearing atom locants (proxy for P-23.2.5 criterion d).
        sub_locs = sorted(locant_map.get(a, 9999) for a in substituent_atoms)
        # Secondary bridge superscript locant pairs (sorted by bridge size desc)
        sec_scored: list[tuple] = []
        for (a, b), interior in sorted(
            secondary_bridges, key=lambda sb: -len(sb[1])
        ):
            la = locant_map.get(a, 9999)
            lb = locant_map.get(b, 9999)
            lo, hi = (la, lb) if la <= lb else (lb, la)
            sec_scored.append((lo, hi))
        all_locs = sorted(locant_map.values())
        return (
            tuple(hetero_locs),
            tuple(unsat_locs),
            tuple(sub_locs),
            tuple(sec_scored),
            tuple(all_locs),
        )

    best_map: dict[int, int] | None = None
    best_score: tuple | None = None

    for bh1, bh2, paths_sorted in triples:
        for bh_swap in (False, True):
            if bh_swap:
                actual_bh1, actual_bh2 = bh2, bh1
                base_paths = [list(reversed(p)) for p in paths_sorted]
            else:
                actual_bh1, actual_bh2 = bh1, bh2
                base_paths = list(paths_sorted)

            runs_all: list[list] = []
            if base_paths:
                cr: list = [base_paths[0]]
                for p in base_paths[1:]:
                    if len(p) == len(cr[0]):
                        cr.append(p)
                    else:
                        runs_all.append(cr)
                        cr = [p]
                runs_all.append(cr)

            run_perms: list[list] = [[]]
            for run in runs_all:
                new_rp: list[list] = []
                for perm in _itertools.permutations(run):
                    for existing in run_perms:
                        new_rp.append(existing + list(perm))
                run_perms = new_rp

            for ordered_paths in run_perms:
                n_br = len(ordered_paths)
                dirs = [i % 2 == 0 for i in range(n_br)]
                lm = _build_locant_map(actual_bh1, actual_bh2, ordered_paths, dirs)
                if lm is None:
                    continue
                ext = _extend_locant_map_over_secondaries(
                    lm, secondary_bridges, total_atoms
                )
                if ext is None:
                    continue
                s = _score(ext)
                if best_score is None or s < best_score:
                    best_score = s
                    best_map = ext

    return best_map


def compute_vb_numberings(
    ring_system: "RingSystem",
    bridge_sizes: tuple[int, ...],
    mol,
) -> "tuple[Numbering, ...]":
    """Compute ALL valid von Baeyer numberings for the ring system.

    Enumerates:
    - Both choices of which bridgehead gets locant 1 (bh1 vs bh2)
    - All orderings of equal-length bridges (P-22.2: when bridges tie in
      length, the one that gives lower locants to substituents is numbered
      second; we enumerate all permutations and let the strategy pick)

    For the first (longest) bridge, traversal is always bh1→bh2 so that
    the atom at locant 2 is adjacent to bh1.  For subsequent (secondary)
    bridges, IUPAC P-23.2.5 permits either direction: the 2nd bridge may
    begin numbering from the atom adjacent to bh2 (continuation of locant
    sequence) or from the atom adjacent to bh1.  Both are enumerated so the
    strategy layer can pick the option with the lowest locants for the
    heteroatoms and substituents — the same criterion IUPAC itself uses.

    Returns a tuple of Numbering objects. Empty tuple on failure.

    Used by numbering.py to supply multiple numbering options so the strategy
    layer can pick the one with lowest locants for substituents/PCG.
    """
    import itertools
    # Import here to avoid circular imports at module level
    from iupac_namer.ring_naming.numbering import _make_numbering

    triples = _find_bridgeheads_and_paths(ring_system, bridge_sizes, mol)

    # When the ring has secondary bridges (tricyclic+) and _find_bridgeheads
    # didn't match (usually because the principal-bridgehead path set doesn't
    # cover all atoms — the secondary atoms are excluded from the main 3
    # paths), derive bridgeheads from the decomposition.
    secondary_bridges = ring_system.secondary_bridges or ()
    if not triples and secondary_bridges:
        from iupac_namer.ring_naming.vb_decompose import decompose_ring_system
        decomps = decompose_ring_system(ring_system.atom_indices, mol)
        if decomps:
            d = decomps[0]
            main_paths = [list(p) for p in d.main_bridges]
            main_paths.sort(key=len, reverse=True)
            triples = [(d.bh1, d.bh2, main_paths)]

    if not triples:
        return ()

    total_atoms = len(ring_system.atom_indices)
    seen: set[tuple] = set()
    all_numberings: list[Numbering] = []

    for bh1, bh2, paths_sorted in triples:
        # Try both bridgehead orderings.
        for bh_swap in (False, True):
            if bh_swap:
                actual_bh1, actual_bh2 = bh2, bh1
                # Reverse each path so that actual_bh1 is path[0]
                base_paths = [list(reversed(p)) for p in paths_sorted]
            else:
                actual_bh1, actual_bh2 = bh1, bh2
                base_paths = list(paths_sorted)

            # Group bridge paths by bridge length (they are already sorted desc).
            # IUPAC P-22.2.4: when bridges tie in length, the one that gives
            # lower locants to double bonds / heteroatoms / substituents is
            # numbered first.  We enumerate all permutations within each
            # equal-length group (across ALL bridges, not just bridge2+) and
            # let the strategy layer pick the lowest-locant winner.
            if len(base_paths) <= 1:
                path_orderings = [base_paths]
            else:
                # Group ALL bridges by length, preserving overall
                # length-descending order.  For each group, enumerate all
                # internal permutations; cross-product across groups.
                #
                # Example: bridges [3, 2, 2] → group1=[3-bridge], group2=[2a, 2b]
                #          → orderings: [3-bridge, 2a, 2b] and [3-bridge, 2b, 2a]
                #
                # Example: bridges [2, 2, 2] → one group=[2a, 2b, 2c]
                #          → 6 orderings (all permutations of 3 bridges)

                # Build runs of equal-length bridges (all of base_paths, not just rest).
                runs_all: list[list] = []
                if base_paths:
                    current_run_all: list = [base_paths[0]]
                    for p in base_paths[1:]:
                        if len(p) == len(current_run_all[0]):
                            current_run_all.append(p)
                        else:
                            runs_all.append(current_run_all)
                            current_run_all = [p]
                    runs_all.append(current_run_all)

                # Cross-product of permutations of each run
                run_perms_list: list[list] = [[]]  # each element is a list of paths
                for run in runs_all:
                    new_perms: list[list] = []
                    for perm in itertools.permutations(run):
                        for existing in run_perms_list:
                            new_perms.append(existing + list(perm))
                    run_perms_list = new_perms

                path_orderings = run_perms_list

            for ordered_paths in path_orderings:
                # IUPAC P-23.2.5 traversal convention: bridges alternate direction.
                # Bridge 0 (the longest) is traversed bh1→bh2 so that locant 2 is
                # adjacent to bh1.  Bridge 1 is traversed bh2→bh1 so that locant
                # (bh2+1) is adjacent to bh2.  Bridge 2 is again bh1→bh2, and so
                # on.  This is the labelling convention OPSIN uses when parsing
                # names back to structures, so emitting names in any other
                # convention produces a name that round-trips to a different
                # molecule.
                n_bridges = len(ordered_paths)
                dirs = [i % 2 == 0 for i in range(n_bridges)]  # True, False, True, ...

                locant_map = _build_locant_map(
                    actual_bh1, actual_bh2, ordered_paths, dirs
                )
                if locant_map is None:
                    continue
                # Extend locants over secondary-bridge interiors when present.
                if secondary_bridges:
                    ext = _extend_locant_map_over_secondaries(
                        locant_map, secondary_bridges, total_atoms
                    )
                    if ext is None:
                        continue
                    locant_map = ext
                ordered_atoms = sorted(locant_map.keys(), key=lambda a: locant_map[a])
                key = tuple(ordered_atoms)
                if key in seen:
                    continue
                seen.add(key)
                nb = _make_numbering(ordered_atoms)
                all_numberings.append(nb)

    return tuple(all_numberings)


def _build_heteroatom_prefix(
    heteroatoms: tuple,  # tuple[HeteroPosition]
    locant_map: dict[int, int],
) -> str | None:
    """Build the heteroatom replacement prefix string for von Baeyer names.

    E.g. for DABCO with N at locants 1 and 4: returns "1,4-diaza"
    For 2-oxabicyclo...: returns "2-oxa"

    Returns None if any heteroatom element is unsupported.
    """
    hw_tables = get_hw_tables()
    prefixes_list = hw_tables.get("prefixes", [])
    elem_to_prefix: dict[str, str] = {}
    for entry in prefixes_list:
        elem_to_prefix[entry["element"]] = entry["prefix"]

    # Get locants for heteroatoms
    hetero_with_locs: list[tuple[int, str]] = []  # (locant, element)
    for hp in heteroatoms:
        loc = locant_map.get(hp.atom_idx)
        if loc is None:
            logger.debug("No locant for heteroatom %s", hp)
            return None
        elem = hp.element
        if elem not in elem_to_prefix:
            logger.debug("No HW prefix for element %s", elem)
            return None
        hetero_with_locs.append((loc, elem))

    if not hetero_with_locs:
        return ""

    # Sort by priority (highest first), then by locant (lowest first)
    hetero_with_locs.sort(key=lambda x: (-_REPLACEMENT_PRIORITY.get(x[1], 0), x[0]))

    # Group by element (in priority order)
    elem_order: list[str] = []
    elem_to_locants: dict[str, list[int]] = {}
    for loc, elem in hetero_with_locs:
        if elem not in elem_to_locants:
            elem_order.append(elem)
            elem_to_locants[elem] = []
        elem_to_locants[elem].append(loc)

    # Sort locants within each element group
    for elem in elem_order:
        elem_to_locants[elem].sort()

    # Build per-element components: e.g. "dioxa" (from "oxa" × 2) or "aza"
    raw_parts: list[tuple[list[int], str]] = []  # (locants, prefix_part)
    for elem in elem_order:
        locs = elem_to_locants[elem]
        pref = elem_to_prefix[elem]
        n_elem = len(locs)
        if n_elem == 1:
            raw_parts.append((locs, pref))
        else:
            multi = get_multiplier(n_elem)
            if multi is None:
                return None
            raw_parts.append((locs, multi + pref))

    # Von Baeyer a-replacement: for multi-element systems, IUPAC (and OPSIN)
    # require each element's locants to be cited as a SEPARATE segment joined
    # with '-', not merged into one combined locant prefix.  A combined
    # "2,5,7,1-trioxaphospha..." is not parseable because OPSIN can only
    # apportion 3 locants to the "tri" multiplier, leaving the "1" orphaned.
    # Instead emit "2,5,7-trioxa-1-phospha...".
    # For single-element systems (n_parts == 1) the original combined form
    # still works (e.g. "1,4-diaza...").
    if len(raw_parts) == 1:
        locs, pref = raw_parts[0]
        loc_str = ",".join(str(loc) for loc in locs)
        return f"{loc_str}-{pref}"

    # Multi-element: build segments "<locs>-<prefix>" joined by '-',
    # applying 'a'-elision between adjacent segments where the *following*
    # segment's leading character is a vowel.  Because each segment begins
    # with a digit/locant after the initial one, elision BETWEEN segments is
    # not applicable — the terminal 'a' of a component is only elided at the
    # VERY END before the parent stem beginning with a vowel (e.g. "aza" +
    # "bicyclo..." → no elision because 'b' is not a vowel).  So per-segment
    # prefixes are kept intact with their trailing 'a'.
    segments: list[str] = []
    for locs, pref in raw_parts:
        loc_str = ",".join(str(loc) for loc in locs)
        segments.append(f"{loc_str}-{pref}")
    return "-".join(segments)


def _detect_vb_unsaturation(
    ring_system: "RingSystem",
    locant_map: dict[int, int],
    mol,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (double_bond_pairs, triple_bond_pairs) for bonds within the ring.

    Each pair is ``(low_locant, high_locant)`` with ``low <= high``.  The bond
    is cited at ``low_locant``; the suffix builder adds the ``(high_locant)``
    disambiguation when ``high != low + 1`` (i.e., the bond crosses a bridge
    or wraps around the ring).

    Uses the VB locant map (atom_idx -> integer_locant).

    Aromatic rings are excluded (all-aromatic bridged systems are unusual, but
    if encountered we return empty lists).  For mixed systems containing some
    aromatic and some saturated rings (e.g., ansa-macrolides where a
    macrocycle bridges a fused aromatic core), aromatic ring bonds must be
    cited as enes after Kekulization so OPSIN reproduces the correct Kekulé
    pattern; otherwise the VB skeleton is named as fully saturated.
    """
    if ring_system.aromatic:
        return [], []

    from rdkit.Chem import BondType

    ring_atom_set = ring_system.atom_indices

    # If any ring atom is aromatic, work on a Kekulized copy so aromatic bonds
    # are exposed as explicit single/double bonds.  We do this without mutating
    # the original mol or its aromatic flags.
    has_aromatic = any(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atom_set
    )
    if has_aromatic:
        from rdkit import Chem as _Chem
        work_mol = _Chem.Mol(mol)
        try:
            _Chem.Kekulize(work_mol, clearAromaticFlags=True)
        except Exception:
            work_mol = mol
    else:
        work_mol = mol

    double_bond_pairs: list[tuple[int, int]] = []
    triple_bond_pairs: list[tuple[int, int]] = []

    seen_bonds: set[tuple[int, int]] = set()
    for atom_idx in ring_atom_set:
        atom = work_mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in ring_atom_set:
                continue
            bond_key = (min(atom_idx, nb_idx), max(atom_idx, nb_idx))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            bond = work_mol.GetBondBetweenAtoms(atom_idx, nb_idx)
            if bond is None:
                continue
            bt = bond.GetBondType()
            if bt not in (BondType.DOUBLE, BondType.TRIPLE):
                continue
            l1 = locant_map.get(atom_idx)
            l2 = locant_map.get(nb_idx)
            if l1 is None or l2 is None:
                continue
            lo, hi = (l1, l2) if l1 <= l2 else (l2, l1)
            if bt == BondType.DOUBLE:
                double_bond_pairs.append((lo, hi))
            else:
                triple_bond_pairs.append((lo, hi))

    return sorted(double_bond_pairs), sorted(triple_bond_pairs)


def _format_vb_locant(pair: tuple[int, int]) -> str:
    """Format an unsaturation locant for VB nomenclature.

    Pair is ``(low, high)`` for the two atom locants of the bond.  When
    ``high == low + 1`` the bond is along the principal numbering and only
    the low locant is cited (e.g. ``"3"``).  Otherwise the bond crosses a
    bridge or wraps around the ring system, and IUPAC P-23 requires the
    disambiguated form ``"low(high)"`` (e.g. ``"19(37)"`` for a bond between
    locants 19 and 37).
    """
    lo, hi = pair
    if hi == lo + 1:
        return str(lo)
    return f"{lo}({hi})"


def _build_vb_unsaturation_suffix(
    double_bond_pairs: list[tuple[int, int]],
    triple_bond_pairs: list[tuple[int, int]],
) -> str:
    """Build the unsaturation suffix for a VB name.

    Unlike monocyclic rings, VB rings always cite locants (even for a single
    double bond).  Each pair is ``(low, high)`` and is formatted by
    :func:`_format_vb_locant`, which adds the ``(high)`` disambiguation when
    the bond does not lie on the principal numbering edge.

    Returns a string like:
    - "-2-ene"               (one double bond at locant 2)
    - "a-2,5-diene"          (two double bonds at locants 2 and 5)
    - "a-2,19(37)-diene"     (one normal, one cross-bridge ene)
    - "-2-yne"               (one triple bond at locant 2)
    - "a-2,4-diene-6-yne"    (mixed, two DBs and one triple)
    Returns "" if no unsaturation.
    """
    nd = len(double_bond_pairs)
    nt = len(triple_bond_pairs)

    if nd == 0 and nt == 0:
        return ""

    mult_d = (get_multiplier(nd, complex=False) or "") if nd > 1 else ""
    mult_t = (get_multiplier(nt, complex=False) or "") if nt > 1 else ""

    d_locs = [_format_vb_locant(p) for p in double_bond_pairs]
    t_locs = [_format_vb_locant(p) for p in triple_bond_pairs]

    if nd > 0 and nt == 0:
        # Pure double bond(s): always cite locants for VB
        if nd == 1:
            return f"-{d_locs[0]}-ene"
        else:
            loc_str = ",".join(d_locs)
            return f"a-{loc_str}-{mult_d}ene"
    elif nd == 0 and nt > 0:
        # Pure triple bond(s)
        if nt == 1:
            return f"-{t_locs[0]}-yne"
        else:
            loc_str = ",".join(t_locs)
            return f"a-{loc_str}-{mult_t}yne"
    else:
        # Mixed: double + triple
        d_loc_str = ",".join(d_locs)
        t_loc_str = ",".join(t_locs)
        if nd == 1:
            d_part = f"-{d_loc_str}-en"
        else:
            d_part = f"a-{d_loc_str}-{mult_d}en"
        if nt == 1:
            t_part = f"-{t_loc_str}-yne"
        else:
            t_part = f"-{t_loc_str}-{mult_t}yne"
        return d_part + t_part


def name_bridged(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Von Baeyer naming for bridged polycyclic ring systems.

    Returns a list with 0 or 1 NamedParent objects.
    """
    if ring_system.bridge_sizes is None:
        return []

    bridge_sizes = ring_system.bridge_sizes
    total_atoms = ring_system.ring_size

    # For heteroatom replacement, we count C atoms as the base chain length.
    # Total atoms = bridge atoms + 2 bridgeheads; the carbocycle base replaces
    # heteroatoms with C → total_atoms is the correct chain length for the stem.
    stem_base = get_chain_stem(total_atoms)
    if stem_base is None:
        logger.debug("No chain stem for %d atoms in bridged ring", total_atoms)
        return []

    # In Von Baeyer nomenclature, the prefix count is the number of rings
    # (circuit rank) of the polycyclic skeleton:
    #   bicyclo = 2 rings → main has 3 bridges, e.g. [2.2.1]
    #   tricyclo = 3 rings → main-3 + 1 secondary, e.g. [2.2.1.0^{2,6}]
    #   tetracyclo = 4 rings → main-3 + 2 secondaries
    # Legacy rank derivation: len(bridge_sizes) - 1 (number of main bridges
    # minus one).  New structured path: rank = main_bridges + secondary_bridges
    # - 1, which agrees for bicyclic and extends to higher ranks.
    n_main_bridges = len(bridge_sizes)
    n_secondary = (
        len(ring_system.secondary_bridges)
        if ring_system.secondary_bridges
        else 0
    )
    n_prefix_cycles = n_main_bridges - 1 + n_secondary
    if n_prefix_cycles < 2:
        n_prefix_cycles = 2  # Minimum is bicyclo

    prefix = _CYCLE_PREFIX.get(n_prefix_cycles)
    if prefix is None:
        # More than 5 cycles: not handled
        logger.debug("Von Baeyer not supported for %d-cycle system", n_prefix_cycles)
        return []

    # Sort main bridge sizes descending (largest bridge first) — IUPAC rule
    sorted_bridges = tuple(sorted(bridge_sizes, reverse=True))
    bridge_str = ".".join(str(b) for b in sorted_bridges)

    # Compute the BEST VB locant map: try all valid numberings and pick the one
    # that gives the lowest locant set for unsaturation (double/triple bonds),
    # then for heteroatoms.  This implements IUPAC P-23.3 unsaturation priority.
    if n_secondary > 0:
        locant_map = _choose_best_vb_locant_map_with_secondaries(
            ring_system, sorted_bridges, mol
        )
    else:
        locant_map = _choose_best_vb_locant_map(ring_system, sorted_bridges, mol)

    # Append secondary-bridge segments to the descriptor string.
    # Each secondary bridge has size k (possibly 0) and endpoints whose
    # locants are p,q (p < q, and p must be the lower-locant endpoint of the
    # bridge under the current numbering; IUPAC convention).
    if n_secondary > 0 and locant_map is not None:
        sec_segments: list[tuple[int, int, int]] = []  # (size, p, q)
        for (a, b), interior in ring_system.secondary_bridges:
            la = locant_map.get(a)
            lb = locant_map.get(b)
            if la is None or lb is None:
                logger.debug("Missing locant for secondary bridge endpoint")
                return []
            p, q = (la, lb) if la <= lb else (lb, la)
            sec_segments.append((len(interior), p, q))
        # Sort by bridge size descending, then by lowest (p, q) ascending
        sec_segments.sort(key=lambda s: (-s[0], s[1], s[2]))
        sec_str = "".join(
            f".{size}^{{{p},{q}}}" for size, p, q in sec_segments
        )
        bridge_str = bridge_str + sec_str

    # Detect unsaturation (double/triple bonds within the ring) via VB locants
    if locant_map is not None:
        dbl_locs, tri_locs = _detect_vb_unsaturation(ring_system, locant_map, mol)
    else:
        dbl_locs, tri_locs = [], []

    # Build the unsaturation suffix: "" for saturated, "-2-ene" etc. for unsaturated
    unsat_suffix = _build_vb_unsaturation_suffix(dbl_locs, tri_locs)

    # Build the base carbocycle name
    # Saturated:   bicyclo[2.2.1]heptane,  stem = bicyclo[2.2.1]heptan
    # Unsaturated: bicyclo[2.2.1]hept-2-ene, stem = bicyclo[2.2.1]hept-2-en
    vb_core = f"{prefix}[{bridge_str}]{stem_base}"  # e.g. "bicyclo[2.2.1]hept"
    if unsat_suffix:
        # Suffix starts with "-" or "a-"; append directly to stem_base (no "an" needed)
        carbocycle_name = vb_core + unsat_suffix
        # stem = carbocycle_name without terminal 'e'
        if carbocycle_name.endswith("e"):
            carbocycle_stem = carbocycle_name[:-1]
        else:
            carbocycle_stem = carbocycle_name
    else:
        carbocycle_name = vb_core + "ane"
        carbocycle_stem = vb_core + "an"

    # Heteroatom replacement nomenclature (P-31.1.3)
    heteroatoms = ring_system.heteroatoms
    if heteroatoms:
        if locant_map is not None:
            hetero_prefix = _build_heteroatom_prefix(heteroatoms, locant_map)
            if hetero_prefix:
                name_str = f"{hetero_prefix}{carbocycle_name}"
                stem = f"{hetero_prefix}{carbocycle_stem}"
            else:
                name_str = carbocycle_name
                stem = carbocycle_stem
        else:
            logger.debug("Could not compute VB numbering for %s", ring_system)
            name_str = carbocycle_name
            stem = carbocycle_stem
    else:
        name_str = carbocycle_name
        stem = carbocycle_stem

    # Method (1) is NOT applicable for bridged rings (P-29.2)
    alkyl_stem = None

    # Capture ring unsaturation bond atom-pairs so the engine can recompute
    # the embedded "-N-en-" / "-N-yn-" locant using the FINAL numbering
    # picked by the strategy layer (which may place the substituent
    # attachment at a different locant than the one baked in here — e.g.
    # bicyclo[2.2.1]hept-5-en-2-yl vs the provisional -2-en-2-yl form).
    # See _recompute_ring_unsaturation_name in engine.py.
    #
    # Restriction: only record when there is EXACTLY one double or triple
    # bond in the ring.  Multi-unsaturated bridged rings (e.g. candicidin's
    # bicyclo[33.3.1]nonatriaconta-heptaene) bake a "-N,M,...-heptaene" list
    # into the name that the single-locant recompute path cannot rewrite.
    # Populating ring_unsaturation_bonds for those would let the strategy
    # layer re-score numberings (strategy.py line 510) against locants that
    # are no longer the ones in the name string, producing substituent
    # locant drift with no compensating en-locant update.  Multi-ene VB
    # rings keep the provisional numbering chosen by
    # _choose_best_vb_locant_map (unchanged behaviour).
    from iupac_namer.ring_naming.monocyclic import get_ring_bond_pairs
    ring_bond_pairs = get_ring_bond_pairs(ring_system, mol)
    if ring_bond_pairs and len(ring_bond_pairs) > 1:
        ring_bond_pairs = ()

    # Pin the ring numbering so that substituent locants, heteroatom locants
    # (baked into name_str) and secondary-bridge superscripts (baked into the
    # descriptor) all agree.  Without this pin the strategy layer would be
    # free to pick any numbering from compute_vb_numberings() — which includes
    # numberings that violate IUPAC P-23.2.5 priority — and would then emit
    # substituent locants drawn from a DIFFERENT numbering than the one that
    # determined the descriptor and heteroatom prefix.  That inconsistency
    # produces names where (e.g.) a heteroatom locant and a substituent locant
    # collide, causing OPSIN round-trip failures.
    #
    # The pinned numbering is the IUPAC-preferred one per P-23.2.5:
    #   (a) main bridgeheads/bridges (set by decomposition)
    #   (b) heteroatoms lowest
    #   (c) unsaturation lowest
    #   (d) substituent-bearing atoms lowest
    #   (e) secondary-bridge superscripts lowest
    # — exactly the tuple scored by _choose_best_vb_locant_map_with_secondaries.
    #
    # For substituent (-yl) use the engine's free-valence handling still applies
    # because the attachment atom itself is one of the substituent-bearing
    # atoms counted in tier (d), so its locant is already minimised under the
    # IUPAC-correct numbering.
    pinned_numberings: tuple[Numbering, ...] = ()
    # Pin the IUPAC-preferred numbering in two cases:
    #
    #   (1) tricyclic+ systems, where an inconsistency between the descriptor
    #       superscripts (e.g. "0^{p,q}") and substituent locants arises; and
    #
    #   (2) heteroatom-replacement systems, where the heteroatom prefix
    #       ("4-thia-1-aza...") AND the unsaturation locant ("hept-2-ene") are
    #       baked into ``name_str`` off the ``_choose_best_vb_locant_map``
    #       numbering.  Without a pin, the strategy layer is free to re-number
    #       to minimise a principal-characteristic-group suffix (e.g. the
    #       β-lactam "-one"), which moves the double bond onto a different edge
    #       than the one named — e.g. 2,3-didehydropenam
    #       (O=C1C[C@H]2SC=CN12) was emitted "...hept-3-en-6-one" (round-trips
    #       to a hypervalent-S structure) instead of the correct
    #       "4-thia-1-azabicyclo[3.2.0]hept-2-en-7-one".  Per P-31.1.4.3.4 the
    #       skeletal-heteroatom and ring-unsaturation locants outrank the
    #       suffix, so the numbering that fixes them must be respected; the
    #       suffix then takes whatever locant remains.
    #
    # Pure carbocyclic bicyclics with no heteroatoms keep the free,
    # free-valence-driven numbering choice (needed for e.g.
    # bicyclo[2.2.1]hept-5-en-2-yl), since they carry no baked-in heteroatom
    # prefix that the suffix numbering could contradict.
    needs_hetero_pin = bool(heteroatoms) and locant_map is not None
    if locant_map is not None and (n_secondary > 0 or needs_hetero_pin):
        from iupac_namer.ring_naming.numbering import _make_numbering
        ordered_atoms = sorted(locant_map.keys(), key=lambda a: locant_map[a])
        pinned_numberings = (_make_numbering(ordered_atoms),)

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=alkyl_stem,
        naming_method="von_baeyer",
        indicated_hydrogen=None,
        numbering_options=pinned_numberings,
        ring_unsaturation_bonds=ring_bond_pairs if ring_bond_pairs else None,
    )]
