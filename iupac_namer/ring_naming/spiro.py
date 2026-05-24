"""
iupac_namer/ring_naming/spiro.py

Spiro nomenclature for spiro ring systems (P-22.3 / P-24.2 / P-24.5).

Produces names like:
  spiro[4.5]decane
  spiro[2.2]pentane
  1,4-dithia-7-azaspiro[4.4]nonane   (a-replacement for heterospiro)
  1-oxa-8-azaspiro[4.5]decane

For single-spiro-atom systems (two rings sharing exactly one atom).

IUPAC spiro numbering (P-24.2.2):
  - Start at the atom next to the spiro atom IN THE SMALLER RING.
  - Number around the smaller ring first, the spiro atom receives the locant
    immediately after the smaller ring (i.e. locant = size_small + 1).
  - Continue around the larger ring.
  - Direction is chosen to give heteroatoms the lowest locant set per
    P-31.1.4.3.4 (heteroatom replacement nomenclature in spiro names).

Heteroatom priority (P-25.3.1 / shared with bridged / HW):
  O > S > Se > Te > N > P > As > Si > Ge
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from iupac_namer.data_loader import get_chain_stem, get_hw_tables, get_multiplier
from iupac_namer.types import Locant, NamedParent, Numbering

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# Heteroatom replacement priority (P-25.3.1).  Higher number = higher priority.
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


def _find_spiro_atom(rings: tuple[frozenset[int], ...]) -> int | None:
    """For a binary spiro ring system, return the atom shared between both rings."""
    if len(rings) != 2:
        return None
    shared = rings[0] & rings[1]
    if len(shared) != 1:
        return None
    return next(iter(shared))


def _all_rings_monocyclic(rings: tuple[frozenset[int], ...], mol) -> bool:
    """True when every ring component is monocyclic (joined only by spiro
    unions — single shared atoms — never by shared edges / fusion).

    Von-Baeyer polyspiro nomenclature (P-24.2) applies only to systems whose
    components are all monocyclic.  If any two rings share two or more atoms
    (a fused / bridged union), the system is named by the P-24.5 component
    composition form instead.
    """
    n = len(rings)
    for i in range(n):
        for j in range(i + 1, n):
            shared = rings[i] & rings[j]
            if len(shared) >= 2:
                # Two shared atoms = a shared edge (fusion) or a bridge.
                return False
    return True


def _walk_ring(
    start: int, spiro: int, ring: frozenset[int], mol
) -> list[int] | None:
    """Walk a ring starting at `start` (adjacent to `spiro`), returning atoms
    in cyclic order [start, ..., last_before_spiro].

    The walk stays within `ring`, avoids the spiro atom (which is the
    terminus), and returns len(ring) - 1 atoms.  Returns None if the walk
    cannot complete.
    """
    ring_size = len(ring)
    # Atoms to visit excluding the spiro atom
    target_len = ring_size - 1
    ordered: list[int] = [start]
    visited: set[int] = {start, spiro}
    current = start
    while len(ordered) < target_len:
        moved = False
        for nb in mol.GetAtomWithIdx(current).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in ring and nb_idx != spiro and nb_idx not in visited:
                ordered.append(nb_idx)
                visited.add(nb_idx)
                current = nb_idx
                moved = True
                break
        if not moved:
            return None
    return ordered


def _enumerate_spiro_numberings(
    rings: tuple[frozenset[int], ...],
    spiro_atom: int,
    mol,
) -> list[dict[int, int]]:
    """Enumerate all valid spiro numberings as atom_idx -> 1-based locant maps.

    IUPAC P-24.2.2: the smaller ring is numbered first.  If the two rings are
    the same size, either ring may come first (both options are enumerated and
    the best is chosen on the heteroatom-locant tiebreak).
    """
    if len(rings) != 2:
        return []

    sizes = (len(rings[0]), len(rings[1]))
    # Candidate (first_ring_idx, second_ring_idx) choices
    if sizes[0] < sizes[1]:
        order_choices = [(0, 1)]
    elif sizes[1] < sizes[0]:
        order_choices = [(1, 0)]
    else:
        order_choices = [(0, 1), (1, 0)]

    maps: list[dict[int, int]] = []
    spiro_neighbors_by_ring: dict[int, list[int]] = {}
    for i, r in enumerate(rings):
        nbs: list[int] = []
        for nb in mol.GetAtomWithIdx(spiro_atom).GetNeighbors():
            if nb.GetIdx() in r and nb.GetIdx() != spiro_atom:
                nbs.append(nb.GetIdx())
        spiro_neighbors_by_ring[i] = nbs

    for first_idx, second_idx in order_choices:
        first_ring = rings[first_idx]
        second_ring = rings[second_idx]
        size_first_minus_one = len(first_ring) - 1
        # Choose each adjacent atom in the first ring as the starting atom;
        # the walk direction is then determined by the ring topology.
        for start_first in spiro_neighbors_by_ring[first_idx]:
            first_walk = _walk_ring(start_first, spiro_atom, first_ring, mol)
            if first_walk is None:
                continue
            # Spiro atom gets locant size_first + 1
            # Second ring continues numbering from locant size_first + 2.
            for start_second in spiro_neighbors_by_ring[second_idx]:
                second_walk = _walk_ring(
                    start_second, spiro_atom, second_ring, mol
                )
                if second_walk is None:
                    continue
                loc_map: dict[int, int] = {}
                for i, atom in enumerate(first_walk):
                    loc_map[atom] = i + 1
                loc_map[spiro_atom] = size_first_minus_one + 1
                base = size_first_minus_one + 2
                for i, atom in enumerate(second_walk):
                    loc_map[atom] = base + i
                maps.append(loc_map)

    return maps


def _score_numbering_for_heteroatoms(
    loc_map: dict[int, int],
    heteroatoms: tuple,  # tuple[HeteroPosition]
) -> tuple:
    """Score a spiro numbering by heteroatom locants (P-25.3.1 / P-31.1.4.3.4).

    Lower score is better.  Compare:
    1. Union locant set of all replacement-priority heteroatoms (ascending).
    2. Per-priority grouping (most-senior element locants, then next, ...).
    This ordering matches P-31.1.4 for spiro a-replacement nomenclature.
    """
    hetero_locs: list[int] = []
    by_prio: dict[int, list[int]] = {}
    for hp in heteroatoms:
        loc = loc_map.get(hp.atom_idx)
        if loc is None:
            continue
        prio = _REPLACEMENT_PRIORITY.get(hp.element, -1)
        hetero_locs.append(loc)
        by_prio.setdefault(prio, []).append(loc)

    hetero_locs.sort()
    # For priority tie-break: iterate priorities from HIGHEST to LOWEST
    # and concatenate their sorted locants.
    prio_order_locants: list[int] = []
    for prio in sorted(by_prio.keys(), reverse=True):
        prio_order_locants.extend(sorted(by_prio[prio]))

    return (tuple(hetero_locs), tuple(prio_order_locants))


def _build_spiro_heteroatom_prefix(
    heteroatoms: tuple,
    loc_map: dict[int, int],
    following_char: str,
    lambda_map: "dict[int, int] | None" = None,
) -> str | None:
    """Build the a-replacement prefix for a spiro name.

    Example: heteroatoms S@1, S@4, N@7 → "1,4-dithia-7-aza" placed before
    "spiro[4.4]nonane" to give "1,4-dithia-7-azaspiro[4.4]nonane".

    `following_char` is the first letter of the component that follows this
    prefix (usually 's' of 'spiro') — used for HW terminal-'a' elision before
    a vowel.  'spiro' starts with 's', not a vowel, so elision normally does
    not apply, but we keep the behaviour correct for safety.

    ``lambda_map`` maps a heteroatom LOCANT to its non-standard valence
    (P-14.1.1 / P-31.1.4.3).  When supplied, that locant is cited inline as
    "<loc>lambda<val>" inside the a-replacement prefix (e.g. S(VI) at locant 5
    → "5lambda6-thia"), so OPSIN can reconstruct the exact valence.

    Returns None if any element has no HW prefix.
    """
    hw_tables = get_hw_tables()
    prefixes_list = hw_tables.get("prefixes", [])
    elem_to_prefix: dict[str, str] = {}
    for entry in prefixes_list:
        elem_to_prefix[entry["element"]] = entry["prefix"]

    items: list[tuple[int, str]] = []  # (locant, element)
    for hp in heteroatoms:
        loc = loc_map.get(hp.atom_idx)
        if loc is None:
            return None
        if hp.element not in elem_to_prefix:
            return None
        items.append((loc, hp.element))

    if not items:
        return ""

    # Group by element, sort by priority (highest first) then locant
    items.sort(key=lambda x: (-_REPLACEMENT_PRIORITY.get(x[1], 0), x[0]))

    elem_order: list[str] = []
    elem_to_locs: dict[str, list[int]] = {}
    for loc, elem in items:
        if elem not in elem_to_locs:
            elem_order.append(elem)
            elem_to_locs[elem] = []
        elem_to_locs[elem].append(loc)
    for elem in elem_order:
        elem_to_locs[elem].sort()

    # IUPAC P-25.3.1.3 / P-23.2.5: locants are cited IN PRIORITY ORDER grouped
    # by element (e.g. "1,4-dithia-7-aza-" — S locants first because S has
    # priority over N in replacement nomenclature).
    raw_parts: list[tuple[list[int], str]] = []
    for elem in elem_order:
        pref = elem_to_prefix[elem]
        locs = elem_to_locs[elem]
        n_elem = len(locs)
        if n_elem == 1:
            raw_parts.append((locs, pref))
        else:
            multi = get_multiplier(n_elem)
            if multi is None:
                return None
            raw_parts.append((locs, multi + pref))

    # Build segments "<locs>-<prefix>" joined by "-", matching the bridged /
    # monocyclic a-replacement pattern.  Apply terminal-'a' elision only at the
    # VERY END (before the following parent component 'spiro').  A locant
    # carrying a non-standard valence is cited inline as "<loc>lambda<val>"
    # (P-14.1.1 / P-31.1.4.3).
    lmap = lambda_map or {}

    def _fmt_loc(loc: int) -> str:
        return f"{loc}lambda{lmap[loc]}" if loc in lmap else str(loc)

    segments: list[str] = []
    for locs, pref in raw_parts:
        loc_str = ",".join(_fmt_loc(loc) for loc in locs)
        segments.append(f"{loc_str}-{pref}")
    joined = "-".join(segments)

    # Elide terminal 'a' of the last prefix if the following character is a vowel.
    if following_char and following_char in "aeiou" and joined.endswith("a"):
        joined = joined[:-1]

    return joined


def name_spiro(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Spiro naming: spiro[X.Y]parent, with optional a-replacement for heteroatoms.

    Returns a list with 0 or 1 NamedParent objects.
    """
    if ring_system.spiro_sizes is None:
        return []

    spiro_sizes = ring_system.spiro_sizes
    total_atoms = ring_system.ring_size

    stem_base = get_chain_stem(total_atoms)
    if stem_base is None:
        logger.debug("No chain stem for %d atoms in spiro ring", total_atoms)
        # Don't return — articulation-split path below may still apply for
        # large polyspiro systems where the chain stem doesn't matter.

    # Sort spiro sizes: smaller first (IUPAC P-22.3.2)
    sorted_sizes = tuple(sorted(spiro_sizes))
    sizes_str = ".".join(str(s) for s in sorted_sizes)

    heteroatoms = ring_system.heteroatoms or ()
    rings = ring_system.rings

    # Attempt proper spiro numbering + a-replacement if this is a binary spiro
    # with heteroatoms we can express.  Falls back to the original all-carbon
    # path on any issue — we never silently drop heteroatoms; if we can't name
    # them, we refuse to emit a spiro name at all.
    hetero_prefix = ""
    best_loc_map: dict[int, int] | None = None
    numbering_options: tuple[Numbering, ...] = ()

    spiro_atom = _find_spiro_atom(rings)
    if heteroatoms:
        if spiro_atom is None:
            # Polyspiro with ≥3 rings (no single shared atom across all rings).
            # P-24.2.2/.3/.4: when every ring component is monocyclic (only
            # spiro unions, no shared edges), the PIN is the von-Baeyer
            # polyspiro form (dispiro/trispiro[a.b.c.d...]alkane with
            # a-replacement for heteroatoms).  Try that first.
            if _all_rings_monocyclic(rings, mol):
                from iupac_namer.ring_naming.polyspiro_vb import (
                    name_polyspiro_vb,
                )
                vb = name_polyspiro_vb(ring_system, candidate, mol)
                if vb:
                    return vb
            # Two or more spiro junctions joining fused/retained components in
            # an unbranched chain -> flat dispiro/trispiro[A-x,y'-B-...] form
            # (P-24.6).  Try this before the single-cut articulation split,
            # which would otherwise produce an invalid NESTED spiro name.
            mc = _try_multicomponent_spiro(ring_system, candidate, mol)
            if mc:
                return mc
            # Otherwise try the articulation-split path (P-24.5):
            #   spiro[<simpler>-N,N'-(<more-complex>)]
            # This handles e.g. spiro[[1,3]dioxolane-2,2'-decalin] where the
            # spiro atom is one articulation cut vertex of the ring graph and
            # one partner is a fused/retained ring system.
            arts = _articulation_atoms_in_ring_graph(
                ring_system.atom_indices, mol
            )
            for art in arts:
                result = _try_articulation_split_spiro(
                    ring_system, candidate, art, mol
                )
                if result:
                    return result
            logger.debug("Polyspiro heteroatom naming not implemented")
            return []  # Avoid silent heteroatom drop
        if stem_base is None:
            return []
        maps = _enumerate_spiro_numberings(rings, spiro_atom, mol)
        if not maps:
            logger.debug("Could not enumerate spiro numberings for heterospiro")
            return []
        # Pick the numbering with lowest heteroatom locants.
        scored = [(_score_numbering_for_heteroatoms(m, heteroatoms), m) for m in maps]
        scored.sort(key=lambda x: x[0])
        best_loc_map = scored[0][1]
        # Lambda convention (P-14.1.1 / P-31.1.4.3): when a skeletal heteroatom
        # — including the spiro centre itself — carries a non-standard valence
        # (e.g. S(IV)/S(VI), P(V)), cite "<loc>lambda<val>" inline in the
        # a-replacement prefix so OPSIN can reconstruct the exact valence.
        # Without it, an S(VI) spiro centre is indistinguishable from S(IV) in
        # "5-thiaspiro[4.4]nona-..." and the name fails to round-trip.
        from iupac_namer.ring_naming.monocyclic import compute_lambda_value_map
        lambda_map = compute_lambda_value_map(
            [
                (best_loc_map[hp_.atom_idx], hp_.element, hp_.atom_idx)
                for hp_ in heteroatoms
                if hp_.atom_idx in best_loc_map
            ],
            mol,
        )
        hp = _build_spiro_heteroatom_prefix(
            heteroatoms, best_loc_map, following_char="s", lambda_map=lambda_map
        )
        if hp is None:
            logger.debug("Could not build heteroatom prefix for spiro")
            return []  # Avoid silent heteroatom drop
        hetero_prefix = hp

        # Build a Numbering object so the engine uses our locants for
        # substituent placement.
        assignments = tuple(
            (atom_idx, Locant.numeric(loc_val))
            for atom_idx, loc_val in sorted(best_loc_map.items())
        )
        locant_set = tuple(Locant.numeric(i + 1) for i in range(total_atoms))
        numbering_options = (Numbering(_assignments=assignments, locant_set=locant_set),)
    else:
        # All-carbon spiro: also provide a proper numbering when we can, so
        # substituents get locants consistent with IUPAC numbering.  (Atom
        # index sorting can give weird locants for the ethyl/methyl etc.)
        if spiro_atom is None:
            # Polyspiro (3+ rings) all-carbon.  P-24.2.2/.3: when every ring
            # component is monocyclic the PIN is the von-Baeyer polyspiro form
            # (dispiro/trispiro[a.b.c.d...]alkane).  Try that first.
            if _all_rings_monocyclic(rings, mol):
                from iupac_namer.ring_naming.polyspiro_vb import (
                    name_polyspiro_vb,
                )
                vb = name_polyspiro_vb(ring_system, candidate, mol)
                if vb:
                    return vb
            # Unbranched chain of fused/retained components joined by >=2 spiro
            # atoms -> flat dispiro/trispiro[...] form (P-24.6), tried before
            # the recursive single-cut articulation split.
            mc = _try_multicomponent_spiro(ring_system, candidate, mol)
            if mc:
                return mc
            # Otherwise try the articulation-split path (P-24.5) for systems
            # where one partner is a fused/retained polycycle.
            arts = _articulation_atoms_in_ring_graph(
                ring_system.atom_indices, mol
            )
            for art in arts:
                result = _try_articulation_split_spiro(
                    ring_system, candidate, art, mol
                )
                if result:
                    return result
            # No articulation split possible — refuse rather than emit a
            # wrong spiro[X.Y] name (spiro_sizes only describes two of the
            # rings; legacy path would silently lose ring atoms).
            return []
        if spiro_atom is not None:
            maps = _enumerate_spiro_numberings(rings, spiro_atom, mol)
            if maps:
                # For all-carbon spiros, all numberings are tied on hetero score;
                # emit them all and let the engine pick lowest-substituent.
                numbering_list: list[Numbering] = []
                for m in maps:
                    assignments = tuple(
                        (atom_idx, Locant.numeric(loc_val))
                        for atom_idx, loc_val in sorted(m.items())
                    )
                    locant_set = tuple(Locant.numeric(i + 1) for i in range(total_atoms))
                    numbering_list.append(
                        Numbering(_assignments=assignments, locant_set=locant_set)
                    )
                numbering_options = tuple(numbering_list)

    # Detect endocyclic unsaturation (double/triple bonds within the spiro ring).
    # For non-aromatic spiro rings with C=N, C=C, or C#C ring bonds, we need to
    # express these as "-1-ene", "a-1,3-diene", etc. appended to the stem.
    # The locant depends on the numbering, so we store ring_unsaturation_bonds and
    # bake in a provisional suffix from best_loc_map (recomputed later if needed).
    from iupac_namer.ring_naming.monocyclic import (
        get_ring_bond_pairs,
        compute_ring_unsaturation_locants_from_numbering,
        _build_ring_unsaturation_suffix,
    )

    ring_bond_pairs = get_ring_bond_pairs(ring_system, mol)
    unsat_suffix = ""
    if ring_bond_pairs and best_loc_map is not None:
        # Compute provisional locants from the chosen numbering
        atom_to_locant_provisional: dict[int, "Locant"] = {
            atom_idx: Locant.numeric(loc_val)
            for atom_idx, loc_val in best_loc_map.items()
        }
        prov_dbl, prov_tri = compute_ring_unsaturation_locants_from_numbering(
            ring_bond_pairs, atom_to_locant_provisional
        )
        if prov_dbl or prov_tri:
            unsat_suffix = _build_ring_unsaturation_suffix(prov_dbl, total_atoms, prov_tri)
    elif ring_bond_pairs and best_loc_map is None and numbering_options:
        # All-carbon spiro: pick the first numbering for the provisional suffix
        first_nb = numbering_options[0]
        prov_dbl, prov_tri = compute_ring_unsaturation_locants_from_numbering(
            ring_bond_pairs, first_nb.atom_to_locant
        )
        if prov_dbl or prov_tri:
            unsat_suffix = _build_ring_unsaturation_suffix(prov_dbl, total_atoms, prov_tri)

    # Build final name: insert unsaturation suffix between stem_base and "ane"
    # e.g. "1,3-diazaspiro[4.4]non" + "-1-ene" -> "1,3-diazaspiro[4.4]non-1-ene"
    # or   "1,3-diazaspiro[4.4]nona-1,3-diene"
    if unsat_suffix:
        name_str = f"{hetero_prefix}spiro[{sizes_str}]{stem_base}{unsat_suffix}"
        stem = name_str[:-1] if name_str.endswith("e") else name_str
    else:
        name_str = f"{hetero_prefix}spiro[{sizes_str}]{stem_base}ane"
        stem = f"{hetero_prefix}spiro[{sizes_str}]{stem_base}an"

    # Method (1) NOT applicable for spiro rings
    alkyl_stem = None

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=alkyl_stem,
        naming_method="spiro_systematic",
        indicated_hydrogen=None,
        numbering_options=numbering_options,
        ring_unsaturation_bonds=ring_bond_pairs if ring_bond_pairs else None,
    )]


# ---------------------------------------------------------------------------
# Articulation-split polyspiro  (P-24.5)
# ---------------------------------------------------------------------------
# Form: spiro[<smaller>-N,N'-<larger>]
# where the system has 3+ rings classified as "spiro" by perception, and the
# topology is two ring sub-systems joined at exactly ONE shared (articulation)
# atom — the spiro atom.  One side is named recursively (typically a fused
# polycycle or a heterocyclic monocycle) and the other side is named recursively
# (typically a carbocyclic monocycle).
#
# Worked example: 1,3-dioxolane spiro to decalin
#   structure : two fused-decalin rings + a 1,3-dioxolane sharing ONE atom
#               with one of the decalin rings (not at a ring-junction).
#   perception: type="spiro", 3 rings, no atom shared by all three.
#   name      : spiro[[1,3]dioxolane-2,2'-decalin]
#                 ^^^^^^^^^^^^^^^^ smaller side, unprimed
#                 ^^^^^^^ from the recursive monocyclic naming
#                                ^^^^ from the recursive fused/retained naming
#                                       ^^^ primed locant on the larger side
#
# Scope: the spiro atom must be carbon (no a-replacement on the spiro atom
# itself).  Heteroatoms in the side sub-mols are handled natively by the
# recursive sub-naming.

def _articulation_atoms_in_ring_graph(
    atom_indices: frozenset[int], mol
) -> list[int]:
    """Articulation atoms of the ring-atom subgraph (alias for the
    pre-existing helper used by ``name_polycyclic_spiro``).
    """
    return _ring_graph_articulation_atoms(atom_indices, mol)


# ---------------------------------------------------------------------------
# Flat multi-component polyspiro  (P-24.6)
# ---------------------------------------------------------------------------
# Form: dispiro[A-l1,l2'-B-l3',l4''-C]  (and trispiro, ...) — an unbranched
# CHAIN of ring-system components joined by spiro atoms, where one or more
# component is polycyclic (fused/retained) so the von-Baeyer descriptor form
# (P-24.2, all-monocyclic) does not apply.
#
# Worked example: dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]
#   topology  : fluorene --(spiro@9)-- cyclohexane --(spiro@4')-- indene
#   the two true spiro atoms are degree-4 carbons that each join two distinct
#   ring-system components.  Removing them decomposes the system into a linear
#   chain of three components.
#
# This replaces the previous recursive articulation split, which produced an
# invalid NESTED ``spiro[A-spiro[B-C]]`` name that OPSIN cannot parse.

def _true_spiro_atoms(atom_indices: frozenset[int], mol) -> list[int]:
    """Degree-4 ring articulation atoms — the spiro junction atoms.

    A spiro atom belongs to exactly two rings and contributes exactly one atom
    to each; in the ring subgraph it has four in-ring neighbours and its
    removal disconnects the subgraph.
    """
    ring_set = set(atom_indices)
    arts = set(_ring_graph_articulation_atoms(atom_indices, mol))
    result: list[int] = []
    for a in atom_indices:
        if a not in arts:
            continue
        atom = mol.GetAtomWithIdx(a)
        in_ring_nbrs = [
            nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() in ring_set
        ]
        if len(in_ring_nbrs) == 4:
            result.append(a)
    return sorted(result)


def _decompose_spiro_components(
    ring_system: "RingSystem", spiro_atoms: list[int], mol
):
    """Decompose a multi-spiro ring system into its ring-system components.

    A component = a maximal set of SSSR rings connected by FUSION (shared
    edges / >= 2 shared atoms).  Rings that share only a single spiro atom are
    placed in different components.  This keeps a monocyclic ring that carries
    two spiro atoms (e.g. the central cyclohexane in
    ``dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]``) as ONE component.

    Returns a list of (component_atoms, incident_spiro_atoms) where
    ``component_atoms`` is the frozenset of NON-spiro atoms of the component
    and ``incident_spiro_atoms`` are the spiro atoms it touches.
    """
    rings = list(ring_system.rings)
    n = len(rings)
    if n == 0:
        return None
    # Union-find over rings, joining rings that share an edge (fusion).
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if len(rings[i] & rings[j]) >= 2:
                union(i, j)

    # Group rings by component root; component atoms = union of ring atoms.
    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).update(rings[i])

    spiro_set = set(spiro_atoms)
    components: list[tuple[frozenset[int], list[int]]] = []
    for atoms in groups.values():
        non_spiro = frozenset(atoms - spiro_set)
        incident = [sp for sp in spiro_atoms if sp in atoms]
        components.append((non_spiro, incident))
    return components


def _try_multicomponent_spiro(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Name an unbranched polyspiro CHAIN of ring-system components (P-24.6).

    Each component is named recursively (retained/fused/monocyclic) and the
    components are composed as a flat ``dispiro[A-l,l'-B-l',l''-C]`` chain.
    Returns [] when the topology is not a clean linear chain of >= 3
    components joined by neutral-carbon spiro atoms.
    """
    spiro_atoms = _true_spiro_atoms(ring_system.atom_indices, mol)
    if len(spiro_atoms) < 2:
        return []  # < 2 spiro atoms -> not polyspiro chain (monospiro / binary)

    # All spiro atoms must be neutral carbon (a-replacement on the spiro atom
    # itself is not expressed by this composition form).
    for sp in spiro_atoms:
        a = mol.GetAtomWithIdx(sp)
        if a.GetAtomicNum() != 6 or a.GetFormalCharge() != 0:
            return []

    comps = _decompose_spiro_components(
        ring_system, spiro_atoms, mol
    )
    if not comps:
        return []
    # Number of components must be number_of_spiro_atoms + 1 for a linear chain.
    if len(comps) != len(spiro_atoms) + 1:
        return []  # branched or cyclic spiro arrangement — not handled here

    # Build component adjacency: each spiro atom links exactly two components.
    sp_to_comps: dict[int, list[int]] = {sp: [] for sp in spiro_atoms}
    for ci, (_, incident) in enumerate(comps):
        for sp in incident:
            sp_to_comps[sp].append(ci)
    for sp, cs in sp_to_comps.items():
        if len(cs) != 2:
            return []  # a spiro atom must join exactly two components

    # Degree of each component in the component graph.
    comp_degree = [0] * len(comps)
    for sp, cs in sp_to_comps.items():
        for ci in cs:
            comp_degree[ci] += 1
    # Linear chain: exactly two terminal components (degree 1), rest degree 2.
    terminals = [ci for ci, d in enumerate(comp_degree) if d == 1]
    if len(terminals) != 2:
        return []
    if any(d > 2 for d in comp_degree):
        return []  # branched

    # Order the chain from one terminal to the other.
    chain_order: list[int] = []
    chain_spiros: list[int] = []  # spiro atom between chain_order[i] and [i+1]
    prev_sp: int | None = None
    current = terminals[0]
    used_sp: set[int] = set()
    while True:
        chain_order.append(current)
        # Find the next spiro atom from this component (not the one we arrived by).
        next_sp = None
        for sp in spiro_atoms:
            if sp in used_sp:
                continue
            if current in sp_to_comps[sp]:
                next_sp = sp
                break
        if next_sp is None:
            break
        used_sp.add(next_sp)
        chain_spiros.append(next_sp)
        # The other component on this spiro atom.
        a, b = sp_to_comps[next_sp]
        current = b if a == current else a
    if len(chain_order) != len(comps):
        return []

    # Name each component sub-mol, recording the spiro-atom locants used.
    # Each component sub-mol includes its incident spiro atom(s).
    named_components: list[dict] = []
    for ci in chain_order:
        comp_atoms, incident = comps[ci]
        # incident spiro atoms in chain context (could be 1 for terminals, 2
        # for internal components).
        inc_in_chain = [sp for sp in chain_spiros if ci in sp_to_comps[sp]]
        full_atoms = frozenset(comp_atoms) | set(inc_in_chain)
        sub = _build_partner_submol(comp_atoms, inc_in_chain[0], mol) if inc_in_chain else None
        # Build sub-mol including ALL incident spiro atoms for this component.
        sub_pack = _build_component_submol(comp_atoms, inc_in_chain, mol)
        if sub_pack is None:
            return []
        sub_mol, sub_to_full = sub_pack
        named = _name_partner(sub_mol)
        if not named:
            return []
        named_components.append({
            "ci": ci,
            "comp_atoms": comp_atoms,
            "incident": inc_in_chain,
            "sub_mol": sub_mol,
            "sub_to_full": sub_to_full,
            "named": named,
        })

    # Pick best NamedParent + locants for each component (lowest spiro-atom
    # locant per P-24.6 / P-24.5.2).  For internal components we need BOTH
    # incident spiro atoms' locants from the SAME numbering.
    result = _compose_multicomponent_name(
        ring_system, candidate, named_components, chain_spiros, mol
    )
    return result


def _build_component_submol(
    comp_atoms: frozenset[int], spiro_atoms_inc: list[int], mol
):
    """Build a sub-mol from ``comp_atoms`` plus all its incident spiro atoms."""
    full_atoms = sorted(set(comp_atoms) | set(spiro_atoms_inc))
    rw = Chem.RWMol()
    full_to_sub: dict[int, int] = {}
    for old in full_atoms:
        orig = mol.GetAtomWithIdx(old)
        new_atom = Chem.Atom(orig.GetAtomicNum())
        if orig.GetFormalCharge():
            new_atom.SetFormalCharge(orig.GetFormalCharge())
        full_to_sub[old] = rw.AddAtom(new_atom)
    added: set[frozenset[int]] = set()
    for old in full_atoms:
        for nb in mol.GetAtomWithIdx(old).GetNeighbors():
            j = nb.GetIdx()
            if j not in full_to_sub:
                continue
            key = frozenset({old, j})
            if key in added:
                continue
            added.add(key)
            bond = mol.GetBondBetweenAtoms(old, j)
            if bond is None:
                continue
            rw.AddBond(full_to_sub[old], full_to_sub[j], bond.GetBondType())
    sub_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(sub_mol)
    except Exception:
        return None
    sub_to_full = {v: k for k, v in full_to_sub.items()}
    return sub_mol, sub_to_full


def _component_name_priority(np) -> int:
    """Naming-method preference for a component (lower = preferred).

    P-24.6 cites each component by its own PIN, which prefers a retained /
    fused ring name (e.g. ``fluorene``, ``1H-indene``) over the systematic
    von-Baeyer ``tricyclo[...]`` form.
    """
    method = getattr(np, "naming_method", "") or ""
    if method == "retained":
        return 0
    if method.startswith("fused"):
        return 1
    if method == "von_baeyer":
        return 3
    return 2


def _spiro_locants_for_component(comp_pack: dict, spiro_atoms_inc: list[int]):
    """Return (chosen_named_parent, ring_system, {full_spiro_idx: locant})
    for one component.

    Selection order: (1) prefer the component's own PIN naming method
    (retained/fused over systematic VB, P-24.6); (2) among equal methods,
    lowest spiro-atom locant set (P-24.5.2).  Returns None on failure."""
    full_to_sub = {v: k for k, v in comp_pack["sub_to_full"].items()}
    sub_spiros = [full_to_sub[s] for s in spiro_atoms_inc if s in full_to_sub]
    if len(sub_spiros) != len(spiro_atoms_inc):
        return None

    best = None
    best_key = None
    for np, rs in comp_pack["named"]:
        method_rank = _component_name_priority(np)
        # Try numbering options exposed by the named parent.
        nbs = list(np.numbering_options)
        if not nbs:
            # Monocyclic / no exposed numbering: walk-based fallback.
            locs = _component_fallback_locants(rs, sub_spiros, comp_pack["sub_mol"])
            if locs is None:
                continue
            sp_locs = {
                comp_pack["sub_to_full"][s]: locs[s] for s in sub_spiros
            }
            key = (method_rank, tuple(sorted(sp_locs.values())))
            if best_key is None or key < best_key:
                best_key = key
                best = (np, rs, sp_locs)
            continue
        for nb in nbs:
            a2l = nb.atom_to_locant
            sp_locs: dict[int, int] = {}
            ok = True
            for s in sub_spiros:
                loc = a2l.get(s)
                if loc is None or not loc.is_numeric or loc._numeric_value is None:
                    ok = False
                    break
                sp_locs[comp_pack["sub_to_full"][s]] = loc._numeric_value
            if not ok:
                continue
            key = (method_rank, tuple(sorted(sp_locs.values())))
            if best_key is None or key < best_key:
                best_key = key
                best = (np, rs, sp_locs)
    return best


def _component_fallback_locants(rs, sub_spiros: list[int], sub_mol):
    """Locants for a monocyclic component whose named parent exposes no
    numbering options.  For a carbocycle the spiro atom is locant 1; for an
    HW heterocycle use the heteroatom-determined numbering."""
    if len(sub_spiros) == 1:
        if rs.heteroatoms:
            return _compute_monocyclic_hw_locants_with_spiro(
                rs.atom_indices, sub_spiros[0], sub_mol
            )
        return _walk_monocyclic_locants_from_spiro(
            rs.atom_indices, sub_spiros[0], sub_mol
        )
    # Internal monocyclic component with two spiro atoms (e.g. the central
    # cyclohexane in dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]): walk
    # the ring so one spiro atom is locant 1 and find the other's locant; try
    # both spiro atoms as the start and keep the lowest locant set.
    best = None
    best_key = None
    for start in sub_spiros:
        locs = _walk_monocyclic_locants_from_spiro(rs.atom_indices, start, sub_mol)
        if locs is None:
            continue
        key = tuple(sorted(locs[s] for s in sub_spiros))
        if best_key is None or key < best_key:
            best_key = key
            best = locs
    return best


def _compose_multicomponent_name(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    named_components: list[dict],
    chain_spiros: list[int],
    mol,
) -> list[NamedParent]:
    """Compose the flat dispiro/trispiro[...] name from the ordered chain."""
    n_comp = len(named_components)
    # Resolve each component's chosen name + spiro-atom locants.
    resolved: list[dict] = []
    for pack in named_components:
        sel = _spiro_locants_for_component(pack, pack["incident"])
        if sel is None:
            return []
        np, rs, sp_locs = sel
        resolved.append({
            "name": np.name,
            "rs": rs,
            "sp_locs": sp_locs,   # full_spiro_idx -> locant within this component
            "comp_atoms": pack["comp_atoms"],
        })

    # Assign prime levels: component 0 unprimed, component 1 primed ('),
    # component 2 double-primed (''), ...
    def primes(level: int) -> str:
        return "'" * level

    # Build the bracket body: for each junction i (chain_spiros[i]) between
    # component i and i+1, cite the spiro atom locant in component i (with i
    # primes) and in component i+1 (with i+1 primes).
    parts: list[str] = []
    parts.append(_format_smaller_partner_name(resolved[0]["name"]))
    for i, sp in enumerate(chain_spiros):
        loc_left = resolved[i]["sp_locs"].get(sp)
        loc_right = resolved[i + 1]["sp_locs"].get(sp)
        if loc_left is None or loc_right is None:
            return []
        left_label = f"{loc_left}{primes(i)}"
        right_label = f"{loc_right}{primes(i + 1)}"
        parts.append(f"{left_label},{right_label}")
        parts.append(_format_smaller_partner_name(resolved[i + 1]["name"]))

    # Stitch: name0-L,R'-name1-L',R''-name2 ...
    body = parts[0]
    idx = 1
    for i in range(len(chain_spiros)):
        body += f"-{parts[idx]}-{parts[idx + 1]}"
        idx += 2

    spiro_word = {2: "dispiro", 3: "trispiro", 4: "tetraspiro",
                  5: "pentaspiro"}.get(len(chain_spiros))
    if spiro_word is None:
        return []
    name_str = f"{spiro_word}[{body}]"
    stem = name_str[:-1] if name_str.endswith("e") else name_str

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=None,
        naming_method="polyspiro_multicomponent",
        indicated_hydrogen=None,
        numbering_options=(),
    )]


def _split_ring_atoms_at(
    ring_atom_indices: frozenset[int], art_atom: int, mol
) -> list[frozenset[int]] | None:
    """Split the ring atom subgraph into connected components after removing
    ``art_atom``.  Returns the list of components (each a frozenset of full-mol
    atom indices, NOT including ``art_atom``).
    """
    remaining = set(ring_atom_indices) - {art_atom}
    if not remaining:
        return None
    components: list[set[int]] = []
    visited: set[int] = set()
    for seed in list(remaining):
        if seed in visited:
            continue
        comp: set[int] = set()
        stack = [seed]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            visited.add(u)
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining and j not in comp:
                    stack.append(j)
        components.append(comp)
    return [frozenset(c) for c in components]


def _build_partner_submol(
    partner_atoms: frozenset[int], art_atom: int, mol
) -> tuple[Chem.Mol, dict[int, int]] | None:
    """Build a sub-mol containing ``partner_atoms`` plus ``art_atom``.

    The articulation atom is included as a regular carbon (its original
    element/charge is preserved if it is carbon; non-carbon spiro atoms are
    not handled here — caller filters those out).

    Returns (sub_mol, sub_to_full) where sub_to_full maps sub-mol atom
    indices back to the full-mol atom indices.  Returns None on sanitization
    failure.
    """
    full_atoms = sorted(set(partner_atoms) | {art_atom})
    rw = Chem.RWMol()
    full_to_sub: dict[int, int] = {}
    for old in full_atoms:
        orig = mol.GetAtomWithIdx(old)
        new_atom = Chem.Atom(orig.GetAtomicNum())
        if orig.GetFormalCharge():
            new_atom.SetFormalCharge(orig.GetFormalCharge())
        full_to_sub[old] = rw.AddAtom(new_atom)
    added: set[frozenset[int]] = set()
    for old in full_atoms:
        for nb in mol.GetAtomWithIdx(old).GetNeighbors():
            j = nb.GetIdx()
            if j not in full_to_sub:
                continue
            key = frozenset({old, j})
            if key in added:
                continue
            added.add(key)
            bond = mol.GetBondBetweenAtoms(old, j)
            if bond is None:
                continue
            rw.AddBond(full_to_sub[old], full_to_sub[j], bond.GetBondType())
    sub_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(sub_mol)
    except Exception:
        return None
    sub_to_full = {v: k for k, v in full_to_sub.items()}
    return sub_mol, sub_to_full


def _name_partner(sub_mol: Chem.Mol):
    """Run the ring-naming dispatcher on a partner sub-mol.

    Returns a list of (NamedParent, RingSystem) pairs from the first ring
    system in the sub-mol, or [] on failure.
    """
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis
    from iupac_namer.ring_naming import name_ring_system
    from iupac_namer.types import CandidateParent as CP

    try:
        aa = AtomAnalysis(sub_mol)
        ra = RingAnalysis(sub_mol, aa)
    except Exception:
        return []
    if not ra.ring_systems:
        return []
    rs = ra.ring_systems[0]
    cand = CP(
        atom_indices=rs.atom_indices,
        type=rs.type,
        length=rs.ring_size,
        ring_system=rs,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )
    nps = name_ring_system(cand, sub_mol)
    return [(np, rs) for np in nps]


def _format_smaller_partner_name(name: str) -> str:
    """Wrap a monocyclic HW partner name in brackets if it starts with a
    locant block (e.g. ``1,3-dioxolane`` → ``[1,3]dioxolane``).

    OPSIN convention: ``spiro[[1,3]dioxolane-...]`` requires the dioxolane to
    be bracketed so the leading ``1,3`` locants are scoped to the partner
    rather than read as fusion locants.  This applies ONLY to simple
    monocyclic HW names — polycyclic VB names (``16,18-dioxapentacyclo[...]``)
    and fused names (``1H-indene``) already have their own brackets or
    disambiguators and must be passed through unchanged.

    Heuristic: wrap only when the name's leading ``<digits>(,<digits>)*-``
    block is followed by a token with NO further ``[`` character (i.e., a
    simple HW stem like ``dioxolane`` or ``thiazole``).
    """
    if not name:
        return name
    if name[0] == "[":
        return name
    # Detect leading "<digits>(,<digits>)*-" pattern
    i = 0
    has_digit = False
    while i < len(name) and (name[i].isdigit() or name[i] == ","):
        if name[i].isdigit():
            has_digit = True
        i += 1
    if has_digit and i < len(name) and name[i] == "-":
        rest = name[i + 1:]
        # Only bracket when the rest is a simple HW stem (no further brackets
        # indicating polycyclic VB / fused notation).
        if "[" not in rest:
            # The leading locants must scope to the partner stem.  When the
            # immediate next token is a hydro-prefix (``dihydro``, ``tetrahydro``,
            # ``hexahydro``, etc.) the locants belong to that prefix, not to
            # the ring stem — bracketing them produces malformed names like
            # ``[1,3]dihydro-2-benzofuran`` that OPSIN cannot parse.  Skip
            # bracketing for hydro-prefix forms; OPSIN accepts the bare
            # ``1,3-dihydro-2-benzofuran-1,9'-xanthene`` form inside spiro[].
            import re as _re_hyd
            if _re_hyd.match(r"^(?:di|tetra|hexa|octa|deca|dodeca)?hydro-", rest):
                return name
            locants_part = name[:i]
            return f"[{locants_part}]{rest}"
    return name


def _spiro_atom_locant_in_partner(
    sub_art_idx: int, named: NamedParent
) -> int | None:
    """Find the LOWEST numeric locant assigned to ``sub_art_idx`` across all
    numbering options of ``named``.  Returns None if not numeric / not found.

    IUPAC P-24.5.2: when the polycyclic partner has multiple valid numberings,
    choose the one that gives the spiro atom the lowest locant.
    """
    best: int | None = None
    for nb in named.numbering_options:
        loc = nb.atom_to_locant.get(sub_art_idx)
        if loc is None:
            continue
        if loc.is_numeric and loc._numeric_value is not None:
            if best is None or loc._numeric_value < best:
                best = loc._numeric_value
    return best


def _best_numbering_option(
    sub_art_idx: int, named: NamedParent
):
    """Return the numbering option whose locant for ``sub_art_idx`` is lowest.

    Breaks ties by first occurrence.  Returns None if no option gives
    ``sub_art_idx`` a numeric locant.
    """
    best = None
    best_loc = None
    for nb in named.numbering_options:
        loc = nb.atom_to_locant.get(sub_art_idx)
        if loc is None or not loc.is_numeric or loc._numeric_value is None:
            continue
        if best_loc is None or loc._numeric_value < best_loc:
            best_loc = loc._numeric_value
            best = nb
    return best


def _walk_monocyclic_locants_from_spiro(
    ring_atoms: frozenset[int], spiro_atom: int, sub_mol
) -> dict[int, int] | None:
    """For a plain carbocyclic monocycle, walk from the spiro atom and assign
    locant 1 to it, then 2, 3, ... around the ring.  Returns sub_mol atom
    index → locant.
    """
    atoms_set = set(ring_atoms) | {spiro_atom}
    cycle: list[int] = [spiro_atom]
    visited = {spiro_atom}
    current = spiro_atom
    while len(cycle) < len(atoms_set):
        moved = False
        for nb in sub_mol.GetAtomWithIdx(current).GetNeighbors():
            j = nb.GetIdx()
            if j in atoms_set and j not in visited:
                cycle.append(j)
                visited.add(j)
                current = j
                moved = True
                break
        if not moved:
            return None
    if len(cycle) != len(atoms_set):
        return None
    return {idx: (i + 1) for i, idx in enumerate(cycle)}


def _compute_monocyclic_hw_locants_with_spiro(
    ring_atoms: frozenset[int], spiro_atom: int, sub_mol
) -> dict[int, int] | None:
    """Compute Hantzsch-Widman locants for a monocyclic ring in sub_mol,
    returning a mapping of sub-mol atom index → locant.

    HW ring numbering: the most senior heteroatom gets locant 1; direction is
    chosen to give heteroatoms the lowest locant set.  If no heteroatoms,
    return None (caller should use atom-index-based cyclo numbering).

    This is a local inline copy of the HW locant logic used to get a numbering
    when the monocyclic naming path doesn't expose one for HW names (see
    monocyclic.py ``_compute_hw_locants``).
    """
    # Build ring cycle starting from spiro atom, walking the ring.
    atoms_set = set(ring_atoms) | {spiro_atom}
    cycle: list[int] = [spiro_atom]
    visited = {spiro_atom}
    current = spiro_atom
    while len(cycle) < len(atoms_set):
        moved = False
        for nb in sub_mol.GetAtomWithIdx(current).GetNeighbors():
            j = nb.GetIdx()
            if j in atoms_set and j not in visited:
                cycle.append(j)
                visited.add(j)
                current = j
                moved = True
                break
        if not moved:
            return None
    if len(cycle) != len(atoms_set):
        return None

    # Find heteroatom positions.
    _HW_PRIORITY = {"O": 8, "S": 7, "Se": 6, "Te": 5, "N": 4, "P": 3, "As": 2, "Si": 1, "Ge": 0}
    hetero_positions: list[tuple[int, int]] = []  # (cycle_pos, priority)
    for pos, idx in enumerate(cycle):
        atom = sub_mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in _HW_PRIORITY:
            hetero_positions.append((pos, _HW_PRIORITY[sym]))
    if not hetero_positions:
        return None

    n = len(cycle)
    # Try every starting position adjacent to a senior heteroatom, both directions.
    # Choose the numbering giving the lowest heteroatom locant set.
    best_assignment: dict[int, int] | None = None
    best_key: tuple | None = None
    # Pick the highest-priority heteroatom; it must get locant 1.
    max_prio = max(p for _, p in hetero_positions)
    senior_positions = [pos for pos, p in hetero_positions if p == max_prio]
    for start in senior_positions:
        for forward in (True, False):
            locs: dict[int, int] = {}
            for step in range(n):
                if forward:
                    p = (start + step) % n
                else:
                    p = (start - step) % n
                locs[cycle[p]] = step + 1
            # Score: locants of all heteroatoms (ascending)
            hlocs = sorted(locs[idx] for idx, _ in [(cycle[pos], None) for pos, _ in hetero_positions])
            key = tuple(hlocs)
            if best_key is None or key < best_key:
                best_key = key
                best_assignment = locs
    return best_assignment


def _classify_partner_complexity(rs) -> tuple:
    """Score a partner for citation order.

    IUPAC P-24.5.1 cites the heterocyclic / smaller / fused-priority partner
    FIRST.  Lower score = cited first (smaller / unprimed side).

    Returns a tuple usable as a sort key:
        (heterocyclic_first, ring_count, ring_size, type_rank)
        - heterocyclic_first: 0 if has heteroatoms else 1 (heterocycles cited first)
        - ring_count, ring_size: smaller goes first on ties
        - type_rank: monocyclic < fused < bridged
    """
    has_hetero = bool(rs.heteroatoms)
    type_rank = {"monocyclic": 0, "fused": 1, "bridged": 2, "spiro": 3}.get(rs.type, 4)
    return (
        0 if has_hetero else 1,
        len(rs.rings),
        rs.ring_size,
        type_rank,
    )


def _try_articulation_split_spiro(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    art_atom: int,
    mol,
) -> list[NamedParent]:
    """Try to name a polyspiro system by splitting at ``art_atom``.

    Returns a single-element NamedParent list on success, or [] on failure.
    """
    art = mol.GetAtomWithIdx(art_atom)
    # Spiro atom must be carbon — heteroatom spiro atoms would need
    # a-replacement which we don't express here.
    if art.GetAtomicNum() != 6 or art.GetFormalCharge() != 0:
        return []

    components = _split_ring_atoms_at(
        ring_system.atom_indices, art_atom, mol
    )
    if not components or len(components) != 2:
        return []
    comp_a, comp_b = components

    # Each side must include at least one ring (≥2 ring atoms after split).
    if len(comp_a) < 2 or len(comp_b) < 2:
        return []

    # Build partner sub-mols and name each via recursion.
    pa = _build_partner_submol(comp_a, art_atom, mol)
    pb = _build_partner_submol(comp_b, art_atom, mol)
    if pa is None or pb is None:
        return []
    sub_a, map_a = pa
    sub_b, map_b = pb

    named_a = _name_partner(sub_a)
    named_b = _name_partner(sub_b)
    if not named_a or not named_b:
        return []

    # Find the articulation atom's index within each sub-mol.
    sub_art_a = None
    for sub_idx, full_idx in map_a.items():
        if full_idx == art_atom:
            sub_art_a = sub_idx
            break
    sub_art_b = None
    for sub_idx, full_idx in map_b.items():
        if full_idx == art_atom:
            sub_art_b = sub_idx
            break
    if sub_art_a is None or sub_art_b is None:
        return []

    # Pick best (NamedParent, RingSystem) for each side: the one whose
    # numbering gives the lowest spiro-atom locant (P-24.5.2).  When a
    # candidate NamedParent does not expose numbering options (e.g. the
    # monocyclic HW naming path), compute locants inline from HW rules.
    def _best_for_side(named_list, sub_art_idx, sub_mol):
        best = None
        best_loc = None
        best_fallback_locs = None
        for np, rs in named_list:
            loc = _spiro_atom_locant_in_partner(sub_art_idx, np)
            fallback_locs = None
            if loc is None and rs.type == "monocyclic":
                if rs.heteroatoms:
                    fallback_locs = _compute_monocyclic_hw_locants_with_spiro(
                        rs.atom_indices, sub_art_idx, sub_mol
                    )
                else:
                    # Plain carbocyclic monocycle (e.g. cyclopentane): the
                    # spiro atom always gets locant 1 by P-24.5.2 (it is the
                    # starting point for numbering the monocyclic side).
                    fallback_locs = _walk_monocyclic_locants_from_spiro(
                        rs.atom_indices, sub_art_idx, sub_mol
                    )
                if fallback_locs is not None:
                    loc = fallback_locs.get(sub_art_idx)
            elif loc is None and rs.type == "bridged":
                # Phase 11 spiro-oxathiolane: ``name_bridged`` only pins
                # numbering_options when the system has secondary bridges
                # (tricyclic+).  Bicyclic VB partners (e.g.
                # ``1-azabicyclo[2.2.2]octane`` = quinuclidine) emit a
                # NamedParent with empty ``numbering_options`` and rely on
                # the strategy layer to enumerate VB numberings.  Inside
                # the spiro path we need the spiro atom's locant NOW to
                # compose the bracketed descriptor, so fall back to
                # ``compute_vb_numberings`` and pick a VB numbering whose
                # heteroatom locants AGREE with the partner name's baked-in
                # prefix (e.g. "1-aza"), then pick the one giving the
                # spiro atom the lowest locant per P-24.5.2.
                from iupac_namer.ring_naming.bridged import compute_vb_numberings
                vb_nbs = compute_vb_numberings(
                    rs, rs.bridge_sizes or (), sub_mol
                )
                if vb_nbs:
                    # Parse the heteroatom locants from the partner name
                    # ("1-aza" → 1, "2,6-dioxa" → 2, 6) so we can constrain
                    # VB numbering selection to numberings consistent with
                    # what the partner stem already names.
                    import re as _re_p11
                    name_hetero_locants: set[int] = set()
                    for m in _re_p11.finditer(
                        r"(\d+(?:,\d+)*)-(?:di|tri|tetra)?(?:aza|oxa|thia|sila|phospha|selena|tellura|stiba|bisma|germa|stanna|plumba|borata)",
                        np.name,
                    ):
                        for s in m.group(1).split(","):
                            try:
                                name_hetero_locants.add(int(s))
                            except ValueError:
                                pass
                    # Filter VB numberings: keep those whose heteroatom locant
                    # set equals the partner-name-stated heteroatom locant set
                    # (when the partner name encodes any heteroatom locants).
                    partner_hetero_atoms = [
                        hp.atom_idx for hp in (rs.heteroatoms or ())
                    ]
                    filtered_nbs = []
                    for nb in vb_nbs:
                        if not name_hetero_locants:
                            filtered_nbs.append(nb)
                            continue
                        # Extract heteroatom locants under this numbering.
                        h_locs_set: set[int] = set()
                        ok = True
                        for ha in partner_hetero_atoms:
                            hl = nb.atom_to_locant.get(ha)
                            if (hl is None or not hl.is_numeric
                                    or hl._numeric_value is None):
                                ok = False
                                break
                            h_locs_set.add(hl._numeric_value)
                        if ok and h_locs_set == name_hetero_locants:
                            filtered_nbs.append(nb)
                    if not filtered_nbs:
                        filtered_nbs = list(vb_nbs)
                    # Among consistent numberings pick the one giving the
                    # spiro atom the lowest locant (P-24.5.2).
                    bridged_best_loc: int | None = None
                    bridged_best_locs: dict[int, int] | None = None
                    for nb in filtered_nbs:
                        nb_loc = nb.atom_to_locant.get(sub_art_idx)
                        if (nb_loc is None or not nb_loc.is_numeric
                                or nb_loc._numeric_value is None):
                            continue
                        cand = nb_loc._numeric_value
                        if bridged_best_loc is None or cand < bridged_best_loc:
                            bridged_best_loc = cand
                            bridged_best_locs = {
                                a: l._numeric_value
                                for a, l in nb.atom_to_locant.items()
                                if l.is_numeric and l._numeric_value is not None
                            }
                    if bridged_best_locs is not None:
                        fallback_locs = bridged_best_locs
                        loc = bridged_best_loc
            if loc is None:
                continue
            if best_loc is None or loc < best_loc:
                best = (np, rs)
                best_loc = loc
                best_fallback_locs = fallback_locs
        return best, best_loc, best_fallback_locs

    best_a, loc_a, fallback_a = _best_for_side(named_a, sub_art_a, sub_a)
    best_b, loc_b, fallback_b = _best_for_side(named_b, sub_art_b, sub_b)
    if best_a is None or best_b is None:
        return []
    np_a, rs_a = best_a
    np_b, rs_b = best_b

    # Decide citation order: smaller / heterocyclic side cited first (unprimed),
    # the larger / carbocyclic side cited second (primed).
    score_a = _classify_partner_complexity(rs_a)
    score_b = _classify_partner_complexity(rs_b)
    if score_a <= score_b:
        first_np, first_rs, first_loc, first_map, first_sub_art, first_fb = np_a, rs_a, loc_a, map_a, sub_art_a, fallback_a
        second_np, second_rs, second_loc, second_map, second_sub_art, second_fb = np_b, rs_b, loc_b, map_b, sub_art_b, fallback_b
    else:
        first_np, first_rs, first_loc, first_map, first_sub_art, first_fb = np_b, rs_b, loc_b, map_b, sub_art_b, fallback_b
        second_np, second_rs, second_loc, second_map, second_sub_art, second_fb = np_a, rs_a, loc_a, map_a, sub_art_a, fallback_a

    first_name = _format_smaller_partner_name(first_np.name)
    second_name = _format_smaller_partner_name(second_np.name)

    # Compose the spiro descriptor.  Form: spiro[<first>-N,N'-<second>]
    name_str = f"spiro[{first_name}-{first_loc},{second_loc}'-{second_name}]"
    stem = name_str[:-1] if name_str.endswith("e") else name_str

    # Build a numbering for the combined system: unprimed locants from the
    # first partner; primed locants from the second partner.  The articulation
    # atom carries BOTH (we use the unprimed for assignment; the engine doesn't
    # need both forms exposed at once).
    total_atoms = len(ring_system.atom_indices)
    atom_to_loc: dict[int, Locant] = {}

    def _assign_from(side_np, side_rs, side_map, side_fb, side_sub_art, side_sub_mol, primed: bool):
        """Assign locants from one partner side into atom_to_loc.

        Uses the chosen numbering option (the one giving the lowest spiro-atom
        locant), falling back to side_fb (HW-computed dict) when the named
        parent has no numbering options.
        """
        suffix = "'" if primed else ""

        def _put(full_idx: int, numeric_loc: int):
            if primed and full_idx == art_atom:
                return  # first side owns the spiro atom's unprimed locant
            atom_to_loc[full_idx] = Locant.numeric(numeric_loc, suffix)

        if side_fb is not None:
            for sub_idx, nloc in side_fb.items():
                full_idx = side_map.get(sub_idx)
                if full_idx is None:
                    continue
                _put(full_idx, nloc)
            return
        chosen = _best_numbering_option(side_sub_art, side_np)
        if chosen is not None:
            for sub_idx, loc in chosen.atom_to_locant.items():
                full_idx = side_map.get(sub_idx)
                if full_idx is None:
                    continue
                if loc.is_numeric and loc._numeric_value is not None:
                    _put(full_idx, loc._numeric_value)
                else:
                    # Non-numeric locants (e.g. fused ring junctions like "4a"):
                    # copy as-is with the primed suffix.
                    if primed and full_idx == art_atom:
                        continue
                    if primed:
                        new_label = loc.label + "'"
                        atom_to_loc[full_idx] = Locant(label=new_label, is_numeric=False, _numeric_value=None, suffix=suffix)
                    else:
                        atom_to_loc[full_idx] = loc

    _assign_from(first_np, first_rs, first_map, first_fb, first_sub_art, sub_a if first_rs is rs_a else sub_b, primed=False)
    _assign_from(second_np, second_rs, second_map, second_fb, second_sub_art, sub_b if second_rs is rs_b else sub_a, primed=True)

    numbering_options: tuple[Numbering, ...] = ()
    if len(atom_to_loc) == total_atoms:
        assignments = tuple(sorted(atom_to_loc.items()))
        sorted_locs = sorted(atom_to_loc.values())
        numbering_options = (
            Numbering(_assignments=assignments, locant_set=tuple(sorted_locs)),
        )

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=None,
        naming_method="polyspiro_articulation",
        indicated_hydrogen=None,
        numbering_options=numbering_options,
    )]


# ---------------------------------------------------------------------------
# Spiro with polycyclic (bridged/fused) partner  (P-24.5)
# ---------------------------------------------------------------------------
# Form: [replacement-prefix]spiro[<partner_A>-<loc>,<loc>'-<partner_B>]
# where one partner is a bridged polycycle (bicyclo[a.b.c]alkane) and the
# other is a simple monocycle.  Perception classifies the combined system
# as "bridged" because of the shared ≥3 atoms inside the polycyclic partner;
# we detect the true spiro topology here by finding an articulation atom.
# Scope: polycyclic partner must be bridged; monocyclic partner carbocycle;
# spiro atom may be C or N+ (azonia).


def _ring_graph_articulation_atoms(
    atom_indices: frozenset[int], mol
) -> list[int]:
    """Return ring-atom articulation points of the ring-atom subgraph.

    Brute-force O(n²) BFS — ring systems here are <50 atoms.  An atom is an
    articulation point if its removal disconnects the ring-atom subgraph.
    """
    atoms = sorted(atom_indices)
    if len(atoms) < 3:
        return []
    ring_set = set(atoms)
    # Precompute in-ring neighbors per atom.
    nbrs: dict[int, list[int]] = {}
    for i in atoms:
        nbrs[i] = [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() in ring_set
        ]
    art: list[int] = []
    for a in atoms:
        remaining = ring_set - {a}
        if not remaining:
            continue
        start = next(iter(remaining))
        visited = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in nbrs[u]:
                if v in remaining and v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < len(remaining):
            art.append(a)
    return art


def _split_at_articulation(
    ring_system: "RingSystem", art_atom: int, mol
) -> tuple[frozenset[int], frozenset[int]] | None:
    """Split ring atoms into two components at articulation atom.

    Returns (comp_a, comp_b) where each component excludes the articulation
    atom but includes all other ring atoms reachable without crossing it.
    Returns None if the articulation atom yields anything other than exactly
    two non-empty components in the ring-atom subgraph.
    """
    atoms = set(ring_system.atom_indices) - {art_atom}
    if len(atoms) < 2:
        return None
    components: list[set[int]] = []
    remaining = set(atoms)
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp: set[int] = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining and j not in comp:
                    stack.append(j)
        components.append(comp)
        remaining -= comp
    if len(components) != 2:
        return None
    return frozenset(components[0]), frozenset(components[1])


def _build_sub_ring_system(
    partner_atoms: frozenset[int], art_atom: int, ring_system: "RingSystem", mol
):
    """Build a RingSystem-like object for a partner (partner_atoms + art_atom).

    Uses the RingAnalysis path for proper classification (bridged vs fused
    vs monocyclic) and bridge-size computation.  Returns the first detected
    ring system within the partner atom set, or None.
    """
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis

    # Build a sub-mol containing only partner atoms + articulation atom,
    # with the articulation atom's charge/element normalized so that
    # perception sees a plain carbocycle (we strip the heteroatom identity;
    # it is re-encoded via a replacement prefix on the combined name).
    full_atoms = sorted(set(partner_atoms) | {art_atom})
    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old in full_atoms:
        orig = mol.GetAtomWithIdx(old)
        if old == art_atom:
            new_atom = Chem.Atom(6)  # C in sub-mol; heteroatom handled via replacement
        else:
            new_atom = Chem.Atom(orig.GetAtomicNum())
            if orig.GetFormalCharge() and orig.GetAtomicNum() != 7:
                new_atom.SetFormalCharge(orig.GetFormalCharge())
        old_to_new[old] = rw.AddAtom(new_atom)
    # Add bonds between atoms that are both in the partner atom set
    added_bonds: set[frozenset[int]] = set()
    for old in full_atoms:
        for nb in mol.GetAtomWithIdx(old).GetNeighbors():
            j = nb.GetIdx()
            if j not in old_to_new:
                continue
            key = frozenset({old, j})
            if key in added_bonds:
                continue
            added_bonds.add(key)
            bond = mol.GetBondBetweenAtoms(old, j)
            if bond is None:
                continue
            rw.AddBond(old_to_new[old], old_to_new[j], bond.GetBondType())
    sub_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(sub_mol)
    except Exception:
        return None, None, None
    new_to_old = {v: k for k, v in old_to_new.items()}
    try:
        aa = AtomAnalysis(sub_mol)
        ra = RingAnalysis(sub_mol, aa)
    except Exception:
        return None, None, None
    if not ra.ring_systems:
        return None, None, None
    return ra.ring_systems[0], new_to_old, sub_mol


def _name_monocyclic_partner(
    ring_size: int, hetero_elements: tuple[str, ...]
) -> str | None:
    """Return an IUPAC name for a simple monocyclic partner.

    Only supports all-carbon saturated cycloalkanes here — heteroatoms
    would require Hantzsch-Widman or a-replacement which is handled via
    the combined-system replacement prefix for the spiro atom only.
    """
    if hetero_elements:
        return None
    stem = get_chain_stem(ring_size)
    if stem is None:
        return None
    return f"cyclo{stem}ane"


def name_polycyclic_spiro(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Emit P-24.5 spiro names when one partner is a bridged polycycle.

    Triggers on ring systems classified as ``bridged`` that actually contain
    a single articulation atom splitting them into a bridged polycyclic
    partner plus a simple monocyclic partner (the "spiro-polycyclic-fused"
    case — e.g. FDA-1386 trospium aglycone kernel).

    Returns ``[]`` on any failure so the normal bridged-naming path stays
    authoritative.  Never silently drops heteroatoms: if a partner has ring
    heteroatoms that would need replacement prefixes we cannot express, the
    function refuses to emit.
    """
    if ring_system.type != "bridged":
        return []

    arts = _ring_graph_articulation_atoms(ring_system.atom_indices, mol)
    if not arts:
        return []
    heteroatoms = ring_system.heteroatoms or ()

    # Phase 11 spiro-oxathiolane: when the ring system contains an
    # articulation atom that splits it into a bridged polycyclic partner
    # plus a HETEROCYCLIC monocyclic partner (e.g. 1,3-oxathiolane spiro-
    # joined to 1-azabicyclo[2.2.2]octane in quinuclidine derivatives),
    # the legacy path below refuses to emit because it cannot express
    # ring heteroatoms outside the spiro atom.  Delegate to the general
    # articulation-split spiro path (used by ``name_spiro`` for true
    # polyspiro inputs) which recursively names each partner sub-system
    # via ``name_ring_system`` and composes
    # ``spiro[<polycyclic>-LOC,LOC'-<monocyclic>]`` with the partners'
    # own a-replacement prefixes + substituent locants intact.
    if heteroatoms:
        for art in arts:
            art_atom_obj = mol.GetAtomWithIdx(art)
            # Spiro atom must be a neutral carbon — articulation-split
            # spiro does not handle azonia / heteroatom spiro atoms.
            if art_atom_obj.GetAtomicNum() != 6 or art_atom_obj.GetFormalCharge() != 0:
                continue
            split_result = _try_articulation_split_spiro(
                ring_system, candidate, art, mol
            )
            if split_result:
                return split_result

    # Candidate spiro atoms: articulation atoms that are either carbon or
    # the allowed charged-N (azonia) case.
    for art in arts:
        atom = mol.GetAtomWithIdx(art)
        if atom.GetAtomicNum() == 6 and atom.GetFormalCharge() == 0:
            spiro_repl: str | None = None  # pure carbocycle spiro atom
        elif atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
            spiro_repl = "azonia"
        else:
            continue

        parts = _split_at_articulation(ring_system, art, mol)
        if parts is None:
            continue
        comp_a, comp_b = parts

        # Ring heteroatoms that are NOT at the spiro atom must be empty —
        # otherwise we'd drop them silently (no replacement expressed).
        other_hetero = [hp for hp in heteroatoms if hp.atom_idx != art]
        if other_hetero:
            return []  # refuse — don't drop ring heteroatoms

        # Classify each partner sub-system
        rs_a, map_a, sub_a = _build_sub_ring_system(comp_a, art, ring_system, mol)
        rs_b, map_b, sub_b = _build_sub_ring_system(comp_b, art, ring_system, mol)
        if rs_a is None or rs_b is None:
            continue

        # Exactly one partner must be bridged (polycyclic).  The other must
        # be monocyclic.
        if rs_a.type == "bridged" and rs_b.type == "monocyclic":
            poly_rs, poly_map, poly_sub = rs_a, map_a, sub_a
            mono_rs, mono_map = rs_b, map_b
        elif rs_b.type == "bridged" and rs_a.type == "monocyclic":
            poly_rs, poly_map, poly_sub = rs_b, map_b, sub_b
            mono_rs, mono_map = rs_a, map_a
        else:
            continue  # both poly or both mono — not handled here

        # Name the polycyclic partner (bridged).  Recurse into bridged naming
        # using a synthesized candidate on the partner sub-mol.
        from iupac_namer.ring_naming.bridged import (
            compute_vb_numberings,
            name_bridged,
        )
        from iupac_namer.types import CandidateParent as CP

        sub_candidate = CP(
            atom_indices=poly_rs.atom_indices,
            type="bridged",
            length=poly_rs.ring_size,
            ring_system=poly_rs,
            unsaturation=None,
            element=None,
            lambda_value=None,
        )
        poly_named = name_bridged(poly_rs, sub_candidate, poly_sub)
        if not poly_named:
            continue
        poly_name = poly_named[0].name

        # Name the monocyclic partner
        mono_name = _name_monocyclic_partner(mono_rs.ring_size, ())
        if mono_name is None:
            continue

        # Compute the polycyclic partner's VB numbering so we can find the
        # locant of the articulation atom (the spiro atom) within the
        # polycyclic partner's numbering.
        vb_nbs = compute_vb_numberings(poly_rs, poly_rs.bridge_sizes or (), poly_sub)
        if not vb_nbs:
            continue
        # The art atom maps to a specific sub-mol index; find its locant.
        sub_art_idx = None
        for sub_idx, full_idx in poly_map.items():
            if full_idx == art:
                sub_art_idx = sub_idx
                break
        if sub_art_idx is None:
            continue
        # Pick the VB numbering that gives the LOWEST locant to the spiro
        # atom (P-24.5: the spiro atom should get the lowest possible
        # locant in the polycyclic partner).
        best_nb = None
        best_loc = None
        for nb in vb_nbs:
            a2l = nb.atom_to_locant
            if sub_art_idx not in a2l:
                continue
            loc = a2l[sub_art_idx]
            if not loc.is_numeric or loc._numeric_value is None:
                continue
            if best_loc is None or loc._numeric_value < best_loc:
                best_loc = loc._numeric_value
                best_nb = nb
        if best_nb is None:
            continue

        spiro_loc_poly = best_loc
        # The unprimed side of the name lists the polycyclic partner.
        # Monocyclic partner gets primed locants starting at 1' at the
        # spiro atom then around the monocyclic ring.
        # Build the full combined numbering: polycyclic atoms keep their VB
        # locants; the monocyclic partner atoms get primed locants assigned
        # by walking the ring from the spiro atom.

        # Map sub-mol poly indices -> full-mol indices, and invert the VB
        # locant map onto full-mol indices.
        atom_to_loc: dict[int, Locant] = {}
        for sub_idx, loc in best_nb.atom_to_locant.items():
            full_idx = poly_map.get(sub_idx)
            if full_idx is None:
                continue
            atom_to_loc[full_idx] = loc

        # Walk the monocyclic partner from the spiro atom.  Order: start at
        # spiro atom (locant 1'), walk one direction around the ring.
        mono_ring = list(mono_rs.atom_indices)
        # Find neighbors of art within the monocyclic partner
        mono_full_atoms = {mono_map[i] for i in mono_rs.atom_indices if i in mono_map}
        # We must have one of the comp_a/comp_b sets match mono_full_atoms.
        # The monocyclic partner atoms are comp_a or comp_b (non-art).
        mono_non_art = mono_full_atoms - {art}
        mono_neighbors_of_art = [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(art).GetNeighbors()
            if nb.GetIdx() in mono_non_art
        ]
        if len(mono_neighbors_of_art) < 1:
            continue
        # Walk the ring starting from one neighbor; this gives monocyclic
        # positions 2', 3', ..., n'.  The spiro atom is position 1'.
        walk_start = mono_neighbors_of_art[0]
        ordered_mono: list[int] = [walk_start]
        visited: set[int] = {walk_start, art}
        current = walk_start
        while len(ordered_mono) < len(mono_non_art):
            moved = False
            for nb in mol.GetAtomWithIdx(current).GetNeighbors():
                j = nb.GetIdx()
                if j in mono_non_art and j not in visited:
                    ordered_mono.append(j)
                    visited.add(j)
                    current = j
                    moved = True
                    break
            if not moved:
                break
        if len(ordered_mono) != len(mono_non_art):
            continue

        # Assign primed locants: spiro atom gets "1'" as an alternative
        # view, but locants on the spiro atom come from the polycyclic
        # side (unprimed).  We only assign primed locants to the mono
        # partner's non-art atoms.
        for i, full_idx in enumerate(ordered_mono, start=2):
            atom_to_loc[full_idx] = Locant.numeric(i, "'")

        # Build the a-replacement prefix for the spiro atom if needed.
        hetero_prefix = ""
        if spiro_repl is not None:
            hetero_prefix = f"{spiro_loc_poly}-{spiro_repl}"

        # Assemble the name: [<replacement>]spiro[<poly>-<locA>,1'-<mono>]
        spiro_bracket = f"spiro[{poly_name}-{spiro_loc_poly},1'-{mono_name}]"
        name_str = f"{hetero_prefix}{spiro_bracket}" if hetero_prefix else spiro_bracket
        stem = name_str[:-1] if name_str.endswith("e") else name_str

        # Build numbering options so the engine places substituents correctly.
        total_atoms = len(ring_system.atom_indices)
        if len(atom_to_loc) != total_atoms:
            continue
        assignments = tuple(sorted(atom_to_loc.items()))
        # locant_set: unprimed 1..n_poly, then primed 2..n_mono+1
        locant_set_list: list[Locant] = []
        for loc in atom_to_loc.values():
            locant_set_list.append(loc)
        locant_set_list.sort()
        locant_set = tuple(locant_set_list)

        numbering_options = (
            Numbering(_assignments=assignments, locant_set=locant_set),
        )

        return [NamedParent(
            candidate=candidate,
            name=name_str,
            stem=stem,
            alkyl_stem=None,
            naming_method="spiro_polycyclic",
            indicated_hydrogen=None,
            numbering_options=numbering_options,
        )]
    return []
