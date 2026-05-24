"""
iupac_namer/ring_naming/monocyclic.py

Systematic monocyclic ring naming.

Two systems:
1. Hantzsch-Widman (P-22.1.2): 3-10 membered heterocycles with standard
   heteroatoms. Produces names like "oxirane", "oxolane", "aziridine", etc.
2. Systematic "cyclo" + chain stem (for all-carbon and non-HW rings).
   E.g. "cyclohexane", "cyclopentane".

The retained-name lookup is attempted first (in __init__.py).  This module
produces systematic alternatives, which are always generated (even when a
retained name exists) so that strategy can pick the best.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.data_loader import get_chain_stem, get_hw_tables, get_multiplier
from iupac_namer.types import NamedParent

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hantzsch-Widman element priority order (for multi-heteroatom rings)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lambda-convention: standard valences for ring heteroatoms (IUPAC P-14.1.1)
# ---------------------------------------------------------------------------

# Standard (lowest normal) valence for each element.  When a ring heteroatom
# has a different actual valence, the lambda convention is applied: e.g. P
# with valence 5 → λ5; S with valence 4 → λ4; S with valence 6 → λ6.
_STANDARD_VALENCE: dict[str, int] = {
    "N": 3, "O": 2, "S": 2, "Se": 2, "Te": 2,
    "P": 3, "As": 3, "Sb": 3,
    "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4,
    "B": 3,
    # Group-13 metals (P-22.2.2.1.1 Table 2.7): Al/Ga/In/Tl are trivalent.
    "Al": 3, "Ga": 3, "In": 3, "Tl": 3,
    # Hg: divalent (standard valence).
    "Hg": 2,
}


# Seniority of heteroatoms for HW naming (P-22.1.3.3): higher number = higher priority
_HW_PRIORITY: dict[str, int] = {
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


def compute_lambda_value_map(
    hetero_entries: "list[tuple[int, str, int]]",
    mol,
) -> "dict[int, int]":
    """Map each non-standard-valence skeletal heteroatom LOCANT to its valence
    (the lambda convention, P-14.1.1 / P-31.1.4.3).

    ``hetero_entries`` is a list of ``(locant, element, atom_idx)`` for each
    heteroatom that participates in the parent skeleton (ring atoms AND, for
    spiro / von-Baeyer a-replacement names, the spiro / skeletal heteroatom).
    The ``locant`` is the integer locant the heteroatom carries in the finished
    numbering; ``atom_idx`` indexes into ``mol`` so the actual valence can be
    read.

    Returns ``{locant: actual_valence}`` for every heteroatom whose actual
    valence differs from the element's standard valence (charge-corrected
    exactly as the monocyclic HW path does, so a ``-ide`` / ``-ium`` charged
    ring atom is NOT mistaken for a non-standard lambda valence).  Returns an
    empty dict when no heteroatom needs a lambda marker.  The caller cites the
    valence inline with the a-replacement locant as ``"<loc>lambda<val>"``.
    """
    result: dict[int, int] = {}
    for loc_val, element, atom_idx in hetero_entries:
        std_val = _STANDARD_VALENCE.get(element)
        if std_val is None:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        actual_val = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        # A non-standard valence that is entirely explained by the formal
        # charge (e.g. a deprotonated ring N, [n-]: val 2, std 3, charge -1)
        # is named via a ring-anion/-cation suffix, NOT a lambda marker.
        if charge < 0 and actual_val == std_val + charge:
            continue
        if charge > 0 and actual_val == std_val + charge:
            continue
        if actual_val != std_val:
            result[loc_val] = actual_val
    return result


def _get_ring_cycle_order(ring_atom_set: frozenset, mol) -> list[int]:
    """Return ring atoms in cyclic (graph traversal) order.

    Starts from the lowest-index atom and walks bonds within the ring.
    Returns a list of atom indices in cycle order.
    """
    atoms = sorted(ring_atom_set)
    if not atoms:
        return []
    start = atoms[0]
    ordered: list[int] = [start]
    visited: set[int] = {start}
    current = start
    while len(ordered) < len(ring_atom_set):
        moved = False
        for nb in mol.GetAtomWithIdx(current).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in ring_atom_set and nb_idx not in visited:
                ordered.append(nb_idx)
                visited.add(nb_idx)
                current = nb_idx
                moved = True
                break
        if not moved:
            break
    return ordered


def _compute_hw_locants(
    ring_cycle: list[int],
    heteroatom_indices: set[int],
    priority_map: dict[int, int],  # atom_idx -> HW priority
    indicated_h_tiebreaker: "frozenset[int] | None" = None,
    lambda_atoms: "frozenset[int] | None" = None,
) -> dict[int, int] | None:
    """Compute HW locant assignment for a monocyclic ring.

    P-31.1.2.2: The most senior heteroatom gets locant 1. The numbering
    direction is chosen to give heteroatoms the lowest possible locant set.

    Tiebreak hierarchy (P-31.1.4.3 / P-14.1.2):
      1. Lowest locants for senior heteroatoms (priority order).
      2. Lowest locants for ALL heteroatoms.
      3. Lowest locants for ``lambda_atoms`` (non-standard-valence carriers,
         per P-14.1.2: "lambda symbols cited at lowest locants") — when
         supplied; otherwise skipped.
      4. Lowest locants for indicated-H carriers (P-31.1.4.3.4) when supplied.
      5. Stable: smallest start_pos rank, then forward direction.

    ``indicated_h_tiebreaker`` is the set of ring atom indices that qualify as
    indicated-H carriers (sp3 atoms whose locant determines the "<N>H-"
    prefix).  When None or empty, that tiebreaker is skipped.

    ``lambda_atoms`` is the set of ring atom indices whose ring-atom valence
    differs from the standard for that element.  When None or empty, the
    lambda-locant tiebreaker is skipped.

    Returns dict mapping atom_idx -> locant (1-indexed), or None on failure.
    """
    n = len(ring_cycle)
    if n == 0:
        return None

    # Find the highest priority among all heteroatoms (for "most senior heteroatom
    # gets locant 1" rule, P-31.1.2.2).
    best_prio = -1
    for idx in ring_cycle:
        if idx in heteroatom_indices:
            prio = priority_map.get(idx, 0)
            if prio > best_prio:
                best_prio = prio

    if best_prio < 0:
        return None

    # Collect ALL ring positions occupied by a most-senior heteroatom; any of them
    # is a valid candidate for locant 1.  P-31.1.4.3 requires the numbering chosen
    # to give the LOWEST locant set to the heteroatoms collectively, so we must
    # try each such start position (forward and reverse) and pick the optimum.
    # Without considering all symmetric starts, asymmetric multi-heteroatom rings
    # (e.g. a 3-N-1-C-2-C ring like 1,2,4-triazine: ring sequence C-N-C-C-N-N
    # cyclic) end up emitted as 1,3,4-triazine because only the first-encountered
    # senior heteroatom is tried as locant 1.
    candidate_starts = [
        pos for pos, idx in enumerate(ring_cycle)
        if idx in heteroatom_indices
        and priority_map.get(idx, 0) == best_prio
    ]
    if not candidate_starts:
        return None

    # Try numbering starting at each candidate position in both directions
    # (clockwise = forward, counter-clockwise = reversed).  Choose the lowest
    # senior-heteroatom locant set first (P-31.1.4.3.2 priority-by-element);
    # then lowest full heteroatom locant set; then indicated-H locant set.
    def senior_locants_in_direction(start_pos: int, forward: bool) -> list[int]:
        """Return locants of MOST-SENIOR heteroatoms only.  Senior locants are
        the primary criterion per P-31.1.4.3.2 (heteroatoms cited in the order
        of priority: O > S > … > N > P > …)."""
        result: list[int] = []
        for step in range(n):
            if forward:
                pos = (start_pos + step) % n
            else:
                pos = (start_pos - step) % n
            idx = ring_cycle[pos]
            if (
                idx in heteroatom_indices
                and priority_map.get(idx, 0) == best_prio
            ):
                result.append(step + 1)  # 1-indexed
        return result

    def locants_in_direction(start_pos: int, forward: bool) -> list[int]:
        """Return locants of ALL heteroatoms for this starting position."""
        result: list[int] = []
        for step in range(n):
            if forward:
                pos = (start_pos + step) % n
            else:
                pos = (start_pos - step) % n
            idx = ring_cycle[pos]
            if idx in heteroatom_indices:
                result.append(step + 1)  # 1-indexed
        return result

    def lambda_locants_in_direction(start_pos: int, forward: bool) -> list[int]:
        """Return sorted locants for lambda atoms (non-standard valence)."""
        if not lambda_atoms:
            return []
        result: list[int] = []
        for step in range(n):
            if forward:
                pos = (start_pos + step) % n
            else:
                pos = (start_pos - step) % n
            idx = ring_cycle[pos]
            if idx in lambda_atoms:
                result.append(step + 1)
        return sorted(result)

    def indicated_h_locants_in_direction(start_pos: int, forward: bool) -> list[int]:
        """Return sorted list of locants for indicated-H carrier atoms."""
        if not indicated_h_tiebreaker:
            return []
        result: list[int] = []
        for step in range(n):
            if forward:
                pos = (start_pos + step) % n
            else:
                pos = (start_pos - step) % n
            idx = ring_cycle[pos]
            if idx in indicated_h_tiebreaker:
                result.append(step + 1)
        return sorted(result)

    # Iterate every (start, direction) and pick the best by lexicographic key:
    #   (senior_locants, full_locants, lambda_locants, indicated_h_locants,
    #    start_pos_rank, direction_rank).
    # Per P-14.1.2 the lambda-locant criterion ranks ABOVE indicated-H, so a
    # ring with both a lambda atom and indicated-H carriers (e.g. hexamethyl
    # triazatriphosphinine) prefers the numbering that places the lambda
    # atom at a lower locant first.  Stable tiebreakers (start_pos, direction)
    # preserve deterministic output for fully-symmetric rings.
    candidates: list[tuple[list[int], list[int], list[int], list[int], int, int, int, bool]] = []
    for cs_rank, start_pos in enumerate(candidate_starts):
        for dir_rank, forward in enumerate((True, False)):
            slocs = senior_locants_in_direction(start_pos, forward)
            locs = locants_in_direction(start_pos, forward)
            llocs = lambda_locants_in_direction(start_pos, forward)
            ih = indicated_h_locants_in_direction(start_pos, forward)
            candidates.append((slocs, locs, llocs, ih, cs_rank, dir_rank, start_pos, forward))

    candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]))
    _, _, _, _, _, _, best_start, best_forward = candidates[0]

    if best_start is None:
        return None

    # Build full locant assignment
    result: dict[int, int] = {}
    for step in range(n):
        if best_forward:
            pos = (best_start + step) % n
        else:
            pos = (best_start - step) % n
        idx = ring_cycle[pos]
        result[idx] = step + 1
    return result


def _compute_hw_locants_all_optima(
    ring_cycle: list[int],
    heteroatom_indices: set[int],
    priority_map: dict[int, int],
    indicated_h_tiebreaker: "frozenset[int] | None" = None,
    lambda_atoms: "frozenset[int] | None" = None,
) -> list[dict[int, int]]:
    """Return EVERY heteroatom-optimal numbering, not just one.

    Same heteroatom-locant criteria as :func:`_compute_hw_locants` (senior >
    full > lambda > indicated-H), but instead of breaking the final tie with the
    stable (start_pos, direction) keys, this returns the full locant map for
    each (start, direction) that ties for the optimal heteroatom key.

    A saturated Hantzsch-Widman heterocycle's NAME fixes the heteroatom locant
    set (e.g. ``1,4-oxazepane`` ⇒ O=1, N=4), but for any given molecule there
    may be several traversal directions that achieve that same optimal set
    (symmetric rings such as morpholine, 1,4-dioxepane).  Pinning only ONE of
    them lets the engine's generic numbering pass instead choose a direction
    that contradicts the cited heteroatom locants when placing substituents
    (it would minimise the substituent locant against the heteroatom
    constraint).  Returning all heteroatom-optimal maps lets the strategy pick
    the lowest substituent locant *among numberings that respect the
    heteroatoms* — P-31.1.4.3.4 then P-14.5.2.
    """
    n = len(ring_cycle)
    if n == 0:
        return []

    best_prio = -1
    for idx in ring_cycle:
        if idx in heteroatom_indices:
            prio = priority_map.get(idx, 0)
            if prio > best_prio:
                best_prio = prio
    if best_prio < 0:
        return []

    candidate_starts = [
        pos for pos, idx in enumerate(ring_cycle)
        if idx in heteroatom_indices
        and priority_map.get(idx, 0) == best_prio
    ]
    if not candidate_starts:
        return []

    def _walk(start_pos: int, forward: bool):
        return [
            ring_cycle[(start_pos + step) % n] if forward
            else ring_cycle[(start_pos - step) % n]
            for step in range(n)
        ]

    def _locs(order: list[int], pred) -> list[int]:
        return [i + 1 for i, idx in enumerate(order) if pred(idx)]

    def _senior(idx: int) -> bool:
        return (idx in heteroatom_indices
                and priority_map.get(idx, 0) == best_prio)

    def _hetero(idx: int) -> bool:
        return idx in heteroatom_indices

    scored: list[tuple[tuple, dict[int, int]]] = []
    for start_pos in candidate_starts:
        for forward in (True, False):
            order = _walk(start_pos, forward)
            slocs = _locs(order, _senior)
            locs = _locs(order, _hetero)
            llocs = sorted(_locs(order, lambda i: bool(lambda_atoms) and i in lambda_atoms))
            ih = sorted(_locs(order, lambda i: bool(indicated_h_tiebreaker)
                              and i in indicated_h_tiebreaker))
            key = (slocs, locs, llocs, ih)
            lmap = {idx: i + 1 for i, idx in enumerate(order)}
            scored.append((key, lmap))

    if not scored:
        return []
    best_key = min(s[0] for s in scored)
    # Dedupe identical maps (a fully symmetric ring yields repeats).
    out: list[dict[int, int]] = []
    seen: set[tuple] = set()
    for key, lmap in scored:
        if key != best_key:
            continue
        sig = tuple(sorted(lmap.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(lmap)
    return out


def try_replacement_nomenclature(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> NamedParent | None:
    """Build a replacement-nomenclature name for large heterocyclic rings (>10 membered).

    IUPAC P-31.1.3: for rings where Hantzsch-Widman does not apply (>10 members),
    use heteroatom replacement prefixes (oxa, aza, thia...) before the carbocyclic
    parent name (cyclo + chain_stem + ane).

    Example: 12-membered ring with 1 O → "1-oxacyclododecane"
             11-membered ring with 1 N → "1-azacycloundecane"
    """
    ring_size = ring_system.ring_size
    # Replacement nomenclature applies to large rings (>10) by default, but
    # also serves as the fallback for 3-10 membered rings where HW could not
    # form a name (e.g., aromatic ring with mixed metals or charged atoms).
    # The decision to use replacement over HW for small rings is left to the
    # caller (name_monocyclic): HW is tried first, replacement only if HW fails.
    if ring_size < 3:
        return None

    heteroatoms = ring_system.heteroatoms
    if not heteroatoms:
        return None  # No heteroatoms, systematic carbocyclic name is correct

    hw_tables = get_hw_tables()
    prefixes_list = hw_tables.get("prefixes", [])
    elem_to_prefix: dict[str, str] = {}
    for entry in prefixes_list:
        elem_to_prefix[entry["element"]] = entry["prefix"]

    # Validate all heteroatoms have replacement prefixes
    elements_used = [hp.element for hp in heteroatoms]
    for elem in elements_used:
        if elem not in elem_to_prefix:
            return None

    # Compute locants for heteroatoms (numbering to give lowest locants to
    # highest-priority heteroatoms, as per P-31.1.2.2)
    ring_atoms = ring_system.atom_indices
    ring_cycle = _get_ring_cycle_order(ring_atoms, mol)
    hetero_set = {hp.atom_idx for hp in heteroatoms}
    prio_map: dict[int, int] = {
        hp.atom_idx: _HW_PRIORITY.get(hp.element, 0)
        for hp in heteroatoms
    }
    locant_map = _compute_hw_locants(ring_cycle, hetero_set, prio_map)

    from iupac_namer.types import Locant, HeteroPosition
    hetero_with_locants = []
    for hp in heteroatoms:
        loc_val = locant_map.get(hp.atom_idx) if locant_map else None
        loc = Locant.numeric(loc_val) if loc_val is not None else hp.locant
        hetero_with_locants.append(HeteroPosition(atom_idx=hp.atom_idx, element=hp.element, locant=loc))

    # Sort by priority (highest first), then locant
    sorted_hetero = sorted(
        hetero_with_locants,
        key=lambda hp: (-_HW_PRIORITY.get(hp.element, -1),
                        hp.locant._numeric_value if hp.locant and hp.locant._numeric_value else 0),
    )

    # Build the carbocyclic base name from ring_size
    # The ring_size includes ALL atoms (C and heteroatoms), and we use that total
    # size as the carbon count for the "cycloNNNane" parent before replacement.
    stem_base = get_chain_stem(ring_size)
    if stem_base is None:
        return None

    # Detect ring unsaturation (endocyclic C=C / C#C within this ring).
    # Replacement-nomenclature macrocycles must reflect endocyclic double bonds
    # in the parent name (e.g. macrolides with C=C in the ring): otherwise
    # downstream stereo-descriptors like (6E) cannot be placed on a bond that
    # does not exist in the parsed structure.  IUPAC P-31.1.3 / P-31.1.2.1:
    # unsaturation suffixes apply the same way to replacement parents as to
    # carbocyclic systematic parents.
    double_bond_locants, triple_bond_locants = _detect_ring_unsaturation(
        ring_system, mol
    )
    ring_bond_pairs = get_ring_bond_pairs(ring_system, mol)

    # Mancude (fully-aromatic) macrocycle: _detect_ring_unsaturation /
    # get_ring_bond_pairs return nothing for an aromatic ring because the
    # Kekulé double bonds are hidden behind RDKit's aromatic flag.  A large
    # aromatic heteromacrocycle (e.g. aza[14]annulene, c1ccccccncccccc1) is
    # the maximally-unsaturated mancude ring, not a saturated one — emitting
    # "1-azacyclotetradecane" is the WRONG structure.  Recover the Kekulé
    # double bonds and place them in the cited heteroatom numbering so the
    # parent becomes "1-azacyclotetradeca-1,3,5,7,9,11,13-heptaene"
    # (P-31.1.4 / P-22.1.4), which OPSIN round-trips to the aromatic input.
    if (
        not double_bond_locants
        and not triple_bond_locants
        and not ring_bond_pairs
        and ring_system.aromatic
        and locant_map is not None
    ):
        kekule_pairs = _kekulized_ring_bond_pairs(ring_system, mol)
        if kekule_pairs:
            ring_bond_pairs = kekule_pairs
            from iupac_namer.types import Locant as _Locant
            # Compute the provisional DB locants from the SAME ``locant_map``
            # that fixes the heteroatom-prefix locants and the pinned
            # numbering_options below, so the initial name string is consistent
            # with what the engine recomputes per the pinned numbering.  (The
            # heteroatom is at locant 1; the Kekulé double-bond placement is
            # read off the actual structure, giving e.g. the heptaene parent
            # of aza[14]annulene.)
            atom_to_loc = {
                idx: _Locant.numeric(loc)
                for idx, loc in locant_map.items()
                if loc is not None
            }
            double_bond_locants, triple_bond_locants = (
                compute_ring_unsaturation_locants_from_numbering(
                    kekule_pairs, atom_to_loc
                )
            )

    if double_bond_locants or triple_bond_locants:
        unsat_suffix = _build_ring_unsaturation_suffix(
            double_bond_locants, ring_size, triple_bond_locants
        )
        # _build_ring_unsaturation_suffix returns forms like:
        #   single DB: "-<loc>-ene"
        #   multi DB:  "a-<locs>-<mult>ene"
        #   single yne: "yne"   (prefix the trailing 'e' onto the stem)
        # We combine these with the carbocyclic base "cyclo" + stem_base so
        # that, e.g., 16-ring with 2 DBs at 4,6 becomes
        # "cyclohexadeca-4,6-diene"; 38-ring with 6 DBs at 6,8,12,14,16,18
        # becomes "cyclooctatriaconta-6,8,12,14,16,18-hexaene".  The leading
        # 'a' in the multi-DB suffix attaches directly to "cyclo...adec" to
        # produce "cyclo...adeca-".
        cycloalkane = "cyclo" + stem_base + unsat_suffix
    else:
        cycloalkane = "cyclo" + stem_base + "ane"

    # Build locant + prefix string
    # Group same-element heteroatoms together for multipliers
    all_have_locants = all(hp.locant is not None for hp in sorted_hetero)
    need_locants = len(sorted_hetero) >= 1  # always include locants for replacement nomenclature

    if need_locants and all_have_locants:
        # IUPAC P-25.2.2.1.2 / P-23.2.5: each heteroatom-element group gets
        # its OWN locant list immediately preceding that element's 'a'-prefix,
        # not a single combined locant list before the whole prefix block.
        # Example (correct):  1-thia-4,7,10,13,16-pentaazacyclooctadecane
        # Example (wrong):    1,4,7,10,13,16-thiapentaazacyclooctadecane
        #
        # Element order follows seniority (sorted_hetero is already in
        # priority order: O > S > Se > Te > N > P > B > ...).  Within each
        # element group the locants are sorted ascending.
        elem_order: list[str] = []
        elem_locants: dict[str, list[Locant]] = {}
        for hp in sorted_hetero:
            if hp.element not in elem_locants:
                elem_order.append(hp.element)
                elem_locants[hp.element] = []
            elem_locants[hp.element].append(hp.locant)

        # Build per-element segments: "<sorted-locants>-[<multi>]<prefix>"
        raw_segments: list[str] = []
        for elem in elem_order:
            pref = elem_to_prefix[elem]
            locs = sorted(
                elem_locants[elem],
                key=lambda l: l._numeric_value if l and l._numeric_value else 0,
            )
            n_elem = len(locs)
            loc_str = ",".join(str(l) for l in locs)
            if n_elem == 1:
                seg = f"{loc_str}-{pref}"
            else:
                multi = get_multiplier(n_elem)
                if multi is None:
                    return None
                seg = f"{loc_str}-{multi}{pref}"
            raw_segments.append(seg)

        # Elision of the terminal 'a' in an 'a'-prefix (aza, oxa, thia, ...)
        # only applies when that 'a' is IMMEDIATELY followed by a vowel-
        # initial character (P-15.2.1.4).  When subsequent segments start
        # with a locant list (e.g. "thia-4,7-aza..."), no elision occurs —
        # the hyphen + digit break prevents vowel adjacency.  The only
        # elision candidate is therefore the FINAL segment's trailing 'a'
        # before the cycloalkane parent stem (e.g. "...pentaaza" +
        # "cyclooctadecane"; the next char is 'c' so NO elision here, but
        # "...oxa" + "indole" would elide to "...ox-indole" in fused
        # systems — not applicable to pure-cyclo parents).
        elided = list(raw_segments)
        if elided and cycloalkane and cycloalkane[0] in "aeiou" and elided[-1].endswith("a"):
            elided[-1] = elided[-1][:-1]

        # Join segments: each subsequent segment starts with a locant list,
        # so it MUST be preceded by a hyphen (e.g. "1-thia-4,7-pentaaza...").
        # The final segment is followed directly by the cycloalkane parent
        # with no hyphen ("...pentaazacyclooctadecane").
        rep_name = "-".join(elided) + cycloalkane
    else:
        # Fallback: no locants
        parts = [elem_to_prefix[e] for e in elements_used]
        for i in range(len(parts)):
            next_start = parts[i + 1][0] if i + 1 < len(parts) else cycloalkane[0]
            if next_start in "aeiou" and parts[i].endswith("a"):
                parts[i] = parts[i][:-1]
        prefix_body = "".join(parts)
        rep_name = prefix_body + cycloalkane

    # Stem (strip terminal 'e')
    if rep_name.endswith("e"):
        stem_str = rep_name[:-1]
    else:
        stem_str = rep_name

    # Build a pre-computed Numbering object where heteroatoms get the lowest locants.
    # This ensures the engine picks the correct numbering without hitting the
    # max_plans cap (large rings have 2*ring_size numberings, easily exceeding 20).
    # The locant_map from _compute_hw_locants already assigns heteroatoms to low
    # positions (O→1, N→1, etc.).
    numbering_options: tuple = ()
    if locant_map:
        try:
            from iupac_namer.types import Numbering
            assignments = tuple(
                (atom_idx, Locant.numeric(loc_val))
                for atom_idx, loc_val in sorted(locant_map.items())
                if loc_val is not None
            )
            locant_set = tuple(Locant.numeric(i + 1) for i in range(ring_size))
            if assignments:
                numbering_options = (Numbering(_assignments=assignments, locant_set=locant_set),)
        except Exception:
            numbering_options = ()

    return NamedParent(
        candidate=candidate,
        name=rep_name,
        stem=stem_str,
        alkyl_stem=None,
        naming_method="replacement",
        indicated_hydrogen=None,
        numbering_options=numbering_options,
        ring_unsaturation_bonds=ring_bond_pairs if ring_bond_pairs else None,
    )


def name_monocyclic(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Generate systematic names for a monocyclic ring.

    Always tries the systematic "cyclo+stem" name.  If heteroatoms are
    present, also tries the Hantzsch-Widman name (for 3-10 membered rings)
    or replacement nomenclature (for >10 membered rings).
    Returns a list (0, 1, or 2 items).
    """
    results: list[NamedParent] = []

    if ring_system.heteroatoms:
        hw = try_hantzsch_widman(ring_system, candidate, mol)
        if hw is not None:
            results.append(hw)
        else:
            # HW doesn't apply (either unsupported size or element).
            # For rings > 10 with heteroatoms, try replacement nomenclature.
            rep = try_replacement_nomenclature(ring_system, candidate, mol)
            if rep is not None:
                results.append(rep)

    syst = name_systematic_monocyclic(ring_system, candidate, mol)
    if syst is not None:
        results.append(syst)

    return results


def _detect_ring_unsaturation(ring_system: "RingSystem", mol) -> tuple[list[int], list[int]]:
    """Return (double_bond_locants, triple_bond_locants) for bonds within the ring.

    Ring atoms are ordered by traversing the ring; the locant of a bond is the
    1-based position of the lower atom in the IUPAC-ordered ring walk.

    Returns empty lists for aromatic rings (they use retained names like
    benzene, not "cyclohexa-1,3,5-triene").
    """
    if ring_system.aromatic:
        return [], []

    ring_atom_set = ring_system.atom_indices

    # Walk the ring to get atoms in order (using the molecular graph).
    ring_atoms_sorted = sorted(ring_atom_set)
    if not ring_atoms_sorted:
        return [], []

    # Build adjacency within ring only.
    adj: dict[int, list[int]] = {a: [] for a in ring_atom_set}
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_atom_set:
                adj[atom_idx].append(nb.GetIdx())

    # Walk the ring to get a cyclic ordered sequence.
    start = ring_atoms_sorted[0]
    ordered: list[int] = [start]
    prev = -1
    curr = start
    while True:
        nexts = [nb for nb in adj[curr] if nb != prev]
        if not nexts:
            break
        nxt = nexts[0]
        if nxt == start:
            break  # completed the ring
        ordered.append(nxt)
        prev = curr
        curr = nxt

    if len(ordered) != len(ring_atom_set):
        # Fallback: just use sorted order
        ordered = ring_atoms_sorted

    # Detect double and triple bonds between consecutive ring atoms.
    # Try both traversal directions (forward and reverse from the starting atom)
    # and pick the one that gives the lowest locant set for unsaturated bonds.
    # This ensures consistency with IUPAC ring numbering, which also picks the
    # direction giving the lowest locants (P-31.1.2.1).
    from rdkit.Chem import BondType

    def _scan_direction(ring_order: list[int]) -> tuple[list[int], list[int]]:
        """Return (double_locants, triple_locants) for bonds in ring_order."""
        dbl: list[int] = []
        tri: list[int] = []
        m = len(ring_order)
        for i in range(m):
            a1 = ring_order[i]
            a2 = ring_order[(i + 1) % m]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                bt = bond.GetBondType()
                if bt == BondType.DOUBLE:
                    dbl.append(i + 1)
                elif bt == BondType.TRIPLE:
                    tri.append(i + 1)
        return sorted(dbl), sorted(tri)

    n = len(ordered)
    # Forward direction (as built by the traversal above)
    fwd_dbl, fwd_tri = _scan_direction(ordered)

    # Reverse direction: same starting atom, traverse the other way around the ring.
    # Reverse by reversing all atoms EXCEPT the first (keep start at index 0).
    ordered_rev = [ordered[0]] + list(reversed(ordered[1:]))
    rev_dbl, rev_tri = _scan_direction(ordered_rev)

    # Choose the direction with the lowest combined locant set.
    # Compare double-bond locants first, then triple-bond locants (lexicographic).
    use_forward = (fwd_dbl + fwd_tri) <= (rev_dbl + rev_tri)
    if use_forward:
        double_bond_locants, triple_bond_locants = fwd_dbl, fwd_tri
    else:
        double_bond_locants, triple_bond_locants = rev_dbl, rev_tri

    return double_bond_locants, triple_bond_locants


def _detect_ring_double_bonds(ring_system: "RingSystem", mol) -> list[int]:
    """Backward-compatible wrapper: return double-bond locants only."""
    dbl, _ = _detect_ring_unsaturation(ring_system, mol)
    return dbl


def get_ring_bond_pairs(ring_system: "RingSystem", mol) -> tuple[tuple[int, int, str], ...]:
    """Return ring double/triple bond pairs as (atom1, atom2, bond_type) tuples.

    bond_type is "double" or "triple".  Returns empty tuple for aromatic rings.
    The pairs are stored by atom indices so that locants can be recomputed
    from any ring numbering.
    """
    if ring_system.aromatic:
        return ()

    from rdkit.Chem import BondType
    ring_atom_set = ring_system.atom_indices
    pairs: list[tuple[int, int, str]] = []
    seen_bonds: set[frozenset] = set()  # avoid double-counting

    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom_idx)
            if other_idx not in ring_atom_set:
                continue
            bond_key = frozenset({atom_idx, other_idx})
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            bt = bond.GetBondType()
            if bt == BondType.DOUBLE:
                pairs.append((atom_idx, other_idx, "double"))
            elif bt == BondType.TRIPLE:
                pairs.append((atom_idx, other_idx, "triple"))

    return tuple(sorted(pairs))


def _kekulized_ring_bond_pairs(
    ring_system: "RingSystem", mol
) -> tuple[tuple[int, int, str], ...]:
    """Return the mancude (Kekulé) endocyclic double-bond pairs of an aromatic
    ring as (atom1, atom2, "double") tuples, by Kekulising a copy of ``mol``.

    Used for large aromatic heteromacrocycles (>10-ring replacement
    nomenclature) where ``get_ring_bond_pairs`` returns nothing because the
    double bonds are hidden behind RDKit's aromatic flag.  Returns () when the
    molecule cannot be Kekulised.  Atom indices are preserved (Kekulisation does
    not renumber atoms), so the pairs map cleanly onto any ring numbering.
    """
    from rdkit import Chem
    from rdkit.Chem import BondType

    ring_atom_set = ring_system.atom_indices
    try:
        mk = Chem.Mol(mol)
        Chem.Kekulize(mk, clearAromaticFlags=True)
    except Exception:
        return ()

    pairs: list[tuple[int, int, str]] = []
    seen: set[frozenset] = set()
    for atom_idx in ring_atom_set:
        atom = mk.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom_idx)
            if other_idx not in ring_atom_set:
                continue
            key = frozenset({atom_idx, other_idx})
            if key in seen:
                continue
            seen.add(key)
            if bond.GetBondType() == BondType.DOUBLE:
                pairs.append((atom_idx, other_idx, "double"))
    return tuple(sorted(pairs))


def compute_ring_unsaturation_locants_from_numbering(
    ring_bond_pairs: tuple[tuple[int, int, str], ...],
    atom_to_locant: dict,
) -> tuple[list[int], list[int]]:
    """Compute double/triple bond locants using an actual ring numbering.

    Returns (double_bond_locants, triple_bond_locants) as sorted lists of ints.

    IUPAC convention: the locant of a double bond is the LOWER-NUMBERED endpoint
    in the sequential ring walk.  For bonds between consecutive positions (k, k+1),
    this is k.  For the wrap-around bond (between position n and position 1 in an
    n-membered ring), the locant is n (NOT 1), because the bond runs from position n
    to the first position.

    Special case: if v1=1 and v2=ring_size (or vice versa), the locant is ring_size
    (the wrap-around convention).  For all other bonds, locant = min(v1, v2).
    """
    double_locants: list[int] = []
    triple_locants: list[int] = []

    # Determine ring size from the locant map (max locant value)
    ring_size = max(
        (loc._numeric_value for loc in atom_to_locant.values()
         if hasattr(loc, '_numeric_value') and loc._numeric_value),
        default=0,
    )

    for a1, a2, btype in ring_bond_pairs:
        loc1 = atom_to_locant.get(a1)
        loc2 = atom_to_locant.get(a2)
        if loc1 is None or loc2 is None:
            continue
        v1 = loc1._numeric_value if hasattr(loc1, '_numeric_value') else None
        v2 = loc2._numeric_value if hasattr(loc2, '_numeric_value') else None
        if v1 is None or v2 is None:
            continue

        # Detect wrap-around bond: one atom at locant 1, other at locant ring_size
        if ring_size > 0 and {v1, v2} == {1, ring_size}:
            bond_locant = ring_size  # wrap-around: locant is n (the larger value)
        else:
            bond_locant = min(v1, v2)

        if btype == "double":
            double_locants.append(bond_locant)
        elif btype == "triple":
            triple_locants.append(bond_locant)

    return sorted(double_locants), sorted(triple_locants)


def _build_ring_unsaturation_suffix(
    double_bond_locants: list[int],
    ring_size: int,
    triple_bond_locants: list[int] | None = None,
) -> str:
    """Build the unsaturation portion of a ring name.

    IUPAC rules (P-31.1.2.1, P-31.1.3.1):
    - One double bond: "-ene" (locant omitted when there's only one DB and
      the ring is unambiguous — i.e., DB could only be at one position).
      In practice: cyclohexene (no locant), cyclopentene (no locant).
      But cyclopent-2-en-1-one (locant cited when FG present) — that's handled
      upstream. For the base parent name, locant is omitted for single DB.
    - Two+ double bonds: "-a" is inserted before "-diene"/"-triene", locants cited.
      e.g. cyclohexa-1,3-diene, cyclopenta-2,4-dien-1-ol
    - Triple bonds: "-yne" for one, "-a-1,5-diyne" for two, etc.
    - Mixed: double and triple bonds both cited, e.g. cyclohexa-1-en-3-yne

    Returns the unsaturation suffix (empty string if fully saturated).
    """
    if triple_bond_locants is None:
        triple_bond_locants = []

    nd = len(double_bond_locants)
    nt = len(triple_bond_locants)

    if nd == 0 and nt == 0:
        return ""

    # Pure triple bond(s): cyclooctyne, cycloocta-1,5-diyne
    if nd == 0:
        mult_t = get_multiplier(nt, complex=False) or ""
        if nt == 1:
            return "yne"
        else:
            loc_str = ",".join(str(l) for l in triple_bond_locants)
            return f"a-{loc_str}-{mult_t}yne"

    # Pure double bond(s)
    n = nd
    mult = get_multiplier(n, complex=False) or ""  # "di", "tri", etc. (n>=2)

    if n == 1 and nt == 0:
        # Single double bond: always include the locant so that assembly can
        # render "cyclohex-2-en-1-one" (when a suffix is present) correctly.
        # For unsubstituted rings the locant is omissible — the assembly strips
        # it via _strip_ring_unsaturation_locant_if_omissible.
        # NOTE: single double bonds use "-{loc}-ene" NOT "a-{loc}-ene".
        # The 'a' insertion is only for multiple double bonds (P-31.1.2.1).
        loc = double_bond_locants[0]
        return f"-{loc}-ene"
    elif nt == 0:
        # Multiple double bonds, no triples: "a" inserted, locants cited
        # e.g. cyclohexa-1,3-diene (di + ene = diene, not diiene)
        locant_str = ",".join(str(loc) for loc in double_bond_locants)
        return f"a-{locant_str}-{mult}ene"
    else:
        # Mixed double + triple bonds: cite both with locants
        mult_t = get_multiplier(nt, complex=False) or ""
        d_loc_str = ",".join(str(l) for l in double_bond_locants)
        t_loc_str = ",".join(str(l) for l in triple_bond_locants)
        return f"a-{d_loc_str}-{mult}en-{t_loc_str}-{mult_t}yne"


def _try_annulene_name(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> NamedParent | None:
    """Build a Blue Book P-22.1.4 ``[N]annulene`` name for a fully-conjugated
    monocyclic carbocyclic polyene.

    The ``[N]annulene`` systematic name applies to an even-membered (size ≥ 8)
    monocyclic all-carbon ring whose every ring atom participates in an
    endocyclic π-bond — i.e. the ring is the maximally-unsaturated cyclic
    polyene with N=2k carbons and k double bonds.  In RDKit, 4n+2 sizes
    (10, 14, 18, 22, 26, …) come back with the aromatic flag set and zero
    explicit double bonds, while 4n sizes (8, 12, 16, 20, 24, …) keep
    Kekulé double bonds.  Both situations qualify.

    For odd N (annulene names are not defined for odd cycles) and N < 8 (those
    are benzene at 6 / cyclopentadiene at 5), this returns None and the
    caller falls through to the cyclo-polyene / cyclo-Xane systematic form.

    The returned ``NamedParent`` has ``naming_method='systematic'`` (per
    Blue Book P-22.1.4 ``[N]annulene`` IS a systematic name), no locants
    in the name itself, no alkyl_stem, and no ``ring_unsaturation_bonds``
    (so the assembly layer will not try to inject ``-x,y,z-Nene`` locants).
    """
    if ring_system.heteroatoms:
        return None
    n = ring_system.ring_size
    if n < 8 or n % 2 != 0:
        return None
    # P-54.2: cyclooctatetraene is the PIN for the 8-ring fully-unsaturated
    # carbocycle — [8]annulene is general nomenclature only.  Fall through
    # to the cyclo-polyene systematic form ("cycloocta-1,3,5,7-tetraene")
    # so the strategy can pick that as the PIN.  Larger annulenes
    # ([10], [14], [18], …) remain handled via the retained-name table.
    if n == 8:
        return None

    # All ring atoms must be carbon (no exotic atoms snuck in).  Carbon is
    # the only element for which a Blue Book ``[N]annulene`` parent is
    # defined; anything else needs replacement nomenclature or HW.
    ring_atoms = ring_system.atom_indices
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 6:
            return None
        # Reject ring carbons that carry an exocyclic substituent or charge:
        # ``[N]annulene`` is the bare parent; substituted annulenes are still
        # valid but their substituent locants depend on the perimeter
        # numbering, which the engine handles via the retained-name path
        # for sizes that have curated atom_locants.  Without curated atom
        # locants, an unsubstituted-only emission is the safe path here.
        if atom.GetFormalCharge() != 0:
            return None

    # Fully-unsaturated criterion: either RDKit aromatized the whole ring
    # (4n+2 case) OR every ring atom carries an endocyclic double bond
    # (4n Kekulé case).  Both correspond structurally to the maximally
    # unsaturated mancude monocyclic polyene with N/2 double bonds.
    all_aromatic = all(
        mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atoms
    )
    if not all_aromatic:
        # Count endocyclic double bonds; require N/2 (i.e. every other
        # ring bond is a DB) AND every ring atom must touch one.
        seen: set[frozenset] = set()
        endo_dbs = 0
        atoms_with_endo_db: set[int] = set()
        for a in ring_atoms:
            atom = mol.GetAtomWithIdx(a)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(a)
                if other not in ring_atoms:
                    continue
                key = frozenset((a, other))
                if key in seen:
                    continue
                seen.add(key)
                bt = bond.GetBondTypeAsDouble()
                # A ring TRIPLE bond is NOT part of the mancude annulene
                # criterion: [N]annulene is the all-DOUBLE-bond polyene
                # (N/2 double bonds, P-22.1.4).  A cyclic enyne (e.g.
                # cyclododeca-1,3,5,7,9-pentaen-11-yne, C1#CC=CC=CC=CC=CC=C1)
                # has a -yne that must be expressed in the cyclo-polyenyne
                # name, so it is NOT an annulene — reject and fall through.
                if bt >= 3.0:
                    return None
                if bt == 2.0:
                    endo_dbs += 1
                    atoms_with_endo_db.add(a)
                    atoms_with_endo_db.add(other)
        if endo_dbs != n // 2:
            return None
        if len(atoms_with_endo_db) != n:
            return None

    annulene_name = f"[{n}]annulene"
    # The stem (used for Method-2 suffix attachment in compound names) is the
    # name minus a terminal 'e' so e.g. "[N]annulen-1-yl"-style attachments
    # could be assembled if a downstream consumer ever needs them.  Substitution
    # of an annulene parent goes through the retained-name path for the
    # curated sizes (data_loader entries).  For sizes outside the curated
    # set the assembly layer will emit the bare parent only — adequate for
    # round-trip equality of the unsubstituted forms which is what the
    # Blue Book P-22.1.4 systematic name covers.
    stem = annulene_name[:-1] if annulene_name.endswith("e") else annulene_name
    return NamedParent(
        candidate=candidate,
        name=annulene_name,
        stem=stem,
        alkyl_stem=None,
        naming_method="systematic",
        indicated_hydrogen=None,
        numbering_options=(),
        ring_unsaturation_bonds=None,
    )


def name_systematic_monocyclic(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> NamedParent | None:
    """Generate "cyclo" + stem name for an all-carbon monocyclic ring.

    Returns None if the chain stem is unavailable.
    Returns None if the ring contains ANY heteroatoms — naming a heterocycle
    as a carbocycle would silently drop heteroatoms (architectural rule:
    no silent atom drops). Heterocyclic rings are named by Hantzsch-Widman
    (3-10 ring atoms) or by skeletal a-prefix replacement nomenclature
    (>10 ring atoms or HW-ineligible). See P-22.1 / P-23.2.5 / P-31.1.3.
    Detects double/triple bonds within the ring and uses appropriate suffix.
    """
    # Architectural guard: heterocycles must NEVER be named as carbocycles.
    # Doing so silently drops heteroatom information from the name.
    if ring_system.heteroatoms:
        return None

    # Blue Book P-22.1.4: large fully-conjugated monocyclic polyenes use the
    # systematic ``[N]annulene`` name (e.g. ``[18]annulene``).  When the ring
    # qualifies, emit that form INSTEAD of the cyclo-polyene/ane systematic
    # name — otherwise an aromatized 4n+2 ring (e.g. 22-membered, RDKit
    # marks aromatic and reports zero explicit double bonds) would silently
    # drop into the saturated "cyclodocosane" branch and fail OPSIN
    # round-trip.  The retained-name path runs higher up in name_monocyclic
    # so any sizes covered by the curated retained table (10/14/16/18/20)
    # already produce ``[N]annulene`` from there; this branch is the
    # uniform systematic fallback for the remaining even sizes (≥ 8).
    annulene = _try_annulene_name(ring_system, candidate, mol)
    if annulene is not None:
        return annulene

    ring_size = ring_system.ring_size
    stem_base = get_chain_stem(ring_size)
    if stem_base is None:
        return None

    cyclo_prefix = "cyclo" + stem_base

    # Detect double and triple bonds in the ring.
    # We compute provisional locants (from the lowest-index-atom traversal) for
    # the initial name/stem; the actual locants will be recomputed per-numbering
    # in the engine once the IUPAC ring numbering is known.  The bond PAIRS
    # (atom indices) are stored in ring_unsaturation_bonds for this purpose.
    double_bond_locants, triple_bond_locants = _detect_ring_unsaturation(ring_system, mol)
    ring_bond_pairs = get_ring_bond_pairs(ring_system, mol)

    if not double_bond_locants and not triple_bond_locants:
        # Fully saturated (or aromatic — retained names handle benzene etc.)
        name_str = cyclo_prefix + "ane"
        stem = cyclo_prefix + "an"
        alkyl_stem = cyclo_prefix  # for -yl substituent
    else:
        unsat_suffix = _build_ring_unsaturation_suffix(
            double_bond_locants, ring_size, triple_bond_locants
        )
        # IUPAC P-31.1.4.2.4: for an unsubstituted monocyclic ring with a
        # SINGLE double bond at the unique unambiguous position (locant 1),
        # the locant is omitted in the standalone parent name.  E.g.
        # "cyclohexene" not "cyclohex-1-ene".  When a FG suffix or attachment
        # later forces the DB to a different locant, the engine recomputes
        # via _recompute_ring_unsaturation_name with the FG-respecting
        # numbering, and `_build_ring_unsaturation_suffix` re-emits the
        # explicit "-N-ene" form (which is then preserved because the locant
        # is no longer 1).
        if (
            len(double_bond_locants) == 1
            and not triple_bond_locants
            and double_bond_locants[0] == 1
            and unsat_suffix == "-1-ene"
        ):
            unsat_suffix = "ene"
        # name ends in 'e' (e.g. "cyclohexene", "cyclohexa-1,3-diene", "cyclooctyne")
        name_str = cyclo_prefix + unsat_suffix
        # stem strips terminal 'e' (for FG suffix attachment)
        if name_str.endswith("e"):
            stem = name_str[:-1]
        else:
            stem = name_str
        # alkyl_stem: the part before "-ene"/"-yne" (for -yl substituent)
        # For unsaturated rings, alkyl_stem = cyclo + stem_base (no saturation suffix)
        alkyl_stem = cyclo_prefix

    return NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=alkyl_stem,
        naming_method="systematic",
        indicated_hydrogen=None,
        numbering_options=(),
        ring_unsaturation_bonds=ring_bond_pairs if ring_bond_pairs else None,
    )


# ---------------------------------------------------------------------------
# Indicated-H detection (P-25.7.1.3)
# ---------------------------------------------------------------------------

# Atoms whose standard (lowest normal) valence is 2 — these atoms are
# *intrinsically* sp3 in any neutral ring (they carry their lone pairs and
# cannot participate in an endocyclic double bond without becoming charged
# or hypervalent).  When such atoms exist in a partially-saturated ring,
# they do NOT count as indicated-H carriers — only atoms whose standard
# valence is ≥3 (C, N, P, As, etc.) and that lack an endocyclic DB qualify.
_DIVALENT_HETERO_BLOCKERS: frozenset[str] = frozenset({"O", "S", "Se", "Te"})

# Monovalent heteroatoms (halogens).  In a ring they carry only their two ring
# single bonds and cannot form an endocyclic double bond — they segment the
# DB-capable atom chain exactly as a divalent chalcogen does.
_MONOVALENT_HALOGENS: frozenset[str] = frozenset({"F", "Cl", "Br", "I", "At"})


def _atom_has_endocyclic_db(atom, ring_atom_set: frozenset, mol) -> bool:
    """True if ``atom`` participates in an endocyclic double/triple bond."""
    for bond in atom.GetBonds():
        other = bond.GetOtherAtomIdx(atom.GetIdx())
        if other in ring_atom_set and bond.GetBondTypeAsDouble() >= 2.0:
            return True
    return False


def _is_ring_db_blocker(atom, ring_atom_set: frozenset, mol) -> bool:
    """True if ``atom`` cannot lie in an endocyclic double bond of the ring's
    mancude tautomer, and is therefore neither an indicated-H carrier nor an
    added-hydrogen (hydro) position.

    Blockers:
      * a neutral divalent chalcogen (O/S/Se/Te at standard valence 2), and
      * a monovalent halogen / mercury that carries no endocyclic DB (only its
        two ring single bonds — no spare valence for a ring double bond).

    These atoms are divalent/monovalent in both the mancude AND the saturated
    ring, so they never represent "added" or "indicated" hydrogen.
    """
    sym = atom.GetSymbol()
    if sym in _DIVALENT_HETERO_BLOCKERS:
        return atom.GetFormalCharge() == 0 and atom.GetTotalValence() == 2
    if sym in _MONOVALENT_HALOGENS or sym == "Hg":
        return not _atom_has_endocyclic_db(atom, ring_atom_set, mol)
    return False


def _ring_endocyclic_db_count(ring_atom_set: frozenset, mol) -> int:
    """Count endocyclic double bonds (C=C, C=N, etc.) within a ring system."""
    seen: set[frozenset] = set()
    n = 0
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(atom_idx)
            if other not in ring_atom_set:
                continue
            key = frozenset({atom_idx, other})
            if key in seen:
                continue
            seen.add(key)
            if bond.GetBondTypeAsDouble() >= 2.0:
                n += 1
    return n


def _kekule_endocyclic_db_count(ring_atom_set: frozenset, mol) -> int:
    """Count the ring's TRUE (Kekulé) endocyclic double bonds.

    ``_ring_endocyclic_db_count`` counts ``GetBondTypeAsDouble() >= 2.0`` on the
    perceived molecule, which is **zero** for any ring RDKit marks aromatic
    (its bonds are flagged AROMATIC, not DOUBLE).  For an aromatic ring we must
    therefore Kekulise a copy before counting, otherwise we cannot tell a real
    mancude ring (with Kekulé double bonds) from a ring that RDKit aromatised
    purely by lone-pair donation and which carries NO double bond at all.

    The latter case is exactly the fully-SATURATED all-heteroatom ring:
    odd-membered all-NH / all-PH / all-chalcogen rings (``N1NNNN1``,
    ``P1PPPP1``, ``S1SSSS1``, the 3-ring analogues, etc.) come back from RDKit
    with every ring atom aromatic yet kekulise to **zero** double bonds —
    structurally they are pentazolidine / pentaphospholane / pentathiolane,
    NOT the mancude pentazole / pentaphosphole (which has two real double bonds
    and a single indicated-H carrier).  Naming such a ring with the unsaturated
    HW stem emits a DIFFERENT structure (P-31.1.4: the mancude reference state
    must reflect the actual ring unsaturation).

    Returns the Kekulé endocyclic double-bond count, or, when the molecule
    cannot be Kekulised, falls back to the perceived double-bond count.
    """
    from rdkit import Chem
    from rdkit.Chem import BondType

    try:
        mk = Chem.Mol(mol)
        Chem.Kekulize(mk, clearAromaticFlags=True)
    except Exception:
        return _ring_endocyclic_db_count(ring_atom_set, mol)

    seen: set[frozenset] = set()
    n = 0
    for atom_idx in ring_atom_set:
        atom = mk.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(atom_idx)
            if other not in ring_atom_set:
                continue
            key = frozenset({atom_idx, other})
            if key in seen:
                continue
            seen.add(key)
            if bond.GetBondType() == BondType.DOUBLE:
                n += 1
    return n


def _max_mancude_db_count(
    ring_atom_set: frozenset,
    mol,
    elements_used: list[str],
    valence_aware: bool = False,
) -> int:
    """Compute the maximum non-cumulative endocyclic double-bond count for
    the ring's mancude (maximally non-cumulative) tautomer.

    Each "DB-incapable" ring atom (a divalent chalcogen O/S/Se/Te) breaks the
    ring into segments of DB-capable atoms; max DBs is sum of floor(seg_len/2)
    over each segment plus the closing wrap-around if no blockers exist.

    For all-DB-capable rings (no chalcogens), the max is floor(n/2).

    When ``valence_aware`` is True, monovalent halogens and mercury are ALSO
    treated as blockers (they have no spare valence for a ring double bond).
    This is needed by the partial-saturation regime decision so that e.g.
    iodinine ([IH]1CC=CC=C1) and mercurinine ([Hg]1CC=CC=C1) report max 2 DBs
    (matching their cur 2) rather than the naive floor(6/2)=3 — without it they
    would be wrongly classified as "partially saturated".  The default (False)
    preserves the chalcogen-only behaviour relied on by the indicated-H path.
    """
    n = len(ring_atom_set)
    if n == 0:
        return 0

    # Sort atoms in cycle order to walk segments
    cycle = _get_ring_cycle_order(ring_atom_set, mol)
    blockers: list[bool] = []
    for atom_idx in cycle:
        atom = mol.GetAtomWithIdx(atom_idx)
        sym = atom.GetSymbol()
        # An atom is a "blocker" only if it is a divalent chalcogen at its
        # standard (neutral) valence — atoms with charge or non-standard
        # valence may participate in DBs and are not blockers.
        is_blocker = False
        if sym in _DIVALENT_HETERO_BLOCKERS:
            if atom.GetFormalCharge() == 0 and atom.GetTotalValence() == 2:
                is_blocker = True
        elif valence_aware and (sym in _MONOVALENT_HALOGENS or sym == "Hg"):
            if not _atom_has_endocyclic_db(atom, ring_atom_set, mol):
                is_blocker = True
        blockers.append(is_blocker)

    if not any(blockers):
        # All atoms are DB-capable: ring max = floor(n/2)
        return n // 2

    # Walk from the first blocker, then collect segments of consecutive
    # DB-capable atoms between blockers.
    start = blockers.index(True)
    segments: list[int] = []
    cur = 0
    for k in range(n):
        idx = (start + k) % n
        if blockers[idx]:
            if cur > 0:
                segments.append(cur)
            cur = 0
        else:
            cur += 1
    if cur > 0:
        segments.append(cur)

    # Each segment of L DB-capable atoms supports floor(L / 2) DBs.
    return sum(seg // 2 for seg in segments)


def _pick_indicated_h_locant(
    ring_atom_set: frozenset,
    locant_map: dict[int, int],
    mol,
) -> int | None:
    """Return the lowest locant of a ring atom that:
      (a) participates in NO endocyclic double bond, AND
      (b) is NOT an intrinsically-divalent blocker (O/S/Se/Te std valence).

    This is the IUPAC indicated-H locant (P-25.7.1.3) for a ring whose
    heavy-atom skeleton has the maximum non-cumulative double-bond count
    but still contains one or more sp3 ring atoms.

    Returns None when no qualifying atom exists (fully unsaturated ring).
    """
    if not locant_map:
        return None
    candidates: list[int] = []
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        # Skip blocker atoms (divalent chalcogens, monovalent halogens, Hg):
        # they cannot lie in a ring DB and are never indicated-H carriers.
        if _is_ring_db_blocker(atom, ring_atom_set, mol):
            continue
        in_endocyclic_db = False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(atom_idx)
            if other in ring_atom_set and bond.GetBondTypeAsDouble() >= 2.0:
                in_endocyclic_db = True
                break
        if not in_endocyclic_db:
            loc = locant_map.get(atom_idx)
            if loc is not None:
                candidates.append(loc)
    if not candidates:
        return None
    return min(candidates)


# Hantzsch-Widman unsaturated stems that OPSIN does NOT recognise as a valid
# standalone parent (the retained carbocyclic-style name displaced them):
#   * 6-membered single-oxygen ring → "oxine"  (retained: pyran)
#   * 6-membered single-nitrogen ring → "azine" (retained: pyridine)
# For these, the mancude / partial-hydro HW forms must NOT be emitted — the
# retained-name path (pyran/pyridine and their hydro derivatives) covers them.
# Probed against OPSIN 2.8.0: every other single-heteroatom HW unsaturated stem
# (thiine, selenine, phosphinine, arsinine, silinine for 6-rings; all 5- and
# 7-ring stems) DOES parse, so this guard is intentionally narrow.
def _hw_unsaturated_stem_is_opsin_rejected(
    ring_size: int, elements_used: list[str]
) -> bool:
    if ring_size != 6:
        return False
    if len(elements_used) != 1:
        return False
    return elements_used[0] in ("O", "N")


def _collect_hydro_locants(
    ring_atom_set: frozenset,
    locant_map: dict[int, int],
    mol,
) -> list[int]:
    """Return sorted IUPAC locants of ring atoms that are "added-hydrogen"
    (saturated) positions relative to the ring's mancude form.

    An added-H position is a ring atom that:
      (a) participates in NO endocyclic double/triple bond, AND
      (b) is NOT an intrinsically-divalent chalcogen blocker (O/S/Se/Te at
          standard valence 2 — these are divalent in BOTH the mancude and the
          saturated ring, so they are never "added-H"), AND
      (c) carries no formal charge.

    These are exactly the ring atoms whose locants appear in a
    ``<locs>-<mult>hydro-`` prefix on the mancude parent (P-31.1.4.2).
    Returns [] if any qualifying atom lacks a locant.
    """
    if not locant_map:
        return []
    locs: list[int] = []
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        # Blocker atoms (chalcogens / halogens / Hg) are divalent/monovalent in
        # both the mancude and the saturated ring — never an added-H position.
        if _is_ring_db_blocker(atom, ring_atom_set, mol):
            continue
        if atom.GetFormalCharge() != 0:
            return []
        in_endocyclic_unsat = False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(atom_idx)
            if other in ring_atom_set and bond.GetBondTypeAsDouble() >= 2.0:
                in_endocyclic_unsat = True
                break
        if in_endocyclic_unsat:
            continue
        loc = locant_map.get(atom_idx)
        if loc is None:
            return []
        locs.append(loc)
    return sorted(locs)


def _mancude_parent_indicated_h_locant(
    ring_atom_set: frozenset,
    locant_map: dict[int, int],
    mol,
) -> int | None:
    """For a partially-saturated HW ring, return the indicated-H locant that
    the MANCUDE PARENT carries, or None when the parent is fully unsaturated.

    The mancude parent has the maximum non-cumulative double-bond count.  If
    the number of DB-capable (non-blocker) ring atoms is ODD, exactly one such
    atom remains sp3 in the parent and carries an indicated hydrogen
    (e.g. ``1H-phosphole``: the 5-ring P-parent leaves the P sp3; ``1H-azepine``:
    the 7-ring N-parent leaves the N sp3).  When the count is EVEN, every
    DB-capable atom pairs into a double bond and the parent needs no
    indicated-H (e.g. the dioxine / thiazepine parents).

    The indicated-H atom is chosen per P-25.7.1.3 as the lowest locant among
    DB-capable non-blocker ring atoms.  Among the actual ring's saturated
    positions, the indicated-H carrier of the parent is the lowest-locant
    DB-capable non-blocker atom; we prefer a heteroatom (whose valence forces
    the H, e.g. P-H, N-H) when one is present at the lowest position so the
    emitted ``<loc>H`` matches OPSIN's parent convention.
    """
    if not locant_map:
        return None
    db_capable: list[tuple[int, bool]] = []  # (locant, is_heteroatom)
    for atom_idx in ring_atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        # Blocker atoms (chalcogens / halogens / Hg) are never DB-capable.
        if _is_ring_db_blocker(atom, ring_atom_set, mol):
            continue
        loc = locant_map.get(atom_idx)
        if loc is None:
            continue
        db_capable.append((loc, atom.GetSymbol() != "C"))
    # ODD count → exactly one residual indicated-H atom in the mancude parent.
    if len(db_capable) % 2 == 0:
        return None
    # Prefer the lowest-locant heteroatom carrier (its valence forces the H);
    # otherwise the lowest-locant DB-capable atom overall.
    hetero = sorted(loc for loc, is_het in db_capable if is_het)
    if hetero:
        return hetero[0]
    return min(loc for loc, _ in db_capable)


def _build_canonical_numbering(
    ring_size: int,
    locant_map: dict[int, int],
):
    """Build a Numbering object pinning the ring atoms to their IUPAC
    locants (1..n) as computed by _compute_hw_locants.  Returns a single-item
    tuple of Numbering, or empty tuple if locant_map is incomplete.
    """
    from iupac_namer.types import Locant, Numbering
    if not locant_map:
        return ()
    try:
        assignments = tuple(
            (atom_idx, Locant.numeric(loc_val))
            for atom_idx, loc_val in sorted(locant_map.items())
            if loc_val is not None
        )
        if len(assignments) != ring_size:
            return ()
        locant_set = tuple(Locant.numeric(i + 1) for i in range(ring_size))
        return (Numbering(_assignments=assignments, locant_set=locant_set),)
    except Exception:
        return ()


def _build_canonical_numbering_multi(
    ring_size: int,
    locant_maps: list[dict[int, int]],
):
    """Build Numbering objects for several heteroatom-optimal locant maps.

    Returns a tuple of Numbering (deduped), or empty tuple if no map is
    complete.  Used to pin every heteroatom-respecting numbering so the
    strategy can still minimise substituent locants among them.
    """
    from iupac_namer.types import Locant, Numbering
    out: list = []
    seen: set[tuple] = set()
    for lmap in locant_maps or ():
        single = _build_canonical_numbering(ring_size, lmap)
        if not single:
            continue
        nb = single[0]
        if nb._assignments in seen:
            continue
        seen.add(nb._assignments)
        out.append(nb)
    return tuple(out)


def try_hantzsch_widman(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> NamedParent | None:
    """Build a Hantzsch-Widman name for a heterocyclic ring (P-22.1.2).

    Supports 3-10 membered rings with a single heteroatom (multi-heteroatom
    support is partial — only the senior element is used for now).

    Returns None if the ring is not HW-eligible.
    """
    ring_size = ring_system.ring_size
    if ring_size < 3 or ring_size > 10:
        return None

    heteroatoms = ring_system.heteroatoms
    if not heteroatoms:
        return None

    hw_tables = get_hw_tables()
    prefixes_list = hw_tables.get("prefixes", [])
    stems = hw_tables.get("stems", {})
    six_groups = hw_tables.get("six_membered_groups", {})

    # Build element -> HW prefix map
    elem_to_prefix: dict[str, str] = {}
    for entry in prefixes_list:
        elem_to_prefix[entry["element"]] = entry["prefix"]

    # Compute HW locants from the ring topology.
    # P-31.1.2.2: the most senior heteroatom gets locant 1; numbering direction
    # is chosen to give heteroatoms the lowest locant set.
    ring_atoms = ring_system.atom_indices
    ring_cycle = _get_ring_cycle_order(ring_atoms, mol)

    hetero_set = {hp.atom_idx for hp in heteroatoms}
    prio_map: dict[int, int] = {
        hp.atom_idx: _HW_PRIORITY.get(hp.element, 0)
        for hp in heteroatoms
    }

    # Indicated-H tiebreaker (P-31.1.4.3.4): when two numbering directions tie
    # on heteroatom locants, prefer the one that gives the lowest indicated-H
    # locant.  Carriers are non-blocker ring atoms with no endocyclic DB; the
    # set is direction-independent, so we can compute it from the ring graph
    # before locants are assigned.
    indicated_h_carriers: set[int] = set()
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        # Blocker atoms (chalcogens / halogens / Hg) are never carriers.
        if _is_ring_db_blocker(atom, ring_atoms, mol):
            continue
        in_endo_db = False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(atom_idx)
            if other in ring_atoms and bond.GetBondTypeAsDouble() >= 2.0:
                in_endo_db = True
                break
        if not in_endo_db:
            indicated_h_carriers.add(atom_idx)

    # Lambda-atom tiebreaker (P-14.1.2): when two numbering directions tie on
    # heteroatom locants, prefer the one that places the non-standard-valence
    # heteroatom (lambda atom) at the lowest locant.  Lambda atoms are ring
    # heteroatoms whose actual valence differs from the element's standard;
    # the formal charge correction matches the lambda block in the post-locant
    # code so that a -ide (charge-reduced) atom is NOT counted as lambda.
    lambda_atom_indices: set[int] = set()
    for hp in heteroatoms:
        std_val = _STANDARD_VALENCE.get(hp.element)
        if std_val is None:
            continue
        atom = mol.GetAtomWithIdx(hp.atom_idx)
        actual_val = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        if charge < 0 and actual_val == std_val + charge:
            continue
        if charge > 0 and actual_val == std_val + charge:
            continue
        if actual_val != std_val:
            lambda_atom_indices.add(hp.atom_idx)

    locant_map = _compute_hw_locants(
        ring_cycle,
        hetero_set,
        prio_map,
        indicated_h_tiebreaker=frozenset(indicated_h_carriers) if indicated_h_carriers else None,
        lambda_atoms=frozenset(lambda_atom_indices) if lambda_atom_indices else None,
    )

    # Attach computed locants to heteroatom objects for sorting
    from iupac_namer.types import Locant, HeteroPosition
    hetero_with_locants = []
    for hp in heteroatoms:
        loc_val = locant_map.get(hp.atom_idx) if locant_map else None
        loc = Locant.numeric(loc_val) if loc_val is not None else hp.locant
        hetero_with_locants.append(HeteroPosition(atom_idx=hp.atom_idx, element=hp.element, locant=loc))

    # Sort heteroatoms by HW priority (highest first), then by locant
    sorted_hetero = sorted(
        hetero_with_locants,
        key=lambda hp: (-_HW_PRIORITY.get(hp.element, -1),
                        hp.locant._numeric_value if hp.locant and hp.locant._numeric_value else 0),
    )

    # Validate all heteroatom elements have HW prefixes
    elements_used = []
    for hp in sorted_hetero:
        elem = hp.element
        if elem not in elem_to_prefix:
            return None  # Unsupported element for HW
        elements_used.append(elem)

    # Build stem based on ring size and saturation
    aromatic = ring_system.aromatic

    # P-31.1.4 structural-saturation override for carbon-free rings that RDKit
    # marks aromatic but which carry NO real (Kekulé) endocyclic double bond.
    # Odd-membered all-NH / all-PH / all-chalcogen rings (N1NNNN1, P1PPPP1,
    # S1SSSS1, the 3-ring analogues, …) are aromatised by RDKit through pure
    # lone-pair donation, yet kekulise to zero double bonds — they ARE the
    # fully-saturated ring (pentazolidine / pentaphospholane / pentathiolane),
    # not the mancude azole/phosphole (two real double bonds + one indicated-H
    # carrier).  Treating ``aromatic`` as True here would pick the unsaturated
    # HW stem and emit a DIFFERENT structure that round-trips wrong.  So when a
    # heterocyclic ring is flagged aromatic but has zero Kekulé endocyclic
    # double bonds, we treat it as non-aromatic for stem / saturation-regime
    # selection.  (Carbon-bearing aromatics — pyrrole, thiophene, pyridine —
    # always kekulise to >0 double bonds and are unaffected; benzene-class rings
    # never reach the HW path.)
    if aromatic and _kekule_endocyclic_db_count(ring_system.atom_indices, mol) == 0:
        aromatic = False

    # Detect endocyclic double bonds (non-aromatic ring with C=C or similar within ring)
    ring_atoms = ring_system.atom_indices
    has_endocyclic_double_bond = False
    if not aromatic:
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            for bond in atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(atom_idx)
                if other_idx in ring_atoms and bond.GetBondTypeAsDouble() >= 2.0:
                    has_endocyclic_double_bond = True
                    break
            if has_endocyclic_double_bond:
                break

    # -----------------------------------------------------------------------
    # Lambda-valence detection (IUPAC P-14.1.1 / P-14.1.2):
    # When a ring heteroatom has a valence different from its standard value,
    # IUPAC requires the λ<n> (lambda) convention.  The parent ring is named
    # using the MAXIMALLY UNSATURATED (aromatic) form, and the lambda marker
    # is prepended as "<locant>lambda<valence>-" (e.g. "4lambda5-").
    #
    # Example: 6-membered N/P ring where one P has valence 5 due to N=P bond:
    #   parent → "1,3,5,2,4,6-triazatriphosphinine" (aromatic form)
    #   lambda → "4lambda5-1,3,5,2,4,6-triazatriphosphinine"
    #
    # This generalises to any heteroatom with non-standard valence: S(IV/VI),
    # As(V), Sb(V), etc.
    #
    # IMPORTANT distinction: the aromatic (maximally unsaturated) parent is
    # used ONLY when the non-standard valence is caused by an ENDOCYCLIC
    # double bond (e.g. N=P in the ring, which makes P valence 5).  When the
    # non-standard valence comes solely from exocyclic bonds (e.g. P=O in
    # oxazaphosphinine, S=O in thiaazine), the SATURATED parent is kept and
    # the lambda marker is still emitted so OPSIN knows the heteroatom
    # valence, but we do NOT switch to the aromatic stem.
    # -----------------------------------------------------------------------
    lambda_descriptors: list[tuple[int, int]] = []  # (locant, actual_valence)
    lambda_endocyclic: list[bool] = []  # True if caused by endocyclic double bond
    ring_atom_set_for_lambda: set[int] = ring_system.atom_indices
    if locant_map:
        for hp in sorted_hetero:
            std_val = _STANDARD_VALENCE.get(hp.element)
            if std_val is None:
                continue
            atom = mol.GetAtomWithIdx(hp.atom_idx)
            actual_val = atom.GetTotalValence()
            # Anionic ring heteroatoms (e.g. [n-] in imidazolate) have valence
            # reduced by one electron pair; IUPAC treats this as a formal
            # charge to be named with a ring-anion suffix (-ide, P-72.2),
            # NOT as a non-standard lambda valence.  Skip lambda emission
            # when the valence difference is entirely explained by the formal
            # charge.  Example: imidazolate c1c[n-]cn1 → [n-] has val=2,
            # std=3, charge=-1 — the deficit matches the charge, so no lambda.
            # Without this guard we would emit "1lambda2-1,3-diazol-3-ide".
            charge = atom.GetFormalCharge()
            if charge < 0 and actual_val == std_val + charge:
                continue
            if charge > 0 and actual_val == std_val + charge:
                continue
            if actual_val != std_val:
                loc_val = locant_map.get(hp.atom_idx)
                if loc_val is not None:
                    # Check if this atom participates in an endocyclic double bond
                    has_endo_dbl = any(
                        mol.GetBondBetweenAtoms(hp.atom_idx, nb.GetIdx()).GetBondTypeAsDouble() >= 2.0
                        for nb in atom.GetNeighbors()
                        if nb.GetIdx() in ring_atom_set_for_lambda
                    )
                    lambda_descriptors.append((loc_val, actual_val))
                    lambda_endocyclic.append(has_endo_dbl)

    # The aromatic (maximally unsaturated) parent stem is used only when at
    # least one non-standard-valence heteroatom has an endocyclic double bond.
    # For purely exocyclic non-standard valence (P=O, S=O, etc.) the
    # saturated parent is correct.
    use_lambda_parent = (
        any(lambda_endocyclic) and not aromatic
        if lambda_endocyclic else False
    )

    # -----------------------------------------------------------------------
    # Indicated-H detection (P-25.7.1.3) — Stage 6 R3-B
    #
    # A monocyclic HW ring is in "indicated-H form" (mancude parent name with
    # an italicised "<N>H-" prefix marking the sp3 ring atom that breaks the
    # otherwise-maximum unsaturation) when:
    #   (a) ring is non-aromatic, AND
    #   (b) ring already has the maximum non-cumulative DB count for its
    #       heteroatom composition, AND
    #   (c) at least one DB-capable ring atom (i.e. not a neutral divalent
    #       chalcogen) carries no endocyclic DB.
    #
    # Distinguishes "2H-1,3-oxazine" (mancude 6-ring with O+N: max 2 DBs;
    # currently 2 DBs; one sp3 carrier) from "1,4-dihydropyridine" (6-ring
    # with one N: max 3 DBs; currently 2 DBs → "dihydro" form, not indH).
    #
    # Only applies to 5- to 8-membered rings on the HW path; larger rings
    # use replacement nomenclature with explicit dihydro/tetrahydro etc.
    # -----------------------------------------------------------------------
    indicated_h_locant: int | None = None
    use_indicated_h_parent = False
    # P-31.1.4 ring-unsaturation regime.  The mancude (maximally non-cumulative
    # unsaturated) parent name is the IUPAC reference state for a heterocycle;
    # the ACTUAL ring unsaturation is then expressed relative to it:
    #   * cur_dbs == 0            → fully saturated → saturated HW stem
    #   * cur_dbs == max_dbs      → fully mancude:
    #         - with an sp3 carrier → indicated-H form (``2H-1,3-oxazine``)
    #         - without a carrier   → bare unsaturated stem (``1,4-dioxine``)
    #   * 0 < cur_dbs < max_dbs   → partially saturated → mancude parent + a
    #                               ``<locs>-<mult>hydro-`` prefix
    #                               (``4,5,6,7-tetrahydro-1,4-thiazepine``)
    # The previous code only handled the first case and the indicated-H subcase
    # of the second; carrier-free mancude rings (dioxine/dithiine/oxathiine)
    # were wrongly emitted SATURATED and partially-saturated rings (thiazepine,
    # phosphole) were wrongly emitted FULLY mancude.
    use_mancude_parent = False
    partial_hydro_locants: list[int] = []
    # Indicated-H locant for a PARTIALLY-saturated ring (P-31.1.4.2.4).  This
    # is the sp3 ring atom that, after the hydro prefix accounts for the
    # double-bond deficit, still carries an "extra" indicated hydrogen because
    # the saturated-atom count is odd.  Usually this equals the mancude
    # parent's indicated-H carrier (e.g. C2 of 2H-pyran), but when that carrier
    # is itself unsaturated in the actual molecule (e.g. the N of azepine is in
    # a C=N here), the indicated-H must instead fall on one of the molecule's
    # real sp3 atoms.  Stored so the assembly step emits the right "<loc>H-".
    partial_indicated_h: int | None = None
    if (
        not aromatic
        and not use_lambda_parent
        and ring_size in (5, 6, 7, 8)
        and locant_map is not None
        and not _hw_unsaturated_stem_is_opsin_rejected(ring_size, elements_used)
    ):
        cur_dbs = _ring_endocyclic_db_count(ring_system.atom_indices, mol)
        max_dbs = _max_mancude_db_count(
            ring_system.atom_indices, mol, elements_used
        )
        # Valence-aware max additionally treats monovalent halogens / Hg as
        # blockers (no spare valence for a ring DB).  It governs ONLY the
        # partial-saturation gate below: a ring is "partially saturated" only
        # when cur < the valence-aware max.  The chalcogen-only ``max_dbs``
        # still governs the indicated-H / carrier-free-mancude (cur==max)
        # decision, so the existing heavy-halide/Hg handling (bare iodinine,
        # mercurinine via the _choose_hw_stem heavy path) is left unchanged.
        max_dbs_va = _max_mancude_db_count(
            ring_system.atom_indices, mol, elements_used, valence_aware=True
        )
        if cur_dbs > 0 and cur_dbs == max_dbs:
            indicated_h_locant = _pick_indicated_h_locant(
                ring_system.atom_indices, locant_map, mol
            )
            if indicated_h_locant is not None:
                use_indicated_h_parent = True
            else:
                # Fully mancude with every ring atom in a DB or an intrinsic
                # divalent-chalcogen blocker — no sp3 carrier, so there is no
                # indicated-H.  The ring still needs the UNSATURATED stem.
                # This is the dioxine / dithiine / oxathiine family
                # (1,4-dioxine = O1C=COC=C1), which OPSIN accepts via the
                # unsaturated HW stem but which the saturated-stem default
                # wrongly emitted as 1,4-dioxane.
                use_mancude_parent = True
        elif 0 < cur_dbs < max_dbs and cur_dbs < max_dbs_va:
            # Partially saturated: name the mancude parent and add a hydro
            # prefix for the sp3 ring atoms (P-31.1.4.2 / P-31.1.4.3).  The
            # number of "missing" double bonds corresponds to an even count of
            # saturated ring atoms (hydro comes in pairs): 2*(max_dbs-cur_dbs)
            # hydrogens are added.  The sp3 ring atoms carry those added H's.
            #
            # The MANCUDE PARENT may itself carry an indicated-H atom (an odd
            # residual DB-capable atom that stays sp3 even at max unsaturation,
            # e.g. the P of 1H-phosphole or the N of 1H-azepine).  That atom is
            # NOT a hydro position — it is sp3 in the parent too — so it must be
            # excluded from the hydro-locant list.  Without this exclusion the
            # hydro count is inflated by 1 (e.g. phosphole's P+2 CH2 = 3 instead
            # of the correct dihydro at 2 carbons).
            mancude_parent_ih = _mancude_parent_indicated_h_locant(
                ring_system.atom_indices, locant_map, mol
            )
            sp3_locants = _collect_hydro_locants(
                ring_system.atom_indices, locant_map, mol
            )
            needed = 2 * (max_dbs_va - cur_dbs)
            if mancude_parent_ih is not None and mancude_parent_ih in sp3_locants:
                # The mancude parent's indicated-H carrier is sp3 here too
                # (e.g. C2 of 2H-pyran): it is the partial ring's indicated-H,
                # not a hydro position.
                partial_indicated_h = mancude_parent_ih
                sp3_locants = [l for l in sp3_locants if l != mancude_parent_ih]
            elif len(sp3_locants) == needed + 1:
                # The saturated-atom count is odd by one, but the mancude
                # parent's indicated-H carrier is unsaturated in this molecule
                # (e.g. azepine's N is in a C=N).  One of the real sp3 atoms
                # therefore carries the indicated hydrogen (P-31.1.4.2.4):
                # choose the lowest-locant sp3 atom, preferring a heteroatom
                # whose valence forces the H, so the remaining even-count set
                # is a well-defined hydro prefix.
                sp3_set = {
                    a.GetIdx() for a in mol.GetAtoms()
                    if a.GetIdx() in ring_system.atom_indices
                    and locant_map.get(a.GetIdx()) in sp3_locants
                }
                het_locs = sorted(
                    locant_map[i] for i in sp3_set
                    if mol.GetAtomWithIdx(i).GetSymbol() != "C"
                )
                partial_indicated_h = het_locs[0] if het_locs else min(sp3_locants)
                sp3_locants = [l for l in sp3_locants if l != partial_indicated_h]
            partial_hydro_locants = sp3_locants
            # Reject when the remaining hydro-atom count is not exactly twice
            # the double-bond deficit (i.e. the cited hydro locants don't form
            # a well-defined hydro prefix), or when a cited sp3 atom couldn't
            # be mapped to a locant.  Use the valence-aware max so a ring whose
            # max is limited by a halogen/Hg blocker still computes the right
            # deficit.
            if len(partial_hydro_locants) != needed:
                partial_hydro_locants = []
                partial_indicated_h = None

    stem_data = stems.get(str(ring_size))
    if stem_data is None:
        return None

    # Choose stem variant — for lambda parents, indicated-H mancude tautomers,
    # carrier-free mancude rings, and partially-saturated rings (whose mancude
    # PARENT name uses the unsaturated stem before the hydro prefix is added),
    # pretend the ring is aromatic so _choose_hw_stem returns the unsaturated/
    # aromatic stem (e.g. "ine" for 6-membered group-A/B rings, "epine" for
    # 7-membered).  For indicated-H, the trailing "-<locant>H-" prefix is later
    # prepended to the assembled name to mark the sp3 carrier (P-25.7.1.3).
    use_partial_hydro = bool(partial_hydro_locants)
    effective_aromatic = (
        aromatic
        or use_lambda_parent
        or use_indicated_h_parent
        or use_mancude_parent
        or use_partial_hydro
    )
    hw_stem = _choose_hw_stem(ring_size, stem_data, six_groups, elements_used, effective_aromatic, has_endocyclic_double_bond)
    if hw_stem is None:
        return None

    # Build the HW name.
    # P-22.1.3.3 / P-31.1.3.1: when there are multiple heteroatoms, locants are
    # cited together in one locant prefix, then the element prefixes are concatenated
    # (with HW elision) and the stem appended directly.
    #
    # For SAME elements:  multiplier + base prefix
    #   2x O at 1,3: "1,3-" + "di" + "ox" (elided) + "olane" = "1,3-dioxolane"
    #   3x O at 1,3,5: "1,3,5-" + "tri" + "ox" (elided) + "ane" = "1,3,5-trioxane"
    #
    # For DIFFERENT elements: concatenate prefixes with per-component elision
    #   O@1, N@3: "1,3-" + "ox" (oxa elided before aza→a) + "az" (aza elided before olid→o)
    #             + "olidine" = "1,3-oxazolidine"
    #
    # For a single heteroatom: no locant needed (e.g. "oxolane", "oxirane").
    all_have_locants = all(hp.locant is not None for hp in sorted_hetero)
    need_locants = len(sorted_hetero) > 1

    if need_locants and all_have_locants:
        # Step 1: group heteroatoms by element in priority order (already done
        # by sorted_hetero = sort by -priority then locant).
        # Step 2: collect locants grouped by element (priority order).
        #
        # P-22.1.3.3 / P-25.3.1: in heteroatom replacement nomenclature, when
        # multiple element types are present, locants are cited in PRIORITY
        # ORDER grouped by element — not by ascending numeric value.  E.g.
        # for O at {1,3,5}, Si at {2,4,6} the locant prefix is
        # "1,3,5,2,4,6-" (O-locants first, then Si-locants), not
        # "1,2,3,4,5,6-" (which OPSIN rejects because it cannot apportion
        # locants to the "tri" multipliers of oxa and sila).
        elem_order: list[str] = []
        elem_groups: dict[str, list[int]] = {}  # element -> list of locant values
        for hp in sorted_hetero:
            loc_val = (hp.locant._numeric_value or 0) if hp.locant else 0
            if hp.element not in elem_groups:
                elem_order.append(hp.element)
                elem_groups[hp.element] = []
            elem_groups[hp.element].append(loc_val)

        # Sort locants within each element group ascending (multiplier semantics
        # require the locants for a given element to be listed in ascending order).
        for elem in elem_order:
            elem_groups[elem].sort()

        # Build locant prefix in priority-grouped order
        locant_order: list[int] = []
        for elem in elem_order:
            locant_order.extend(elem_groups[elem])
        loc_prefix = ",".join(str(loc) for loc in locant_order) + "-"

        # Build prefix parts (before elision)
        raw_parts: list[str] = []
        for elem in elem_order:
            pref = elem_to_prefix[elem]
            n_elem = len(elem_groups[elem])
            if n_elem == 1:
                raw_parts.append(pref)
            else:
                multi = get_multiplier(n_elem)
                if multi is None:
                    return None  # Can't express multiplier
                raw_parts.append(multi + pref)  # e.g. "dioxa"

        # Apply HW elision: terminal 'a' of each component is elided before the
        # FIRST LETTER of the NEXT COMPONENT (or stem).
        elided_parts: list[str] = list(raw_parts)
        for i in range(len(elided_parts)):
            next_start: str
            if i + 1 < len(elided_parts):
                next_start = elided_parts[i + 1][0]
            elif hw_stem:
                next_start = hw_stem[0]
            else:
                next_start = ""
            if next_start in "aeiou" and elided_parts[i].endswith("a"):
                elided_parts[i] = elided_parts[i][:-1]

        prefix_body = "".join(elided_parts)
        hw_name = loc_prefix + prefix_body + hw_stem

    else:
        # Single heteroatom or locants unavailable: simple concatenation (legacy)
        prefix_components_raw: list[str] = [elem_to_prefix[e] for e in elements_used]

        # HW elision rule (P-22.1.3.2): terminal 'a' of each heteroatom prefix is
        # elided before a following component (another prefix or the stem) that
        # begins with a vowel (a, e, i, o, u).
        # Apply elision PER COMPONENT to handle multi-heteroatom cases correctly.
        parts: list[str] = list(prefix_components_raw)
        for i in range(len(parts)):
            next_start = parts[i + 1][0] if i + 1 < len(parts) else (hw_stem[0] if hw_stem else "")
            if next_start in "aeiou" and parts[i].endswith("a"):
                parts[i] = parts[i][:-1]

        prefix_str = "".join(parts)
        hw_name = prefix_str + hw_stem

    # Apply lambda-convention prefix (P-14.1.1/P-14.1.2): prepend
    # "<locant>lambda<valence>-" for each heteroatom with non-standard valence.
    # Descriptors are sorted by locant (ascending) so the ordering is canonical.
    if lambda_descriptors:
        lambda_descriptors.sort(key=lambda d: d[0])
        lambda_str = ",".join(
            f"{loc}lambda{val}" for loc, val in lambda_descriptors
        ) + "-"
        hw_name = lambda_str + hw_name

    # Apply indicated-H prefix (P-25.7.1.3): for partially-saturated mancude
    # tautomers, prepend "<locant>H-" to the assembled HW name.  The locant
    # is chosen by _pick_indicated_h_locant (lowest locant of the sp3 ring
    # atom that breaks max unsaturation).  E.g. for `2H-1,3-oxazine` the
    # body is "1,3-oxazine" (built above with the unsaturated stem) and the
    # "2H-" is prepended here.
    if use_indicated_h_parent and indicated_h_locant is not None:
        hw_name = f"{indicated_h_locant}H-" + hw_name

    # Apply partial-saturation hydro prefix (P-31.1.4.2 / P-31.1.4.3): a
    # partially-saturated HW ring is named on its MANCUDE parent (the
    # unsaturated stem chosen above) with a ``<locs>-<mult>hydro-`` prefix
    # citing the saturated ring positions.  When the mancude PARENT itself
    # requires an indicated-H (e.g. ``1H-phosphole`` for the 5-ring P parent,
    # whose P carries the residual H), that ``<loc>H-`` is inserted between
    # the hydro prefix and the stem: ``2,3-dihydro-1H-phosphole``.
    if use_partial_hydro:
        hydro_mult = get_multiplier(len(partial_hydro_locants))
        if hydro_mult is None:
            return None
        hydro_loc_str = ",".join(str(l) for l in partial_hydro_locants)
        # Indicated-H carrier for the partially-saturated ring: the residual
        # sp3 atom (odd-count leftover) that carries an H after the hydro
        # prefix accounts for the double-bond deficit.  Computed alongside
        # partial_hydro_locants so it falls on a REAL sp3 atom even when the
        # mancude parent's carrier is unsaturated in this molecule.
        mancude_ih = partial_indicated_h
        if mancude_ih is not None:
            # "...hydro-<loc>H-<stem>": the H-prefix needs a trailing hyphen
            # (it is followed by the italic locant token).
            hw_name = f"{hydro_loc_str}-{hydro_mult}hydro-{mancude_ih}H-" + hw_name
        else:
            # "...hydro<stem>": the hydro prefix attaches DIRECTLY to the stem
            # with no separating hyphen (P-31.1.4.2 — a multiplied "hydro" is a
            # detachable prefix written solid against the parent, e.g.
            # "2,3-dihydrooxepine", "1,4-dithiine"→"2,3-dihydro-1,4-dithiine"
            # only gets a hyphen because the next token is a locant).
            sep = "-" if hw_name[:1].isdigit() else ""
            hw_name = f"{hydro_loc_str}-{hydro_mult}hydro{sep}" + hw_name

    # stem (for Method 2 suffix attachment): remove trailing 'e'
    if hw_name.endswith("e"):
        stem_str = hw_name[:-1]
    else:
        stem_str = hw_name

    # alkyl_stem: HW names generally do NOT support Method (1)
    # (P-29.2 restricts Method 1 to acyclic and monocyclic saturated hydrocarbons)
    alkyl_stem: str | None = None
    if not aromatic and not any(e in elements_used for e in ("N", "O", "S")):
        # Very conservative: only allow for simple saturated rings
        alkyl_stem = None  # Keep None for now

    # Pin the heteroatom-determined numbering so substituent locants always
    # respect the heteroatom locants that the HW name CITES.  The HW name fixes
    # the heteroatom locant set (e.g. ``1,4-oxazepane`` ⇒ O=1, N=4); if we leave
    # numbering_options empty, the engine's generic monocyclic numbering pass
    # enumerates all 2n traversals and minimises the substituent locant against
    # the heteroatom constraint — emitting e.g. "2-chloro-1,4-oxazepane" for a
    # chlorine that the cited 1,4 numbering actually places at locant 7
    # (round-trips to the wrong regioisomer).  Pinning EVERY heteroatom-optimal
    # numbering (symmetric rings expose several) lets the strategy still pick
    # the lowest substituent locant — but only among numberings consistent with
    # the cited heteroatoms (P-31.1.4.3.4 then P-14.5.2).
    #
    # The indicated-H / lambda criteria, when present, already pick a single
    # canonical numbering (the indicated-H carrier or lambda atom must align
    # with the prepended "<N>H-" / "<loc>lambda<val>-" descriptor), so keep the
    # single-map form there.
    numbering_options_out: tuple = ()
    pin_single = use_indicated_h_parent or bool(lambda_descriptors)
    # Partially-saturated and carrier-free-mancude HW rings (which set
    # effective_aromatic=True so the unsaturated stem is chosen) still have
    # their heteroatom locants FIXED by the cited HW name (e.g. 1,4-thiazepine
    # ⇒ S=1, N=4; 1,4-dioxine ⇒ O=1,O=4).  Pin every heteroatom-optimal
    # numbering so substituent locants follow the cited heteroatoms, exactly
    # as for the saturated path — otherwise the generic numbering pass would
    # minimise the substituent locant against the heteroatom constraint and
    # round-trip to the wrong regioisomer.
    pin_hetero_multi = use_mancude_parent or use_partial_hydro
    if pin_single and locant_map is not None:
        numbering_options_out = _build_canonical_numbering(ring_size, locant_map)
    elif locant_map is not None and (not effective_aromatic or pin_hetero_multi):
        # Pin the heteroatom-determined numbering for SATURATED / partially-
        # saturated / carrier-free-mancude HW heterocycles.  For these, the
        # cited heteroatom locants fix the traversal direction, so the
        # substituent locants must follow.  Genuinely aromatic HW rings are
        # excluded (pin_hetero_multi is False for them): they compete with a
        # retained aromatic name (e.g. thiophene over "thiole"), and pinning
        # the HW alternative's numbering perturbs the strategy's plan ranking,
        # which can flip a correctly-numbered retained-name plan to a worse one.
        all_optima = _compute_hw_locants_all_optima(
            ring_cycle, hetero_set, prio_map,
            indicated_h_tiebreaker=(
                frozenset(indicated_h_carriers) if indicated_h_carriers else None
            ),
            lambda_atoms=(
                frozenset(lambda_atom_indices) if lambda_atom_indices else None
            ),
        )
        numbering_options_out = _build_canonical_numbering_multi(
            ring_size, all_optima
        )

    return NamedParent(
        candidate=candidate,
        name=hw_name,
        stem=stem_str,
        alkyl_stem=alkyl_stem,
        naming_method="hantzsch_widman",
        indicated_hydrogen=None,
        numbering_options=numbering_options_out,
    )


def _choose_hw_stem(
    ring_size: int,
    stem_data: dict,
    six_groups: dict,
    elements: list[str],
    aromatic: bool,
    has_endocyclic_double_bond: bool = False,
) -> str | None:
    """Choose the correct HW stem given ring size, elements, and saturation.

    For non-aromatic rings with endocyclic double bonds (e.g. maleimide ring
    C1=CCNC1), the 'unsaturated' stem is used (e.g. 'azole') rather than the
    saturated stem.  This matches IUPAC P-22.1.2 Table 2-5 which uses the same
    stem for aromatic and partially-unsaturated rings in the HW system.
    """
    if ring_size == 6:
        # Six-membered rings have three groups (A, B, C).
        # P-22.1.2.2 Table 2-8: stem depends on the MOST JUNIOR group present
        # (A < B < C).  That is:
        #   all-group-A          → group_A stem ("ane"/"ine")   e.g. 1,3,5-trioxane
        #   any group-B (no C)   → group_B stem ("inane"/"ine") e.g. 1,3,5-triazinane, trisilinane
        #   any group-C          → group_C stem ("inane"/"inine") e.g. phosphinane
        #
        # For saturated 6-mem rings group-B and group-C both use "inane", so the
        # distinction only matters for aromatic/unsaturated forms (B: "ine",
        # C: "inine").
        #
        # NOTE: For 6-membered rings with LIGHT heteroatoms (N, O, S, Se, Te,
        # P, B), RDKit can kekulize and set the aromatic flag. A genuinely
        # aromatic ring reaches us with aromatic=True; a merely partially
        # unsaturated ring reaches us with aromatic=False AND has ring
        # double bonds, and we must keep the SATURATED stem (OPSIN rejects
        # HW "azine"/"pyran"-type names for non-aromatic light-heteroatom
        # rings).
        #
        # HOWEVER: for HEAVY heteroatoms (As, Sb, Bi, Ge, Sn, Pb, halogens,
        # Al, Ga, In, Tl, Hg) RDKit cannot kekulize/aromatize the ring even
        # when it is structurally fully unsaturated. In that case the
        # aromatic flag is False despite the ring having the maximum
        # non-cumulative double-bond pattern. For these elements we must
        # also route to the unsaturated stem when endocyclic double bonds
        # are present -- otherwise we emit "arsinane"/"bismane"/"iodinane"
        # when OPSIN expects "arsinine"/"bismine"/"iodinine" (Stage 6 R2-D).
        group_A = set(six_groups.get("A", []))
        group_B = set(six_groups.get("B", []))
        group_C = set(six_groups.get("C", []))
        elements_set = set(elements)

        any_C = bool(elements_set & group_C)
        any_B = bool(elements_set & group_B)

        if any_C:
            grp = stem_data.get("group_C", {})
        elif any_B:
            grp = stem_data.get("group_B", {})
        else:
            grp = stem_data.get("group_A", {})

        # Heavy heteroatoms that RDKit cannot kekulize at 6-ring size.  For
        # these, aromatic=False + has_endocyclic_double_bond=True is the
        # expected state for a fully unsaturated ring (structural Kekulé),
        # and we must pick the unsaturated stem.  Light heteroatoms
        # (N/O/S/Se/Te/P/B) are excluded so that e.g. `N1CC=CC=C1` keeps
        # using the saturated stem (no regression to OPSIN-rejected
        # "azine").
        _HEAVY_NON_KEKULIZABLE_6RING = {
            "F", "Cl", "Br", "I",        # halogens
            "As", "Sb", "Bi",            # heavy pnictogens
            "Ge", "Sn", "Pb",            # heavy group-14
            "Al", "Ga", "In", "Tl",      # group-13 metals
            "Hg",                        # group-12
        }
        use_heavy_unsat_path = (
            has_endocyclic_double_bond
            and bool(elements_set & _HEAVY_NON_KEKULIZABLE_6RING)
        )
        effective_aromatic = aromatic or use_heavy_unsat_path
        return grp.get("unsaturated") if effective_aromatic else grp.get("saturated")

    elif ring_size in (3, 4, 5, 7, 8, 9, 10):
        # Check for N in elements (affects stem choice)
        has_N = "N" in elements
        use_unsaturated = aromatic or has_endocyclic_double_bond
        if use_unsaturated:
            return stem_data.get("unsaturated")
        else:
            if has_N:
                return stem_data.get("saturated_with_N") or stem_data.get("saturated_default")
            else:
                return stem_data.get("saturated_default")

    return None
