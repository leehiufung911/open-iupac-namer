"""
iupac_namer/ring_naming/benzo_fused_bridged.py

Specialised naming for benzo-fused bridged-bicyclic ring systems
(e.g., dezocine's 5,11-methanobenzocyclodecene skeleton).

The generic bridged-VB path yields tricyclo[...] names that collapse the
aromatic benzene into a saturated ring — losing aromaticity and therefore
failing OPSIN round-trip.  This module detects the pattern

    benzene (aromatic 6-ring) ortho-fused to one macrocycle that is
    itself bridged by a small methano/ethano bridge

and emits the IUPAC-preferred name

    <hydro-prefix>-<bridge-locs>-methano-benzocyclo[N]ene

with an explicit atom → locant pin that the engine uses for substituent
locant assignment.  The aromatic benzene carbons keep locants 1–4
(with 4a, 12a as fusion), and the aliphatic ring is numbered
5, 6, 7, …, (N+4), with the bridge atoms continuing after.

Scope guard:
    Only activates when the ring system is:
      - classified "bridged" (may be ambiguous with "fused")
      - contains exactly ONE fully-aromatic 6-membered SSSR ring
      - that ring shares exactly 2 bonded atoms with the rest of the system
      - the non-benzene portion + fusion atoms forms a bicyclic whose two
        paths between the bridgeheads enclose the benzene fusion edge
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.data_loader import get_chain_stem
from iupac_namer.types import Locant, NamedParent, Numbering

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)

# Bridge name for a single-atom bridge (count → IUPAC "bridge-element" prefix).
_BRIDGE_PREFIX: dict[int, str] = {
    1: "methano",
    2: "ethano",
    3: "propano",
    4: "butano",
}


def _find_benzene_ring(
    ring_system: "RingSystem", mol
) -> frozenset[int] | None:
    """Return the atoms of a fully-aromatic 6-membered ring in the system, or None.

    Only returns when there is exactly one such ring (otherwise the system is
    a multi-arene fused system which is out of scope for this helper).
    """
    benzenes: list[frozenset[int]] = []
    for r in ring_system.rings:
        if len(r) != 6:
            continue
        # All-carbon aromatic (benzene).  Heteroaromatic six-rings would
        # need different naming (pyrido-, pyrimido- etc.), out of scope.
        all_arom = True
        all_c = True
        for a in r:
            atom = mol.GetAtomWithIdx(a)
            if not atom.GetIsAromatic():
                all_arom = False
                break
            if atom.GetSymbol() != "C":
                all_c = False
        if all_arom and all_c:
            benzenes.append(r)
    if len(benzenes) == 1:
        return benzenes[0]
    return None


def _fusion_atoms(
    benzene: frozenset[int], ring_system: "RingSystem", mol
) -> tuple[int, int] | None:
    """Return the two fusion atoms (bonded pair shared with rest of system).

    None if the benzene does not share exactly 2 bonded atoms with the rest.
    """
    other_atoms: set[int] = set()
    for r in ring_system.rings:
        if r == benzene:
            continue
        other_atoms.update(r)
    shared = benzene & other_atoms
    if len(shared) != 2:
        return None
    a, b = sorted(shared)
    bond = mol.GetBondBetweenAtoms(a, b)
    if bond is None:
        return None
    return (a, b)


def _analyze_sub_bicyclic(
    sub_atoms: frozenset[int],
    fusion: tuple[int, int],
    mol,
) -> tuple[int, int, list[list[int]]] | None:
    """Treat sub_atoms (non-benzene atoms of the system + fusion atoms) as a
    bridged bicyclic and return (bh1, bh2, sorted_paths).

    Where bh1, bh2 are the two "high-valence" atoms (each in 2+ sub-rings or
    each a bridgehead with at least 3 sub-atom neighbours).  Paths are sorted
    descending by length.

    Returns None if sub_atoms is not a bicyclic.
    """
    # Degree within sub_atoms
    deg: dict[int, int] = {}
    for a in sub_atoms:
        d = 0
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in sub_atoms:
                d += 1
        deg[a] = d
    # Bridgeheads in a bicyclic have degree 3 within the subgraph
    bridgeheads = [a for a, d in deg.items() if d >= 3]
    if len(bridgeheads) < 2:
        return None

    # Find all simple paths between each candidate pair
    def _paths(start: int, end: int) -> list[list[int]]:
        all_paths: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            if curr == end and len(path) > 1:
                all_paths.append(path)
                continue
            for nb in mol.GetAtomWithIdx(curr).GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx not in sub_atoms:
                    continue
                if nb_idx in path:
                    continue
                stack.append((nb_idx, path + [nb_idx]))
        return all_paths

    # Try each bridgehead pair; a valid bicyclic has exactly 3 disjoint paths
    # covering all sub_atoms (bicyclo has 3 bridges).
    best: tuple[int, int, list[list[int]]] | None = None
    for i in range(len(bridgeheads)):
        for j in range(i + 1, len(bridgeheads)):
            bh1, bh2 = bridgeheads[i], bridgeheads[j]
            paths = _paths(bh1, bh2)
            # Keep paths that don't pass through other bridgeheads
            other_bhs = set(bridgeheads) - {bh1, bh2}
            valid = [p for p in paths if not any(x in other_bhs for x in p[1:-1])]
            if len(valid) != 3:
                continue
            covered: set[int] = set()
            for p in valid:
                covered.update(p)
            if covered != set(sub_atoms):
                continue
            valid_sorted = sorted(valid, key=len, reverse=True)
            # We want the configuration where the fusion edge is on one of
            # the main macrocyclic paths (longest or middle), not entirely
            # inside a bridge.
            fusion_set = set(fusion)
            # Identify which path covers the fusion edge
            fusion_on_path: list[int] = []
            for pi, p in enumerate(valid_sorted):
                p_pairs = {
                    tuple(sorted((p[k], p[k + 1]))) for k in range(len(p) - 1)
                }
                if tuple(sorted(fusion)) in p_pairs:
                    fusion_on_path.append(pi)
            # Benzo fusion must lie on exactly one of the two macrocycle
            # paths (not on the shortest = bridge) for the naming to make
            # sense.  If fusion is on a path, and that path is not the
            # shortest bridge, accept.
            if not fusion_on_path:
                continue
            # Don't accept if fusion edge is only on the shortest bridge.
            short_idx = len(valid_sorted) - 1
            if all(i == short_idx for i in fusion_on_path):
                continue
            best = (bh1, bh2, valid_sorted)
            break
        if best is not None:
            break
    return best


def _pick_cycle_stem(size: int) -> str | None:
    """Return the ``cyclo...ene`` stem for a size N macrocycle.

    benzocyclodecene, benzocycloundecene, etc.  Uses the chain stem with
    an "ene" suffix.  Returns the stem, e.g. "decen" (for -ene before '-3-ol').
    """
    s = get_chain_stem(size)
    if s is None:
        return None
    # OPSIN uses "-en" before hyphen and a locant; we want ".../<stem>en"
    # Example: "decane" → "dec", we want "deca" + "n" + "ene" → "decen"
    # Actually chain_stems returns 'dec' for 10.  For 'decene' we need 'dec' + 'ene'.
    return s


def name_benzo_fused_bridged(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Detect benzo-fused-bridged-bicyclic and emit IUPAC-preferred name.

    Returns [] when the pattern doesn't match.
    """
    # Only bridged (or ambiguous-bridged-with-fused-alt) systems qualify
    if ring_system.type != "bridged":
        return []

    benzene = _find_benzene_ring(ring_system, mol)
    if benzene is None:
        return []

    fusion = _fusion_atoms(benzene, ring_system, mol)
    if fusion is None:
        return []

    all_atoms = set(ring_system.atom_indices)
    benzene_only = benzene - set(fusion)  # 4 non-fusion benzene atoms
    # sub_atoms = non-benzene atoms of system + 2 fusion atoms
    sub_atoms = frozenset(all_atoms - benzene_only)
    if len(sub_atoms) < 5:
        return []  # too small, no meaningful bridge

    analysis = _analyze_sub_bicyclic(sub_atoms, fusion, mol)
    if analysis is None:
        return []

    bh1, bh2, paths = analysis
    # paths sorted desc: [longest, middle, shortest]
    # The macrocycle consists of the two paths that include the fusion edge.
    # Actually the macrocycle = longest + middle (both share bridgeheads
    # bh1/bh2 but the fusion edge sits on one of them).  The shortest path
    # is the bridge.
    macrocycle_paths = paths[:2]
    bridge_path = paths[-1]

    fusion_set = set(fusion)
    # Which of the two macrocycle paths contains the fusion edge?
    def _path_has_fusion(p: list[int]) -> bool:
        for k in range(len(p) - 1):
            pair = tuple(sorted((p[k], p[k + 1])))
            if pair == tuple(sorted(fusion)):
                return True
        return False

    fusion_paths = [p for p in macrocycle_paths if _path_has_fusion(p)]
    if len(fusion_paths) != 1:
        return []
    fusion_path = fusion_paths[0]
    other_path = [p for p in macrocycle_paths if p is not fusion_path][0]

    # Macrocycle size = (fusion_path length - 1) + (other_path length - 1)
    #                 = total atoms traversed in the full cycle - 1 shared vertex
    # Actually macrocycle = full aliphatic ring containing the fusion edge and
    # the other_path going from bh1 back to bh2.  Size = number of distinct
    # atoms in fusion_path ∪ other_path.
    macro_atoms = set(fusion_path) | set(other_path)
    macro_size = len(macro_atoms)

    # Bridge size = intermediate atoms count in the shortest bridge
    bridge_size = len(bridge_path) - 2

    if bridge_size < 1 or bridge_size > 4:
        # Too short (no bridge) or too long (no retained -ano- name)
        return []

    cycle_stem = _pick_cycle_stem(macro_size)
    if cycle_stem is None:
        return []

    bridge_prefix = _BRIDGE_PREFIX.get(bridge_size)
    if bridge_prefix is None:
        return []

    # ------------------------------------------------------------------
    # Build the locant assignment.
    #
    # Benzocyclodecene numbering (IUPAC):
    #   Aromatic benzene carbons: 1, 2, 3, 4
    #   Fusion atoms: 4a (adjacent to 4) and (macro_size+4)a (adjacent to 5)
    #   Aliphatic ring: 5, 6, 7, ..., (macro_size+4)
    #   Bridge atoms follow: (macro_size+5), (macro_size+6), ...
    #
    # We need to pick the numbering that gives bridgeheads the lowest locant
    # pair, and substituents low locants.  Standard IUPAC convention puts
    # the aliphatic-ring locant 5 at the carbon adjacent to a fusion atom.
    # ------------------------------------------------------------------

    # Number the benzene: start at a benzene atom adjacent to one of the
    # fusion atoms (call it f_high), going AWAY from fusion → locants 1..4,
    # then fusion atom f_high = 4a, then fusion atom f_low = (macro_size+4)a.
    # The "high" fusion atom is the one that gets locant 4a (bonded to 4),
    # and the "low" fusion atom (locant (macro+4)a) is bonded to locant 5.

    # Walk the benzene ring starting from a non-fusion atom adjacent to a
    # fusion atom.  The benzene has 4 non-fusion atoms arranged as a chain
    # between the two fusion atoms.
    f_a, f_b = fusion
    # Find the chain of 4 non-fusion benzene atoms from neighbour-of-f_a to
    # neighbour-of-f_b.
    def _walk_benzene(start_fusion: int, end_fusion: int) -> list[int] | None:
        # BFS through benzene from start_fusion, excluding end_fusion and the
        # direct bond to end_fusion.  Return path [non_fus_1, ..., non_fus_4].
        for nb in mol.GetAtomWithIdx(start_fusion).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx == end_fusion:
                continue
            if nb_idx not in benzene:
                continue
            # Walk forward
            path = [nb_idx]
            prev = start_fusion
            curr = nb_idx
            while True:
                next_atom = None
                for nb2 in mol.GetAtomWithIdx(curr).GetNeighbors():
                    nb2_idx = nb2.GetIdx()
                    if nb2_idx in benzene and nb2_idx != prev:
                        next_atom = nb2_idx
                        break
                if next_atom is None or next_atom == end_fusion:
                    return path
                if next_atom in path:
                    return None
                path.append(next_atom)
                prev = curr
                curr = next_atom
        return None

    # Pick orientation: we want the locant that eventually becomes 5 (the
    # carbon adjacent to the macrocycle path, on the aliphatic side) to be
    # BH-adjacent so bridgeheads get low locants.  Try both orientations.
    # Orientation 1: f_a = "low" fusion (adjacent to 5), f_b = "high" (4a).
    # Orientation 2: swap.

    # Determine which fusion atom is adjacent (via the macrocycle path's
    # aliphatic side) to the bridgehead that we want to be "5" - adjacent.
    # Actually bridgeheads bh1, bh2 are on the aliphatic ring.  The aliphatic
    # locants go 5, 6, ..., (macro+4), traversing the ring.  One of bh1/bh2
    # should get a lower locant than the other.

    best: tuple[dict[int, int], int, int] | None = None  # (locant_map, bh_lo, bh_hi)

    for f_low, f_high in ((f_a, f_b), (f_b, f_a)):
        # Benzene chain: 4 non-fusion atoms in order starting from neighbour
        # of f_high (locant 1) to neighbour of f_low (locant 4).
        bz_chain = _walk_benzene(f_high, f_low)
        if bz_chain is None or len(bz_chain) != 4:
            continue

        # Build the aliphatic-side walk: from f_low, go around the aliphatic
        # ring (the path NOT containing fusion edge) back to f_high.
        # other_path runs bh1 → ... → bh2 (bridgeheads).  The fusion_path
        # runs bh1 → ... (through fusion edge) ... → bh2.
        # We need to assemble a walk of macrocycle atoms: f_low,
        # its non-fusion aliphatic neighbour, around through bh1/bh2, back
        # to f_high.

        # The macrocycle has atoms macro_atoms.  Walk starting at f_low,
        # going to its ONLY non-benzene, in-macro neighbour (i.e. a
        # non-fusion aliphatic neighbour in macro_atoms), then continue
        # hopping along macro_atoms, avoiding f_high (which is also in
        # macro_atoms — it's the OTHER fusion atom and it must be the
        # END of the walk).

        def _walk_macro(start: int, end: int) -> list[int] | None:
            visited = {start}
            curr = start
            path = [start]
            # We expect macro_size - 1 hops to reach end.
            for _ in range(macro_size):
                next_atoms: list[int] = []
                for nb in mol.GetAtomWithIdx(curr).GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx not in macro_atoms:
                        continue
                    if nb_idx in visited:
                        continue
                    # Don't take the fusion edge short-cut
                    if (curr, nb_idx) == tuple(sorted((f_low, f_high))) or \
                       (nb_idx, curr) == tuple(sorted((f_low, f_high))):
                        continue
                    next_atoms.append(nb_idx)
                if not next_atoms:
                    break
                # A simple ring has a unique next atom per step; if ambiguity,
                # pick the one that doesn't end the walk prematurely.
                # For our bicyclic sub-system, at bridgeheads there are 2
                # macro neighbours (excluding the already-visited one) + 1
                # bridge neighbour (not in macro_atoms anyway).
                # Choose ANY candidate not equal to end (we want end last).
                chosen = None
                for c in next_atoms:
                    if c != end:
                        chosen = c
                        break
                if chosen is None:
                    chosen = next_atoms[0]
                path.append(chosen)
                visited.add(chosen)
                curr = chosen
                if curr == end:
                    return path
            if path[-1] == end:
                return path
            return None

        macro_walk = _walk_macro(f_low, f_high)
        if macro_walk is None:
            continue
        if len(macro_walk) != macro_size:
            continue
        if macro_walk[0] != f_low or macro_walk[-1] != f_high:
            continue

        # Locant assignment
        locant_map: dict[int, int] = {}
        # Benzene: 1..4 on bz_chain, 4a on f_high, (macro_size+4)a on f_low
        for i, a in enumerate(bz_chain):
            locant_map[a] = i + 1  # 1..4
        # f_high = 4a; f_low = (macro_size+4)a
        # Aliphatic ring: macro_walk[1..-2] are atoms 5..(macro_size+3),
        # where macro_walk[0]=f_low and macro_walk[-1]=f_high, and we want
        # locants starting at 5 adjacent to f_low.
        # macro_walk has macro_size atoms: [f_low, a5, a6, ..., f_high].
        # So the aliphatic atoms (macro_size - 2 of them) get locants
        # 5, 6, ..., (5 + macro_size - 3) = (macro_size + 2).
        # Plus f_high = 4a, f_low = (macro_size + 4)a as fusion locants.
        for i, a in enumerate(macro_walk[1:-1]):  # skip endpoints
            locant_map[a] = 5 + i  # 5, 6, ...
        # Fusion locants are compound — we won't put them in the integer
        # locant_map since they're not substituent-eligible in practice.
        # Leaving them unmapped means substituents attached there would
        # fall through, but fusion atoms typically carry no substituents
        # for benzo-fused systems (their valence is filled by 3 ring bonds).

        # Bridge atoms: bridge_path has length = bridge_size + 2 (bh1 at start,
        # bh2 at end, intermediates in between).  We number the bridge
        # intermediates starting from (macro_size + 3).
        # Example for dezocine: macro=10, bridge_size=1 → locant 13.
        # Number from the endpoint that has the LOWER bridgehead locant.
        bh_a, bh_b = bridge_path[0], bridge_path[-1]
        la = locant_map.get(bh_a)
        lb = locant_map.get(bh_b)
        if la is None or lb is None:
            continue
        if la <= lb:
            bridge_seq = bridge_path[1:-1]
        else:
            bridge_seq = list(reversed(bridge_path[1:-1]))
        next_loc = macro_size + 3
        for a in bridge_seq:
            locant_map[a] = next_loc
            next_loc += 1

        # Bridgehead locants in main ring
        bh_locs = sorted([la, lb])

        if best is None:
            best = (locant_map, bh_locs[0], bh_locs[1])
        else:
            # Prefer the orientation with the lower bridgehead locant pair
            if (bh_locs[0], bh_locs[1]) < (best[1], best[2]):
                best = (locant_map, bh_locs[0], bh_locs[1])

    if best is None:
        return []

    locant_map, bh_lo, bh_hi = best

    # Build the hydro prefix: locants of saturated ring atoms.
    # For "octahydro-5,6,7,8,9,10,11,12" in a benzocyclodecene with
    # fully-aromatic benzene + fully-saturated macrocycle, the hydro
    # locants are the aliphatic ring atoms (5..(macro_size+2)).
    hydro_locants = list(range(5, macro_size + 3))
    hydro_count = len(hydro_locants)
    from iupac_namer.data_loader import get_multiplier
    hydro_mult = get_multiplier(hydro_count) or ""
    # Convert "di" → "di", "hexa" → "hexa", etc.  For 8 → "octa".
    hydro_loc_str = ",".join(str(l) for l in hydro_locants)

    # Bridge descriptor: <bh_lo>,<bh_hi>-methano
    bridge_desc = f"{bh_lo},{bh_hi}-{bridge_prefix}"

    # Parent ene stem: "benzocyclo<stem>ene"
    # e.g. cycle_stem="dec" → "benzocyclodecene"
    parent_core = f"benzocyclo{cycle_stem}ene"
    # stem form (for "-N-ol", "-N-amine") strips terminal "e".
    parent_stem = f"benzocyclo{cycle_stem}en"

    # Full hydride name:
    #   <hydro_loc_str>-<hydro_mult>hydro-<bridge_desc>benzocyclo<stem>ene
    name_str = (
        f"{hydro_loc_str}-{hydro_mult}hydro-"
        f"{bridge_desc}{parent_core}"
    )
    stem = (
        f"{hydro_loc_str}-{hydro_mult}hydro-"
        f"{bridge_desc}{parent_stem}"
    )

    # Build Numbering with all integer locants (fusion atoms, which carry
    # compound locants 4a/12a, are not placed in the substituent map — they
    # are never substituent carriers in our detected pattern because they
    # are aromatic CH atoms already fully bonded).
    #
    # Verify: no fusion atom has a substituent.  If it does, give up.
    benzene_fusion = set(fusion)
    for fa in benzene_fusion:
        atom = mol.GetAtomWithIdx(fa)
        # Its neighbours outside the ring system indicate an external substituent.
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in all_atoms:
                logger.debug("Benzo fusion atom %d has external substituent; skipping", fa)
                return []

    # Build assignments for the Numbering object.
    # Include fusion atoms with compound locants so the engine has a full
    # atom → locant map (useful for later passes that enumerate locants).
    # f_high = 4a, f_low = (macro_size+4)a.  Determine which is which from
    # the 1..4 chain adjacency.
    # Find the benzene neighbour of each fusion atom that has locant 1 vs 4.
    f_high_atom = None
    f_low_atom = None
    for fa in benzene_fusion:
        for nb in mol.GetAtomWithIdx(fa).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in locant_map:
                loc = locant_map[nb_idx]
                if loc == 1:
                    f_high_atom = fa
                elif loc == 4:
                    f_low_atom = fa
    # But wait: locant 4 is on benzene; locant 5 is on aliphatic.  f_low is
    # adjacent to locant 5.  f_high is adjacent to locants 1 AND 4a's partner
    # on benzene (i.e. locant 4 in the benzene chain).  Actually f_high is
    # adjacent to benzene-locants 1 AND f_low, and to aliphatic locant via
    # being a fusion atom.  Let me re-derive.
    # In benzocyclodecene:
    #   locants 1-2-3-4 are the 4 non-fusion benzene C's in order.
    #   locant 4a is a fusion C, bonded to 4 and to 5 (on aliphatic side).
    #   BUT 5 is adjacent to f_low (not f_high)!
    # So: f_high is bonded to 1 and to (macro_size+2) (= 12 for N=10).
    #     f_low is bonded to 4 and to 5.
    # Hmm, let me re-check from the OPSIN structure above:
    #   5,6,7,8,9,10,11,12-octahydrobenzocyclodecene → c1ccc2c(c1)CCCCCCCC2
    #   Numbering: 1-2-3-4-4a-5-6-7-8-9-10-11-12-12a-1
    #   (ring of 14 atoms)
    # Here 4a is between 4 and 5, 12a is between 12 and 1.  So 4a is adjacent
    # to both 4 and 5 ⇒ 4a = f_low (bonded to aliphatic 5).
    # And 12a is adjacent to both 12 and 1 ⇒ 12a = f_high (bonded to
    # aliphatic 12 and benzene 1).
    # So: f_low was already the one adjacent to aliphatic 5 (locant 5 in
    # macro_walk[1]), and its benzene neighbour with locant 4 is the end of
    # the bz_chain.  f_high is adjacent to bz_chain[0] (locant 1) and to
    # macro_walk[-2] (locant macro_size+2).
    # So we should swap: f_low_atom = 4a = adjacent to 4 and 5.
    #                    f_high_atom = 12a = adjacent to 1 and 12.
    # Our detection above IS consistent: the fusion atom with a neighbour at
    # locant 4 is f_low (gets locant 4a), and at locant 1 is f_high (gets
    # (macro+4)a).

    # Actually our earlier walk assigned:
    #   bz_chain = _walk_benzene(f_high, f_low): 4 atoms from neighbour of
    #              f_high (locant 1) to neighbour of f_low (locant 4).
    # So the benzene atom at locant 1 is adjacent to f_high; locant 4 is
    # adjacent to f_low.  So:
    #   f_low gets compound locant "4a"
    #   f_high gets compound locant "(macro_size+4)a" = e.g. "12a" for N=10
    # But the f_high_atom/f_low_atom we derived: checking neighbour's
    # integer locant == 1 ⇒ that fusion atom is f_high.  == 4 ⇒ f_low.
    # So f_high_atom → compound locant (macro_size+4) with 'a' suffix.
    #    f_low_atom  → compound locant 4 with 'a' suffix.

    assignments: list[tuple[int, Locant]] = []
    for atom_idx, loc in sorted(locant_map.items(), key=lambda x: x[1]):
        assignments.append((atom_idx, Locant.numeric(loc)))
    # Append fusion atoms with compound locants (position in sort by numeric
    # value of leading digit).
    if f_low_atom is not None:
        assignments.append((f_low_atom, Locant.numeric(4, "a")))
    if f_high_atom is not None:
        assignments.append((f_high_atom, Locant.numeric(macro_size + 4, "a")))

    # Sort final assignments by locant order (compound "4a" comes after "4").
    assignments.sort(key=lambda x: x[1])

    locant_set = tuple(loc for _, loc in assignments)
    numbering = Numbering(_assignments=tuple(assignments), locant_set=locant_set)

    return [NamedParent(
        candidate=candidate,
        name=name_str,
        stem=stem,
        alkyl_stem=None,
        naming_method="benzo_fused_bridged",
        indicated_hydrogen=None,
        numbering_options=(numbering,),
        ring_unsaturation_bonds=None,
    )]
